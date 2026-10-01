"""PromptSRC + HyperPrompt model for cross-scene HSI classification."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

from clip_utils import clip_preprocess
from trainers.hsi_rgb_adapter import HSIRGBAdapter
from trainers.patch_coop import CoOpTextEncoder
from trainers.pclra_utils import inject_pclra_text, lora_param_count
from trainers.sam_backbone import SAMBackbone
from trainers.upsamplers import get_upsampler


class HSIHyperPromptPromptSRC(nn.Module):
    """Apply HyperPrompt to PromptSRC with independent patch/pixel prompts."""

    def __init__(
        self,
        in_channels,
        num_classes,
        classnames,
        clip_name,
        sam_model,
        ctx_len_patch,
        ctx_len_pixel,
        class_token_position="end",
        csc=False,
        ctx_init=None,
        temperature=1.0,
        upsampler_type="bilinear",
        upsampler_weights=None,
        tcdm_last_k_layers=4,
        use_cls_only=False,
        pclra_enabled=False,
        pclra_rank=0,
        pclra_alpha=None,
        pclra_prompt_dim=256,
        pclra_last_n_layers=-1,
        pclra_target_keys=("q_proj", "v_proj"),
        pclra_dropout=0.0,
        pclra_tau=0.07,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = float(temperature)

        self.rgb_proj = HSIRGBAdapter(in_channels)

        self.clip_branch1 = CLIPModel.from_pretrained(clip_name)
        self.tokenizer1 = CLIPTokenizer.from_pretrained(clip_name)
        self.clip_branch2 = CLIPModel.from_pretrained(clip_name)
        self.tokenizer2 = CLIPTokenizer.from_pretrained(clip_name)
        for clip_model in (self.clip_branch1, self.clip_branch2):
            for parameter in clip_model.parameters():
                parameter.requires_grad = False

        clip_dim = self.clip_branch1.config.projection_dim
        self.text_encoder1 = CoOpTextEncoder(
            clip_model=self.clip_branch1,
            tokenizer=self.tokenizer1,
            classnames=classnames,
            n_ctx=ctx_len_patch,
            ctx_init=ctx_init,
            class_token_position=class_token_position,
            csc=csc,
        )
        self.text_encoder2 = CoOpTextEncoder(
            clip_model=self.clip_branch2,
            tokenizer=self.tokenizer2,
            classnames=classnames,
            n_ctx=ctx_len_pixel,
            ctx_init=ctx_init,
            class_token_position=class_token_position,
            csc=csc,
        )

        self.sam = SAMBackbone(
            sam_model,
            tcdm_last_k_layers=tcdm_last_k_layers,
            use_cls_only=use_cls_only,
        )
        self.sam_proj = nn.Linear(768, clip_dim)
        self.reconstruction_head = nn.Conv2d(768, 3, kernel_size=1)
        self.upsampler = get_upsampler(
            upsampler_type,
            dim=3,
            weight_path=upsampler_weights,
            device="cpu",
        )
        for parameter in self.upsampler.parameters():
            parameter.requires_grad = True

        def image_adapter():
            return nn.Sequential(
                nn.Linear(clip_dim, clip_dim // 2),
                nn.BatchNorm1d(clip_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(clip_dim // 2, clip_dim),
            )

        self.image_adapter1 = image_adapter()
        self.image_adapter2 = image_adapter()

        # Build frozen PromptSRC teachers before PCLRA replaces the student
        # text-attention projections.
        zero_shot_text_features1 = self._zero_shot_text_features(
            self.clip_branch1, self.tokenizer1, classnames
        )
        zero_shot_text_features2 = self._zero_shot_text_features(
            self.clip_branch2, self.tokenizer2, classnames
        )

        self.pclra = None
        if pclra_enabled and pclra_rank > 0:
            def get_layers(encoder):
                return encoder.clip.text_model.encoder.layers

            def get_prompt_tokens(encoder):
                return encoder.ctx

            self.pclra = inject_pclra_text(
                encoder1=self.text_encoder1,
                encoder2=self.text_encoder2,
                get_layers=get_layers,
                get_prompt_tokens=get_prompt_tokens,
                hidden_dim=self.text_encoder1.hidden_dim,
                r=pclra_rank,
                alpha=pclra_alpha if pclra_alpha is not None else float(pclra_rank),
                prompt_dim=pclra_prompt_dim,
                last_n_layers=pclra_last_n_layers,
                target_keys=pclra_target_keys,
                dropout=pclra_dropout,
                tau=pclra_tau,
            )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.alpha_mlp = nn.Sequential(
            nn.Linear(clip_dim * 2, clip_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(clip_dim // 4, 1),
        )

        self.register_buffer(
            "zero_shot_text_features1",
            zero_shot_text_features1,
        )
        self.register_buffer(
            "zero_shot_text_features2",
            zero_shot_text_features2,
        )
        lora_param_count(self)

    @staticmethod
    def _zero_shot_text_features(clip_model, tokenizer, classnames):
        features = []
        with torch.no_grad():
            for classname in classnames:
                tokens = tokenizer(
                    [f"a photo of a {classname}"],
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                )
                outputs = clip_model.text_model(
                    input_ids=tokens.input_ids.to(clip_model.device),
                    attention_mask=tokens.attention_mask.to(clip_model.device),
                )
                features.append(clip_model.text_projection(outputs.pooler_output))
        return torch.cat(features, dim=0)

    def encode_mean_prompt(self):
        if self.pclra is not None:
            return self.pclra.encode_mean_prompt()
        return self.text_encoder1(), self.text_encoder2()

    def forward(self, images):
        rgb = self.rgb_proj(images)
        rgb_clip = clip_preprocess(rgb, image_resolution=224)

        vision_outputs = self.clip_branch1.vision_model(
            pixel_values=rgb_clip,
            return_dict=True,
            output_hidden_states=True,
        )
        patch_base = self.clip_branch1.visual_projection(vision_outputs.pooler_output)
        zs_img_feat1 = F.normalize(patch_base, dim=-1)
        img_feat1 = F.normalize(self.image_adapter1(patch_base), dim=-1)

        if self.pclra is not None:
            text_feat1, text_feat2 = self.pclra()
        else:
            text_feat1, text_feat2 = self.text_encoder1(), self.text_encoder2()
        text_feat1 = F.normalize(text_feat1, dim=-1)
        text_feat2 = F.normalize(text_feat2, dim=-1)

        rgb_sam = F.interpolate(rgb, size=(224, 224), mode="bicubic", align_corners=False)
        sam_feat = self.sam(rgb_sam, tcdm_hidden_states=vision_outputs.hidden_states)
        sam_spatial = sam_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 14, 14)
        recon_rgb = self.upsampler(self.reconstruction_head(sam_spatial), rgb_sam)

        pixel_base = self.sam_proj(sam_feat)
        zs_img_feat2 = F.normalize(pixel_base, dim=-1)
        img_feat2 = F.normalize(self.image_adapter2(pixel_base), dim=-1)

        zs_text_feat1 = F.normalize(self.zero_shot_text_features1, dim=-1)
        zs_text_feat2 = F.normalize(self.zero_shot_text_features2, dim=-1)
        scale = self.logit_scale.exp().clamp(max=20)
        logits1 = scale * (img_feat1 @ text_feat1.t())
        logits2 = scale * (img_feat2 @ text_feat2.t())
        teacher_logits1 = scale * (zs_img_feat1 @ zs_text_feat1.t())
        teacher_logits2 = scale * (zs_img_feat2 @ zs_text_feat2.t())

        alpha = torch.sigmoid(self.alpha_mlp(torch.cat([img_feat1, img_feat2], dim=-1)))
        logits = alpha * F.log_softmax(logits1, dim=-1) + (1 - alpha) * F.log_softmax(logits2, dim=-1)

        return (
            logits1, img_feat1, text_feat1,
            logits2, img_feat2, text_feat2,
            logits, recon_rgb, rgb_sam,
            teacher_logits1, teacher_logits2,
            zs_img_feat1, zs_img_feat2,
            zs_text_feat1, zs_text_feat2,
        )
