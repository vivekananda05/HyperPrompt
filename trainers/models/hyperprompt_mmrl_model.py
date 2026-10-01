"""
HyperPrompt MMRL model with Prompt-Conditioned Low-Rank Adapter.

Branch 1 uses a CLIP vision/text branch with MMRL prompt learning.
Branch 2 uses a SAM pixel branch with MMRL-style visual conditioning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel, CLIPTokenizer

from trainers.hsi_rgb_adapter import HSIRGBAdapter
from trainers.sam_backbone import SAMBackbone
from trainers.patch_mmrl import MultiModalRepresentationLearner, TextEncoder_MMRL, VisualEncoder_MMRL
from trainers.pclra_utils import inject_pclra_text, lora_param_count
from clip_utils import clip_preprocess
from trainers.upsamplers import get_upsampler


class HSIHyperPromptMMRL(nn.Module):
    """HyperPrompt HSI classification model built around MMRL prompts."""

    def __init__(
        self,
        in_channels,
        num_classes,
        classnames,
        clip_name,
        sam_model,
        ctx_len_patch,
        ctx_len_pixel,
        class_token_position,
        csc,
        ctx_init,
        n_rep_tokens=8,
        rep_dim=512,
        n_layers=12,
        upsampler_type="bilinear",
        upsampler_weights=None,
        tcdm_last_k_layers=4,
        pclra_enabled: bool = False,
        pclra_rank: int = 0,
        pclra_alpha: float = None,
        pclra_prompt_dim: int = 256,
        pclra_last_n_layers: int = -1,
        pclra_target_keys=("q_proj", "v_proj"),
        pclra_dropout: float = 0.0,
        pclra_tau: float = 0.07,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.classnames = classnames
        self.ctx_len_patch = ctx_len_patch
        self.ctx_len_pixel = ctx_len_pixel
        self.class_token_position = class_token_position
        self.csc = csc

        self.pclra_rank = pclra_rank
        self.pclra_alpha = pclra_alpha if pclra_alpha is not None else float(pclra_rank)

        self.rgb_proj = HSIRGBAdapter(in_channels)

        self.clip_branch1 = CLIPModel.from_pretrained(clip_name).float()
        self.tokenizer1 = CLIPTokenizer.from_pretrained(clip_name)
        for p in self.clip_branch1.parameters():
            p.requires_grad = False

        clip_dim = int(self.clip_branch1.config.projection_dim)

        self.representation_learner1 = MultiModalRepresentationLearner(
            tokenizer=self.tokenizer1,
            classnames=self.classnames,
            clip_model=self.clip_branch1,
            n_rep_tokens=n_rep_tokens,
            rep_dim=rep_dim,
            n_layers=n_layers,
            ctx_init=ctx_init,
            class_token_position=class_token_position,
            csc=csc,
        )
        self.text_encoder1 = TextEncoder_MMRL(clip_model=self.clip_branch1)
        self.visual_encoder1 = VisualEncoder_MMRL(clip_model=self.clip_branch1)

        self.sam = SAMBackbone(sam_model, tcdm_last_k_layers=tcdm_last_k_layers)
        self.sam_proj = nn.Linear(768, clip_dim)
        self.reconstruction_head = nn.Conv2d(768, 3, kernel_size=1)
        self.upsampler = get_upsampler(
            upsampler_type,
            dim=3,
            weight_path=upsampler_weights,
            device="cpu",
        )
        for p in self.upsampler.parameters():
            p.requires_grad = True

        self.clip_branch2 = CLIPModel.from_pretrained(clip_name).float()
        self.tokenizer2 = CLIPTokenizer.from_pretrained(clip_name)
        for p in self.clip_branch2.parameters():
            p.requires_grad = False

        self.representation_learner2 = MultiModalRepresentationLearner(
            tokenizer=self.tokenizer2,
            classnames=self.classnames,
            clip_model=self.clip_branch2,
            n_rep_tokens=n_rep_tokens,
            rep_dim=rep_dim,
            n_layers=n_layers,
            ctx_init=ctx_init,
            class_token_position=class_token_position,
            csc=csc,
        )
        self.text_encoder2 = TextEncoder_MMRL(clip_model=self.clip_branch2)

        self.visual_prompt_to_sam2 = nn.Linear(self.clip_branch2.vision_model.config.hidden_size, 768)
        self.sam_prompt_scale2 = nn.Linear(768, 768)
        self.sam_prompt_bias2 = nn.Linear(768, 768)
        nn.init.normal_(self.sam_prompt_scale2.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.sam_prompt_scale2.bias)
        nn.init.normal_(self.sam_prompt_bias2.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.sam_prompt_bias2.bias)

        prompts = [f"{text.replace('_', ' ')}." for text in classnames]
        tokenized = self.tokenizer1(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        self.register_buffer("tokenized_prompts", tokenized.input_ids)
        with torch.no_grad():
            prompt_embeddings = self.clip_branch1.text_model.embeddings.token_embedding(
                self.tokenized_prompts
            ).type(self.clip_branch1.dtype)
        self.register_buffer("prompt_embeddings", prompt_embeddings)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.alpha_mlp = nn.Sequential(
            nn.Linear(clip_dim * 2, clip_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(clip_dim // 4, 1),
        )

        self.pclra = None
        if pclra_enabled and pclra_rank > 0:
            def _get_layers(enc):
                return enc.text_model.encoder.layers

            def _get_prompt_tokens(enc):
                if enc is self.text_encoder1:
                    return self.representation_learner1.compound_rep_tokens
                if enc is self.text_encoder2:
                    return self.representation_learner2.compound_rep_tokens
                raise AttributeError("Unable to locate MMRL prompt tokens for PCLRA")

            self.pclra = inject_pclra_text(
                encoder1=self.text_encoder1,
                encoder2=self.text_encoder2,
                get_layers=_get_layers,
                get_prompt_tokens=_get_prompt_tokens,
                hidden_dim=self.text_encoder1.text_model.config.hidden_size,
                r=pclra_rank,
                alpha=pclra_alpha if pclra_alpha is not None else float(pclra_rank),
                prompt_dim=pclra_prompt_dim,
                last_n_layers=pclra_last_n_layers,
                target_keys=pclra_target_keys,
                dropout=pclra_dropout,
                tau=pclra_tau,
            )

        lora_param_count(self)

    def _encode_sam_with_prompts(self, sam_feat, compound_rep_tokens_visual):
        if isinstance(compound_rep_tokens_visual, (list, tuple)):
            compound_rep_tokens_visual = torch.stack(compound_rep_tokens_visual, dim=0)
        prompt_tokens = compound_rep_tokens_visual.mean(dim=0).mean(dim=0)
        prompt_tokens = self.visual_prompt_to_sam2(prompt_tokens)
        prompt_tokens = prompt_tokens.to(sam_feat.dtype).unsqueeze(0)

        scale = 1.0 + 0.1 * torch.tanh(self.sam_prompt_scale2(prompt_tokens))
        bias = 0.1 * torch.tanh(self.sam_prompt_bias2(prompt_tokens))
        sam_feat_mod = sam_feat * scale + bias
        img_feat = F.normalize(self.sam_proj(sam_feat_mod), dim=-1)
        return img_feat

    def forward(self, images):
        rgb = self.rgb_proj(images)
        rgb_clip = clip_preprocess(rgb, image_resolution=224)

        vision_outputs1 = self.clip_branch1.vision_model(
            pixel_values=rgb_clip,
            return_dict=True,
            output_hidden_states=True,
        )
        pooled1 = vision_outputs1.pooler_output
        patch_hidden_states1 = vision_outputs1.hidden_states
        img_feat1_frozen = F.normalize(self.clip_branch1.visual_projection(pooled1), dim=-1)

        compound_rep_tokens_text1, compound_rep_tokens_visual1 = self.representation_learner1()
        if self.pclra is not None:
            text_feat1 = self.pclra(
                enc1_kwargs={
                    "prompts": self.prompt_embeddings,
                    "tokenized_prompts": self.tokenized_prompts,
                    "compound_rep_tokens_text": compound_rep_tokens_text1,
                },
                enc2_kwargs={
                    "prompts": self.prompt_embeddings,
                    "tokenized_prompts": self.tokenized_prompts,
                    "compound_rep_tokens_text": compound_rep_tokens_text1,
                },
            )[0]
        else:
            text_feat1 = self.text_encoder1(
                self.prompt_embeddings,
                self.tokenized_prompts,
                compound_rep_tokens_text1,
            )

        _, img_feat1 = self.visual_encoder1(
            pixel_values=rgb_clip,
            compound_rep_tokens_visual=compound_rep_tokens_visual1,
        )

        rgb_sam = F.interpolate(rgb, size=(224, 224), mode="bicubic", align_corners=False)
        sam_feat = self.sam(rgb_sam, tcdm_hidden_states=patch_hidden_states1)
        img_feat2_frozen = F.normalize(self.sam_proj(sam_feat), dim=-1)
        sam_feat_spatial = sam_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 14, 14)
        recon_lowres = self.reconstruction_head(sam_feat_spatial)
        recon_rgb = self.upsampler(recon_lowres, rgb_sam)

        compound_rep_tokens_text2, compound_rep_tokens_visual2 = self.representation_learner2()
        if self.pclra is not None:
            text_feat2 = self.pclra(
                enc1_kwargs={
                    "prompts": self.prompt_embeddings,
                    "tokenized_prompts": self.tokenized_prompts,
                    "compound_rep_tokens_text": compound_rep_tokens_text2,
                },
                enc2_kwargs={
                    "prompts": self.prompt_embeddings,
                    "tokenized_prompts": self.tokenized_prompts,
                    "compound_rep_tokens_text": compound_rep_tokens_text2,
                },
            )[1]
        else:
            text_feat2 = self.text_encoder2(
                self.prompt_embeddings,
                self.tokenized_prompts,
                compound_rep_tokens_text2,
            )

        img_feat2 = self._encode_sam_with_prompts(sam_feat, compound_rep_tokens_visual2)

        logit_scale = self.logit_scale.exp().clamp(max=20)
        logits1 = logit_scale * (img_feat1 @ F.normalize(text_feat1, dim=-1).t())
        logits2 = logit_scale * (img_feat2 @ F.normalize(text_feat2, dim=-1).t())

        gate_in = torch.cat([img_feat1, img_feat2], dim=-1)
        alpha_l = torch.sigmoid(self.alpha_mlp(gate_in))
        logits = alpha_l * logits1 + (1 - alpha_l) * logits2

        return (
            logits1,
            img_feat1,
            text_feat1,
            logits2,
            img_feat2,
            text_feat2,
            logits,
            img_feat1_frozen,
            img_feat2_frozen,
            recon_rgb,
            rgb_sam,
        )