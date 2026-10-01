"""
HyperPrompt PromptKD + PCLRA model for HSI classification.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel, CLIPTokenizer

from trainers.hsi_rgb_adapter import HSIRGBAdapter
from trainers.sam_backbone import SAMBackbone
from trainers.patch_promptkd import PromptLearner, ZeroShotCLIP
from trainers.pclra_utils import inject_pclra_text, lora_param_count
from clip_utils import clip_preprocess
from trainers.upsamplers import get_upsampler




class HSIHyperPromptPromptKD(nn.Module):
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
        temperature=4.0,
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
        self.temperature = temperature

        self.rgb_proj = HSIRGBAdapter(in_channels)

        self.clip_branch1 = CLIPModel.from_pretrained(clip_name)
        self.tokenizer1 = CLIPTokenizer.from_pretrained(clip_name)
        for p in self.clip_branch1.parameters():
            p.requires_grad = False

        clip_dim = self.clip_branch1.config.projection_dim

        prompt_cfg1 = {
            "ctx_len": ctx_len_patch,
            "class_token_position": class_token_position,
            "csc": csc,
            "ctx_init": ctx_init,
        }
        self.prompt_learner1 = PromptLearner(prompt_cfg1, classnames, self.clip_branch1, self.tokenizer1)
        self.zs_clip1 = ZeroShotCLIP(classnames, self.clip_branch1, self.tokenizer1)

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

        self.clip_branch2 = CLIPModel.from_pretrained(clip_name)
        self.tokenizer2 = CLIPTokenizer.from_pretrained(clip_name)
        for p in self.clip_branch2.parameters():
            p.requires_grad = False

        prompt_cfg2 = {
            "ctx_len": ctx_len_pixel,
            "class_token_position": class_token_position,
            "csc": csc,
            "ctx_init": ctx_init,
        }
        self.prompt_learner2 = PromptLearner(prompt_cfg2, classnames, self.clip_branch2, self.tokenizer2)
        self.zs_clip2 = ZeroShotCLIP(classnames, self.clip_branch2, self.tokenizer2)

        self.pclra = None
        if pclra_enabled and pclra_rank > 0:
            def _get_layers(enc):
                return enc.clip.text_model.encoder.layers

            def _get_prompt_tokens(enc):
                return enc.ctx

            hidden_dim = self.prompt_learner1.hidden_dim
            self.pclra = inject_pclra_text(
                encoder1=self.prompt_learner1,
                encoder2=self.prompt_learner2,
                get_layers=_get_layers,
                get_prompt_tokens=_get_prompt_tokens,
                hidden_dim=hidden_dim,
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

        lora_param_count(self)

    @staticmethod
    def _promptkd_logits(img_feat, text_feat, logit_scale):
        return logit_scale * (img_feat @ text_feat.t())

    def forward(self, images):
        logit_scale = self.logit_scale.exp().clamp(max=20)
        rgb = self.rgb_proj(images)

        rgb_clip = clip_preprocess(rgb, image_resolution=224)

        vision_outputs1 = self.clip_branch1.vision_model(pixel_values=rgb_clip, return_dict=True, output_hidden_states=True)
        pooled1 = vision_outputs1.pooler_output
        patch_hidden_states1 = vision_outputs1.hidden_states
        img_feat1 = F.normalize(self.clip_branch1.visual_projection(pooled1), dim=-1)

        if self.pclra is not None:
            student_text_feat1, student_text_feat2 = self.pclra()
        else:
            student_text_feat1 = self.prompt_learner1()
            student_text_feat2 = self.prompt_learner2()

        teacher_logits1 = self.zs_clip1(rgb_clip)
        student_logits1 = self._promptkd_logits(img_feat1, student_text_feat1, logit_scale)

        rgb_sam = F.interpolate(rgb, size=(224, 224), mode="bicubic", align_corners=False)

        sam_feat = self.sam(rgb_sam, tcdm_hidden_states=patch_hidden_states1)  # [B, D] after pooling
        
        # Expand pooled features to spatial [B, D, 14, 14] for reconstruction head
        sam_feat_spatial = sam_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 14, 14)  # [B, D, 14, 14]
        sam_feat_pooled = sam_feat  # [B, D]

        recon_lowres = self.reconstruction_head(sam_feat_spatial)
        recon_rgb = self.upsampler(recon_lowres, rgb_sam)
        img_feat2 = F.normalize(self.sam_proj(sam_feat_pooled), dim=-1)  # [B, D]

        # Use TCDM-adapted img_feat2 with zero-shot text features (teacher)
        # This ensures both teacher and student use the same TCDM-adapted features
        zs_text_feat2 = F.normalize(self.zs_clip2.text_features.to(img_feat2.device), dim=-1)
        teacher_logits2 = self._promptkd_logits(img_feat2, zs_text_feat2, logit_scale)
        
        # Student: TCDM-adapted img_feat2 with learned text features
        student_logits2 = self._promptkd_logits(img_feat2, student_text_feat2, logit_scale)

        gate_in = torch.cat([img_feat1, img_feat2], dim=-1)
        alpha_l = torch.sigmoid(self.alpha_mlp(gate_in))
        log_p1 = F.log_softmax(student_logits1, dim=-1)
        log_p2 = F.log_softmax(student_logits2, dim=-1)
        logits = alpha_l * log_p1 + (1 - alpha_l) * log_p2

        # Keep output contract stable for downstream code
        # Return per-branch logits AND image features for visualization and analysis

        return (
            student_logits1, img_feat1, teacher_logits1,
            student_logits2, img_feat2, teacher_logits2,
            logits,
            recon_rgb,
            rgb_sam,
        )
