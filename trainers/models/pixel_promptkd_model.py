# trainers/models/pixel_promptkd_model.py
"""
Single-Branch SAM HSI Classification Model with PromptKD (Pixel Branch)
========================================================================
A simplified single-branch model using only the SAM vision backbone
for pixel-level / region-level features.
Uses PromptKD (knowledge distillation with learned prompts) for text encoding.

Architecture:
- HSIRGBAdapter       : spectral → 3-channel projection
- SAMBackbone   : TCDM-capable frozen region-level vision encoder
- Reconstruction head : spatial features → low-res 3-channel RGB
- Upsampler           : low-res → full-res RGB
- PromptKD text encoder: learnable prompts + knowledge distillation
- Contrastive logits  : normalized image feat @ normalized text feat
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel, CLIPTokenizer

from trainers.hsi_rgb_adapter import HSIRGBAdapter
from trainers.sam_backbone import SAMBackbone
from trainers.patch_promptkd import PromptLearner, ZeroShotCLIP, FeatureTransModuleTwoLayer
from trainers.upsamplers import get_upsampler


# ============================================================
# Single-Branch SAM with PromptKD Text Encoder  (Pixel Branch)
# ============================================================

class HSIPixelPromptKD(nn.Module):
    """
    Single-branch HSI classification model using SAMBackbone
    paired with PromptKD learnable text prompts and knowledge distillation.

    Args
    ────
    in_channels : int
        Number of HSI spectral bands.
    num_classes : int
        Number of classification classes.
    classnames : list[str]
        List of class names for text encoding.
    clip_name : str
        Pretrained CLIP model name (e.g., 'openai/clip-vit-base-patch32').
    sam_model : str
        HuggingFace model id or local path for SAM-2.
    ctx_len : int
        Number of learnable context tokens for PromptKD.
    class_token_position : str
        Position of class token: "end", "middle", or "front".
    csc : bool
        Whether to use class-specific context (one ctx vector per class).
    ctx_init : str, optional
        Initialization string for context tokens.
    temperature : float
        Temperature for scaling logits (default: 4.0).
    kd_weight : float
        Weight for knowledge distillation loss (default: 1.0).
    upsampler_type : str
        Upsampler type for RGB reconstruction (default: "bilinear").
    upsampler_weights : str, optional
        Path to pretrained upsampler weights.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        classnames: list,
        clip_name: str,
        sam_model: str,
        ctx_len: int,
        class_token_position: str,
        csc: bool,
        ctx_init: str = None,
        temperature: float = 4.0,
        kd_weight: float = 1.0,
        use_feature_transform: bool = True,
        upsampler_type: str = "bilinear",
        upsampler_weights=None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.classnames  = classnames
        self.temperature = float(temperature)
        self.kd_weight   = float(kd_weight)

        # ====================================================
        # Spectral Adapter  (HSI → RGB projection)
        # ====================================================
        self.rgb_proj = HSIRGBAdapter(in_channels)

        # ====================================================
        # SAM Vision Backbone  (frozen)
        # ====================================================
        self.sam = SAMBackbone(sam_model)

        # ====================================================
        # Reconstruction head + Upsampler
        # SAM hidden dim is 768; reconstruct 3-channel low-res RGB
        # ====================================================
        self.reconstruction_head = nn.Conv2d(768, 3, kernel_size=1)

        self.upsampler = get_upsampler(
            upsampler_type,
            dim=3,
            weight_path=upsampler_weights,
            device="cpu",
        )
        for p in self.upsampler.parameters():
            p.requires_grad = True

        # ====================================================
        # CLIP  (text side only)
        # ====================================================
        self.clip_model = CLIPModel.from_pretrained(clip_name)
        self.tokenizer  = CLIPTokenizer.from_pretrained(clip_name)
        clip_dim        = self.clip_model.config.projection_dim

        # Freeze entire CLIP model — only PromptKD context tokens are trained
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # ====================================================
        # SAM feature projection → CLIP embedding space
        # SAM hidden dim 768 → CLIP projection dim
        # ====================================================
        self.sam_proj = nn.Linear(768, clip_dim)
        self.sam_proj.requires_grad_(False)

        # ====================================================
        # Learnable feature transform for SAM features
        # ====================================================
        self.use_feature_transform = bool(use_feature_transform)
        if self.use_feature_transform:
            self.image_feature_transform = FeatureTransModuleTwoLayer(
                input_dim=clip_dim,
                out_dim=clip_dim,
            )
        else:
            self.image_feature_transform = nn.Identity()

        # ====================================================
        # PromptKD Components
        # ====================================================
        # Student: learnable prompt
        cfg_dict = {
            'ctx_len': ctx_len,
            'class_token_position': class_token_position,
            'csc': csc,
            'ctx_init': ctx_init,
        }
        self.prompt_learner = PromptLearner(
            cfg_dict,
            classnames,
            self.clip_model,
            self.tokenizer,
        )

        # Teacher: zero-shot CLIP
        self.zeroshot_clip = ZeroShotCLIP(
            classnames,
            self.clip_model,
            self.tokenizer,
        )

        # ====================================================
        # Learnable log-temperature  (log 1/τ)
        # ====================================================
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    # ================================================================
    # Forward
    # ================================================================

    def forward(self, images: torch.Tensor):
        """
        Args
        ────
        images : [B, C_spec, H, W]  raw HSI cube

        Returns
        ───────
        logits        : [B, C]           contrastive classification logits (student)
        teacher_logits: [B, C]           zero-shot teacher logits
        img_feat      : [B, D]           L2-normalised SAM image features
        recon_rgb     : [B, 3, 224, 224] reconstructed RGB from SAM branch
        rgb_sam       : [B, 3, 224, 224] bicubic-resized RGB (reconstruction target)
        """

        # ── Step 1: HSI → RGB ──────────────────────────────────────
        rgb = self.rgb_proj(images)                          # [B, 3, H, W]

        # ── Step 2: Resize for SAM ─────────────────────────────────
        rgb_sam = F.interpolate(
            rgb, size=(224, 224), mode="bicubic", align_corners=False
        )                                                    # [B, 3, 224, 224]

        # ── Step 3: SAM forward → pooled features ──────────────────
        sam_feat = self.sam(rgb_sam)                         # [B, 768]

        # ── Step 4: Reconstruction ─────────────────────────────────
        sam_feat_spatial = sam_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 14, 14)
                                                               # [B, 768, 14, 14]
        recon_lowres = self.reconstruction_head(sam_feat_spatial)  # [B, 3, 14, 14]
        recon_rgb    = self.upsampler(recon_lowres, rgb_sam)       # [B, 3, 224, 224]

        # ── Step 5: Project SAM features → CLIP space ──────────────
        # Project to CLIP space (unnormalized)
        img_feat = self.sam_proj(sam_feat)                         # [B, D]
        # Apply feature transform to unnormalized features
        img_feat = self.image_feature_transform(img_feat)          # [B, D]
        # Normalize after transformation
        img_feat = F.normalize(img_feat, dim=-1)                   # [B, D]

        # ── Step 6: Student PromptKD text features ─────────────────
        # PromptLearner returns already-normalized text features
        text_feat = self.prompt_learner()                          # [C, D]

        # ── Step 7: Student contrastive logits ─────────────────────
        logit_scale = self.logit_scale.exp().clamp(max=20)
        logits = logit_scale * (img_feat @ text_feat.t())          # [B, C]

        # ── Step 8: Teacher zero-shot logits ───────────────────────
        teacher_feat = self.zeroshot_clip.text_features             # [C, D]
        teacher_logits = logit_scale * (img_feat @ teacher_feat.t())# [B, C]

        return logits, teacher_logits, img_feat, recon_rgb, rgb_sam
