# trainers/models/pixel_kgcoop_model.py
"""
Single-Branch SAM HSI Classification Model with KgCoOp (Pixel Branch)
======================================================================
A simplified single-branch model using only the SAM vision backbone
for pixel-level / region-level features.
Uses KgCoOp (Knowledge-Guided Context Optimization) for learnable text prompts.
No CLIP vision encoder, no cross-attention fusion, no LoRA, no TCDM.

Architecture:
- HSIRGBAdapter       : spectral → 3-channel projection
- SAMBackbone   : TCDM-capable frozen region-level vision encoder
- Reconstruction head : spatial features → low-res 3-channel RGB
- Upsampler           : low-res → full-res RGB
- KgCoOp text encoder : learnable context tokens + knowledge regularization
- Contrastive logits  : normalized image feat @ normalized text feat
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel, CLIPTokenizer

from trainers.hsi_rgb_adapter import HSIRGBAdapter
from trainers.sam_backbone import SAMBackbone
from trainers.patch_kgcoop import KgCoOpTextEncoder
from trainers.upsamplers import get_upsampler


# ============================================================
# Single-Branch SAM with KgCoOp Text Encoder  (Pixel Branch)
# ============================================================

class HSIPixelKgCoOp(nn.Module):
    """
    Single-branch HSI classification model using SAMBackbone
    (no TCDM) paired with KgCoOp learnable text prompts.

    Args
    ────
    in_channels : int
        Number of HSI spectral bands.
    num_classes : int
        Number of classification classes.
    classnames : list[str]
        List of class names for text encoding.
    clip_name : str
        Pretrained CLIP model name used for the text encoder.
    sam_model : str
        HuggingFace model id or local path for SAM-2.
    ctx_len : int
        Number of learnable context tokens for KgCoOp.
    class_token_position : str
        Position of class token: "end", "middle", or "front".
    csc : bool
        Whether to use class-specific context (one ctx vector per class).
    ctx_init : str, optional
        Initialization string for context tokens (e.g. "a photo of a").
    dataset_name : str, optional
        Dataset identifier used to pick the KgCoOp knowledge template.
    knowledge_weight : float
        Weight of the KgCoOp knowledge regularization term.
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
        dataset_name=None,
        knowledge_weight: float = 0.1,
        upsampler_type: str = "bilinear",
        upsampler_weights=None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.classnames = classnames
        self.knowledge_weight = knowledge_weight

        # ====================================================
        # Spectral Adapter  (HSI → RGB projection)
        # ====================================================
        self.rgb_proj = HSIRGBAdapter(in_channels)

        # ====================================================
        # SAM Vision Backbone  (frozen, no TCDM)
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
        # CLIP  (text side only — vision encoder is NOT used)
        # ====================================================
        self.clip_model = CLIPModel.from_pretrained(clip_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_name)
        clip_dim = self.clip_model.config.projection_dim

        # Freeze entire CLIP model — only KgCoOp context tokens are trained
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # ====================================================
        # SAM feature projection → CLIP embedding space
        # SAM hidden dim 768 → CLIP projection dim
        # ====================================================
        self.sam_proj = nn.Linear(768, clip_dim)
        self.sam_proj.requires_grad_(False)

        # ====================================================
        # KgCoOp Text Encoder
        # ====================================================
        self.text_encoder = KgCoOpTextEncoder(
            clip_model=self.clip_model,
            tokenizer=self.tokenizer,
            classnames=classnames,
            n_ctx=ctx_len,
            ctx_init=ctx_init,
            class_token_position=class_token_position,
            csc=csc,
            dataset_name=dataset_name,
            knowledge_weight=knowledge_weight,
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
        logits        : [B, C]           contrastive classification logits
        knowledge_loss: scalar          KgCoOp regularization loss
        text_feat     : [C, D]           L2-normalised KgCoOp text features
        img_feat      : [B, D]           L2-normalised SAM image features
        recon_rgb     : [B, 3, 224, 224] reconstructed RGB from SAM branch
        rgb_sam       : [B, 3, 224, 224] bicubic-resized RGB (reconstruction target)
        """

        # ── Step 1: HSI → RGB ──────────────────────────────────────
        rgb = self.rgb_proj(images)  # [B, 3, H, W]

        # ── Step 2: Resize for SAM ─────────────────────────────────
        rgb_sam = F.interpolate(
            rgb, size=(224, 224), mode="bicubic", align_corners=False
        )  # [B, 3, 224, 224]

        # ── Step 3: SAM forward → pooled features ──────────────────
        sam_feat = self.sam(rgb_sam)  # [B, 768]

        # ── Step 4: Reconstruction ─────────────────────────────────
        sam_feat_spatial = sam_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 14, 14)
        recon_lowres = self.reconstruction_head(sam_feat_spatial)  # [B, 3, 14, 14]
        recon_rgb = self.upsampler(recon_lowres, rgb_sam)  # [B, 3, 224, 224]

        # ── Step 5: Project SAM features → CLIP space ──────────────
        img_feat = F.normalize(self.sam_proj(sam_feat), dim=-1)  # [B, D]

        # ── Step 6: KgCoOp text features ───────────────────────────
        text_feat, knowledge_loss = self.text_encoder(img_feat)  # [C, D], scalar
        text_feat = F.normalize(text_feat, dim=-1)

        # ── Step 7: Contrastive logits ─────────────────────────────
        logit_scale = self.logit_scale.exp().clamp(max=20)
        logits = logit_scale * (img_feat @ text_feat.t())  # [B, C]

        return logits, knowledge_loss, text_feat, img_feat, recon_rgb, rgb_sam
