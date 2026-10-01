# trainers/models/single_coop_model.py
"""
Single-Branch CLIP HSI Classification Model with CoOp
======================================================
A simplified single-branch model using only CLIP vision encoder.
Uses CoOp (Context Optimization) for learnable text prompts.
No cross attention fusions, no SAM branch.

Architecture:
- CLIP vision encoder (RGB projected from HSI)
- CoOp text encoder with learnable context tokens
- Direct contrastive learning without fusion mechanisms
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel, CLIPTokenizer

from trainers.hsi_rgb_adapter import HSIRGBAdapter
from trainers.patch_coop import CoOpTextEncoder
from clip_utils import clip_preprocess


# ============================================================
# Single-Branch CLIP with CoOp Text Encoder
# (No Cross-Attention, No SAM, No LoRA by default)
# ============================================================

class HSIPatchCoOp(nn.Module):
    """
    A single-branch HSI classification model using CLIP vision encoder
    with CoOp learnable text prompts.

    This model uses:
    1. CLIP vision encoder - for RGB-projected HSI features
    2. CoOp text encoder - learnable context tokens + class names
    3. Direct contrastive learning - without fusion mechanisms

    No cross-attention fusion, no SAM branch, no prompt learning, no LoRA by default.

    Args
    ────
    in_channels : int
        Number of HSI spectral bands
    num_classes : int
        Number of classification classes
    classnames : list[str]
        List of class names for text encoding
    clip_name : str
        Pretrained CLIP model name (e.g., 'openai/clip-vit-base-patch32')
    ctx_len : int
        Number of learnable context tokens
    class_token_position : str
        Position of class token: "end", "middle", or "front"
    csc : bool
        Whether to use class-specific context
    ctx_init : str, optional
        Initialization template for context tokens
    lora_clip_vision_enabled : bool
        Enable LoRA on vision encoder
    lora_clip_text_enabled : bool
        Enable LoRA on text encoder
    lora_rank : int
        LoRA rank (0 = disabled)
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        classnames,
        clip_name,
        ctx_len,
        class_token_position,
        csc,
        ctx_init,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.classnames  = classnames

        # ====================================================
        # Shared Spectral Adapter (HSI → RGB projection)
        # ====================================================
        self.rgb_proj = HSIRGBAdapter(in_channels)

        # ====================================================
        # CLIP Vision Encoder (Single Branch)
        # ====================================================
        self.clip_model = CLIPModel.from_pretrained(clip_name)
        self.tokenizer  = CLIPTokenizer.from_pretrained(clip_name)
        self.clip_dim = int(self.clip_model.visual_projection.weight.shape[0])

        # Freeze entire CLIP model initially
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Unfreeze vision projection and layer norm
        for p in self.clip_model.visual_projection.parameters():
            p.requires_grad = False
        for p in self.clip_model.vision_model.post_layernorm.parameters():
            p.requires_grad = False

        # Unfreeze text projection for adaptation
        for p in self.clip_model.text_projection.parameters():
            p.requires_grad = False
        for p in self.clip_model.text_model.final_layer_norm.parameters():
            p.requires_grad = False

        self.clip_model.logit_scale.requires_grad = False

        # ====================================================
        # CoOp Text Encoder
        # ====================================================
        self.text_encoder = CoOpTextEncoder(
            clip_model=self.clip_model,
            tokenizer=self.tokenizer,
            classnames=classnames,
            n_ctx=ctx_len,
            ctx_init=ctx_init,
            class_token_position=class_token_position,
            csc=csc,
        )

        # ====================================================
        # Classification
        # ====================================================
        # Shared learnable log-temperature (log 1/τ)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    # ================================================================
    # Text Encoding Helper
    # ================================================================

    def encode_text_features(self):
        """
        Get text embeddings from CoOp text encoder.

        Returns
        ───────
        text_feat : [C, D]  L2-normalised class embeddings
        """
        return self.text_encoder()   # [C, D]

    # ================================================================
    # Forward Pass
    # ================================================================

    def forward(self, images):
        """
        Forward pass with single CLIP branch using CoOp text encoder.

        Args
        ────
        images : torch.Tensor
            [B, H, W]  or  [B, C, H, W]  raw HSI cube

        Returns
        ───────
        logits     : [B, C]   Contrastive classification logits
        img_feat   : [B, D]   L2-normalised image features
        rgb        : [B, 3, H, W]  Reconstructed RGB from HSI projection
        """

        # ============================================================
        # Step 1: Project HSI to RGB
        # ============================================================
        rgb = self.rgb_proj(images)  # [B, 3, H, W]

        # ============================================================
        # CLIP Vision Encoder
        # ============================================================
        rgb_clip = clip_preprocess(rgb, image_resolution=224)  # [B, 3, 224, 224]

        vision_outputs = self.clip_model.vision_model(
            pixel_values=rgb_clip,
            return_dict=True,
        )
        pooled = vision_outputs.pooler_output  # [B, 768]
        img_feat = F.normalize(
            self.clip_model.visual_projection(pooled), dim=-1
        )  # [B, D]

        # ============================================================
        # Classification with CoOp Text Embeddings
        # ============================================================
        text_feat = self.encode_text_features()  # [C, D]

        # Temperature-scaled logits
        logit_scale = self.logit_scale.exp().clamp(max=20)
        logits = logit_scale * img_feat @ text_feat.t()  # [B, C]

        return logits
