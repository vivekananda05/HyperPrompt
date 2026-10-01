# trainers/models/kgcoop_model.py
"""
Single-Branch CLIP HSI Classification Model with KgCoOp
=========================================================
A simplified single-branch model using only CLIP vision encoder.
Uses KgCoOp (Knowledge-Guided Context Optimization) for knowledge-guided text prompts.
No cross attention fusions, no SAM branch.

Architecture:
- CLIP vision encoder (RGB projected from HSI)
- KgCoOp text encoder with knowledge regularization
- Direct contrastive learning with knowledge guidance
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel, CLIPTokenizer

from trainers.hsi_rgb_adapter import HSIRGBAdapter
from trainers.patch_kgcoop import KgCoOpTextEncoder
from clip_utils import clip_preprocess


# ============================================================
# Single-Branch CLIP with KgCoOp Text Encoder
# (Knowledge-Guided Text Prompts)
# ============================================================

class HSIPatchKgCoOp(nn.Module):
    """
    A single-branch HSI classification model using CLIP vision encoder
    with KgCoOp (Knowledge-Guided Context Optimization).

    This model extends CoCoOp by adding a knowledge-guided regularization term
    that encourages learned text features to remain close to a static knowledge base
    (template-based embeddings).

    Architecture:
    1. CLIP vision encoder - for RGB-projected HSI features
    2. KgCoOp text encoder - class-level learnable context
                              + knowledge base regularization
    3. Contrastive learning - with knowledge guidance

    Key advantage: KgCoOp prevents excessive prompt drift while keeping
    prompt optimization simple and stable.

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
    knowledge_weight : float
        Weight of knowledge regularization in total loss
        higher = stronger regularization toward knowledge base
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
        dataset_name=None,
        knowledge_weight=0.1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.classnames  = classnames
        self.knowledge_weight = knowledge_weight

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

        # Keep CLIP vision projection and post-layer norm frozen
        for p in self.clip_model.visual_projection.parameters():
            p.requires_grad = False
        for p in self.clip_model.vision_model.post_layernorm.parameters():
            p.requires_grad = False

        # Keep CLIP text projection and final layer norm frozen
        for p in self.clip_model.text_projection.parameters():
            p.requires_grad = False
        for p in self.clip_model.text_model.final_layer_norm.parameters():
            p.requires_grad = False

        self.clip_model.logit_scale.requires_grad = False

        # ====================================================
        # KgCoOp Text Encoder with class-level prompts + knowledge base
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
        # Classification
        # ====================================================
        # Shared learnable log-temperature (log 1/τ)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    # ================================================================
    # Internal: KgCoOp logits computation
    # ================================================================

    @staticmethod
    def _kgcoop_logits(img_feat, text_feat, logit_scale):
        """
        Compute per-image logits from KgCoOp text features.

        Args
        ────
        img_feat    : [B, D]      L2-normalised image features
        text_feat   : [C, D] or [B, C, D]
                  class-level text features (preferred) or legacy per-image
        logit_scale : scalar

        Returns
        ───────
        logits : [B, C]
        """
        if text_feat.dim() == 2:
            logits = img_feat @ text_feat.t()                     # [B, C]
        elif text_feat.dim() == 3:
            logits = (img_feat.unsqueeze(1) * text_feat).sum(dim=-1)  # [B, C]
        else:
            raise ValueError(
                f"Unsupported text_feat shape {tuple(text_feat.shape)}; "
                "expected [C, D] or [B, C, D]"
            )
        return logit_scale * logits

    # ================================================================
    # Forward Pass
    # ================================================================

    def forward(self, images):
        """
        Training forward pass with knowledge guidance.

        KgCoOp returns:
        1. Class-level text features [C, D]
        2. Knowledge loss (regularization term to add to training loss)

        Args
        ────
        images : torch.Tensor
            [B, H, W]  or  [B, C, H, W]  raw HSI cube

        Returns
        ───────
        logits          : [B, C]   Contrastive classification logits
        knowledge_loss  : scalar   Regularization loss (add to training loss)
        img_feat        : [B, D]   L2-normalised image features
        rgb             : [B, 3, H, W]  Reconstructed RGB from HSI projection
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
        # Classification with KgCoOp Text Embeddings
        # ============================================================
        # KgCoOp returns BOTH text features AND knowledge loss.
        text_feat, knowledge_loss = self.text_encoder(img_feat)  # [C, D], scalar

        # Temperature-scaled logits using KgCoOp per-image dot product
        logit_scale = self.logit_scale.exp().clamp(max=20)
        logits = self._kgcoop_logits(img_feat, text_feat, logit_scale)

        return logits, knowledge_loss, text_feat, img_feat, rgb

    # ================================================================
    # Inference helper (without knowledge loss)
    # ================================================================

    def forward_inference(self, images):
        """
        Inference forward pass (returns only logits, no knowledge loss).

        Args
        ────
        images : torch.Tensor  [B, H, W] or [B, C, H, W]

        Returns
        ───────
        logits : [B, C]
        """
        rgb = self.rgb_proj(images)
        rgb_clip = clip_preprocess(rgb, image_resolution=224)

        vision_outputs = self.clip_model.vision_model(
            pixel_values=rgb_clip,
            return_dict=True,
        )
        pooled = vision_outputs.pooler_output
        img_feat = F.normalize(
            self.clip_model.visual_projection(pooled), dim=-1
        )

        text_feat, _ = self.text_encoder(img_feat)
        logit_scale = self.logit_scale.exp().clamp(max=20)
        logits = self._kgcoop_logits(img_feat, text_feat, logit_scale)

        return logits
