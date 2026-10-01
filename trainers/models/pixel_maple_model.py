# trainers/models/pixel_maple_model.py
"""
Single-Branch SAM HSI Classification Model with MaPLe (Pixel Branch)
=====================================================================
A simplified single-branch model using only the SAM vision backbone
for pixel-level / region-level features.
Uses MaPLe (Modular Prompt Learning) for learnable text prompts.
No CLIP vision encoder, no cross-attention fusion, no LoRA, no TCDM.

Architecture:
- HSIRGBAdapter       : spectral → 3-channel projection
- SAMBackbone   : TCDM-capable frozen region-level vision encoder
- Reconstruction head : spatial features → low-res 3-channel RGB
- Upsampler           : low-res → full-res RGB
- MaPLe text encoder  : layer-specific learnable context tokens
- Contrastive logits  : normalized image feat @ normalized text feat
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel, CLIPTokenizer

from trainers.hsi_rgb_adapter import HSIRGBAdapter
from trainers.sam_backbone import SAMBackbone
from trainers.patch_maple import MapleTextEncoder
from trainers.upsamplers import get_upsampler


# ============================================================
# Single-Branch SAM with MaPLe Text Encoder  (Pixel Branch)
# ============================================================

class HSIPixelMaPLe(nn.Module):
    """
    Single-branch HSI classification model using SAMBackbone
    (no TCDM) paired with MaPLe learnable text prompts.

    This model pairs MaPLe text prompts with a SAM visual branch.
    MaPLe visual prompts are mapped to a lightweight conditioning signal that
    modulates SAM pooled features before projection to CLIP embedding space.

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
        Number of learnable context tokens for MaPLe.
    prompt_depth : int
        Number of transformer layers to inject prompts (1-12).
    class_token_position : str
        Position of class token: "end", "middle", or "front".
    csc : bool
        Whether to use class-specific context (one ctx vector per class).
    ctx_init : str, optional
        Initialization string for context tokens (e.g. "a photo of a").
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
        prompt_depth: int,
        class_token_position: str,
        csc: bool,
        ctx_init: str = None,
        upsampler_type: str = "bilinear",
        upsampler_weights=None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.classnames = classnames

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

        # Freeze entire CLIP model — only MaPLe context tokens are trained
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # ====================================================
        # SAM feature projection → CLIP embedding space
        # SAM hidden dim 768 → CLIP projection dim
        # ====================================================
        self.sam_proj = nn.Linear(768, clip_dim)
        self.sam_proj.requires_grad_(False)

        # ====================================================
        # MaPLe Text Encoder
        # ====================================================
        self.prompt_depth = prompt_depth
        self.text_encoder = MapleTextEncoder(
            clip_model=self.clip_model,
            tokenizer=self.tokenizer,
            classnames=classnames,
            n_ctx=ctx_len,
            prompt_depth=prompt_depth,
            ctx_init=ctx_init,
            class_token_position=class_token_position,
            csc=csc,
        )

        # ====================================================
        # MaPLe visual prompts -> SAM feature conditioning
        # ====================================================
        sam_dim = 768
        vision_dim = self.clip_model.vision_model.config.hidden_size
        self.visual_prompt_to_sam = nn.Linear(vision_dim, sam_dim)
        self.sam_prompt_scale = nn.Linear(sam_dim, sam_dim)
        self.sam_prompt_bias = nn.Linear(sam_dim, sam_dim)

        # Start from near-identity conditioning for stable optimization.
        nn.init.zeros_(self.sam_prompt_scale.weight)
        nn.init.zeros_(self.sam_prompt_scale.bias)
        nn.init.zeros_(self.sam_prompt_bias.weight)
        nn.init.zeros_(self.sam_prompt_bias.bias)

        # ====================================================
        # Learnable log-temperature  (log 1/τ)
        # ====================================================
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    # ================================================================
    # Forward
    # ================================================================

    def _build_prompt_condition(self, shallow_visual_ctx, deep_visual_prompts):
        """Aggregate MaPLe visual prompts according to prompt_depth semantics."""
        prompt_tokens = [shallow_visual_ctx]

        # MaPLe deep prompts correspond to layers 1..prompt_depth-1.
        n_deep = max(0, self.prompt_depth - 1)
        if n_deep > 0:
            prompt_tokens.extend(deep_visual_prompts[:n_deep])

        visual_prompt = torch.cat(prompt_tokens, dim=0).mean(dim=0)
        return self.visual_prompt_to_sam(visual_prompt)

    def _encode_sam_with_prompts(self, sam_feat, shallow_visual_ctx, deep_visual_prompts):
        """Condition SAM pooled features using MaPLe visual prompts."""
        prompt_cond = self._build_prompt_condition(shallow_visual_ctx, deep_visual_prompts)
        prompt_cond = prompt_cond.to(sam_feat.dtype).unsqueeze(0)

        # Residual affine modulation keeps SAM branch dominant and stable.
        scale = 1.0 + 0.1 * torch.tanh(self.sam_prompt_scale(prompt_cond))
        bias = 0.1 * torch.tanh(self.sam_prompt_bias(prompt_cond))
        sam_feat_mod = sam_feat * scale + bias

        img_feat = F.normalize(self.sam_proj(sam_feat_mod), dim=-1)
        return img_feat

    def forward(self, images: torch.Tensor):
        """
        Args
        ────
        images : [B, C_spec, H, W]  raw HSI cube

        Returns
        ───────
        logits    : [B, C]           contrastive classification logits
        img_feat  : [B, D]           L2-normalised SAM image features
        recon_rgb : [B, 3, 224, 224] reconstructed RGB from SAM branch
        rgb_sam   : [B, 3, 224, 224] bicubic-resized RGB (reconstruction target)
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

        # ── Step 5: SAM feature conditioning with MaPLe visual prompts ─
        shallow_visual_ctx, deep_visual_prompts = self.text_encoder.get_visual_prompts()
        img_feat = self._encode_sam_with_prompts(
            sam_feat,
            shallow_visual_ctx,
            deep_visual_prompts,
        )

        # ── Step 6: MaPLe text features ────────────────────────────
        text_feat = self.text_encoder()  # [C, D]
        text_feat = F.normalize(text_feat, dim=-1)  # [C, D]

        # ── Step 7: Contrastive logits ─────────────────────────────
        logit_scale = self.logit_scale.exp().clamp(max=20)
        logits = logit_scale * (img_feat @ text_feat.t())  # [B, C]

        return logits, img_feat, recon_rgb, rgb_sam

    def encode_text_features(self):
        """
        Get text embeddings from MaPLe text encoder.

        Returns
        ───────
        text_feat : [C, D]  L2-normalised class embeddings
        """
        return self.text_encoder()  # [C, D]
