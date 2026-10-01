# trainers/models/pixel_promptsrc_model.py
"""
Single-Branch SAM HSI Classification Model with PromptSRC (Pixel Branch)
========================================================================
A simplified single-branch model using only the SAM vision backbone
for pixel-level / region-level features.
Uses PromptSRC (Self-Reinforcing Contextualization) for text encoding.

Architecture:
- HSIRGBAdapter       : spectral → 3-channel projection
- SAMBackbone   : TCDM-capable frozen region-level vision encoder
- Reconstruction head : spatial features → low-res 3-channel RGB
- Upsampler           : low-res → full-res RGB
- PromptSRC text encoder: learnable prompts with SRC loss
- Contrastive logits  : normalized image feat @ normalized text feat
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from transformers import CLIPModel, CLIPTokenizer

from trainers.hsi_rgb_adapter import HSIRGBAdapter
from trainers.sam_backbone import SAMBackbone
from trainers.patch_promptsrc import TextEncoder
from trainers.upsamplers import get_upsampler


# ============================================================
# Vision-Language Prompt Learner for PromptSRC
# ============================================================

class VLPromptLearner(nn.Module):
    """
    Independent Vision-Language prompt learner for PromptSRC.
    
    Learns separate prompt contexts for vision and language branches.
    """

    def __init__(
        self,
        num_classes,
        clip_model,
        tokenizer,
        classnames,
        ctx_len_text=4,
        ctx_len_vision=4,
        ctx_init=None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.ctx_len_text = ctx_len_text
        self.ctx_len_vision = ctx_len_vision
        self.classnames = classnames
        self.tokenizer = tokenizer

        dtype = clip_model.dtype
        ctx_dim = clip_model.text_model.config.hidden_size

        # Initialize text context vectors
        if ctx_init is not None:
            ctx_init = ctx_init.replace("_", " ")
            init_tokens = tokenizer(ctx_init, return_tensors="pt")
            with torch.no_grad():
                embeddings = clip_model.text_model.embeddings
                ctx_emb = embeddings.token_embedding(init_tokens.input_ids)
            # Count available tokens (excluding SOS/EOS)
            n_init_tokens = init_tokens.input_ids.size(1) - 2
            
            if n_init_tokens > ctx_len_text:
                warnings.warn(
                    f"ctx_init '{ctx_init}' tokenizes to {n_init_tokens} tokens "
                    f"but ctx_len_text={ctx_len_text}. Only the first {ctx_len_text} tokens "
                    f"will be used to initialize the context; the rest are discarded. "
                    f"Increase ctx_len_text to use the full init string.",
                    UserWarning, stacklevel=2,
                )
            
            # Initialize with random, then overwrite with available tokens
            ctx_vectors = torch.empty(ctx_len_text, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            n_use = min(n_init_tokens, ctx_len_text)
            ctx_vectors[:n_use] = ctx_emb[0, 1 : 1 + n_use, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(ctx_len_text, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * ctx_len_text)

        self.ctx = nn.Parameter(ctx_vectors)
        self.prompt_prefix = prompt_prefix

    def construct_prompts(self):
        """Construct full text prompts from learned context and class names."""
        ctx = self.ctx
        if ctx.dim() == 2:
            # Expand context for each class
            ctx = ctx.unsqueeze(0).expand(self.num_classes, -1, -1)

        # Build text prompts with context and class names
        prompts = []
        for i, classname in enumerate(self.classnames):
            prompt_str = f"{self.prompt_prefix} {classname}"
            prompts.append(prompt_str)

        return prompts

    def forward(self):
        """Return learned prompts as text strings."""
        prompts = self.construct_prompts()
        return prompts


# ============================================================
# Single-Branch SAM with PromptSRC Text Encoder  (Pixel Branch)
# ============================================================

class HSIPixelPromptSRC(nn.Module):
    """
    Single-branch HSI classification model using SAMBackbone
    paired with PromptSRC learnable text prompts and self-reinforcing
    contextualization.

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
        Number of learnable context tokens for PromptSRC.
    class_token_position : str
        Position of class token (for compatibility, not used in PromptSRC).
    csc : bool
        Whether to use class-specific context (for compatibility).
    ctx_init : str, optional
        Initialization string for context tokens.
    temperature : float
        Temperature for scaling logits (default: 1.0).
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
        class_token_position: str = "end",
        csc: bool = False,
        ctx_init: str = None,
        temperature: float = 1.0,
        upsampler_type: str = "bilinear",
        upsampler_weights=None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.classnames  = classnames
        self.temperature = temperature

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

        # Freeze entire CLIP model — only PromptSRC context tokens are trained
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # ====================================================
        # SAM feature projection → CLIP embedding space
        # SAM hidden dim 768 → CLIP projection dim
        # ====================================================
        self.sam_proj = nn.Linear(768, clip_dim)
        self.sam_proj.requires_grad_(False)

        # ====================================================
        # Learnable image adapter (transforms frozen features → student)
        # Creates student vs teacher distinction for image features
        # ====================================================
        self.image_adapter = nn.Sequential(
            nn.Linear(clip_dim, clip_dim // 2),
            nn.BatchNorm1d(clip_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(clip_dim // 2, clip_dim),
        )

        # ====================================================
        # PromptSRC Components
        # ====================================================
        self.prompt_learner = VLPromptLearner(
            num_classes=num_classes,
            clip_model=self.clip_model,
            tokenizer=self.tokenizer,
            classnames=classnames,
            ctx_len_text=ctx_len,
            ctx_len_vision=ctx_len,
            ctx_init=ctx_init,
        )

        self.text_encoder = TextEncoder(
            self.clip_model,
            self.tokenizer,
        )

        # ====================================================
        # Learnable log-temperature  (log 1/τ)
        # ====================================================
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # ====================================================
        # Precompute zero-shot text features (frozen teacher)
        # ====================================================
        with torch.no_grad():
            zs_text_features = []
            for classname in classnames:
                prompt = f"a photo of a {classname}"
                tokens = self.tokenizer(
                    [prompt],
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                )
                text_outputs = self.clip_model.text_model(
                    input_ids=tokens.input_ids.to(self.clip_model.device),
                    attention_mask=tokens.attention_mask.to(self.clip_model.device),
                )
                text_feat = self.clip_model.text_projection(text_outputs.pooler_output)
                zs_text_features.append(text_feat)

            zs_features = torch.cat(zs_text_features, dim=0)
            self.register_buffer("zero_shot_text_features", zs_features)

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
        logits           : [B, C]           student contrastive classification logits
        zero_shot_logits : [B, C]           teacher (zero-shot) logits
        recon_rgb        : [B, 3, 224, 224] reconstructed RGB from SAM branch
        rgb_sam          : [B, 3, 224, 224] bicubic-resized RGB (reconstruction target)
        img_feat         : [B, D]           L2-normalised SAM image features (student)
        text_feat        : [C, D]           L2-normalised learned text features (student)
        zs_text_feat     : [C, D]           L2-normalised frozen text features (teacher)
        zs_img_feat      : [B, D]           L2-normalised frozen image features (teacher)
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
        sam_feat_proj = self.sam_proj(sam_feat)                       # [B, D]

        # Teacher: frozen image features
        zs_img_feat = F.normalize(sam_feat_proj, dim=-1)              # [B, D]

        # Student: apply learnable adapter to unnormalized features
        img_feat = self.image_adapter(sam_feat_proj)
        img_feat = F.normalize(img_feat, dim=-1)                      # [B, D]

        # ── Step 6: PromptSRC text features ────────────────────────
        text_prompts = self.prompt_learner()                       # List of text strings
        text_feat = self.text_encoder(text_prompts)               # [C, D]
        text_feat = F.normalize(text_feat, dim=-1)                # [C, D]

        # ── Step 7: Contrastive logits ─────────────────────────────
        logit_scale = self.logit_scale.exp().clamp(max=20)
        logits = logit_scale * (img_feat @ text_feat.t())          # [B, C]

        # ── Step 8: Zero-shot (frozen) logits for self-reinforcing loss ──
        zs_text_feat = self.zero_shot_text_features
        zs_text_feat = F.normalize(zs_text_feat, dim=-1)           # [C, D]
        zero_shot_logits = logit_scale * (zs_img_feat @ zs_text_feat.t())  # [B, C]

        # Return features for SRC loss computation
        # student_logits from (learned adapter image + learned text)
        # teacher_logits from (frozen image + frozen text)
        return logits, zero_shot_logits, recon_rgb, rgb_sam, img_feat, text_feat, zs_text_feat, zs_img_feat
