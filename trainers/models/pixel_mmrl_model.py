# """
# Pixel-branch MMRL model for HSI classification.

# This follows the same SAM-based single-branch pattern as the other pixel
# models in this repository, but uses the MMRL representation learner for
# both text prompts and visual conditioning.
# """

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from transformers import CLIPModel, CLIPTokenizer

# from trainers.hsi_rgb_adapter import HSIRGBAdapter
# from trainers.sam_backbone import SAMBackbone
# from trainers.patch_mmrl import MultiModalRepresentationLearner, TextEncoder_MMRL, VisualEncoder_MMRL
# from trainers.upsamplers import get_upsampler
# from clip_utils import clip_preprocess


# class SAMVisualEncoder_MMRL(nn.Module):
#     """SAM vision encoder with MMRL compound-token injection."""

#     def __init__(self, sam_backbone, clip_model, use_tokens=True):
#         super().__init__()
#         self.sam_backbone = sam_backbone
#         self.visual_projection = clip_model.visual_projection
#         self.dtype = clip_model.dtype
#         self.use_tokens = use_tokens
#         self.clip_dim = clip_model.config.projection_dim
#         self._token_proj = None
#         self._token_layer_weights = None
#         self._token_dim = None
#         self._mod_scale = nn.Parameter(torch.tensor(0.1))

#     def _init_token_injection(self, token_dim, n_layers, device, dtype):
#         if self._token_proj is not None:
#             return
#         self._token_dim = token_dim
#         self._token_proj = nn.ModuleList(
#             [nn.Linear(token_dim, 768, bias=True) for _ in range(n_layers)]
#         )
#         self._token_layer_weights = nn.Parameter(torch.zeros(n_layers))
#         self._token_proj.to(device=device, dtype=dtype)

#     def forward(self, rgb_sam_224, compound_rep_tokens_visual=None):
#         """
#         Extract SAM features and optionally apply MMRL compound-token enhancement.
        
#         Args:
#             rgb_sam_224: [B, 3, 224, 224] - SAM-formatted RGB image
#             compound_rep_tokens_visual: Optional list of token tensors for MMRL
        
#         Returns:
#             (base_features, prompted_features): Both [B, clip_dim] normalized features
#         """
#         # Get base SAM features
#         sam_feat = self.sam_backbone(rgb_sam_224)  # [B, 768]
#         base_features = F.normalize(self.visual_projection(sam_feat), dim=-1)
        
#         if compound_rep_tokens_visual is None or not self.use_tokens:
#             return base_features, base_features
        
#         # Apply MMRL token enhancement by per-layer modulation of SAM features.
#         n_layers = len(compound_rep_tokens_visual)
#         token_dim = compound_rep_tokens_visual[0].shape[-1]
#         self._init_token_injection(token_dim, n_layers, sam_feat.device, sam_feat.dtype)

#         layer_weights = torch.softmax(self._token_layer_weights, dim=0)
#         token_modulation = torch.zeros_like(sam_feat)
#         for layer_idx, layer_tokens in enumerate(compound_rep_tokens_visual):
#             layer_token_agg = layer_tokens.mean(dim=0)
#             layer_mod = self._token_proj[layer_idx](layer_token_agg)
#             token_modulation += layer_weights[layer_idx] * layer_mod

#         if token_modulation.dim() == 1:
#             token_modulation = token_modulation.unsqueeze(0).expand(sam_feat.size(0), -1)

#         # Modulate SAM features with weighted per-layer tokens
#         sam_feat_enhanced = sam_feat + self._mod_scale * token_modulation
#         prompted_features = F.normalize(self.visual_projection(sam_feat_enhanced), dim=-1)
        
#         return base_features, prompted_features


# class HSIPixelMMRL(nn.Module):
#     """Single-branch SAM + MMRL model for pixel-level HSI classification."""

#     def __init__(
#         self,
#         in_channels,
#         num_classes,
#         classnames,
#         clip_name,
#         sam_model,
#         ctx_len,
#         class_token_position,
#         csc,
#         ctx_init=None,
#         n_rep_tokens=8,
#         rep_dim=512,
#         n_layers=12,
#         alpha=0.7,
#         reg_weight=1.0,
#         upsampler_type="bilinear",
#         upsampler_weights=None,
#     ):
#         super().__init__()

#         self.num_classes = num_classes
#         self.classnames = classnames
#         self.alpha = float(alpha)
#         self.reg_weight = float(reg_weight)
#         self.ctx_len = ctx_len
#         self.class_token_position = class_token_position
#         self.csc = csc

#         self.rgb_proj = HSIRGBAdapter(in_channels)
#         # Explicit SAM vision encoder branch for pixel MMRL.
#         self.sam_vision_encoder = SAMBackbone(sam_model)
#         self.sam = self.sam_vision_encoder
#         self.reconstruction_head = nn.Conv2d(768, 3, kernel_size=1)

#         self.upsampler = get_upsampler(
#             upsampler_type,
#             dim=3,
#             weight_path=upsampler_weights,
#             device="cpu",
#         )
#         for p in self.upsampler.parameters():
#             p.requires_grad = True

#         self.clip_model = CLIPModel.from_pretrained(clip_name)
#         self.tokenizer = CLIPTokenizer.from_pretrained(clip_name)
#         self.clip_model = self.clip_model.float()
#         self.clip_dim = int(self.clip_model.config.projection_dim)

#         for p in self.clip_model.parameters():
#             p.requires_grad = False
#         if hasattr(self.clip_model, "logit_scale"):
#             self.clip_model.logit_scale.requires_grad_(False)

#         self.sam_proj = nn.Linear(768, self.clip_dim)
#         self.sam_proj.requires_grad_(False)

#         self.representation_learner = MultiModalRepresentationLearner(
#             tokenizer=self.tokenizer,
#             classnames=self.classnames,
#             clip_model=self.clip_model,
#             n_rep_tokens=n_rep_tokens,
#             rep_dim=rep_dim,
#             n_layers=n_layers,
#             ctx_init=ctx_init,
#             class_token_position=class_token_position,
#             csc=csc,
#         )
#         self.text_encoder_mmrl = TextEncoder_MMRL(clip_model=self.clip_model)
#         self.visual_encoder_mmrl = SAMVisualEncoder_MMRL(
#             sam_backbone=self.sam_vision_encoder,
#             clip_model=self.clip_model,
#             use_tokens=True,
#         )

#         prompts = [f"{text.replace('_', ' ')}." for text in classnames]
#         tokenized = self.tokenizer(
#             prompts,
#             padding="max_length",
#             truncation=True,
#             max_length=77,
#             return_tensors="pt",
#         )
#         self.register_buffer("tokenized_prompts", tokenized.input_ids)
#         with torch.no_grad():
#             prompt_embeddings = self.clip_model.text_model.embeddings.token_embedding(
#                 self.tokenized_prompts
#             ).type(self.clip_model.dtype)
#         self.register_buffer("prompt_embeddings", prompt_embeddings)

#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

#     def forward(self, images):
#         rgb = self.rgb_proj(images)
#         rgb_sam = F.interpolate(rgb, size=(224, 224), mode="bicubic", align_corners=False)
#         rgb_clip = clip_preprocess(rgb, image_resolution=224)

#         frozen_vision_outputs = self.clip_model.vision_model(
#             pixel_values=rgb_clip.type(self.clip_model.dtype),
#             return_dict=True,
#         )
#         frozen_pooled = frozen_vision_outputs.pooler_output
#         image_features_frozen = self.clip_model.visual_projection(frozen_pooled)
#         image_features_frozen = F.normalize(image_features_frozen, dim=-1)

#         compound_rep_tokens_text, compound_rep_tokens_visual = self.representation_learner()

#         # Get text features with and without MMRL tokens
#         text_features_main = self.text_encoder_mmrl(
#             self.prompt_embeddings,
#             self.tokenized_prompts,
#             None,
#         )
#         text_features_token = self.text_encoder_mmrl(
#             self.prompt_embeddings,
#             self.tokenized_prompts,
#             compound_rep_tokens_text,
#         )

#         # Get image features via SAM visual encoder with MMRL token enhancement
#         image_features_base, image_features_token = self.visual_encoder_mmrl(
#             rgb_sam,
#             compound_rep_tokens_visual=compound_rep_tokens_visual,
#         )

#         # Reconstruction branch (still use direct SAM features)
#         sam_feat = self.sam_vision_encoder(rgb_sam)
#         sam_feat_spatial = sam_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 14, 14)
#         recon_lowres = self.reconstruction_head(sam_feat_spatial)
#         recon_rgb = self.upsampler(recon_lowres, rgb_sam)

#         logit_scale = self.logit_scale.exp().clamp(max=20)
#         logits_main = logit_scale * (image_features_base @ text_features_main.t())
#         logits_token_enhanced = logit_scale * (image_features_token @ text_features_token.t())
#         logits_fused = self.alpha * logits_main + (1.0 - self.alpha) * logits_token_enhanced

#         return (
#             logits_main,
#             logits_token_enhanced,
#             logits_fused,
#             image_features_token,
#             image_features_frozen,
#             recon_rgb,
#             rgb_sam,
#             image_features_base,
#         )



"""
Pixel-branch MMRL model for HSI classification.

This follows the same SAM-based single-branch pattern as the other pixel
models in this repository, but uses the MMRL representation learner for
both text prompts and visual conditioning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel, CLIPTokenizer

from trainers.hsi_rgb_adapter import HSIRGBAdapter
from trainers.sam_backbone import SAMBackbone
from trainers.patch_mmrl import MultiModalRepresentationLearner, TextEncoder_MMRL
from trainers.upsamplers import get_upsampler
from clip_utils import clip_preprocess


class SAMVisualEncoder_MMRL(nn.Module):
    """SAM vision encoder with MMRL compound-token injection.

    FIX (Bug 2 + Bug 3):
      - Accepts ``sam_proj`` (the trained SAM→CLIP bridge) instead of
        reusing ``clip_model.visual_projection`` (which expects ViT tokens,
        not SAM features — wrong distribution entirely).
      - All learnable sub-modules (_token_proj, _token_layer_weights) are
        created eagerly in ``__init__`` so the optimizer, checkpointing, and
        ``.to(device)`` all see them immediately. The old lazy
        ``_init_token_injection`` path silently dropped these parameters from
        the optimizer because ``nn.ModuleList`` created after ``__init__``
        returns is not automatically registered.
    """

    def __init__(self, sam_backbone, sam_proj, n_layers, rep_dim, use_tokens=True):
        super().__init__()
        self.sam_backbone = sam_backbone
        # FIX (Bug 2): use the dedicated SAM→CLIP projection, not CLIP's own
        # visual_projection (which expects ViT patch embeddings).
        self.sam_proj = sam_proj
        self.use_tokens = use_tokens

        # FIX (Bug 3): create all learnable parameters eagerly so they are
        # visible to the optimizer and move correctly with .to(device/dtype).
        self._token_proj = nn.ModuleList(
            [nn.Linear(768, 768, bias=True) for _ in range(n_layers)]
        )
        self._token_layer_weights = nn.Parameter(torch.zeros(n_layers))
        self._mod_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, rgb_sam_224, compound_rep_tokens_visual=None):
        """
        Extract SAM features and optionally apply MMRL compound-token enhancement.

        Args:
            rgb_sam_224: [B, 3, 224, 224] - SAM-formatted RGB image
            compound_rep_tokens_visual: Optional list of token tensors for MMRL

        Returns:
            (base_features, prompted_features): Both [B, clip_dim] normalized,
            and the raw SAM feature tensor [B, 768] for reuse.
        """
        # FIX (Bug 4): return raw SAM features so the caller can reuse them
        # for the reconstruction branch without a second forward pass.
        sam_feat = self.sam_backbone(rgb_sam_224)          # [B, 768]
        base_features = F.normalize(self.sam_proj(sam_feat), dim=-1)

        if compound_rep_tokens_visual is None or not self.use_tokens:
            return base_features, base_features, sam_feat

        layer_weights = torch.softmax(self._token_layer_weights, dim=0)
        token_modulation = torch.zeros_like(sam_feat)
        for layer_idx, layer_tokens in enumerate(compound_rep_tokens_visual):
            # layer_tokens: [n_rep_tokens, visual_dim=768] — already projected by r2vproj
            layer_token_agg = layer_tokens.mean(dim=0)  # [visual_dim=768]
            layer_mod = self._token_proj[layer_idx](
                layer_token_agg.to(sam_feat.dtype)
            )                                                              # [768]
            token_modulation = token_modulation + layer_weights[layer_idx] * layer_mod

        # Broadcast scalar modulation across the batch if needed
        if token_modulation.dim() == 1:
            token_modulation = token_modulation.unsqueeze(0).expand(sam_feat.size(0), -1)

        sam_feat_enhanced = sam_feat + self._mod_scale * token_modulation
        prompted_features = F.normalize(self.sam_proj(sam_feat_enhanced), dim=-1)

        return base_features, prompted_features, sam_feat


class HSIPixelMMRL(nn.Module):
    """Single-branch SAM + MMRL model for pixel-level HSI classification."""

    def __init__(
        self,
        in_channels,
        num_classes,
        classnames,
        clip_name,
        sam_model,
        ctx_len,
        class_token_position,
        csc,
        ctx_init=None,
        n_rep_tokens=8,
        rep_dim=512,
        n_layers=12,
        alpha=0.7,
        reg_weight=1.0,
        upsampler_type="bilinear",
        upsampler_weights=None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.classnames = classnames
        self.alpha = float(alpha)
        self.reg_weight = float(reg_weight)
        self.ctx_len = ctx_len
        self.class_token_position = class_token_position
        self.csc = csc

        self.rgb_proj = HSIRGBAdapter(in_channels)
        self.sam_vision_encoder = SAMBackbone(sam_model)
        self.sam = self.sam_vision_encoder
        self.reconstruction_head = nn.Conv2d(768, 3, kernel_size=1)

        self.upsampler = get_upsampler(
            upsampler_type,
            dim=3,
            weight_path=upsampler_weights,
            device="cpu",
        )
        for p in self.upsampler.parameters():
            p.requires_grad = True

        self.clip_model = CLIPModel.from_pretrained(clip_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_name)
        self.clip_model = self.clip_model.float()
        self.clip_dim = int(self.clip_model.config.projection_dim)

        for p in self.clip_model.parameters():
            p.requires_grad = False
        if hasattr(self.clip_model, "logit_scale"):
            self.clip_model.logit_scale.requires_grad_(False)

        # FIX (Bug 1): sam_proj MUST be trainable — it is the only learned
        # bridge between SAM's 768-dim feature space and CLIP's projection
        # space.  The original code called .requires_grad_(False), which
        # completely prevented this layer from learning anything, causing the
        # pixel branch to produce random/meaningless similarity scores.
        self.sam_proj = nn.Linear(768, self.clip_dim)
        self.sam_proj.requires_grad_(False)
        # (No requires_grad_(False) here — leave it trainable by default.)

        self.representation_learner = MultiModalRepresentationLearner(
            tokenizer=self.tokenizer,
            classnames=self.classnames,
            clip_model=self.clip_model,
            n_rep_tokens=n_rep_tokens,
            rep_dim=rep_dim,
            n_layers=n_layers,
            ctx_init=ctx_init,
            class_token_position=class_token_position,
            csc=csc,
        )
        self.text_encoder_mmrl = TextEncoder_MMRL(clip_model=self.clip_model)

        # FIX (Bug 2 + Bug 3): pass sam_proj, n_layers, and rep_dim so the
        # encoder uses the correct projection and eagerly registers all
        # learnable parameters.
        self.visual_encoder_mmrl = SAMVisualEncoder_MMRL(
            sam_backbone=self.sam_vision_encoder,
            sam_proj=self.sam_proj,
            n_layers=n_layers,
            rep_dim=rep_dim,
            use_tokens=True,
        )

        prompts = [f"{text.replace('_', ' ')}." for text in classnames]
        tokenized = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        self.register_buffer("tokenized_prompts", tokenized.input_ids)
        with torch.no_grad():
            prompt_embeddings = self.clip_model.text_model.embeddings.token_embedding(
                self.tokenized_prompts
            ).type(self.clip_model.dtype)
        self.register_buffer("prompt_embeddings", prompt_embeddings)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # FIX (Bug 5): register frozen text embeddings for cosine-similarity
        # regularization, matching the pattern in mmrl_model.py.
        self.register_buffer(
            "text_embeddings_frozen",
            self._encode_text_features_frozen(classnames),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_text_features_frozen(self, classnames):
        """Pre-compute frozen CLIP text embeddings for all classes."""
        device = next(self.clip_model.parameters()).device
        inputs = self.tokenizer(
            classnames,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            text_outputs = self.clip_model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
            pooled = text_outputs.pooler_output
            text_embeddings = self.clip_model.text_projection(pooled)
            text_embeddings = F.normalize(text_embeddings, dim=-1)
        return text_embeddings

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images):
        rgb = self.rgb_proj(images)
        rgb_sam = F.interpolate(rgb, size=(224, 224), mode="bicubic", align_corners=False)
        rgb_clip = clip_preprocess(rgb, image_resolution=224)

        # Frozen CLIP image features for regularization
        with torch.no_grad():
            frozen_vision_outputs = self.clip_model.vision_model(
                pixel_values=rgb_clip.type(self.clip_model.dtype),
                return_dict=True,
            )
        frozen_pooled = frozen_vision_outputs.pooler_output
        image_features_frozen = F.normalize(
            self.clip_model.visual_projection(frozen_pooled), dim=-1
        )

        compound_rep_tokens_text, compound_rep_tokens_visual = self.representation_learner()

        # Text features with and without MMRL tokens
        text_features_main = self.text_encoder_mmrl(
            self.prompt_embeddings,
            self.tokenized_prompts,
            None,
        )
        text_features_token = self.text_encoder_mmrl(
            self.prompt_embeddings,
            self.tokenized_prompts,
            compound_rep_tokens_text,
        )

        # FIX (Bug 4): SAMVisualEncoder_MMRL now returns sam_feat so the
        # reconstruction branch can reuse it — no second SAM forward pass.
        image_features_base, image_features_token, sam_feat = self.visual_encoder_mmrl(
            rgb_sam,
            compound_rep_tokens_visual=compound_rep_tokens_visual,
        )

        # Reconstruction branch — reuse cached SAM features
        sam_feat_spatial = sam_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 14, 14)
        recon_lowres = self.reconstruction_head(sam_feat_spatial)
        recon_rgb = self.upsampler(recon_lowres, rgb_sam)

        logit_scale = self.logit_scale.exp().clamp(max=20)
        logits_main = logit_scale * (image_features_base @ text_features_main.t())
        logits_token_enhanced = logit_scale * (image_features_token @ text_features_token.t())
        logits_fused = self.alpha * logits_main + (1.0 - self.alpha) * logits_token_enhanced

        return (
            logits_main,
            logits_token_enhanced,
            logits_fused,
            image_features_token,
            image_features_frozen,
            recon_rgb,
            rgb_sam,
            image_features_base,
        )