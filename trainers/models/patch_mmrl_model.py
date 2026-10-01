"""
MMRL single-branch HSI model.

Structure aligned with existing single-branch models in this repository.

Multi-Modal Representation Learning (MMRL) for hyperspectral image (HSI)
classification using CLIP with learnable compound representation tokens
that bridge visual and text modalities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel, CLIPTokenizer

from trainers.hsi_rgb_adapter import HSIRGBAdapter
from trainers.patch_mmrl import (
    MultiModalRepresentationLearner,
    TextEncoder_MMRL,
    VisualEncoder_MMRL,
    MMRL_Loss,
)
from clip_utils import clip_preprocess


class HSIPatchMMRL(nn.Module):
    """
    HSI classification model using MMRL (Multi-Modal Representation Learning).

    Combines learnable compound representation tokens that are projected into
    both visual and text embedding spaces, enabling richer cross-modal fusion
    and improved classification performance.

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
    n_rep_tokens : int
        Number of learnable representation tokens (default: 8)
    rep_dim : int
        Dimension of representation token embeddings (default: 512)
    n_layers : int
        Number of transformer layers to inject tokens into (default: 12)
    alpha : float
        Interpolation weight for fusing main and token-enhanced logits (0.0-1.0)
        (default: 0.7)
    reg_weight : float
        Weight for cosine similarity regularization (default: 1.0)
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        classnames,
        clip_name,
        ctx_len=4,
        ctx_init=None,
        class_token_position="end",
        csc=False,
        n_rep_tokens=8,
        rep_dim=512,
        n_layers=12,
        alpha=0.7,
        reg_weight=1.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.classnames = classnames
        self.alpha = alpha
        self.reg_weight = reg_weight
        self.ctx_len = ctx_len
        self.ctx_init = ctx_init
        self.class_token_position = class_token_position
        self.csc = csc

        # ──────────────────────────────────────────────────────────────────────
        # HSI to RGB Projection
        # ──────────────────────────────────────────────────────────────────────
        self.rgb_proj = HSIRGBAdapter(in_channels)

        # ──────────────────────────────────────────────────────────────────────
        # CLIP Model Initialization
        # ──────────────────────────────────────────────────────────────────────
        self.clip_model = CLIPModel.from_pretrained(clip_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_name)
        self.clip_dim = int(self.clip_model.config.projection_dim)

        # Cast CLIP to float32 for numeric stability during training
        self.clip_model = self.clip_model.float()

        # Freeze all CLIP parameters (representation tokens remain learnable)
        for p in self.clip_model.parameters():
            p.requires_grad_(False)
        if hasattr(self.clip_model, "logit_scale"):
            self.clip_model.logit_scale.requires_grad_(False)

        # ──────────────────────────────────────────────────────────────────────
        # MMRL Components
        # ──────────────────────────────────────────────────────────────────────
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

        self.text_encoder_mmrl = TextEncoder_MMRL(
            clip_model=self.clip_model,
        )

        self.visual_encoder_mmrl = VisualEncoder_MMRL(
            clip_model=self.clip_model,
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

        # Separate learnable logit scale (CLIP's own is frozen)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Prepare text embeddings for all classes (computed once during initialization)
        self.register_buffer(
            "text_embeddings_frozen",
            self._encode_text_features_frozen(self.classnames),
        )

    def _encode_text_features_frozen(self, classnames):
        """
        Pre-compute frozen text embeddings for all classes using CLIP.

        Args
        ────
        classnames : list[str]
            Class names to encode

        Returns
        ───────
        text_embeddings : Tensor [C, embed_dim]
            L2-normalized class embeddings
        """
        device = next(self.clip_model.parameters()).device
        dtype = next(self.clip_model.parameters()).dtype

        inputs = self.tokenizer(
                classnames,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

        text_outputs = self.clip_model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
        pooled = text_outputs.pooler_output  # [C, hidden_dim]
        text_embeddings = self.clip_model.text_projection(pooled)  # [C, embed_dim]
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        return text_embeddings

    def forward(self, images, labels=None, **kwargs):
        """
        Forward pass computing classification logits.

        Args
        ────
        images : Tensor [B, C_hsi, H, W]
            HSI input
        labels : LongTensor [B], optional
            Ground truth labels (used only during training for auxiliary computations)

        Returns
        ───────
        logits_main : Tensor [B, num_classes]
            Logits from main visual-text similarity
        logits_token_enhanced : Tensor [B, num_classes]
            Logits incorporating representation token fusion
        logits_fused : Tensor [B, num_classes]
            Alpha-blended fusion of main and token-enhanced logits
        image_features : Tensor [B, embed_dim]
            L2-normalized learned image features
        text_features : Tensor [C, embed_dim]
            L2-normalized learned text features
        """
        # ──────────────────────────────────────────────────────────────────────
        # HSI → RGB → CLIP Pre-processing
        # ──────────────────────────────────────────────────────────────────────
        rgb = self.rgb_proj(images)  # [B, 3, H, W]
        rgb_clip = clip_preprocess(rgb, image_resolution=224)  # [B, 3, 224, 224]

    
        frozen_vision_outputs = self.clip_model.vision_model(
                pixel_values=rgb_clip.type(self.clip_model.dtype),
                return_dict=True,
            )
        frozen_pooled = frozen_vision_outputs.pooler_output
        image_features_frozen = self.clip_model.visual_projection(frozen_pooled)
        image_features_frozen = F.normalize(image_features_frozen, dim=-1)

        # ──────────────────────────────────────────────────────────────────────
        # Representation Tokens
        # ──────────────────────────────────────────────────────────────────────
        compound_rep_tokens_text, compound_rep_tokens_visual = self.representation_learner()

        # Prompted and vanilla encodings, matching the CoOp/MaPLe-style structure
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

        image_features_main, image_features_token = self.visual_encoder_mmrl(
            pixel_values=rgb_clip.type(self.clip_model.dtype),
            compound_rep_tokens_visual=compound_rep_tokens_visual,
        )

        text_features_frozen_norm = self.text_embeddings_frozen  # [C, D]

        # ──────────────────────────────────────────────────────────────────────
        # Logits Computation
        # ──────────────────────────────────────────────────────────────────────
        logit_scale = self.logit_scale.exp().clamp(max=20)

        # Main logits: vanilla prompted branch
        logits_main = logit_scale * (image_features_main @ text_features_main.t())

        # Token-enhanced logits: deep-prompt branch
        logits_token_enhanced = logit_scale * (image_features_token @ text_features_token.t())

        # Fused logits: alpha-blended combination
        logits_fused = (
            self.alpha * logits_main + (1.0 - self.alpha) * logits_token_enhanced
        )  # [B, C]

        # Return prompted image features and frozen CLIP features for regularization.
        return (
            logits_main,
            logits_token_enhanced,
            logits_fused,
            image_features_token,
            image_features_frozen,
        )

    def encode_text_features(self) -> torch.Tensor:
        """
        Get text embeddings for all classes in CLIP projection space.

        Uses frozen pre-computed embeddings suitable for nearest-neighbor
        retrieval or prototype-based inference.

        Returns
        ───────
        text_embeddings : Tensor [C, D]
            L2-normalized class embeddings
        """
        return self.text_embeddings_frozen.clone()  # [C, D]

    def get_image_features(self, images) -> torch.Tensor:
        """
        Extract L2-normalized image features without representation tokens.

        Useful for retrieval tasks or analysis.

        Args
        ────
        images : Tensor [B, C_hsi, H, W]
            HSI input

        Returns
        ───────
        image_features : Tensor [B, D]
            L2-normalized image features
        """
        rgb = self.rgb_proj(images)
        rgb_clip = clip_preprocess(rgb, image_resolution=224)

        with torch.no_grad():
            vision_outputs = self.clip_model.vision_model(
                pixel_values=rgb_clip.type(self.clip_model.dtype),
                return_dict=True,
            )
            pooled = vision_outputs.pooler_output
            image_features = self.clip_model.visual_projection(pooled)
            image_features = F.normalize(image_features, dim=-1)

        return image_features

    def clear_eval_cache(self):
        """Clear cached representation tokens (call if switching train/eval modes)."""
        self._rep_tokens_text_cache = None
        self._rep_tokens_visual_cache = None
