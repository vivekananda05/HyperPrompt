"""
MaPLe (Modular Prompt Learning) for HSI Classification

Model wrapper for HSI classification using CLIP with MaPLe prompt learning.
This module imports prompt learning components from trainers.patch_maple and wraps
them into an HSI-specialized model.

Paper: MaPLe - Modular Prompt Learning for Vision-Language Models
Reference: He et al., CVPR 2023

Architecture:
- HSI → RGB projection (spectral adapter)
- CLIP vision encoder
- MaPLe text encoder (deep compound prompts at each layer)
- Contrastive learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel, CLIPTokenizer

from trainers.hsi_rgb_adapter import HSIRGBAdapter
from clip_utils import clip_preprocess
from trainers.patch_maple import MapleTextEncoder


# ========================================================================
# HSI Single-Branch CLIP with MaPLe
# ========================================================================

class HSIPatchMaPLe(nn.Module):
    """
    HSI classification model using CLIP with MaPLe prompt learning.

    Single-branch architecture following CoOp/CoCoOp patterns:
    - HSI → RGB projection (spectral adapter)
    - CLIP vision encoder (frozen)
    - MaPLe text encoder (layer-specific learnable prompts)
    - Contrastive learning

    MaPLe key feature: Learns layer-specific context tokens (deep compound prompts)
    for each transformer layer, enabling hierarchical feature adaptation.

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
        Number of learnable context tokens (default: 4)
    prompt_depth : int
        Number of layers to inject prompts (default: 1, range: 1-12)
    class_token_position : str
        Position of class token: "end", "middle", or "front" (default: "end")
    csc : bool
        Whether to use class-specific context (default: False)
    ctx_init : str, optional
        Initialization template for context tokens
    temperature : float
        Temperature for softmax scaling (default: 0.07)
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        classnames,
        clip_name,
        ctx_len=4,
        prompt_depth=1,
        class_token_position="end",
        csc=False,
        ctx_init=None,
        temperature=0.07,
        **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.classnames = classnames
        self.temperature = temperature

        # ====================================================
        # HSI → RGB Projection
        # ====================================================
        self.rgb_proj = HSIRGBAdapter(in_channels)

        # ====================================================
        # CLIP Model
        # ====================================================
        self.clip_model = CLIPModel.from_pretrained(clip_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_name)
        self.clip_dim = int(self.clip_model.visual_projection.weight.shape[0])

        # Freeze entire CLIP
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # ====================================================
        # MaPLe Text Encoder
        # ====================================================
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
        # Learnable logit_scale
        # ====================================================
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @staticmethod
    def _extract_hidden(layer_output):
        if hasattr(layer_output, "last_hidden_state"):
            return layer_output.last_hidden_state
        if isinstance(layer_output, tuple):
            return layer_output[0]
        if torch.is_tensor(layer_output):
            return layer_output
        raise TypeError(f"Unsupported encoder layer output type: {type(layer_output)}")

    def _encode_vision_with_prompts(self, rgb_clip, shallow_visual_ctx, deep_visual_prompts):
        """Encode images with MaPLe-style shallow/deep prompt injection in vision layers."""
        vision_model = self.clip_model.vision_model

        hidden = vision_model.embeddings(pixel_values=rgb_clip)
        hidden = vision_model.pre_layrnorm(hidden)

        batch_size = hidden.size(0)
        n_ctx = shallow_visual_ctx.size(0)

        shallow = shallow_visual_ctx.to(hidden.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        hidden = torch.cat([hidden[:, :1, :], shallow, hidden[:, 1:, :]], dim=1)

        for layer_idx, encoder_layer in enumerate(vision_model.encoder.layers):
            if layer_idx < len(deep_visual_prompts):
                deep_prompt = deep_visual_prompts[layer_idx].to(hidden.dtype)
                deep_prompt = deep_prompt.unsqueeze(0).expand(batch_size, -1, -1)

                hidden_with_prompt = torch.cat(
                    [hidden[:, :1, :], deep_prompt, hidden[:, 1:, :]],
                    dim=1,
                )

                layer_output = encoder_layer(
                    hidden_with_prompt,
                    attention_mask=None,
                    causal_attention_mask=None,
                    output_attentions=False,
                    return_dict=True,
                )
                hidden_out = self._extract_hidden(layer_output)

                hidden = torch.cat(
                    [hidden_out[:, :1, :], hidden_out[:, 1 + n_ctx:, :]],
                    dim=1,
                )
            else:
                layer_output = encoder_layer(
                    hidden,
                    attention_mask=None,
                    causal_attention_mask=None,
                    output_attentions=False,
                    return_dict=True,
                )
                hidden = self._extract_hidden(layer_output)

        pooled = vision_model.post_layernorm(hidden[:, 0, :])
        img_feat = F.normalize(self.clip_model.visual_projection(pooled), dim=-1)
        return img_feat


    def forward(self, images):
        """
        Forward pass for MaPLe model.

        Args
        ────
        images : torch.Tensor
            [B, C_hsi, H, W]  raw HSI cube

        Returns
        ───────
        logits : [B, C]  Classification logits
        rgb : [B, 3, H, W] Reconstructed RGB from HSI
        """

        # ============================================================
        # Step 1: Project HSI to RGB
        # ============================================================
        rgb = self.rgb_proj(images)  # [B, 3, H, W]

        # ============================================================
        # Step 2: CLIP Vision Encoding
        # ============================================================
        rgb_clip = clip_preprocess(rgb, image_resolution=224)  # [B, 3, 224, 224]

        shallow_visual_ctx, deep_visual_prompts = self.text_encoder.get_visual_prompts()
        img_feat = self._encode_vision_with_prompts(
            rgb_clip,
            shallow_visual_ctx,
            deep_visual_prompts,
        )

        # ============================================================
        # Step 3: MaPLe Text Features
        # ============================================================
        text_feat = self.text_encoder()  # [C, D]

        # ============================================================
        # Step 4: Compute Logits
        # ============================================================
        logit_scale = self.logit_scale.exp().clamp(max=20)
        
        # Logits: img_feat [B, D] @ text_feat [C, D].T = [B, C]
        logits = logit_scale * img_feat @ text_feat.t()  # [B, C]

        return logits, rgb


    def encode_text_features(self):
        """
        Get text embeddings from MaPLe text encoder.

        Returns
        ───────
        text_feat : [C, D]  L2-normalised class embeddings
        """
        return self.text_encoder()  # [C, D]
