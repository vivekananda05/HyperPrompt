# models/hsi_hyperprompt_maple_model.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel, CLIPTokenizer

from trainers.hsi_rgb_adapter import HSIRGBAdapter
from trainers.sam_backbone import SAMBackbone
from trainers.patch_maple import MapleTextEncoder
from trainers.pclra_utils import (
    inject_pclra_text,
    lora_param_count,
)
from clip_utils import clip_preprocess
from trainers.upsamplers import get_upsampler




# ============================================================
# Dual Prompt Cross Model  —  MaPLe version  +  LoRA
# ============================================================

class HSIHyperPromptMaPLe(nn.Module):
    """
    Two-branch HSI classification model using MaPLe (modular deep prompts),
    with optional LoRA fine-tuning on CLIP text encoders and the
    SAM vision backbone.

    MaPLe-specific args
    ───────────────────
    ctx_len                number of learnable context tokens per layer
    prompt_depth           number of transformer layers to inject prompts
    class_token_position   position of class token ("end", "middle", "front")
    csc                    class-specific context (True/False)
    ctx_init               optional string initialization for context

    PCLRA args  (Self-Guided Prompt-Conditioned Low-Rank Adapter for text encoders)
    ─────────────────────────────────────────────────────────────
    pclra_enabled            whether to enable PCLRA on text encoders
    pclra_rank               rank r for PCLRA matrices  (0 = disable)
    pclra_alpha              alpha scaling  (default = pclra_rank)
    pclra_prompt_dim         dimension of learned prompts
    pclra_last_n_layers      number of layers to inject PCLRA
    pclra_target_keys        which attention projections to target
    pclra_dropout            dropout on PCLRA path
    pclra_tau                temperature for loss gradient focus
    """

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
        # ── MaPLe-specific ────────────────────────────────────────
        prompt_depth=3,
        # ── upsampler ─────────────────────────────────────────────
        upsampler_type="bilinear",
        upsampler_weights=None,        tcdm_last_k_layers=4,       
        # ── Self-Guided Prompt-Conditioned Low-Rank Adapter (text encoders, fixed rank) ────        
        pclra_enabled: bool         = False,        
        pclra_rank: int             = 0,
        pclra_alpha: float          = None,
        pclra_prompt_dim: int       = 256,
        pclra_last_n_layers: int    = -1,
        pclra_target_keys           = ("q_proj", "v_proj"),
        pclra_dropout: float        = 0.0,
        pclra_tau: float            = 0.07,
    ):
        super().__init__()

        self.num_classes = num_classes

        # Store PCLRA config
        self.pclra_rank   = pclra_rank
        self.pclra_alpha  = pclra_alpha if pclra_alpha is not None else float(pclra_rank)

        # ====================================================
        # Shared Spectral Adapter
        # ====================================================
        self.rgb_proj = HSIRGBAdapter(in_channels)

        # ====================================================
        # ──────────────── BRANCH 1 ──────────────────────────
        # ====================================================
        self.clip_branch1 = CLIPModel.from_pretrained(clip_name)
        self.tokenizer1   = CLIPTokenizer.from_pretrained(clip_name)

        # Freeze CLIP-1
        for p in self.clip_branch1.parameters():
            p.requires_grad = False

        clip_dim = self.clip_branch1.config.projection_dim

        # ── MaPLe text encoder for branch 1 ──────────────────────────
        self.text_encoder1 = MapleTextEncoder(
            clip_model           = self.clip_branch1,
            tokenizer            = self.tokenizer1,
            classnames           = classnames,
            n_ctx                = ctx_len_patch,
            prompt_depth         = prompt_depth,
            ctx_init             = ctx_init,
            class_token_position = class_token_position,
            csc                  = csc,
        )

        # ====================================================
        # ──────────────── BRANCH 2 ──────────────────────────
        # ====================================================
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
        self.tokenizer2   = CLIPTokenizer.from_pretrained(clip_name)

        # Freeze CLIP-2
        for p in self.clip_branch2.parameters():
            p.requires_grad = False

        # ── MaPLe text encoder for branch 2 ──────────────────────────
        self.text_encoder2 = MapleTextEncoder(
            clip_model           = self.clip_branch2,
            tokenizer            = self.tokenizer2,
            classnames           = classnames,
            n_ctx                = ctx_len_pixel,
            prompt_depth         = prompt_depth,
            ctx_init             = ctx_init,
            class_token_position = class_token_position,
            csc                  = csc,
        )

        # ====================================================
        # Fusion + Classifier
        # ====================================================
        self.classifier   = nn.Linear(clip_dim, num_classes)

        # ====================================================
        # Prompt-Conditioned Low-Rank Adapter  (dual text encoders)
        # ====================================================
        self.pclra: "DualTextEncoderPCLRA | None" = None
        if pclra_enabled and pclra_rank > 0:
            # ── MaPLe-specific callables ────────────────────────────────────
            def _maple_get_layers(enc):
                return enc.clip.text_model.encoder.layers

            def _maple_get_prompt_tokens(enc):
                # MaPLe stores deep prompts; for aggregation use them all
                # or fall back to shallow prompts if deep not available
                if hasattr(enc, 'deep_compound_prompts'):
                    # Return deep prompts [depth-1] as a stacked tensor [depth-1, n_ctx, D]
                    if len(enc.deep_compound_prompts) > 0:
                        return torch.stack(list(enc.deep_compound_prompts), dim=0)
                    return enc.ctx
                elif hasattr(enc, 'compound_prompts_text'):
                    return enc.compound_prompts_text
                elif hasattr(enc, 'ctx'):
                    # Fall back to shallow prompts
                    return enc.ctx
                else:
                    raise RuntimeError("MaPLe encoder has no recognized prompt attributes")

            _clip_hidden_dim = self.text_encoder1.hidden_dim

            self.pclra = inject_pclra_text(
                encoder1          = self.text_encoder1,
                encoder2          = self.text_encoder2,
                get_layers        = _maple_get_layers,
                get_prompt_tokens = _maple_get_prompt_tokens,
                hidden_dim        = _clip_hidden_dim,
                r                 = pclra_rank,
                alpha             = pclra_alpha if pclra_alpha is not None else float(pclra_rank),
                prompt_dim        = pclra_prompt_dim,
                last_n_layers     = pclra_last_n_layers,
                target_keys       = pclra_target_keys,
                dropout           = pclra_dropout,
                tau               = pclra_tau,
            )

        # Shared learnable log-temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Print LoRA summary if enabled
        lora_param_count(self)

        self.alpha_mlp = nn.Sequential(
            nn.Linear(clip_dim * 2, clip_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(clip_dim // 4, 1),
        )

        # Store text features from last forward pass for inference
        self.register_buffer('_text_feat1', None)
        self.register_buffer('_text_feat2', None)

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
        """Encode branch-1 CLIP vision features with shallow/deep MaPLe prompts."""
        vision_model = self.clip_branch1.vision_model

        hidden = vision_model.embeddings(pixel_values=rgb_clip)
        hidden = vision_model.pre_layrnorm(hidden)

        batch_size = hidden.size(0)
        n_ctx = shallow_visual_ctx.size(0)

        shallow = shallow_visual_ctx.to(hidden.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        hidden = torch.cat([hidden[:, :1, :], shallow, hidden[:, 1:, :]], dim=1)

        patch_hidden_states = []
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

            patch_hidden_states.append(hidden)

        self._patch_hidden_states1 = patch_hidden_states
        pooled = vision_model.post_layernorm(hidden[:, 0, :])
        img_feat = F.normalize(self.clip_branch1.visual_projection(pooled), dim=-1)
        return img_feat

    # ================================================================
    # Inference helper
    # ================================================================

    def encode_mean_prompt(self):
        """
        Return [C, D] text embeddings from both encoders.

        Returns
        ───────
        text_feat1 : [C, D]  L2-normalised, from encoder1
        text_feat2 : [C, D]  L2-normalised, from encoder2
        """
        if self.pclra is not None:
            return self.pclra.encode_mean_prompt()

        # Use stored text features from forward pass
        if self._text_feat1 is None or self._text_feat2 is None:
            raise RuntimeError("Text features not initialized. Run forward() first or during training.")

        return self._text_feat1, self._text_feat2

    # ================================================================
    # Forward
    # ================================================================

    def forward(self, images):
        """
        Training forward pass.

        Returns
        ───────
        img_feat1             : [B, D]   L2-normalised CLIP image features
        text_feat1            : [C, D]   MaPLe text embeddings from encoder 1
        img_feat2             : [B, D]   L2-normalised SAM image features
        text_feat2            : [C, D]   MaPLe text embeddings from encoder 2
        logits                : [B, C]   fused classification logits
        recon_rgb             : [B, 3, H, W]  reconstructed RGB from SAM branch
        rgb_sam               : [B, 3, 224, 224]  bilinear-resized RGB (MSE target)
        """
        device = images.device
        rgb = self.rgb_proj(images)

        # ============================================================
        # Branch 1: patch-level CLIP
        # ============================================================
        rgb_clip = clip_preprocess(rgb, image_resolution=224)
        shallow_visual_ctx1, deep_visual_prompts1 = self.text_encoder1.get_visual_prompts()
        img_feat1 = self._encode_vision_with_prompts(
            rgb_clip,
            shallow_visual_ctx1,
            deep_visual_prompts1,
        )
        patch_hidden_states1 = self._patch_hidden_states1

        # MaPLe text encoding (deep compound prompts)
        if self.pclra is not None:
            text_feat1, text_feat2 = self.pclra()
        else:
            text_feat1 = self.text_encoder1()
            text_feat2 = self.text_encoder2()

        # Store for inference
        if not self.training:
            self._text_feat1 = text_feat1.detach()
            self._text_feat2 = text_feat2.detach()

        # ============================================================
        # Branch 2: SAM region-level
        # ============================================================
        rgb_sam = F.interpolate(
            rgb, size=(224, 224), mode="bicubic", align_corners=False
        )

        sam_feat = self.sam(rgb_sam, tcdm_hidden_states=patch_hidden_states1)  # [B, D] after pooling
        
        # Expand pooled features to spatial [B, D, 14, 14] for reconstruction head
        sam_feat_spatial = sam_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 14, 14)  # [B, D, 14, 14]
        sam_feat_pooled = sam_feat  # [B, D]

        
        recon_lowres = self.reconstruction_head(sam_feat_spatial)
        recon_rgb = self.upsampler(recon_lowres, rgb_sam)
        img_feat2 = F.normalize(self.sam_proj(sam_feat_pooled), dim=-1)  # [B, D]

        # ============================================================
        # Branch logits + logits fusion
        # ============================================================
        text_feat1_norm = F.normalize(text_feat1, dim=-1)
        text_feat2_norm = F.normalize(text_feat2, dim=-1)

        logit_scale = self.logit_scale.exp().clamp(max=20)
        logits1 = logit_scale * img_feat1 @ text_feat1_norm.t()
        logits2 = logit_scale * img_feat2 @ text_feat2_norm.t()

        gate_in = torch.cat([img_feat1, img_feat2], dim=-1)
        alpha_l = torch.sigmoid(self.alpha_mlp(gate_in))

        log_p1 = F.log_softmax(logits1, dim=-1)
        log_p2 = F.log_softmax(logits2, dim=-1)
        logits = alpha_l * log_p1 + (1 - alpha_l) * log_p2

        # Keep output contract stable for downstream code
        # Return per-branch logits AND image features for visualization and analysis

        return (
            logits1, img_feat1, text_feat1,
            logits2, img_feat2, text_feat2,
            logits,
            recon_rgb,
            rgb_sam,
        )
