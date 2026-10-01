# models/sam_backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Sam2Model


class SAMBackbone(nn.Module):
    """
    SAM-2 image encoder with the Token-wise Cross-Scale Distillation Module
    (TCDM) used by HyperPrompt.

    Uses:
        outputs.last_hidden_state  →  global-average-pooled feature vector

    Args:
        model_name          : HuggingFace model id for Sam2Model
    """

    def __init__(
        self,
        model_path: str,
        tcdm_last_k_layers: int = None,
        use_cls_only: bool = False,
    ):
        super().__init__()

        self.sam = Sam2Model.from_pretrained(model_path)
        self.use_cls_only = use_cls_only

        # Use only vision encoder
        self.vision_encoder = self.sam.vision_encoder

        # ----------------------------------------------------------
        # Step 1: Freeze everything
        # ----------------------------------------------------------
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        vision_cfg = self.vision_encoder.config
        hidden_size = getattr(vision_cfg, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(vision_cfg, "fpn_hidden_size", None)
        if hidden_size is None and hasattr(vision_cfg, "hidden_sizes"):
            hidden_sizes = getattr(vision_cfg, "hidden_sizes")
            if isinstance(hidden_sizes, (list, tuple)) and len(hidden_sizes) > 0:
                hidden_size = hidden_sizes[-1]
        if hidden_size is None:
            raise AttributeError(
                "Could not infer SAM vision hidden size from config. "
                "Expected one of: hidden_size, fpn_hidden_size, hidden_sizes."
            )

        self.tcdm_in = nn.Linear(hidden_size, hidden_size)
        self.tcdm_out = nn.Linear(hidden_size, 2 * hidden_size)
        nn.init.zeros_(self.tcdm_out.weight)
        nn.init.zeros_(self.tcdm_out.bias)
        self.tcdm_last_k_layers = tcdm_last_k_layers

    def _pool_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Pool token/grid features to [B, D] regardless of layout.
        If use_cls_only=True, extract CLS token (first token) instead of mean pooling.
        """
        if x.dim() == 4:
            # [B, H, W, D] -> [B, D]
            if self.use_cls_only:
                # For spatial features, take top-left corner as "CLS-like"
                return x[:, 0, 0, :]
            return x.mean(dim=(1, 2))
        if x.dim() == 3:
            # [B, N, D] -> [B, D]
            if self.use_cls_only:
                # Take CLS token (first token)
                return x[:, 0, :]
            return x.mean(dim=1)
        if x.dim() == 2:
            # [B, D]
            return x
        raise ValueError(f"Unsupported tensor shape for pooling: {tuple(x.shape)}")

    @staticmethod
    def _apply_tcdm(target: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Apply TCDM directly without residual connection: feat' = gamma * feat + beta"""
        if target.dim() == 4:
            return gamma[:, None, None, :] * target + beta[:, None, None, :]
        if target.dim() == 3:
            return gamma[:, None, :] * target + beta[:, None, :]
        raise ValueError(f"Unsupported target shape for TCDM: {tuple(target.shape)}")

    @staticmethod
    def _match_feature_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Deterministically map [B, D_in] to [B, target_dim] without lazy params."""
        d_in = x.size(-1)
        if d_in == target_dim:
            return x
        
        # Reshape using adaptive pooling, then verify exact output size
        x_1d = x.unsqueeze(1)  # [B, 1, D_in]
        x_resized = F.adaptive_avg_pool1d(x_1d, target_dim)  # [B, 1, target_dim]
        x_out = x_resized.squeeze(1)  # [B, target_dim]
        
        # Verify the output has exactly the target dimension
        assert x_out.size(-1) == target_dim, f"Resize failed: got {x_out.size(-1)}, expected {target_dim}"
        return x_out


    # ---------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        tcdm_tokens: torch.Tensor = None,
        tcdm_hidden_states=None,
        tcdm_last_k_layers: int = None,
    ) -> torch.Tensor:
        """
        Args:
            images : (B, 3, H, W)  range [0, 1]
            tcdm_tokens : Optional patch-encoder tokens for ViT-based TCDM bridging.
                         Shape (B, N, D_ctx) or (B, D_ctx)
            tcdm_hidden_states : Optional list/tuple of patch hidden states
                         for layer-wise TCDM bridging. Each element is
                         typically shape (B, N, D_ctx).
            tcdm_last_k_layers : Optional runtime override for selecting only
                         the last k aligned layers for TCDM aggregation.
                         Use -1 to aggregate all aligned layers.

        Returns:
            feat : (B, H, W, D) spatial features from last_hidden_state with TCDM modulation applied
        """
        outputs = self.vision_encoder(
            pixel_values=images,
            return_dict=True,
            output_hidden_states=tcdm_hidden_states is not None,
        )

        # Shape: (B, H', W', D)
        feat = outputs.last_hidden_state

        if tcdm_hidden_states is not None:
            if not isinstance(tcdm_hidden_states, (list, tuple)):
                raise ValueError("tcdm_hidden_states must be a list or tuple of tensors")

            sam_hidden_states = outputs.hidden_states
            if sam_hidden_states is None or len(sam_hidden_states) == 0:
                raise RuntimeError("SAM vision encoder did not return hidden states")

            sam_states = list(sam_hidden_states)
            patch_states = list(tcdm_hidden_states)

            # Hidden-state tuples usually contain an embedding-stage output at index 0.
            # For strict layer-wise mapping, align encoder block outputs in forward order.
            if len(sam_states) > 1:
                sam_states = sam_states[1:]
            if len(patch_states) > 1:
                patch_states = patch_states[1:]

            n_layers = min(len(sam_states), len(patch_states))
            if n_layers == 0:
                return feat

            sam_states = sam_states[:n_layers]
            patch_states = patch_states[:n_layers]

            k_layers = tcdm_last_k_layers
            if k_layers is None:
                k_layers = self.tcdm_last_k_layers
            if k_layers is not None:
                if k_layers == -1:
                    k_layers = n_layers
                elif k_layers <= 0:
                    raise ValueError("tcdm_last_k_layers must be -1 (all) or > 0")
                else:
                    k_layers = min(k_layers, n_layers)
                sam_states = sam_states[-k_layers:]
                patch_states = patch_states[-k_layers:]

            gamma_layers = []
            beta_layers = []
            target_dim = feat.size(-1)

            # Aggregate TCDM parameters from all selected layers
            for i in range(len(sam_states)):
                patch_state = patch_states[i].detach()

                if patch_state.dim() == 3 and patch_state.size(1) > 1:
                    if self.use_cls_only:
                        # Use CLS token only: [B, N, D] -> [B, D]
                        patch_context = patch_state[:, 0, :]
                    else:
                        # Drop CLS token and use patches only: [B, N, D] -> [B, D]
                        patch_context = patch_state[:, 1:, :].mean(dim=1)
                else:
                    patch_context = self._pool_tokens(patch_state)

                # Generate layer-specific TCDM parameters
                patch_context = self._match_feature_dim(patch_context, self.tcdm_in.in_features)
                tcdm_hidden = F.gelu(self.tcdm_in(patch_context))
                gamma, beta = self.tcdm_out(tcdm_hidden).chunk(2, dim=-1)
                
                # Resize to match final feat dimension - ensures consistent shape
                gamma = self._match_feature_dim(gamma, target_dim)
                beta = self._match_feature_dim(beta, target_dim)
                
                # Verify shapes before appending
                assert gamma.shape[-1] == target_dim, f"gamma shape mismatch: {gamma.shape} vs target {target_dim}"
                assert beta.shape[-1] == target_dim, f"beta shape mismatch: {beta.shape} vs target {target_dim}"
                
                gamma_layers.append(gamma)
                beta_layers.append(beta)

            # Stack requires all tensors to have same shape
            gamma_agg = torch.stack(gamma_layers, dim=0).mean(dim=0)  # [B, D]
            beta_agg = torch.stack(beta_layers, dim=0).mean(dim=0)    # [B, D]
            
            # Apply aggregated TCDM to final SAM output
            feat = self._apply_tcdm(feat, gamma_agg, beta_agg)
            
            # Pool spatial features to [B, D]
            feat = self._pool_tokens(feat)
            return feat

        if tcdm_tokens is not None:
            if tcdm_tokens.dim() == 3:
                if self.use_cls_only:
                    # Use CLS token only
                    context = tcdm_tokens.detach()[:, 0, :]
                else:
                    # Mean pool all tokens
                    context = tcdm_tokens.detach().mean(dim=1)
            elif tcdm_tokens.dim() == 2:
                context = tcdm_tokens.detach()
            else:
                raise ValueError("tcdm_tokens must have shape (B, N, D_ctx) or (B, D_ctx)")

            context = self._match_feature_dim(context, self.tcdm_in.in_features)
            tcdm_hidden = F.gelu(self.tcdm_in(context))
            gamma, beta = self.tcdm_out(tcdm_hidden).chunk(2, dim=-1)
            target_dim = feat.size(-1)
            gamma = self._match_feature_dim(gamma, target_dim)
            beta = self._match_feature_dim(beta, target_dim)
            feat = self._apply_tcdm(feat, gamma, beta)

        # Pixel-only models do not provide patch tokens. Keep their interface
        # consistent by returning a pooled feature vector from the same
        # canonical TCDM backbone.
        return self._pool_tokens(feat)
