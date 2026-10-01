"""
Loss function for the pixel-branch MMRL (Multi-Modal Representation Learning) model.

Uses fused classification logits, cosine similarity regularization to the frozen
CLIP image features, and RGB reconstruction loss.

Model output tuple (HSIPixelMMRL.forward):
    [0] logits_main            [B, C]       — SAM base × text_main logits
    [1] logits_token_enhanced  [B, C]       — SAM token × text_token logits
    [2] logits_fused           [B, C]       — alpha-blended fusion of [0] and [1]
    [3] image_features_token   [B, D]       — SAM features after MMRL token modulation
    [4] image_features_frozen  [B, D]       — frozen CLIP ViT features (regularization anchor)
    [5] recon_rgb              [B,3,H,W]    — upsampled RGB reconstruction
    [6] rgb_sam                [B,3,224,224]— SAM input RGB (reconstruction target)
    [7] image_features_base    [B, D]       — SAM features before token modulation
"""

import torch
import torch.nn.functional as F


def get_pixel_mmrl_loss_fn():
    """Return the loss computation function for the pixel MMRL model."""

    def pixel_mmrl_loss_fn(model_outputs, labels, model, cfg):
        if not isinstance(model_outputs, (tuple, list)) or len(model_outputs) < 5:
            raise ValueError(
                f"Expected at least 5 outputs from pixel MMRL model, "
                f"got {type(model_outputs)} / "
                f"{len(model_outputs) if isinstance(model_outputs, (tuple, list)) else 'n/a'}"
            )

        logits_main          = model_outputs[0]
        logits_token_enhanced = model_outputs[1]
        logits_fused         = model_outputs[2]
        image_features_token = model_outputs[3]   # learned; MMRL-modulated SAM features
        image_features_frozen = model_outputs[4]  # frozen CLIP ViT features — reg anchor
        recon_rgb            = model_outputs[5] if len(model_outputs) > 5 else None
        rgb_sam              = model_outputs[6] if len(model_outputs) > 6 else None
        # image_features_base (index 7) is SAM-before-modulation — also a trained
        # layer, so it is NOT used as the frozen reference for regularization.

        # ── Classification loss ───────────────────────────────────────────────
        # FIX (Bug F): apply cfg.LAMBDA_CLS, matching mmrl_loss_fn.py and
        # hyperprompt_mmrl_loss_fn.py. Original code had an implicit coefficient
        # of 1.0, breaking cross-model config parity.
        lambda_cls = getattr(cfg, 'LAMBDA_CLS', 1.0)
        loss_cls = F.cross_entropy(logits_fused, labels)

        # ── Cosine similarity regularization ──────────────────────────────────
        # FIX (Bug B): regularize image_features_token against image_features_frozen
        # (true frozen CLIP ViT features, index 4), NOT against image_features_base
        # (index 7, which is the SAM pre-modulation output — also a trained layer
        # that changes during training and therefore makes a meaningless anchor).
        img_norm        = F.normalize(image_features_token, dim=-1)
        frozen_img_norm = F.normalize(image_features_frozen, dim=-1)
        cossim_reg = 1.0 - torch.mean(
            F.cosine_similarity(img_norm, frozen_img_norm, dim=-1)
        )

        # ── Reconstruction loss ───────────────────────────────────────────────
        loss_recon = torch.tensor(0.0, device=logits_fused.device)
        if recon_rgb is not None and rgb_sam is not None:
            loss_recon = F.mse_loss(recon_rgb, rgb_sam)

        # ── Combine ───────────────────────────────────────────────────────────
        reg_weight = getattr(cfg, 'LAMBDA_REG', getattr(cfg, 'REG_WEIGHT', 0.2))
        lambda_mse = getattr(cfg, 'LAMBDA_MSE', 0.1)

        total_loss = (
            lambda_cls * loss_cls
            + reg_weight * cossim_reg
            + lambda_mse * loss_recon
        )

        loss_dict = {
            'total': total_loss.item(),
            'cls':   loss_cls.item(),
            'reg':   cossim_reg.item(),
            'recon': loss_recon.item(),
        }

        return total_loss, loss_dict

    return pixel_mmrl_loss_fn