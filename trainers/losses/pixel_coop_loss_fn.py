"""
Loss function for CoOp Pixel (SAM) model.
Uses classification loss + RGB reconstruction loss.
No PCLRA diversity term (single branch, no LoRA).
"""

import torch
import torch.nn.functional as F


def get_pixel_coop_loss_fn():
    """
    Get the loss computation function for the CoOp Pixel (SAM) model.

    Returns:
        loss_fn: A function that computes loss given model outputs and labels.
    """

    def pixel_coop_loss_fn(model_outputs, labels, model, cfg):
        """
        CoOp Pixel loss: cross-entropy classification + MSE reconstruction.

        model_outputs tuple:
            (logits, img_feat, recon_rgb, rgb_sam)

            logits    : [B, C]          classification logits
            img_feat  : [B, D]          L2-normalised SAM features (unused in loss)
            recon_rgb : [B, 3, H, W]    reconstructed RGB from SAM branch
            rgb_sam   : [B, 3, 224, 224] bicubic-resized RGB (reconstruction target)
        """
        logits    = model_outputs[0]
        # model_outputs[1] = img_feat  (not used in loss)
        recon_rgb = model_outputs[2] if len(model_outputs) > 2 else None
        rgb_sam   = model_outputs[3] if len(model_outputs) > 3 else None

        # ── Classification loss ──────────────────────────────────────
        loss_cls = F.cross_entropy(logits, labels)

        # ── Reconstruction loss ──────────────────────────────────────
        loss_recon = torch.tensor(0.0, device=logits.device)
        if recon_rgb is not None and rgb_sam is not None:
            loss_recon = F.mse_loss(recon_rgb, rgb_sam)

        # ── Total loss ───────────────────────────────────────────────
        total_loss = (
            cfg.LAMBDA_CLS * loss_cls +
            cfg.LAMBDA_MSE * loss_recon
        )

        loss_dict = {
            'total': total_loss.item(),
            'cls':   loss_cls.item(),
            'recon': loss_recon.item(),
        }

        return total_loss, loss_dict

    return pixel_coop_loss_fn