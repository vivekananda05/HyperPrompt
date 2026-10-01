"""
Loss function for KgCoOp Pixel (SAM) model.
Uses classification loss + knowledge regularization loss + RGB reconstruction loss.
Based on pixel_coop_loss_fn with added KgCoOp knowledge term.
"""

import torch
import torch.nn.functional as F


def get_pixel_kgcoop_loss_fn():
    """
    Get the loss computation function for the KgCoOp Pixel (SAM) model.

    Returns:
        loss_fn: A function that computes loss given model outputs and labels.
    """

    def kgpixel_coop_loss_fn(model_outputs, labels, model, cfg):
        """
        KgCoOp Pixel loss: cross-entropy classification + knowledge regularization + MSE reconstruction.

        model_outputs tuple:
            (logits, knowledge_loss, text_feat, img_feat, recon_rgb, rgb_sam)

            logits        : [B, C]          classification logits
            knowledge_loss: scalar          KgCoOp regularization loss
            text_feat     : [C, D]          L2-normalised KgCoOp text features (unused in loss)
            img_feat      : [B, D]          L2-normalised SAM features (unused in loss)
            recon_rgb     : [B, 3, H, W]    reconstructed RGB from SAM branch
            rgb_sam       : [B, 3, 224, 224] bicubic-resized RGB (reconstruction target)
        """
        logits         = model_outputs[0]
        knowledge_loss = model_outputs[1] if len(model_outputs) > 1 else None
        # model_outputs[2] = text_feat  (not used in loss)
        # model_outputs[3] = img_feat   (not used in loss)
        recon_rgb      = model_outputs[4] if len(model_outputs) > 4 else None
        rgb_sam        = model_outputs[5] if len(model_outputs) > 5 else None

        # ── Classification loss ──────────────────────────────────────
        loss_cls = F.cross_entropy(logits, labels)

        # ── Knowledge regularization loss ────────────────────────────
        if not isinstance(knowledge_loss, torch.Tensor):
            loss_knowledge = torch.tensor(0.0, device=logits.device)
        else:
            loss_knowledge = knowledge_loss.to(logits.device)
            if loss_knowledge.ndim > 0:
                loss_knowledge = loss_knowledge.mean()

        # ── Reconstruction loss ──────────────────────────────────────
        loss_recon = torch.tensor(0.0, device=logits.device)
        if recon_rgb is not None and rgb_sam is not None:
            loss_recon = F.mse_loss(recon_rgb, rgb_sam)

        # ── Total loss ───────────────────────────────────────────────
        lambda_cls = getattr(cfg, 'LAMBDA_CLS', 1.0)
        lambda_knowledge = getattr(cfg, 'LAMBDA_KNOWLEDGE', None)
        if lambda_knowledge is None:
            lambda_knowledge = getattr(model, 'knowledge_weight', 1.0)
        lambda_mse = getattr(cfg, 'LAMBDA_MSE', 1.0)

        total_loss = (
            lambda_cls * loss_cls +
            lambda_knowledge * loss_knowledge +
            lambda_mse * loss_recon
        )

        loss_dict = {
            'total': total_loss.item(),
            'cls': loss_cls.item(),
            'knowledge': loss_knowledge.item() if isinstance(loss_knowledge, torch.Tensor) else 0.0,
            'recon': loss_recon.item(),
        }

        return total_loss, loss_dict

    return kgpixel_coop_loss_fn
