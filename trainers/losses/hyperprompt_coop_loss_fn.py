"""
Loss function for Dual CoOp PCLRA model.
Aligned with Dual CoOp non-PCLRA loss behavior (without PCLRA term).
"""

import torch
import torch.nn.functional as F


def get_hyperprompt_coop_loss_fn():
    """
    Get the loss computation function for Dual CoOp PCLRA model.

    Returns:
        loss_fn: A function that computes loss given model outputs and labels.
    """

    def hyperprompt_coop_loss_fn(model_outputs, labels, model, cfg):
        """
        Dual CoOp PCLRA loss matching Dual CoOp base objective.

        model_outputs tuple (NEW STRUCTURE with per-branch logits and img_features):
            (logits1, img_feat1, text_feat1, logits2, img_feat2, text_feat2, logits, recon_rgb, rgb_sam)
        """
        # Extract outputs from new structure
        logits1 = model_outputs[0]
        img_feat1 = model_outputs[1]
        text_feat1 = model_outputs[2]
        logits2 = model_outputs[3]
        img_feat2 = model_outputs[4]
        text_feat2 = model_outputs[5]
        logits = model_outputs[6]
        recon_rgb = model_outputs[7] if len(model_outputs) > 7 else None
        rgb_sam = model_outputs[8] if len(model_outputs) > 8 else None

        loss_ce = F.cross_entropy(logits, labels)

        loss_recon = torch.tensor(0.0, device=cfg.DEVICE)
        if recon_rgb is not None and rgb_sam is not None:
            loss_recon = F.mse_loss(recon_rgb, rgb_sam)

        loss_total = (
            cfg.LAMBDA_CLS * loss_ce +
            cfg.LAMBDA_MSE * loss_recon
        )

        lambda_hor = getattr(cfg, 'LAMBDA_HOR', 1.0)
        pclra = getattr(model, 'pclra', None)
        loss_hor = torch.tensor(0.0, device=loss_total.device)
        if pclra is not None and hasattr(pclra, 'loss_hor'):
            hor_out = pclra.loss_hor()
            if isinstance(hor_out, tuple):
                hor_out = hor_out[0]
            if torch.is_tensor(hor_out):
                loss_hor = hor_out
            else:
                loss_hor = torch.tensor(float(hor_out), device=loss_total.device)
        loss_hor = lambda_hor * loss_hor
        loss_total = loss_total + loss_hor

        loss_dict = {
            'loss_total': loss_total.item(),
            'loss_ce': loss_ce.item(),
            'loss_recon': loss_recon.item(),
            'loss_hor': loss_hor.item(),
        }

        return loss_total, loss_dict

    return hyperprompt_coop_loss_fn
