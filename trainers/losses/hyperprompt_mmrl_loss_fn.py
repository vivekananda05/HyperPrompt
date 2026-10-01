"""
Loss function for the dual MMRL + PCLRA model.
"""

import torch
import torch.nn.functional as F


def get_hyperprompt_mmrl_loss_fn():
    """Return the loss function for the dual MMRL PCLRA model."""

    def hyperprompt_mmrl_loss_fn(model_outputs, labels, model, cfg):
        if not isinstance(model_outputs, (tuple, list)) or len(model_outputs) < 11:
            raise ValueError(
                f"Expected dual MMRL PCLRA outputs with at least 11 elements, got {type(model_outputs)} / {len(model_outputs) if isinstance(model_outputs, (tuple, list)) else 'n/a'}"
            )

        logits1 = model_outputs[0]
        img_feat1 = model_outputs[1]
        text_feat1 = model_outputs[2]
        logits2 = model_outputs[3]
        img_feat2 = model_outputs[4]
        text_feat2 = model_outputs[5]
        logits = model_outputs[6]
        img_feat1_frozen = model_outputs[7]
        img_feat2_frozen = model_outputs[8]
        recon_rgb = model_outputs[9]
        rgb_sam = model_outputs[10]

        loss_ce = F.cross_entropy(logits, labels)
        loss_recon = F.mse_loss(recon_rgb, rgb_sam)

        reg_img1 = 1.0 - torch.mean(F.cosine_similarity(F.normalize(img_feat1, dim=-1), F.normalize(img_feat1_frozen, dim=-1), dim=-1))
        reg_img2 = 1.0 - torch.mean(F.cosine_similarity(F.normalize(img_feat2, dim=-1), F.normalize(img_feat2_frozen, dim=-1), dim=-1))
        loss_reg = 0.5 * (reg_img1 + reg_img2)

        loss_total = (
            cfg.LAMBDA_CLS * loss_ce
            + getattr(cfg, 'LAMBDA_REG', getattr(cfg, 'REG_WEIGHT', 0.2)) * loss_reg
            + cfg.LAMBDA_MSE * loss_recon
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
            'reg': loss_reg.item(),
            'loss_recon': loss_recon.item(),
            'loss_hor': loss_hor.item(),
        }

        return loss_total, loss_dict

    return hyperprompt_mmrl_loss_fn
