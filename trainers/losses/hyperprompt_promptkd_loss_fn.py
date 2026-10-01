"""
Loss function for Dual PromptKD PCLRA model.
"""

import torch
import torch.nn.functional as F

from trainers.patch_promptkd import PromptKDLoss


def get_hyperprompt_promptkd_loss_fn():
    def hyperprompt_promptkd_loss_fn(model_outputs, labels, model, cfg):
        # NEW STRUCTURE: (logits1, img_feat1, teacher_logits1, logits2, img_feat2, teacher_logits2, logits, recon_rgb, rgb_sam)
        student_logits1 = model_outputs[0]
        img_feat1 = model_outputs[1]
        teacher_logits1 = model_outputs[2]
        student_logits2 = model_outputs[3]
        img_feat2 = model_outputs[4]
        teacher_logits2 = model_outputs[5]
        logits = model_outputs[6]
        recon_rgb = model_outputs[7] if len(model_outputs) > 7 else None
        rgb_sam = model_outputs[8] if len(model_outputs) > 8 else None

        loss_ce = F.cross_entropy(logits, labels)

        temperature = getattr(cfg, 'PROMPTKD_TEMPERATURE', getattr(cfg, 'TEMPERATURE', 4.0))
        kd_weight = getattr(cfg, 'KD_WEIGHT', getattr(cfg, 'LAMBDA_KL', 1.0))

        patch_promptkd = PromptKDLoss(temperature=temperature, kd_weight=kd_weight, ce_weight=1.0)
        loss_kd1 = patch_promptkd.kd_term(student_logits1, teacher_logits1)
        loss_kd2 = patch_promptkd.kd_term(student_logits2, teacher_logits2)

        loss_recon = torch.tensor(0.0, device=cfg.DEVICE)
        if recon_rgb is not None and rgb_sam is not None:
            loss_recon = F.mse_loss(recon_rgb, rgb_sam)

        loss_total = (
            cfg.LAMBDA_CLS * loss_ce +
            kd_weight * (loss_kd1 + loss_kd2) +
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
            'kd1': loss_kd1.item(),
            'kd2': loss_kd2.item(),
            'loss_recon': loss_recon.item(),
            'loss_hor': loss_hor.item(),
        }

        return loss_total, loss_dict

    return hyperprompt_promptkd_loss_fn
