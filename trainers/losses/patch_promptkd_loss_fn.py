"""
Loss function for PromptKD model.
"""

import torch
import torch.nn.functional as F

from trainers.patch_promptkd import PromptKDLoss


def get_promptkd_loss_fn():
    def promptkd_loss_fn(model_outputs, labels, model, cfg):
        if isinstance(model_outputs, tuple) and len(model_outputs) >= 2:
            student_logits = model_outputs[0]
            teacher_logits = model_outputs[1]
        else:
            student_logits = model_outputs
            teacher_logits = None

        temperature = getattr(cfg, 'PROMPTKD_TEMPERATURE', getattr(cfg, 'TEMPERATURE', 4.0))
        kd_weight = getattr(
            cfg,
            'KD_WEIGHT',
            getattr(cfg, 'LAMBDA_KL', getattr(cfg, 'LAMBDA_KNOWLEDGE', 1.0)),
        )
        ce_weight = getattr(cfg, 'LAMBDA_CLS', 1.0)

        kd_loss_module = PromptKDLoss(
            temperature=temperature,
            kd_weight=kd_weight,
            ce_weight=ce_weight,
        )

        if teacher_logits is not None:
            loss_ce, loss_kd, total_loss = kd_loss_module(student_logits, teacher_logits, labels)
        else:
            loss_ce = F.cross_entropy(student_logits, labels)
            loss_kd = torch.tensor(0.0, device=student_logits.device)
            total_loss = ce_weight * loss_ce

        return total_loss, {
            'total': total_loss.item(),
            'ce': loss_ce.item(),
            'kd': loss_kd.item(),
        }

    return promptkd_loss_fn
