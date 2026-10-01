"""
Loss function for PromptKD Pixel (SAM) model.
Uses knowledge distillation loss + classification loss + RGB reconstruction loss.
Based on pixel_coop_loss_fn with PromptKD KD support.
"""

import torch
import torch.nn.functional as F

from trainers.patch_promptkd import PromptKDLoss


def get_pixel_promptkd_loss_fn():
    """
    Get the loss computation function for the PromptKD Pixel (SAM) model.

    Returns:
        loss_fn: A function that computes loss given model outputs and labels.
    """

    def pixel_promptkd_loss_fn(model_outputs, labels, model, cfg):
        """
        PromptKD Pixel loss: KD loss + cross-entropy classification + MSE reconstruction.

        model_outputs tuple:
            (logits, teacher_logits, img_feat, recon_rgb, rgb_sam)

            logits         : [B, C]          student classification logits
            teacher_logits : [B, C]          teacher classification logits
            img_feat       : [B, D]          L2-normalised SAM features (unused in loss)
            recon_rgb      : [B, 3, H, W]    reconstructed RGB from SAM branch
            rgb_sam        : [B, 3, 224, 224] bicubic-resized RGB (reconstruction target)
        """
        student_logits = model_outputs[0]
        teacher_logits = model_outputs[1] if len(model_outputs) > 1 else None
        # model_outputs[2] = img_feat  (not used in loss)
        recon_rgb      = model_outputs[3] if len(model_outputs) > 3 else None
        rgb_sam        = model_outputs[4] if len(model_outputs) > 4 else None

        # ── Get loss parameters ──────────────────────────────────────
        temperature = getattr(cfg, 'PROMPTKD_TEMPERATURE', getattr(cfg, 'TEMPERATURE', 4.0))
        kd_weight = getattr(cfg, 'LAMBDA_KL', getattr(cfg, 'LAMBDA_KNOWLEDGE', 1.0))
        ce_weight = getattr(cfg, 'LAMBDA_CLS', 1.0)
        lambda_mse = getattr(cfg, 'LAMBDA_MSE', 1.0)

        # ── Create KD loss module ────────────────────────────────────
        kd_loss_module = PromptKDLoss(
            temperature=temperature,
            kd_weight=kd_weight,
            ce_weight=ce_weight,
        )

        # ── KD and CE losses ─────────────────────────────────────────
        if teacher_logits is not None:
            loss_ce, loss_kd, _ = kd_loss_module(student_logits, teacher_logits, labels)
        else:
            loss_ce = F.cross_entropy(student_logits, labels)
            loss_kd = torch.tensor(0.0, device=student_logits.device)
        loss_promptkd = ce_weight * loss_ce + kd_weight * loss_kd

        # ── Reconstruction loss ──────────────────────────────────────
        loss_recon = torch.tensor(0.0, device=student_logits.device)
        if recon_rgb is not None and rgb_sam is not None:
            loss_recon = F.mse_loss(recon_rgb, rgb_sam)

        # ── Total loss ───────────────────────────────────────────────
        total_loss = loss_promptkd + lambda_mse * loss_recon

        loss_dict = {
            'total': total_loss.item(),
            'ce': loss_ce.item(),
            'kd': loss_kd.item(),
            'recon': loss_recon.item(),
        }

        return total_loss, loss_dict

    return pixel_promptkd_loss_fn
