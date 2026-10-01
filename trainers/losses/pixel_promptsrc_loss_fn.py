"""
Loss function for PromptSRC Pixel (SAM) model.
Uses 4-term SRC loss (CE + text consistency + image consistency + logit KL) + RGB reconstruction.
Based on standard PromptSRC loss with pixel model extensions.
"""

import torch
import torch.nn.functional as F


def get_pixel_promptsrc_loss_fn():
    """
    Get the loss computation function for the PromptSRC Pixel (SAM) model.

    Returns:
        loss_fn: A function that computes loss given model outputs and labels.
    """

    def pixel_promptsrc_loss_fn(model_outputs, labels, model, cfg):
        """
        PromptSRC Pixel loss: 4-term SRC loss + MSE reconstruction.

        model_outputs tuple (8 values):
            logits           : [B, C]           student contrastive logits
            zero_shot_logits : [B, C]           teacher (zero-shot) logits
            recon_rgb        : [B, 3, 224, 224] reconstructed RGB
            rgb_sam          : [B, 3, 224, 224] reconstruction target
            img_feat         : [B, D]           L2-normalized student image features
            text_feat        : [C, D]           L2-normalized student text features
            zs_text_feat     : [C, D]           L2-normalized teacher text features
            zs_img_feat      : [B, D]           L2-normalized teacher image features

        Loss components:
            1. L_CE: Cross-entropy on student logits
            2. L_SCL_text: L1 loss between learned and frozen text features
            3. L_SCL_image: L1 loss between learned and frozen image features
            4. L_SCL_logits: KL divergence between learned and frozen logits
            5. L_RECON: MSE reconstruction loss (pixel model specific)
        """
        logits           = model_outputs[0]
        zero_shot_logits = model_outputs[1]
        recon_rgb        = model_outputs[2] if len(model_outputs) > 2 else None
        rgb_sam          = model_outputs[3] if len(model_outputs) > 3 else None
        img_feat         = model_outputs[4] if len(model_outputs) > 4 else None
        text_feat        = model_outputs[5] if len(model_outputs) > 5 else None
        zs_text_feat     = model_outputs[6] if len(model_outputs) > 6 else None
        zs_img_feat      = model_outputs[7] if len(model_outputs) > 7 else None

        # ── Classification loss ──────────────────────────────────────
        loss_cls = F.cross_entropy(logits, labels)
        # ── Self-Reinforcing Contextualization (SRC) losses ──────────
        loss_scl_text = torch.tensor(0.0, device=logits.device)
        loss_scl_image = torch.tensor(0.0, device=logits.device)
        loss_scl_logits = torch.tensor(0.0, device=logits.device)

        # L1 loss: learned text vs frozen text
        if text_feat is not None and zs_text_feat is not None:
            loss_scl_text = F.l1_loss(
                text_feat,
                zs_text_feat.detach(),
                reduction="mean",
            )

        # L1 loss: learned image vs frozen image
        if img_feat is not None and zs_img_feat is not None:
            loss_scl_image = F.l1_loss(
                img_feat,
                zs_img_feat.detach(),
                reduction="mean",
            )

        # KL divergence: learned logits vs frozen logits
        if logits is not None and zero_shot_logits is not None:
            temperature = getattr(cfg, 'TEMPERATURE', 1.0)
            loss_scl_logits = F.kl_div(
                F.log_softmax(logits / temperature, dim=1),
                F.log_softmax(zero_shot_logits.detach() / temperature, dim=1),
                reduction="sum",
                log_target=True,
            ) * (temperature * temperature) / logits.numel()
        # ── Reconstruction loss ──────────────────────────────────────
        loss_recon = torch.tensor(0.0, device=logits.device)
        if recon_rgb is not None and rgb_sam is not None:
            loss_recon = F.mse_loss(recon_rgb, rgb_sam)
        # ── Weight SRC losses ────────────────────────────────────────
        lambda_text = getattr(cfg, 'LAMBDA_SRC_TEXT', 1.0)
        lambda_image = getattr(cfg, 'LAMBDA_SRC_IMAGE', 1.0)
        lambda_logit = getattr(cfg, 'LAMBDA_SRC_LOGIT', 1.0)

        loss_scl_text = loss_scl_text * lambda_text
        loss_scl_image = loss_scl_image * lambda_image
        loss_scl_logits = loss_scl_logits * lambda_logit

        # ── Total loss ───────────────────────────────────────────────
        total_loss = (
            cfg.LAMBDA_CLS * loss_cls +
            loss_scl_text +
            loss_scl_image +
            loss_scl_logits +
            cfg.LAMBDA_MSE * loss_recon
        )

        loss_dict = {
            'total': total_loss.item(),
            'cls': loss_cls.item(),
            'scl_text': loss_scl_text.item(),
            'scl_image': loss_scl_image.item(),
            'scl_logits': loss_scl_logits.item(),
            'recon': loss_recon.item(),
        }

        return total_loss, loss_dict

    return pixel_promptsrc_loss_fn
