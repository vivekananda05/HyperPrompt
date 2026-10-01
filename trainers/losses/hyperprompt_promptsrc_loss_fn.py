"""Loss for PromptSRC + HyperPrompt."""

import torch
import torch.nn.functional as F


def get_hyperprompt_promptsrc_loss_fn():
    def hyperprompt_promptsrc_loss_fn(model_outputs, labels, model, cfg):
        (
            logits1, img_feat1, text_feat1,
            logits2, img_feat2, text_feat2,
            logits, recon_rgb, rgb_sam,
            teacher_logits1, teacher_logits2,
            zs_img_feat1, zs_img_feat2,
            zs_text_feat1, zs_text_feat2,
        ) = model_outputs[:15]

        loss_ce = F.cross_entropy(logits, labels)
        loss_scl_text = 0.5 * (
            F.l1_loss(text_feat1, zs_text_feat1.detach())
            + F.l1_loss(text_feat2, zs_text_feat2.detach())
        )
        loss_scl_image = 0.5 * (
            F.l1_loss(img_feat1, zs_img_feat1.detach())
            + F.l1_loss(img_feat2, zs_img_feat2.detach())
        )

        temperature = getattr(cfg, "PROMPTSRC_TEMPERATURE", getattr(cfg, "TEMPERATURE", 1.0))
        loss_scl_logits = 0.5 * sum(
            F.kl_div(
                F.log_softmax(student / temperature, dim=1),
                F.log_softmax(teacher.detach() / temperature, dim=1),
                reduction="sum",
                log_target=True,
            ) * (temperature * temperature) / student.numel()
            for student, teacher in ((logits1, teacher_logits1), (logits2, teacher_logits2))
        )

        loss_scl_text = getattr(cfg, "LAMBDA_SRC_TEXT", 1.0) * loss_scl_text
        loss_scl_image = getattr(cfg, "LAMBDA_SRC_IMAGE", 1.0) * loss_scl_image
        loss_scl_logits = getattr(cfg, "LAMBDA_SRC_LOGIT", 1.0) * loss_scl_logits
        loss_recon = F.mse_loss(recon_rgb, rgb_sam)

        loss_hor = torch.zeros((), device=logits.device)
        pclra = getattr(model, "pclra", None)
        if pclra is not None:
            hor_output = pclra.loss_hor()
            loss_hor = hor_output[0] if isinstance(hor_output, tuple) else hor_output
        loss_hor = getattr(cfg, "LAMBDA_HOR", 1.0) * loss_hor

        loss_total = (
            getattr(cfg, "LAMBDA_CLS", 1.0) * loss_ce
            + loss_scl_text
            + loss_scl_image
            + loss_scl_logits
            + getattr(cfg, "LAMBDA_MSE", 0.1) * loss_recon
            + loss_hor
        )
        return loss_total, {
            "loss_total": loss_total.item(),
            "loss_ce": loss_ce.item(),
            "loss_scl_text": loss_scl_text.item(),
            "loss_scl_image": loss_scl_image.item(),
            "loss_scl_logits": loss_scl_logits.item(),
            "loss_recon": loss_recon.item(),
            "loss_hor": loss_hor.item(),
        }

    return hyperprompt_promptsrc_loss_fn
