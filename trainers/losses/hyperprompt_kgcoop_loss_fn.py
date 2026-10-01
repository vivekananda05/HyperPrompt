"""
Loss function for Dual KgCoOp PCLRA model.
Aligned with Dual KgCoOp base loss behavior plus PCLRA diversity loss.
"""

import torch
import torch.nn.functional as F


def get_hyperprompt_kgcoop_loss_fn():
    """
    Get the loss computation function for Dual KgCoOp PCLRA model.

    Returns:
        loss_fn: A function that computes loss given model outputs and labels.
    """

    def hyperprompt_kgcoop_loss_fn(model_outputs, labels, model, cfg):
        # NEW STRUCTURE: (logits1, img_feat1, text_feat1, knowledge_loss1, logits2, img_feat2, text_feat2, knowledge_loss2, logits, recon_rgb, rgb_sam)
        logits1 = model_outputs[0]
        img_feat1 = model_outputs[1]
        text_feat1 = model_outputs[2]
        knowledge_loss1 = model_outputs[3] if len(model_outputs) > 3 else None
        logits2 = model_outputs[4]
        img_feat2 = model_outputs[5]
        text_feat2 = model_outputs[6]
        knowledge_loss2 = model_outputs[7] if len(model_outputs) > 7 else None
        logits = model_outputs[8]
        recon_rgb = model_outputs[9] if len(model_outputs) > 9 else None
        rgb_sam = model_outputs[10] if len(model_outputs) > 10 else None

        loss_ce = F.cross_entropy(logits, labels)

        loss_recon = torch.tensor(0.0, device=labels.device)
        if recon_rgb is not None and rgb_sam is not None:
            loss_recon = F.mse_loss(recon_rgb, rgb_sam)

        if not isinstance(knowledge_loss1, torch.Tensor):
            knowledge_loss1 = torch.tensor(0.0, device=labels.device)
        else:
            knowledge_loss1 = knowledge_loss1.to(labels.device)
            if knowledge_loss1.ndim > 0:
                knowledge_loss1 = knowledge_loss1.mean()

        if not isinstance(knowledge_loss2, torch.Tensor):
            knowledge_loss2 = torch.tensor(0.0, device=labels.device)
        else:
            knowledge_loss2 = knowledge_loss2.to(labels.device)
            if knowledge_loss2.ndim > 0:
                knowledge_loss2 = knowledge_loss2.mean()

        loss_knowledge = (knowledge_loss1 + knowledge_loss2) / 2.0

        lambda_cls = getattr(cfg, 'LAMBDA_CLS', 1.0)
        lambda_mse = getattr(cfg, 'LAMBDA_MSE', 0.0)
        lambda_knowledge = getattr(cfg, 'LAMBDA_KNOWLEDGE', None)
        if lambda_knowledge is None:
            lambda_knowledge = getattr(model, 'knowledge_weight', 1.0)

        loss_total = (
            lambda_cls * loss_ce +
            lambda_mse * loss_recon +
            lambda_knowledge * loss_knowledge
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
            'knowledge': loss_knowledge.item(),
            'knowledge1': knowledge_loss1.item() if isinstance(knowledge_loss1, torch.Tensor) else 0.0,
            'knowledge2': knowledge_loss2.item() if isinstance(knowledge_loss2, torch.Tensor) else 0.0,
            'loss_hor': loss_hor.item(),
        }

        return loss_total, loss_dict

    return hyperprompt_kgcoop_loss_fn
