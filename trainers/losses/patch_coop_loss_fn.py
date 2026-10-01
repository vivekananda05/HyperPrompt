"""
Loss function for CoOp model.
Uses only classification loss with cross-entropy.
"""

import torch
import torch.nn.functional as F


def get_coop_loss_fn():
    """
    Get the loss computation function for CoOp model.
    
    Returns:
        loss_fn: A function that computes loss given model outputs and labels
    """
    def coop_loss_fn(model_outputs, labels, model, cfg):
        # Single-branch models return just logits
        logits = model_outputs
        
        # Classification loss only
        loss_cls = F.cross_entropy(logits, labels)

        # Total loss (with lambda weight)
        total_loss = cfg.LAMBDA_CLS * loss_cls

        loss_dict = {
            'total': total_loss.item(),
            'cls': loss_cls.item(),
        }
        
        return total_loss, loss_dict

    return coop_loss_fn
