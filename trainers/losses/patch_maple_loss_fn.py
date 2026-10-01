"""
Loss function for MaPLe model.
Uses classification loss with cross-entropy.
"""

import torch
import torch.nn.functional as F


def get_maple_loss_fn():
    """
    Get the loss computation function for MaPLe model.
    
    Returns:
        loss_fn: A function that computes loss given model outputs and labels
    """
    def maple_loss_fn(model_outputs, labels, model, cfg):
        # MaPLe model can return either:
        #   - logits tensor
        #   - tuple where first item is logits (e.g., logits, rgb)
        if isinstance(model_outputs, (tuple, list)):
            logits = model_outputs[0]
        else:
            logits = model_outputs

        if not torch.is_tensor(logits):
            raise TypeError(
                f"Expected logits to be a Tensor, got {type(logits)}"
            )
        
        # Classification loss only
        loss_cls = F.cross_entropy(logits, labels)

        # Total loss (with lambda weight)
        total_loss = cfg.LAMBDA_CLS * loss_cls

        loss_dict = {
            'total': total_loss.item(),
            'cls': loss_cls.item(),
        }
        
        return total_loss, loss_dict

    return maple_loss_fn
