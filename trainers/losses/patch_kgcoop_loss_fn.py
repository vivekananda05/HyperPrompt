"""
Loss function for KgCoOp model.
Uses classification loss with cross-entropy.
"""

import torch
import torch.nn.functional as F


def get_kgcoop_loss_fn():
    """
    Get the loss computation function for KgCoOp model.
    
    Returns:
        loss_fn: A function that computes loss given model outputs and labels
    """
    def kgcoop_loss_fn(model_outputs, labels, model, cfg):
        """
        KgCoOp model returns:
          0. logits           : [B, C] classification logits
          1. knowledge_loss   : scalar  knowledge regularization loss
          2. text_feat        : [C, D] or [B, C, D] text features
          3. img_feat         : [B, D] image features
          4. rgb              : [B, 3, H, W] reconstructed RGB
        """
        # Extract model outputs
        logits = model_outputs[0]
        knowledge_loss = model_outputs[1] if len(model_outputs) > 1 else None
        text_feat = model_outputs[2] if len(model_outputs) > 2 else None
        img_feat = model_outputs[3] if len(model_outputs) > 3 else None
        rgb = model_outputs[4] if len(model_outputs) > 4 else None
        
        # Classification loss
        loss_cls = F.cross_entropy(logits, labels)
        
        # Knowledge loss (regularization from KgCoOp text encoder)
        if not isinstance(knowledge_loss, torch.Tensor):
            knowledge_loss = torch.tensor(0.0, device=labels.device)
        else:
            knowledge_loss = knowledge_loss.to(labels.device)
            if knowledge_loss.ndim > 0:
                knowledge_loss = knowledge_loss.mean()

        lambda_cls = getattr(cfg, 'LAMBDA_CLS', 1.0)
        lambda_knowledge = getattr(cfg, 'LAMBDA_KNOWLEDGE', None)
        if lambda_knowledge is None:
            lambda_knowledge = getattr(model, 'knowledge_weight', 1.0)

        # Total loss with weighted components
        total_loss = (
            lambda_cls * loss_cls +
            lambda_knowledge * knowledge_loss
        )

        loss_dict = {
            'total': total_loss.item(),
            'cls': loss_cls.item(),
            'knowledge': knowledge_loss.item() if isinstance(knowledge_loss, torch.Tensor) else 0.0,
        }
        
        return total_loss, loss_dict

    return kgcoop_loss_fn
