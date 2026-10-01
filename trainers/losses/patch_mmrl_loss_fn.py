"""
Loss function for the single-branch MMRL (Multi-Modal Representation Learning) model.

Uses fused classification logits plus cosine similarity regularization to the
frozen CLIP image features.
"""

import torch
import torch.nn.functional as F


def get_mmrl_loss_fn():
    """
    Get the loss computation function for MMRL model.
    
    Returns:
        loss_fn: A function that computes loss given model outputs and labels
    """
    def mmrl_loss_fn(model_outputs, labels, model, cfg):
        # model returns: (logits_main, logits_token_enhanced, logits_fused,
        #                 image_features, image_features_frozen)
        if not isinstance(model_outputs, (tuple, list)) or len(model_outputs) < 5:
            raise ValueError(
                f"Expected at least 5 outputs from MMRL model, got {type(model_outputs)} / {len(model_outputs) if isinstance(model_outputs, (tuple, list)) else 'n/a'}"
            )

        logits_main, logits_token_enhanced, logits_fused, image_features, image_features_frozen = model_outputs[:5]
        
        # Classification losses
        loss_cls = F.cross_entropy(logits_fused, labels)


        # Regularization: keep the learned branch close to the frozen CLIP image space
        # while still allowing the representation tokens to move the features.
        img_norm = F.normalize(image_features, dim=-1)
        frozen_img_norm = F.normalize(image_features_frozen, dim=-1)
        cossim_reg = 1.0 - torch.mean(
            F.cosine_similarity(img_norm, frozen_img_norm, dim=-1)
        )
        # Regularization weight
        reg_weight = getattr(cfg, 'LAMBDA_REG', getattr(cfg, 'REG_WEIGHT', 0.2))
        
        # Total loss
        total_loss = loss_cls + reg_weight * cossim_reg
        
        loss_dict = {
            'total': total_loss.item(),
            'cls': loss_cls.item(),
            'reg': cossim_reg.item(),
        }
        
        return total_loss, loss_dict

    return mmrl_loss_fn
