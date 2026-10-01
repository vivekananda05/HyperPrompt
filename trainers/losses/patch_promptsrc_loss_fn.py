"""
Loss function for PromptSRC single-branch model.
"""

import torch
import torch.nn.functional as F

from trainers.patch_promptsrc import PromptSRCLoss


def get_promptsrc_loss_fn():
    """
    Get loss function for single-branch PromptSRC model.
    
    Returns:
        loss_fn: Function that computes loss from model outputs using PromptSRCLoss
    """
    def promptsrc_loss_fn(model_outputs, labels, model, cfg):
        # Unpack extended model outputs with features
        if len(model_outputs) >= 7:
            student_logits, teacher_logits, rgb, img_feat, text_features, zs_text_feat, zs_img_feat = model_outputs[:7]
        elif len(model_outputs) >= 6:
            # Backward compatibility: old model without zs_img_feat
            student_logits, teacher_logits, rgb, img_feat, text_features, zs_text_feat = model_outputs[:6]
            zs_img_feat = img_feat  # Use student features as fallback
        else:
            # Fallback for very old models
            student_logits, teacher_logits, rgb = model_outputs[:3]
            img_feat = None
            text_features = None
            zs_text_feat = None
            zs_img_feat = None
        
        # Get loss parameters from config
        temperature = getattr(cfg, 'PROMPTSRC_TEMPERATURE', getattr(cfg, 'TEMPERATURE', 1.0))
        text_loss_weight = getattr(cfg, 'TEXT_LOSS_WEIGHT', getattr(cfg, 'LAMBDA_TEXT', 1.0))
        image_loss_weight = getattr(cfg, 'IMAGE_LOSS_WEIGHT', getattr(cfg, 'LAMBDA_IMAGE', 1.0))
        logit_loss_weight = getattr(cfg, 'LOGIT_LOSS_WEIGHT', getattr(cfg, 'LAMBDA_LOGITS', 1.0))
        
        # Create SRC loss module
        src_loss = PromptSRCLoss(
            text_loss_weight=text_loss_weight,
            image_loss_weight=image_loss_weight,
            logit_loss_weight=logit_loss_weight,
            temperature=temperature,
        )
        
        # For single-branch, use learnable image adapter for student features
        # and frozen image features for teacher
        if img_feat is not None and text_features is not None and zs_text_feat is not None and zs_img_feat is not None:
            # Use full SRC loss with all components
            # student: adapted image features + learned text
            # teacher: frozen image features + frozen text
            loss_ce, loss_text, loss_image, loss_logits, total_loss = src_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                student_text_features=text_features,  # Learned from learned prompts
                teacher_text_features=zs_text_feat,   # Frozen zero-shot text
                student_image_features=img_feat,       # Learned from image adapter
                teacher_image_features=zs_img_feat,    # Frozen CLIP image features
                labels=labels,
            )
            loss_dict = {
                'total': total_loss.item(),
                'cls': loss_ce.item(),
                'text': loss_text.item(),
                'image': loss_image.item(),
                'logits_kl': loss_logits.item(),
            }
        else:
            # Fallback: just use CE loss
            loss_ce = F.cross_entropy(student_logits, labels)
            total_loss = loss_ce
            loss_dict = {
                'total': total_loss.item(),
                'cls': loss_ce.item(),
            }
        
        return total_loss, loss_dict
    
    return promptsrc_loss_fn
