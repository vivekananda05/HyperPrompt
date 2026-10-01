"""
PromptSRC utilities and loss functions.

This module implements the PromptSRC training components following the
independent vision-language prompt learning approach with self-reinforcing
contextualization (SRC).

Reference:
  - PromptSRC: Towards Robust Prompt Learning for Vision-Language Models via Self-Reinforcing Contextualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """CLIP text encoder that processes prompts with positional embeddings."""

    def __init__(self, clip_model, tokenizer):
        super().__init__()
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.dtype = clip_model.dtype

    def forward(self, prompts_text):
        """
        Process text through CLIP text encoder.

        Args:
            prompts_text: List of text strings or tokenized prompts

        Returns:
            text_features: [C, D] normalized text features
        """
        if isinstance(prompts_text, list):
            # Tokenize text prompts
            tokens = self.tokenizer(
                prompts_text,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
        else:
            tokens = prompts_text

        text_outputs = self.clip_model.text_model(
            input_ids=tokens.input_ids.to(self.clip_model.device),
            attention_mask=tokens.attention_mask.to(self.clip_model.device),
            return_dict=True,
        )
        text_features = self.clip_model.text_projection(text_outputs.pooler_output)
        return text_features


class PromptSRCLoss(nn.Module):
    """
    PromptSRC loss combining multiple objectives:

    1. L_CE: Cross-entropy on student logits
    2. L_SCL_text: L1 loss between learned and frozen text features
    3. L_SCL_image: L1 loss between learned and frozen image features  
    4. L_SCL_logits: KL divergence between learned and frozen logits

    Total loss = L_CE + (L_SCL_text + L_SCL_image + L_SCL_logits)
    """

    def __init__(
        self,
        text_loss_weight=1.0,
        image_loss_weight=1.0,
        logit_loss_weight=1.0,
        temperature=1.0,
    ):
        super().__init__()
        self.text_loss_weight = float(text_loss_weight)
        self.image_loss_weight = float(image_loss_weight)
        self.logit_loss_weight = float(logit_loss_weight)
        self.temperature = float(temperature)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_text_features: torch.Tensor,
        teacher_text_features: torch.Tensor,
        student_image_features: torch.Tensor,
        teacher_image_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple:
        """
        Compute PromptSRC loss.

        Args:
            student_logits: [B, C] student model logits
            teacher_logits: [B, C] frozen/teacher model logits
            student_text_features: [B, D] learned text features
            teacher_text_features: [B, D] frozen text features
            student_image_features: [B, D] learned image features
            teacher_image_features: [B, D] frozen image features
            labels: [B] class labels

        Returns:
            Tuple of (loss_ce, loss_scl_text, loss_scl_image, loss_scl_logits, total_loss)
        """
        # Cross-entropy loss
        loss_ce = F.cross_entropy(student_logits, labels)

        # Self-Reinforcing Contextualization (SRC) losses
        loss_scl_text = F.l1_loss(
            student_text_features,
            teacher_text_features.detach(),
            reduction="mean",
        ) * self.text_loss_weight

        loss_scl_image = F.l1_loss(
            student_image_features,
            teacher_image_features.detach(),
            reduction="mean",
        ) * self.image_loss_weight

        # KL divergence on logits
        loss_scl_logits = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.log_softmax(teacher_logits.detach() / self.temperature, dim=1),
            reduction="sum",
            log_target=True,
        ) * (self.temperature * self.temperature) / student_logits.numel()
        loss_scl_logits = loss_scl_logits * self.logit_loss_weight

        total_loss = loss_ce + loss_scl_text + loss_scl_image + loss_scl_logits

        return loss_ce, loss_scl_text, loss_scl_image, loss_scl_logits, total_loss


class GaussianProgressAnnealing:
    """
    Gaussian Progress Annealing (GPA) scheduler.

    Applies a Gaussian weighting schedule over training epochs for model
    averaging. Models from earlier epochs receive lower weights under a
    Gaussian curve centered at a specified mean epoch.

    Args:
        total_epochs: Total number of training epochs
        mean: Mean epoch for Gaussian (typically around 0.7-0.8 * total_epochs)
        std: Standard deviation of the Gaussian
    """

    def __init__(self, total_epochs, mean, std):
        self.total_epochs = int(total_epochs)
        self.mean = float(mean)
        self.std = float(std)

        # Precompute Gaussian weights for all epochs
        import numpy as np

        def gauss_fn(x):
            return (1.0 / (self.std * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((x - self.mean) / self.std) ** 2
            )

        self.weights = np.array([gauss_fn(epoch) for epoch in range(1, self.total_epochs + 1)])
        self.weights = self.weights / np.sum(self.weights)

    def __getitem__(self, epoch: int) -> float:
        """Get weight for epoch (1-indexed)."""
        if 1 <= epoch <= self.total_epochs:
            return float(self.weights[epoch - 1])
        return 0.0

    def __len__(self):
        return self.total_epochs


def average_state_dicts(
    state_dicts: list,
    weights: list = None,
    prompt_only: bool = False,
) -> dict:
    """
    Average multiple state dicts with optional weights.

    Args:
        state_dicts: List of model state dicts to average
        weights: Optional list of weights (will be normalized)
        prompt_only: If True, only average prompt-related parameters

    Returns:
        Averaged state dict
    """
    if not state_dicts:
        raise ValueError("state_dicts cannot be empty")

    if weights is None:
        weights = [1.0] * len(state_dicts)
    else:
        weights = list(weights)

    # Normalize weights
    total = sum(weights)
    weights = [w / total for w in weights]

    # Initialize result with first state dict (scaled)
    averaged = {}
    for key in state_dicts[0].keys():
        if prompt_only and "prompt" not in key.lower():
            continue
        averaged[key] = state_dicts[0][key] * weights[0]

    # Add remaining state dicts
    for state_dict, weight in zip(state_dicts[1:], weights[1:]):
        for key in state_dict.keys():
            if prompt_only and "prompt" not in key.lower():
                if key not in averaged:
                    averaged[key] = state_dict[key] * weight
            else:
                averaged[key] = averaged.get(key, 0) + state_dict[key] * weight

    return averaged
