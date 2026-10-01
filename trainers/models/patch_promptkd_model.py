"""
PromptKD single-branch HSI model.

Structure is aligned with existing single-branch models in this repository.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel, CLIPTokenizer

from trainers.hsi_rgb_adapter import HSIRGBAdapter
from trainers.patch_promptkd import PromptLearner, ZeroShotCLIP, FeatureTransModuleTwoLayer
from clip_utils import clip_preprocess


class HSIPatchPromptKD(nn.Module):
    """
    HSI classification model using PromptKD-style teacher-student distillation.

    Returns student and teacher logits so loss computation can apply KD.
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        classnames,
        clip_name,
        ctx_len=4,
        class_token_position="end",
        csc=False,
        ctx_init=None,
        temperature=4.0,
        kd_weight=1.0,
        use_feature_transform=True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.classnames = classnames
        self.temperature = float(temperature)
        self.kd_weight = float(kd_weight)

        self.rgb_proj = HSIRGBAdapter(in_channels)

        self.clip_model = CLIPModel.from_pretrained(clip_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_name)
        self.clip_dim = int(self.clip_model.visual_projection.weight.shape[0])

        for p in self.clip_model.parameters():
            p.requires_grad = False

        prompt_cfg = {
            "ctx_len": ctx_len,
            "class_token_position": class_token_position,
            "csc": csc,
            "ctx_init": ctx_init,
        }
        self.prompt_learner = PromptLearner(
            prompt_cfg,
            classnames,
            self.clip_model,
            self.tokenizer,
        )

        self.zs_clip = ZeroShotCLIP(classnames, self.clip_model, self.tokenizer)

        self.use_feature_transform = bool(use_feature_transform)
        if self.use_feature_transform:
            self.image_feature_transform = FeatureTransModuleTwoLayer(
                input_dim=self.clip_dim,
                out_dim=self.clip_dim,
            )
        else:
            self.image_feature_transform = nn.Identity()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images):
        rgb = self.rgb_proj(images)
        rgb_clip = clip_preprocess(rgb, image_resolution=224)

        vision_outputs = self.clip_model.vision_model(
            pixel_values=rgb_clip,
            return_dict=True,
        )
        pooled = vision_outputs.pooler_output
        img_feat = self.clip_model.visual_projection(pooled)
        img_feat = self.image_feature_transform(img_feat)
        img_feat = F.normalize(img_feat, dim=-1)

        student_text_features = self.prompt_learner()

        logit_scale = self.logit_scale.exp().clamp(max=20)
        student_logits = logit_scale * img_feat @ student_text_features.t()

        with torch.no_grad():
            teacher_logits = self.zs_clip(rgb_clip)

        return student_logits, teacher_logits, rgb

    def encode_text_features(self):
        return self.prompt_learner()
