"""
PromptSRC single-branch HSI model with independent vision-language prompt learning.

Structure aligns with existing single-branch models in this repository.
Implements the SRC (Self-Reinforcing Contextualization) approach with independent
vision and language prompt depths.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel, CLIPTokenizer

from trainers.hsi_rgb_adapter import HSIRGBAdapter
from trainers.patch_promptsrc import TextEncoder
from clip_utils import clip_preprocess


class VLPromptLearner(nn.Module):
    """
    Independent Vision-Language prompt learner for PromptSRC.

    Learns separate prompt contexts for vision and language branches.
    """

    def __init__(
        self,
        num_classes,
        clip_model,
        tokenizer,
        classnames,
        ctx_len_text=4,
        ctx_len_vision=4,
        ctx_init=None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.ctx_len_text = ctx_len_text
        self.ctx_len_vision = ctx_len_vision
        self.classnames = classnames
        self.tokenizer = tokenizer

        dtype = clip_model.dtype
        ctx_dim = clip_model.text_model.config.hidden_size

        # Initialize text context vectors
        if ctx_init is not None:
            ctx_init = ctx_init.replace("_", " ")
            init_tokens = tokenizer(ctx_init, return_tensors="pt")
            with torch.no_grad():
                embeddings = clip_model.text_model.embeddings
                ctx_emb = embeddings.token_embedding(init_tokens.input_ids)
            # Count available tokens (excluding SOS/EOS)
            n_init_tokens = init_tokens.input_ids.size(1) - 2
            
            if n_init_tokens > ctx_len_text:
                # Warn if init string has more tokens than context slots
                import warnings
                warnings.warn(
                    f"ctx_init '{ctx_init}' tokenizes to {n_init_tokens} tokens "
                    f"but ctx_len_text={ctx_len_text}. Only the first {ctx_len_text} tokens "
                    f"will be used to initialize the context; the rest are discarded. "
                    f"Increase ctx_len_text to use the full init string.",
                    UserWarning, stacklevel=2,
                )
            
            # Initialize with random, then overwrite with available tokens
            ctx_vectors = torch.empty(ctx_len_text, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            n_use = min(n_init_tokens, ctx_len_text)
            ctx_vectors[:n_use] = ctx_emb[0, 1 : 1 + n_use, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(ctx_len_text, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * ctx_len_text)

        self.ctx = nn.Parameter(ctx_vectors)
        self.prompt_prefix = prompt_prefix

    def construct_prompts(self):
        """Construct full text prompts from learned context and class names."""
        ctx = self.ctx
        if ctx.dim() == 2:
            # Expand context for each class
            ctx = ctx.unsqueeze(0).expand(self.num_classes, -1, -1)

        # Build text prompts with context and class names
        prompts = []
        for i, classname in enumerate(self.classnames):
            prompt_str = f"{self.prompt_prefix} {classname}"
            prompts.append(prompt_str)

        return prompts

    def forward(self):
        """Return learned prompts as text strings."""
        prompts = self.construct_prompts()
        return prompts


class HSIPatchPromptSRC(nn.Module):
    """
    HSI classification with PromptSRC-style independent vision-language prompts.

    Returns logits along with teacher logits for self-reinforcing contextualization.
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        classnames,
        clip_name,
        ctx_len_text=4,
        ctx_len_vision=4,
        ctx_init=None,
        temperature=1.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.classnames = classnames
        self.temperature = float(temperature)

        self.rgb_proj = HSIRGBAdapter(in_channels)

        self.clip_model = CLIPModel.from_pretrained(clip_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_name)
        self.clip_dim = int(self.clip_model.visual_projection.weight.shape[0])

        # Freeze CLIP encoders
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Initialize prompt learner
        self.prompt_learner = VLPromptLearner(
            num_classes=num_classes,
            clip_model=self.clip_model,
            tokenizer=self.tokenizer,
            classnames=classnames,
            ctx_len_text=ctx_len_text,
            ctx_len_vision=ctx_len_vision,
            ctx_init=ctx_init,
        )

        # Text encoder for processing learned prompts
        self.text_encoder = TextEncoder(self.clip_model, self.tokenizer)

        # Add learnable image feature adapter (transforms frozen CLIP features -> student features)
        # This creates student vs teacher distinction for image features
        self.image_adapter = nn.Sequential(
            nn.Linear(self.clip_dim, self.clip_dim // 2),
            nn.BatchNorm1d(self.clip_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.clip_dim // 2, self.clip_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Precompute zero-shot text features (frozen teacher)
        with torch.no_grad():
            zs_text_features = []
            for classname in classnames:
                prompt = f"a photo of a {classname}"
                tokens = self.tokenizer(
                    [prompt],
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                )
                text_outputs = self.clip_model.text_model(
                    input_ids=tokens.input_ids.to(self.clip_model.device),
                    attention_mask=tokens.attention_mask.to(self.clip_model.device),
                )
                text_feat = self.clip_model.text_projection(text_outputs.pooler_output)
                zs_text_features.append(text_feat)

            zs_features = torch.cat(zs_text_features, dim=0)
            self.register_buffer("zero_shot_text_features", zs_features)

    def forward(self, images):
        """
        Forward pass.

        Args:
            images: [B, C, H, W] HSI images

        Returns:
            logits: [B, num_classes] student logits
            zero_shot_logits: [B, num_classes] teacher (zero-shot) logits
            rgb: [B, 3, H, W] RGB projection
        """
        rgb = self.rgb_proj(images)
        rgb_clip = clip_preprocess(rgb, image_resolution=224)

        # Vision encoder
        vision_outputs = self.clip_model.vision_model(
            pixel_values=rgb_clip,
            return_dict=True,
        )
        pooled = vision_outputs.pooler_output
        zs_img_feat = self.clip_model.visual_projection(pooled)
        zs_img_feat = F.normalize(zs_img_feat, dim=-1)  # Teacher: frozen image features

        # Student: apply learnable adapter to frozen features
        img_feat = self.image_adapter(zs_img_feat)
        img_feat = F.normalize(img_feat, dim=-1)

        # Learned prompts
        text_prompts = self.prompt_learner()

        # Text encoder processes learned prompts
        text_features = self.text_encoder(text_prompts)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = self.logit_scale.exp().clamp(max=20)
        logits = logit_scale * img_feat @ text_features.t()

        # Zero-shot (frozen) logits for self-reinforcing loss
        zs_text_feat = self.zero_shot_text_features
        zs_text_feat = F.normalize(zs_text_feat, dim=-1)
        zero_shot_logits = logit_scale * zs_img_feat @ zs_text_feat.t()

        # Return features for SRC loss computation
        # student_logits from (learned adapter image + learned text)
        # teacher_logits from (frozen image + frozen text)
        return logits, zero_shot_logits, rgb, img_feat, text_features, zs_text_feat, zs_img_feat

    def encode_text_features(self):
        """Return learned text features."""
        text_prompts = self.prompt_learner()
        return self.text_encoder(text_prompts)
