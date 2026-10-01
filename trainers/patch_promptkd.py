"""
PromptKD utilities.

This module mirrors the project style used by other prompt-learning trainers
while keeping the implementation lightweight for the current training pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer


# ========================================================================
# Utilities
# ========================================================================

def _make_causal_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Build a causal (upper-triangular) attention mask of shape [1, 1, seq_len, seq_len].
    Masked positions are filled with -inf so they are ignored by softmax.
    This mirrors what HuggingFace CLIP does internally when input_ids are provided.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)          # upper triangle = -inf
    return mask.unsqueeze(0).unsqueeze(0)        # [1, 1, seq_len, seq_len]


# ========================================================================
# Simple Text Encoder (frozen, for zero-shot baseline)
# ========================================================================

class TextEncoder(nn.Module):
    """
    Simple text encoder wrapper for computing text embeddings.
    
    Used for encoding text prompts in both teacher (zero-shot) and student
    (learnable context prompts) models.
    """
    
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.text_model.encoder
        self.positional_embedding = clip_model.text_model.embeddings.position_embedding
        self.ln_final = clip_model.text_model.final_layer_norm
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        """
        Args:
            prompts: [B, L, D]  — embedded prompts
            tokenized_prompts: [B, L]  — token indices (for EOS position)
        
        Returns:
            text_features: [B, D]  — L2-normalized text embeddings
        """
        x = prompts
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(
            inputs_embeds=x,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # Pool at EOS token position
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


# ========================================================================
# Learnable Prompt Encoder (CoOp-style)
# ========================================================================

class PromptLearner(nn.Module):
    """
    Learnable prompt encoder following CoOp's design.
    
    Manages learnable context embeddings and produces prompt embeddings
    for the student model. Supports multiple class-token positions:
    - "end": [SOS | ctx | class_name | EOS]
    - "front": [SOS | class_name | ctx | EOS]
    - "middle": [SOS | ctx_half1 | class_name | ctx_half2 | EOS]
    """
    
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        
        self.clip = clip_model
        self.tokenizer = tokenizer
        self.classnames = classnames
        
        self.n_cls = len(classnames)
        self.n_ctx = cfg.get("ctx_len", 16)
        self.ctx_init = cfg.get("ctx_init", None)
        self.class_token_position = cfg.get("class_token_position", "end")
        self.csc = cfg.get("csc", False)
        
        self.hidden_dim = clip_model.text_model.config.hidden_size
        self.dtype = clip_model.dtype

        # ====================================================
        # Context Initialization
        # ====================================================
        if self.ctx_init:
            ctx_init_text = self.ctx_init.replace("_", " ")
            init_tokens = tokenizer(ctx_init_text, return_tensors="pt")
            with torch.no_grad():
                embedding = clip_model.text_model.embeddings.token_embedding(
                    init_tokens.input_ids
                )
            n_init_tokens = init_tokens.input_ids.size(1) - 2  # excl. SOS/EOS
            if n_init_tokens > self.n_ctx:
                import warnings
                warnings.warn(
                    f"ctx_init tokenizes to {n_init_tokens} tokens but n_ctx={self.n_ctx}. "
                    f"Only the first {self.n_ctx} tokens will be used.",
                    UserWarning, stacklevel=2,
                )
            n_use = min(n_init_tokens, self.n_ctx)
            base_ctx = torch.empty(self.n_ctx, self.hidden_dim)
            nn.init.normal_(base_ctx, std=0.02)
            base_ctx[:n_use] = embedding[0, 1:1+n_use, :]
        else:
            base_ctx = torch.empty(self.n_ctx, self.hidden_dim)
            nn.init.normal_(base_ctx, std=0.02)

        # Class-specific or shared context
        if self.csc:
            ctx = base_ctx.unsqueeze(0).repeat(self.n_cls, 1, 1)  # [C, n_ctx, D]
        else:
            ctx = base_ctx  # [n_ctx, D]

        self.ctx = nn.Parameter(ctx)

        # ====================================================
        # Pre-tokenize class prompts
        # ====================================================
        prompts = [f"{name}." for name in classnames]
        tokenized = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        
        self.register_buffer("input_ids", tokenized.input_ids)
        self.register_buffer("attention_mask", tokenized.attention_mask)

    def forward(self) -> torch.Tensor:
        """
        Generate prompt embeddings for all classes.
        
        Returns:
            text_features: [C, D]  — L2-normalized class embeddings
        """
        device = self.ctx.device
        C = self.n_cls
        n_ctx = self.n_ctx

        token_embed = self.clip.text_model.embeddings.token_embedding
        pos_embed = self.clip.text_model.embeddings.position_embedding

        input_ids = self.input_ids.to(device)
        text_embed = token_embed(input_ids)  # [C, L_orig, D]

        # Expand context to [C, n_ctx, D]
        if self.csc:
            ctx = self.ctx
        else:
            ctx = self.ctx.unsqueeze(0).expand(C, -1, -1)

        sos = text_embed[:, :1, :]       # [C, 1, D]
        rest = text_embed[:, 1:, :]      # [C, L_orig-1, D]

        # ====================================================
        # Build sequence based on class_token_position
        # ====================================================
        if self.class_token_position == "end":
            # [SOS | ctx | class_name | period | EOS | PAD...]
            x = torch.cat([sos, ctx, rest], dim=1)
            eos_id = self.tokenizer.eos_token_id
            eos_in_rest = (input_ids == eos_id).int().argmax(dim=1) - 1
            eos_pos_in_x = 1 + n_ctx + eos_in_rest

        elif self.class_token_position == "front":
            # [SOS | class_name | period | ctx | EOS]
            eos_id = self.tokenizer.eos_token_id
            real_len = self.attention_mask.to(device).sum(dim=1) - 1
            max_body = real_len.max().item() - 1
            seq_len = 1 + max_body + n_ctx + 1

            x = torch.zeros(C, seq_len, self.hidden_dim, device=device, dtype=ctx.dtype)
            eos_pos_in_x = torch.zeros(C, dtype=torch.long, device=device)
            eos_embed = token_embed(torch.tensor([eos_id], device=device)).squeeze(0)

            for i in range(C):
                body_len = real_len[i].item() - 1
                body = rest[i, :body_len, :]
                row = torch.cat([sos[i], body, ctx[i], eos_embed.unsqueeze(0)], dim=0)
                x[i, :row.size(0)] = row
                eos_pos_in_x[i] = 1 + body_len + n_ctx

        elif self.class_token_position == "middle":
            # [SOS | ctx_half1 | class_name | period | ctx_half2 | EOS]
            half1 = n_ctx // 2
            half2 = n_ctx - half1
            ctx1 = ctx[:, :half1, :]
            ctx2 = ctx[:, half1:, :]

            eos_id = self.tokenizer.eos_token_id
            real_len = self.attention_mask.to(device).sum(dim=1) - 1
            max_body = real_len.max().item() - 1
            seq_len = 1 + half1 + max_body + half2 + 1

            x = torch.zeros(C, seq_len, self.hidden_dim, device=device, dtype=ctx.dtype)
            eos_pos_in_x = torch.zeros(C, dtype=torch.long, device=device)
            eos_embed = token_embed(torch.tensor([eos_id], device=device)).squeeze(0)

            for i in range(C):
                body_len = real_len[i].item() - 1
                body = rest[i, :body_len, :]
                row = torch.cat([
                    sos[i], ctx1[i], body, ctx2[i], eos_embed.unsqueeze(0)
                ], dim=0)
                x[i, :row.size(0)] = row
                eos_pos_in_x[i] = 1 + half1 + body_len + half2

        else:
            raise ValueError(f"Invalid class_token_position: {self.class_token_position}")

        # ====================================================
        # Add positional embeddings and pass through transformer
        # ====================================================
        seq_len = x.size(1)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(C, -1)
        x = x + pos_embed(pos_ids)

        causal_mask = _make_causal_mask(seq_len, x.dtype, device).expand(C, -1, -1, -1)

        hidden = self.clip.text_model.encoder(
            inputs_embeds=x,
            attention_mask=None,
            causal_attention_mask=causal_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state

        hidden = self.clip.text_model.final_layer_norm(hidden)

        # Pool at EOS position
        text_features = hidden[
            torch.arange(C, device=device),
            eos_pos_in_x,
        ]

        # Project to CLIP space and normalize
        text_features = self.clip.text_projection(text_features)
        text_features = F.normalize(text_features, dim=-1)

        return text_features


# ========================================================================
# Zero-Shot CLIP (Frozen Teacher)
# ========================================================================

class ZeroShotCLIP(nn.Module):
    """
    Zero-shot CLIP model (teacher).
    
    Frozen reference model that provides stable guidance signals
    for the student during PromptKD training.
    """
    
    def __init__(self, classnames, clip_model, tokenizer):
        super().__init__()
        
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.classnames = classnames
        
        # Generate zero-shot prompts
        prompts = [f"a photo of a {name}." for name in classnames]
        tokenized = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        
        # Compute and cache text features
        with torch.no_grad():
            text_features = self._encode_text(tokenized.input_ids)
            text_features = F.normalize(text_features, dim=-1)
        
        self.register_buffer("text_features", text_features)

    def _encode_text(self, input_ids):
        """Helper to encode text given token IDs."""
        with torch.no_grad():
            outputs = self.clip_model.text_model(input_ids=input_ids)
            # Get last hidden state and apply final layer norm
            hidden = outputs.last_hidden_state  # [B, L, D]
            hidden = self.clip_model.text_model.final_layer_norm(hidden)
            
            # Pool at EOS token position
            eos_positions = input_ids.argmax(dim=-1)  # [B]
            batch_size = hidden.shape[0]
            text_features = hidden[torch.arange(batch_size), eos_positions]  # [B, D]
            
            # Project to CLIP space
            text_features = self.clip_model.text_projection(text_features)
        return text_features

    def forward(self, image):
        """
        Args:
            image: [B, 3, H, W]
        
        Returns:
            logits: [B, C]
        """
        image_features = self.clip_model.vision_model(pixel_values=image)
        image_features = image_features.pooler_output
        image_features = self.clip_model.visual_projection(image_features)  # [B, D_clip]
        image_features = F.normalize(image_features, dim=-1)

        logit_scale = self.clip_model.logit_scale.exp()
        text_features = self.text_features.to(image_features.device)
        logits = logit_scale * image_features @ text_features.t()

        return logits


class FeatureTransModuleTwoLayer(nn.Module):
    """
    Two-layer feature transform used in PromptKD reference implementations.

    Input:  [B, D]
    Output: [B, out_dim]
    """

    def __init__(self, input_dim=512, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PromptKDLoss(nn.Module):
    """
    PromptKD loss: CE(student, y) + kd_weight * KL(student || teacher).

    KL term follows the PromptKD-style scaling:
      KL = kl_div(log_softmax(s/T), softmax(t/T)) * T^2 / numel
    """

    def __init__(self, temperature=4.0, kd_weight=1.0, ce_weight=1.0):
        super().__init__()
        self.temperature = float(temperature)
        self.kd_weight = float(kd_weight)
        self.ce_weight = float(ce_weight)

    def kd_term(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        if student_logits is None or teacher_logits is None:
            device = None
            if torch.is_tensor(student_logits):
                device = student_logits.device
            elif torch.is_tensor(teacher_logits):
                device = teacher_logits.device
            return torch.tensor(0.0, device=device)

        t = self.temperature
        kd = F.kl_div(
            F.log_softmax(student_logits / t, dim=1),
            F.softmax(teacher_logits / t, dim=1),
            reduction="sum",
        )
        kd = kd * (t * t) / student_logits.numel()
        return kd

    def forward(self, student_logits, teacher_logits, labels):
        loss_ce = F.cross_entropy(student_logits, labels)
        loss_kd = self.kd_term(student_logits, teacher_logits)
        total = self.ce_weight * loss_ce + self.kd_weight * loss_kd
        return loss_ce, loss_kd, total
