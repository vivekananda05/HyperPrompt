# trainers/patch_kgcoop.py
"""
KgCoOp (Knowledge-Guided Context Optimization) Text Encoder

KgCoOp extends CoCoOp by adding knowledge guidance through a regularization
term that measures the divergence between learned text features and a static
knowledge base (original template-based text embeddings).

Key features:
- Learnable class-level prompt context
- Knowledge-guided regularization term measuring feature drift
- Optional knowledge distillation to retain semantic consistency

Architecture:
- Learnable context tokens for class prompts
- Static text embeddings from templates (knowledge base)
- Similarity-based regularization between learned and static features

References:
"Knowledge-Guided Context Optimization for Few-Shot Classification"
Similar concepts in: CoOp, CoCoOp, and knowledge distillation literature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


CUSTOM_TEMPLATES = {
    "houston": "a hyperspectral image of {}.",
    "pavia": "a hyperspectral image of {}.",
    "hyrank": "a hyperspectral image of {}.",
}


def _make_causal_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Build a causal (upper-triangular) attention mask of shape [1, 1, seq_len, seq_len].
    Masked positions are filled with -inf so they are ignored by softmax.
    This mirrors what HuggingFace CLIP does internally when input_ids are provided.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)          # upper triangle = -inf
    return mask.unsqueeze(0).unsqueeze(0)        # [1, 1, seq_len, seq_len]


class KgCoOpTextEncoder(nn.Module):
    """
    KgCoOp Text Encoder — Knowledge-Guided Context Optimization.

    An extension of CoCoOp that adds a knowledge-guided regularization term.
    The key innovation is maintaining a static text embedding from templates
    (the "knowledge base") and regularizing learned features to stay close to it.

    Architecture:
    1. Learnable context tokens + class names → learned embeddings
    2. Template-based static embeddings → knowledge base
    3. Regularization: minimize divergence between learned and static embeddings

    This helps prevent feature drift and improves generalization to unseen domains.

    Forward returns TWO outputs:
    - text_features : [C, D_proj]  learned class text embeddings
    - knowledge_loss : scalar  divergence from knowledge base (for regularization)

    Args
    ────
    clip_model           : HuggingFace CLIPModel (freeze externally)
    tokenizer            : matching CLIPTokenizer
    classnames           : list[str]  downstream class names
    n_ctx                : number of learnable context tokens
    ctx_init             : optional text initialiser
    class_token_position : "end" | "front" | "middle"
    csc                  : class-specific context
    knowledge_weight     : weight of knowledge regularization term
                          (higher = stronger regularization)
    """

    def __init__(
        self,
        clip_model,
        tokenizer,
        classnames,
        n_ctx,
        ctx_init=None,
        class_token_position="end",
        csc=False,
        dataset_name=None,
        knowledge_weight=0.1,
    ):
        super().__init__()

        # ── validate ──────────────────────────────────────────────────
        if class_token_position not in ("end", "front", "middle"):
            raise ValueError(
                f"class_token_position must be 'end', 'front', or 'middle', "
                f"got '{class_token_position}'."
            )

        self.clip                 = clip_model
        self.tokenizer            = tokenizer
        self.classnames           = classnames
        self.n_ctx                = n_ctx
        self.class_token_position = class_token_position
        self.csc                  = csc
        self.dataset_name         = dataset_name
        self.knowledge_weight     = knowledge_weight

        self.num_classes = len(classnames)
        self.hidden_dim  = clip_model.text_model.config.hidden_size   # D_text

        # ── ctx_init (identical to CoOp) ──────────────────────────────
        if ctx_init is not None:
            ctx_init = ctx_init.replace("_", " ")
            init_tokens = tokenizer(ctx_init, return_tensors="pt")
            with torch.no_grad():
                embedding = clip_model.text_model.embeddings.token_embedding(
                    init_tokens.input_ids
                )                                           # [1, len+2, D]
            n_init_tokens = init_tokens.input_ids.size(1) - 2  # excl. SOS/EOS
            if n_init_tokens > n_ctx:
                import warnings
                warnings.warn(
                    f"ctx_init '{ctx_init}' tokenises to {n_init_tokens} tokens "
                    f"but n_ctx={n_ctx}. Only the first {n_ctx} tokens will be used.",
                    UserWarning, stacklevel=2,
                )
            n_use         = min(n_init_tokens, n_ctx)
            base_ctx      = torch.empty(n_ctx, self.hidden_dim)
            nn.init.normal_(base_ctx, std=0.02)
            base_ctx[:n_use] = embedding[0, 1 : 1 + n_use, :]
        else:
            base_ctx = torch.empty(n_ctx, self.hidden_dim)
            nn.init.normal_(base_ctx, std=0.02)

        # CSC
        if csc:
            ctx = base_ctx.unsqueeze(0).repeat(self.num_classes, 1, 1)  # [C, n_ctx, D]
        else:
            ctx = base_ctx                                               # [n_ctx, D]

        self.ctx = nn.Parameter(ctx)

        # ── pre-tokenise class names ──────────────────────────────────
        prompts   = [f"{name}." for name in classnames]
        tokenized = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        self.register_buffer("input_ids",      tokenized.input_ids)       # [C, L]
        self.register_buffer("attention_mask", tokenized.attention_mask)  # [C, L]

        # ── KNOWLEDGE BASE: Pre-compute static text embeddings ────────
        # These serve as the reference/knowledge for the learned prompts
        # Computed once at init time using template-based prompts
        self._precompute_knowledge_base()

    def _precompute_knowledge_base(self):
        """
        Pre-compute static text embeddings from templates as the knowledge base.
        These embeddings serve as the target for knowledge regularization.
        
        The knowledge base is fixed and doesn't change during training.
        This helps guide the learned prompts while preventing excessive drift
        from the semantic meaning of the class names.
        """
        device = self.ctx.device
        dataset_key = str(self.dataset_name).lower() if self.dataset_name is not None else ""
        template = CUSTOM_TEMPLATES.get(dataset_key, "a hyperspectral image of {}.")
        prompts = [template.format(name.replace("_", " ")) for name in self.classnames]
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            # Encode class names using CLIP without any learned prompts
            outputs = self.clip.text_model(
                input_ids=tokenized.input_ids,
                attention_mask=tokenized.attention_mask,
                return_dict=True,
            )
            text_hidden = outputs.last_hidden_state  # [C, seq_len, D]

            # Pool at EOS and project
            eos_id = self.tokenizer.eos_token_id
            eos_pos = (tokenized.input_ids == eos_id).int().argmax(dim=1)  # [C]
            text_features = text_hidden[
                torch.arange(self.num_classes, device=device),
                eos_pos,
            ]  # [C, D]

            # Project and normalize
            knowledge_base = F.normalize(
                self.clip.text_projection(text_features), dim=-1
            )  # [C, D_proj]

        self.register_buffer("knowledge_base", knowledge_base)

    # ================================================================
    # Sequence Building (identical to CoCoOp)
    # ================================================================

    def _build_sequence(self, ctx, position):
        """
        Build the token-embedding sequence for all C classes with context block `ctx`.

        Args
        ────
        ctx      : [C, n_ctx, D]  — already shifted and expanded
        position : "end" | "front" | "middle"

        Returns
        ───────
        x            : [C, seq_len, D]
        eos_pos_in_x : [C]  long — 0-indexed EOS position per class
        """
        device      = ctx.device
        C           = ctx.size(0)
        n_ctx       = self.n_ctx
        eos_id      = self.tokenizer.eos_token_id
        token_embed = self.clip.text_model.embeddings.token_embedding
        input_ids   = self.input_ids.to(device)

        text_embed  = token_embed(input_ids)               # [C, L_orig, D]
        sos  = text_embed[:, :1, :]                        # [C, 1, D]
        rest = text_embed[:, 1:, :]                        # [C, L_orig-1, D]

        if position == "end":
            x = torch.cat([sos, ctx, rest], dim=1)
            eos_in_rest  = (input_ids == eos_id).int().argmax(dim=1) - 1
            eos_pos_in_x = 1 + n_ctx + eos_in_rest

        elif position == "front":
            real_len    = self.attention_mask.to(device).sum(dim=1) - 1
            max_body    = real_len.max().item() - 1
            seq_len_new = 1 + max_body + n_ctx + 1

            x            = torch.zeros(C, seq_len_new, self.hidden_dim,
                                       device=device, dtype=ctx.dtype)
            eos_pos_in_x = torch.zeros(C, dtype=torch.long, device=device)
            eos_emb      = token_embed(
                torch.tensor([eos_id], device=device)
            ).squeeze(0)

            for i in range(C):
                body_len = real_len[i].item() - 1
                body = rest[i, :body_len, :]
                row  = torch.cat([sos[i], body, ctx[i], eos_emb.unsqueeze(0)], dim=0)
                x[i, :row.size(0)] = row
                eos_pos_in_x[i]    = 1 + body_len + n_ctx

        else:  # "middle"
            half1 = n_ctx // 2
            half2 = n_ctx - half1
            ctx1  = ctx[:, :half1, :]
            ctx2  = ctx[:, half1:, :]

            real_len    = self.attention_mask.to(device).sum(dim=1) - 1
            max_body    = real_len.max().item() - 1
            seq_len_new = 1 + half1 + max_body + half2 + 1

            x            = torch.zeros(C, seq_len_new, self.hidden_dim,
                                       device=device, dtype=ctx.dtype)
            eos_pos_in_x = torch.zeros(C, dtype=torch.long, device=device)
            eos_emb      = token_embed(
                torch.tensor([eos_id], device=device)
            ).squeeze(0)

            for i in range(C):
                body_len = real_len[i].item() - 1
                body = rest[i, :body_len, :]
                row  = torch.cat([
                    sos[i], ctx1[i], body, ctx2[i], eos_emb.unsqueeze(0)
                ], dim=0)
                x[i, :row.size(0)] = row
                eos_pos_in_x[i]    = 1 + half1 + body_len + half2

        return x, eos_pos_in_x

    def _run_transformer(self, x, eos_pos_in_x):
        """
        Add positional embeddings, run CLIP transformer, pool at EOS,
        project, L2-normalise.

        Args
        ────
        x            : [C, seq_len, D]
        eos_pos_in_x : [C]  long

        Returns
        ───────
        features : [C, D_proj]  L2-normalised
        """
        device  = x.device
        C       = x.size(0)
        seq_len = x.size(1)

        pos_embed = self.clip.text_model.embeddings.position_embedding
        pos_ids   = torch.arange(seq_len, device=device).unsqueeze(0).expand(C, -1)
        x = x + pos_embed(pos_ids)

        causal_mask = _make_causal_mask(seq_len, x.dtype, device).expand(C, -1, -1, -1)

        hidden = self.clip.text_model.encoder(
            inputs_embeds=x,
            attention_mask=None,
            causal_attention_mask=causal_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state                                # [C, seq_len, D]

        hidden   = self.clip.text_model.final_layer_norm(hidden)
        features = hidden[torch.arange(C, device=device), eos_pos_in_x]  # [C, D]
        features = self.clip.text_projection(features)    # [C, D_proj]
        return F.normalize(features, dim=-1)

    # ================================================================
    # Forward  —  Returns BOTH text_features and knowledge_loss
    # ================================================================

    def forward(self, image_features: torch.Tensor = None) -> tuple:
        """
        Knowledge-guided text encoding with class-level prompts.

        Args
        ────
        image_features : optional, unused
            Kept only for API compatibility with call sites that pass image
            features (e.g. CoCoOp-style wrappers).

        Returns
        ───────
        text_features : [C, D_proj]  class text features, L2-normalised
        knowledge_loss : scalar  divergence from static knowledge base
        """
        if self.csc:
            ctx = self.ctx                                   # [C, n_ctx, D]
        else:
            ctx = self.ctx.unsqueeze(0).expand(self.num_classes, -1, -1)

        x, eos_pos = self._build_sequence(ctx, self.class_token_position)
        text_features = self._run_transformer(x, eos_pos)    # [C, D_proj]

        # Cosine-distance regularization against static template features.
        cos_sim = (text_features * self.knowledge_base).sum(dim=-1)  # [C]
        knowledge_loss = (1.0 - cos_sim).mean()

        return text_features, knowledge_loss
