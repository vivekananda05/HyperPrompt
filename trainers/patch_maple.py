# trainers/patch_maple.py
"""
MaPLe (Modular Prompt Learning) Text Encoder

MaPLe introduces layer-specific learnable prompts at each layer of the CLIP
text transformer. Unlike CoOp which has a single set of context tokens at the
input level, MaPLe maintains separate learned prompts for each transformer layer.

Key Architecture Features
──────────────────────────
1. Deep Compound Prompts: Separate learnable context per transformer layer
   - Each layer receives its own modular prompt tokens
   - Enables fine-grained control and better feature interaction
   
2. Shared Context Design: Context can be shared across classes or class-specific
   - Shared: [depth, n_ctx, D] — one prompt per layer
   - Class-Specific: [depth, C, n_ctx, D] — per-class prompts per layer

3. Position Flexibility: Supports "end", "front", "middle" positions
   - Compatible with CoOp-style prompt positioning

4. Multi-level Feature Adaptation: Prompts injected and refined at each layer
   - Better alignment with CLIP's transformer depth
   - Enables hierarchical prompt learning

References
──────────
MaPLe: Modular Prompt Learning for Vision-Language Models
He et al., CVPR 2023 and related work on multi-modal prompt learning.

Differences from Original MaPLe
───────────────────────────────
• Text-only encoder (original includes vision branch)
• Uses HuggingFace transformers CLIP (original uses OpenAI CLIP)
• Integrated with existing CoOp/CoCoOp infrastructure
• Simplified for HSI-CLIP fusion context (original is general VLM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_causal_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Build a causal (upper-triangular) attention mask of shape [1, 1, seq_len, seq_len].
    Masked positions are filled with -inf so they are ignored by softmax.
    This mirrors what HuggingFace CLIP does internally when input_ids are provided.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)          # upper triangle = -inf
    return mask.unsqueeze(0).unsqueeze(0)        # [1, 1, seq_len, seq_len]


class MapleTextEncoder(nn.Module):
    """
    MaPLe (Modular Prompt Learning) Text Encoder.

    Learns layer-specific context tokens (deep compound prompts) for each layer
    of the CLIP text encoder. Each transformer layer receives its own set of
    learnable prompts that enable hierarchical feature adaptation.

    The architecture includes:
    • Shallow prompt: Single learnable context at input level (like CoOp)
    • Deep prompts: Additional learnable prompts at each intermediate layer
    • Optional projection: Projects prompts between different dimensions

    Args
    ────
    clip_model : transformers.CLIPModel
        Pretrained CLIP model (freeze weights externally)
    tokenizer : transformers.CLIPTokenizer
        Matching tokenizer
    classnames : list[str]
        Downstream class names
    n_ctx : int
        Number of learnable context tokens
    prompt_depth : int
        Number of layers to inject prompts (≥1, typically 1-12)
        1 = shallow only (like CoOp)
        12 = all transformer layers (full depth)
    ctx_init : str, optional
        Text string to initialize context tokens
        Underscores are replaced with spaces
    class_token_position : str
        Position of class token: "end" (default), "middle", or "front"
    csc : bool
        Whether to use class-specific context
        If True: context shape [C, n_ctx, D]
        If False: context shape [n_ctx, D]

    Output
    ──────
    text_features : [C, D]  L2-normalised class embeddings
    """

    def __init__(
        self,
        clip_model,
        tokenizer,
        classnames,
        n_ctx,
        prompt_depth=1,
        ctx_init=None,
        class_token_position="end",
        csc=False,
    ):
        super().__init__()

        self.clip = clip_model
        self.tokenizer = tokenizer
        self.classnames = classnames
        self.n_ctx = n_ctx
        self.class_token_position = class_token_position
        self.csc = csc
        self.prompt_depth = max(1, min(prompt_depth, clip_model.text_model.config.num_hidden_layers))

        self.num_classes = len(classnames)
        self.num_layers = clip_model.text_model.config.num_hidden_layers
        self.hidden_dim = clip_model.text_model.config.hidden_size
        self.vision_hidden_dim = clip_model.vision_model.config.hidden_size

        # ====================================================
        # Shallow Context Initialization (input level)
        # ====================================================
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
                    f"but n_ctx={n_ctx}. Only the first {n_ctx} tokens will be used "
                    f"to initialise the context; the rest are discarded. "
                    f"Increase n_ctx to use the full init string.",
                    UserWarning, stacklevel=2,
                )
            n_use = min(n_init_tokens, n_ctx)
            base_ctx = torch.empty(n_ctx, self.hidden_dim)
            nn.init.normal_(base_ctx, std=0.02)             # random fallback
            base_ctx[:n_use] = embedding[0, 1 : 1 + n_use, :]  # overwrite first n_use
        else:
            base_ctx = torch.empty(n_ctx, self.hidden_dim)
            nn.init.normal_(base_ctx, std=0.02)

        # ====================================================
        # Shallow context (first layer / input level)
        # ====================================================
        if csc:
            ctx = base_ctx.unsqueeze(0).repeat(self.num_classes, 1, 1)  # [C, n_ctx, D]
        else:
            ctx = base_ctx  # [n_ctx, D]

        self.ctx = nn.Parameter(ctx)

        # Project shared shallow text context into vision-token space.
        self.shared_ctx_proj = nn.Linear(self.hidden_dim, self.vision_hidden_dim)

        # ====================================================
        # Deep Compound Prompts (MaPLe-specific)
        # ====================================================
        # For layers beyond the shallow prompt, create separate learned prompts
        # Original: prompt_depth can go up to num_layers
        # We create (prompt_depth - 1) additional deep prompts
        #
        # Layer structure:
        # - Layer 0 (input): shallow context (self.ctx)
        # - Layer 1..prompt_depth-1: deep compound prompts
        # - Layer prompt_depth..num_layers-1: no prompts (or reuse last)

        if self.prompt_depth > 1:
            # Create deep compound prompts for layers 1 to prompt_depth-1
            # Shape: [prompt_depth-1, n_ctx, hidden_dim]
            self.deep_compound_prompts = nn.ParameterList([
                nn.Parameter(torch.empty(n_ctx, self.hidden_dim))
                for _ in range(self.prompt_depth - 1)
            ])
            # Initialize with small random values
            for param in self.deep_compound_prompts:
                nn.init.normal_(param, std=0.02)

            # One projection per deep text prompt for the vision branch.
            self.deep_prompt_projections = nn.ModuleList([
                nn.Linear(self.hidden_dim, self.vision_hidden_dim)
                for _ in range(self.prompt_depth - 1)
            ])
        else:
            self.deep_compound_prompts = nn.ParameterList([])
            self.deep_prompt_projections = nn.ModuleList([])

        # ====================================================
        # Pre-tokenise class prompts
        # ====================================================
        prompts = [f"{name}." for name in self.classnames]

        tokenized = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

        self.register_buffer("input_ids", tokenized.input_ids)          # [C, L_orig]
        self.register_buffer("attention_mask", tokenized.attention_mask) # [C, L_orig]

    # ============================================================
    # Sequence Building (adapted from CoOp)
    # ============================================================

    def _build_sequence(self, ctx, position):
        """
        Build the token-embedding sequence for all C classes using context
        block `ctx`, following CoOp's layout logic.

        Args
        ────
        ctx      : [C, n_ctx, D]
        position : "end" | "front" | "middle"

        Returns
        ───────
        x            : [C, seq_len, D]
        eos_pos_in_x : [C]  long — 0-indexed EOS position per class
        """
        device = ctx.device
        C = ctx.size(0)
        n_ctx = self.n_ctx
        eos_id = self.tokenizer.eos_token_id
        token_embed = self.clip.text_model.embeddings.token_embedding
        input_ids = self.input_ids.to(device)         # [C, L_orig]

        text_embed = token_embed(input_ids)           # [C, L_orig, D]
        sos  = text_embed[:, :1, :]       # [C, 1, D]
        rest = text_embed[:, 1:, :]       # [C, L_orig-1, D]

        if position == "end":
            # [SOS | ctx | class_tokens, period, EOS, PAD ...]
            x = torch.cat([sos, ctx, rest], dim=1)          # [C, 1+n_ctx+L_orig-1, D]
            eos_in_rest = (input_ids == eos_id).int().argmax(dim=1) - 1
            eos_pos_in_x = 1 + n_ctx + eos_in_rest         # [C]

        elif position == "front":
            eos_id = self.tokenizer.eos_token_id
            real_len = self.attention_mask.to(device).sum(dim=1) - 1   # [C]
            max_body = real_len.max().item() - 1
            seq_len_new = 1 + max_body + n_ctx + 1

            x = torch.zeros(C, seq_len_new, self.hidden_dim, device=device, dtype=ctx.dtype)
            eos_pos_in_x = torch.zeros(C, dtype=torch.long, device=device)
            eos_emb = token_embed(torch.tensor([eos_id], device=device)).squeeze(0)  # [D]

            for i in range(C):
                body_len = real_len[i].item() - 1
                body = rest[i, :body_len, :]
                row = torch.cat([sos[i], body, ctx[i], eos_emb.unsqueeze(0)], dim=0)
                x[i, :row.size(0)] = row
                eos_pos_in_x[i] = 1 + body_len + n_ctx

        elif position == "middle":
            half1 = n_ctx // 2
            half2 = n_ctx - half1
            ctx1  = ctx[:, :half1, :]
            ctx2  = ctx[:, half1:, :]

            real_len = self.attention_mask.to(device).sum(dim=1) - 1   # [C]
            max_body = real_len.max().item() - 1
            seq_len_new = 1 + half1 + max_body + half2 + 1

            x = torch.zeros(C, seq_len_new, self.hidden_dim, device=device, dtype=ctx.dtype)
            eos_pos_in_x = torch.zeros(C, dtype=torch.long, device=device)
            eos_emb = token_embed(torch.tensor([eos_id], device=device)).squeeze(0)

            for i in range(C):
                body_len = real_len[i].item() - 1
                body = rest[i, :body_len, :]
                row = torch.cat([
                    sos[i], ctx1[i], body, ctx2[i], eos_emb.unsqueeze(0)
                ], dim=0)
                x[i, :row.size(0)] = row
                eos_pos_in_x[i] = 1 + half1 + body_len + half2

        else:
            raise ValueError(
                f"Invalid class_token_position '{position}'. "
                "Choose from: 'end', 'front', 'middle'."
            )

        return x, eos_pos_in_x

    # ============================================================
    # Forward Pass with Deep Compound Prompt Injection
    # ============================================================

    def forward(self) -> torch.Tensor:
        """
        Forward pass with shallow and deep prompt injection.

        Shallow prompt (self.ctx) is injected at the input level.
        Deep compound prompts are injected at their respective layers.

        Returns
        ───────
        text_features : [C, D]  L2-normalised class embeddings
        """
        device = self.ctx.device
        C = self.num_classes
        n_ctx = self.n_ctx

        # ====================================================
        # Build the input embedding sequence (shallow prompts)
        # ====================================================
        token_embed = self.clip.text_model.embeddings.token_embedding
        pos_embed   = self.clip.text_model.embeddings.position_embedding

        input_ids = self.input_ids.to(device)           # [C, L_orig]
        text_embed = token_embed(input_ids)             # [C, L_orig, D]

        # Expand shallow context to [C, n_ctx, D] if needed
        if self.csc:
            ctx = self.ctx                              # [C, n_ctx, D]
        else:
            ctx = self.ctx.unsqueeze(0).expand(C, -1, -1)  # [C, n_ctx, D]

        # Build initial sequence with shallow context
        x, eos_pos_in_x = self._build_sequence(ctx, self.class_token_position)
        seq_len_orig = x.size(1)

        # Add positional embeddings
        pos_ids = torch.arange(seq_len_orig, device=device).unsqueeze(0).expand(C, -1)
        x = x + pos_embed(pos_ids)  # [C, seq_len_orig, D]

        # ====================================================
        # Pass through transformer layers with deep prompt injection
        # ====================================================
        causal_mask_orig = _make_causal_mask(seq_len_orig, x.dtype, device).expand(C, -1, -1, -1)

        text_model = self.clip.text_model
        encoder_layers = text_model.encoder.layers

        hidden = x  # [C, seq_len_orig, D]

        def _extract_hidden(layer_output):
            """
            HF version compatibility:
            encoder_layer may return
              - BaseModelOutput (has .last_hidden_state)
              - tuple where first item is hidden states
              - tensor directly (hidden states)
            """
            if hasattr(layer_output, "last_hidden_state"):
                return layer_output.last_hidden_state
            if isinstance(layer_output, tuple):
                return layer_output[0]
            if torch.is_tensor(layer_output):
                return layer_output
            raise TypeError(
                f"Unsupported encoder layer output type: {type(layer_output)}"
            )

        for layer_idx, encoder_layer in enumerate(encoder_layers):
            # Check if this layer should have a deep prompt injected
            if layer_idx < self.prompt_depth - 1:
                # This layer gets a deep compound prompt
                deep_prompt = self.deep_compound_prompts[layer_idx]  # [n_ctx, D]
                deep_prompt = deep_prompt.unsqueeze(0).expand(C, -1, -1)  # [C, n_ctx, D]

                # Inject deep prompt: concatenate after SOS token
                hidden_with_prompt = torch.cat([
                    hidden[:, :1, :],           # [C, 1, D] — SOS
                    deep_prompt,                # [C, n_ctx, D] — deep prompt
                    hidden[:, 1:, :]            # [C, seq_len_orig-1, D] — rest
                ], dim=1)
                seq_len_with_prompt = hidden_with_prompt.size(1)

                # Update causal mask for new sequence length
                causal_mask = _make_causal_mask(seq_len_with_prompt, hidden.dtype, device)
                causal_mask = causal_mask.expand(C, -1, -1, -1)

                # Run transformer layer with deep prompt
                layer_output = encoder_layer(
                    hidden_with_prompt,
                    attention_mask=None,
                    causal_attention_mask=causal_mask,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                hidden_out = _extract_hidden(layer_output)  # [C, seq_len_with_prompt, D]

                # Remove injected deep prompt for next layer
                # Keep: [SOS | rest of sequence]
                hidden = torch.cat([hidden_out[:, :1, :], hidden_out[:, 1 + n_ctx:, :]], dim=1)
            else:
                # No deep prompt at this layer — pass through normally
                layer_output = encoder_layer(
                    hidden,
                    attention_mask=None,
                    causal_attention_mask=causal_mask_orig,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                hidden = _extract_hidden(layer_output)

        # ====================================================
        # Final layer norm and EOS pooling
        # ====================================================
        hidden = text_model.final_layer_norm(hidden)  # [C, seq_len_orig, D]

        # Pool at EOS position (from original sequence structure)
        text_features = hidden[
            torch.arange(C, device=device),
            eos_pos_in_x,
        ]  # [C, D]

        # Project to CLIP shared embedding space and L2-normalise
        text_features = self.clip.text_projection(text_features)  # [C, D_proj]
        text_features = F.normalize(text_features, dim=-1)

        return text_features

    def get_visual_prompts(self):
        """
        Build vision-side prompts from learned text prompts.

        Returns
        -------
        shallow_visual_ctx : [n_ctx, Dv]
            Shared shallow prompt tokens for vision input sequence.
        deep_visual_prompts : list[[n_ctx, Dv]]
            Deep prompt tokens for intermediate vision layers.
        """
        if self.csc:
            # Vision branch uses shared prompts, so aggregate class-specific ctx.
            shared_text_ctx = self.ctx.mean(dim=0)
        else:
            shared_text_ctx = self.ctx

        shallow_visual_ctx = self.shared_ctx_proj(shared_text_ctx)

        deep_visual_prompts = []
        for proj, deep_text_prompt in zip(self.deep_prompt_projections, self.deep_compound_prompts):
            deep_visual_prompts.append(proj(deep_text_prompt))

        return shallow_visual_ctx, deep_visual_prompts
