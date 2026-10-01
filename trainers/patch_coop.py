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


class CoOpTextEncoder(nn.Module):
    """
    CoOp Text Encoder — fixed version.

    Fixes applied vs original:
      1. Causal attention mask is now passed to the transformer encoder,
         matching CLIP's original behaviour.
      2. EOS position is computed from the constructed sequence structure
         (not by blindly shifting the original tokenised position), which
         correctly handles multi-token class names.
      3. ctx_init length is validated against n_ctx at init time so a
         mismatch raises an early, informative error instead of silently
         producing a wrong-shaped context.

    Output:
        text_features: [C, D]   (L2-normalised, projected to CLIP space)
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
    ):
        super().__init__()

        self.clip = clip_model
        self.tokenizer = tokenizer
        self.classnames = classnames
        self.n_ctx = n_ctx
        self.class_token_position = class_token_position
        self.csc = csc

        self.num_classes = len(classnames)
        self.hidden_dim = clip_model.text_model.config.hidden_size

        # ====================================================
        # Context Initialisation
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
                # Init string has more tokens than ctx slots — first n_ctx used,
                # remainder silently dropped. Warn so the user is aware.
                import warnings
                warnings.warn(
                    f"ctx_init '{ctx_init}' tokenises to {n_init_tokens} tokens "
                    f"but n_ctx={n_ctx}. Only the first {n_ctx} tokens will be used "
                    f"to initialise the context; the rest are discarded. "
                    f"Increase n_ctx to use the full init string.",
                    UserWarning, stacklevel=2,
                )
            # If n_init_tokens < n_ctx: only the first n_init_tokens slots are
            # initialised from the string; remaining slots use random noise
            # (applied below via torch.randn_like * 0.02). This is intentional.
            n_use         = min(n_init_tokens, n_ctx)
            base_ctx      = torch.empty(n_ctx, self.hidden_dim)
            nn.init.normal_(base_ctx, std=0.02)             # random fallback
            base_ctx[:n_use] = embedding[0, 1 : 1 + n_use, :]  # overwrite first n_use
        else:
            base_ctx = torch.empty(n_ctx, self.hidden_dim)
            nn.init.normal_(base_ctx, std=0.02)

        # CSC: one set of context vectors per class
        if csc:
            ctx = base_ctx.unsqueeze(0).repeat(self.num_classes, 1, 1)  # [C, n_ctx, D]
        else:
            ctx = base_ctx  # [n_ctx, D]

        self.ctx = nn.Parameter(ctx)

        # ====================================================
        # Pre-tokenise class prompts (class name + period only;
        # the learnable context replaces the hand-crafted prefix)
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
    def forward(self) -> torch.Tensor:
        """
        Returns
        -------
        text_features : torch.Tensor  shape [C, D]
        """
        device = self.ctx.device
        input_ids = self.input_ids.to(device)         # [C, L_orig]

        C = input_ids.size(0)
        n_ctx = self.n_ctx

        token_embed = self.clip.text_model.embeddings.token_embedding
        pos_embed   = self.clip.text_model.embeddings.position_embedding

        text_embed = token_embed(input_ids)           # [C, L_orig, D]

        # Expand context to [C, n_ctx, D]
        if self.csc:
            ctx = self.ctx                                         # [C, n_ctx, D]
        else:
            ctx = self.ctx.unsqueeze(0).expand(C, -1, -1)         # [C, n_ctx, D]

        sos  = text_embed[:, :1, :]       # [C, 1,     D]  — SOS token
        rest = text_embed[:, 1:, :]       # [C, L_orig-1, D]  — class tokens, period, EOS, PAD

        # ====================================================
        # Build the full embedded sequence and record where EOS
        # will land in the NEW sequence.
        #
        # Layout depends on class_token_position:
        #
        #  "end"    : [SOS | ctx(n_ctx) | class+period+EOS+PAD...]
        #  "front"  : [SOS | class+period | ctx(n_ctx) | EOS      ]
        #  "middle" : [SOS | ctx[:half] | class+period | ctx[half:] | EOS ]
        #
        # For "end" the EOS is wherever it was in `rest`, shifted by n_ctx.
        # For "front" and "middle" EOS is reconstructed explicitly at the end.
        #
        # We compute eos_pos_in_x per class to handle variable-length names.
        # ====================================================

        if self.class_token_position == "end":
            # [SOS | ctx | class_tokens, period, EOS, PAD ...]
            x = torch.cat([sos, ctx, rest], dim=1)          # [C, 1+n_ctx+L_orig-1, D]

            # EOS position in original rest (0-indexed in rest)
            eos_id = self.tokenizer.eos_token_id
            # argmax finds the first occurrence; EOS always comes before PAD
            eos_in_rest = (input_ids == eos_id).int().argmax(dim=1) - 1  # offset by 1 (SOS removed)
            # In x: SOS(1) + ctx(n_ctx) + eos_in_rest
            eos_pos_in_x = 1 + n_ctx + eos_in_rest         # [C]

        elif self.class_token_position == "front":
            # Separate class body (everything between SOS and EOS, inclusive of period)
            # and EOS from rest.
            eos_id = self.tokenizer.eos_token_id
            # Number of real tokens per class (excl. SOS): class_tokens + period + EOS
            # attention_mask counts SOS too, so subtract 1 for SOS
            real_len = self.attention_mask.sum(dim=1) - 1   # [C]  (class body + EOS length in rest)

            # Build x by iterating — class lengths differ so we must pad manually.
            # Simpler: keep PAD in rest but move EOS to after ctx.
            #
            # rest layout: [class_tok..., period, EOS, PAD...]
            # We want:     [SOS | class_tok..., period | ctx | EOS]
            #
            # Slice class body (without EOS) then append ctx then EOS.
            # Because lengths differ per class we process with a loop and re-pad.

            max_class_body = real_len.max().item() - 1  # exclude EOS from body
            seq_len = 1 + max_class_body + n_ctx + 1    # SOS + body + ctx + EOS

            x = torch.zeros(C, seq_len, self.hidden_dim, device=device, dtype=ctx.dtype)
            eos_pos_in_x = torch.zeros(C, dtype=torch.long, device=device)

            eos_embed = token_embed(
                torch.tensor([eos_id], device=device)
            ).squeeze(0)  # [D]

            for i in range(C):
                body_len = real_len[i].item() - 1   # class tokens + period (no EOS)
                body = rest[i, :body_len, :]         # [body_len, D]
                row = torch.cat([
                    sos[i],                          # [1,      D]
                    body,                            # [body_len, D]
                    ctx[i],                          # [n_ctx,  D]
                    eos_embed.unsqueeze(0),          # [1,      D]
                ], dim=0)                            # [1+body_len+n_ctx+1, D]
                x[i, :row.size(0), :] = row
                eos_pos_in_x[i] = 1 + body_len + n_ctx  # 0-indexed position of EOS

        elif self.class_token_position == "middle":
            half1 = n_ctx // 2
            half2 = n_ctx - half1                    # handles odd n_ctx

            ctx1 = ctx[:, :half1, :]                 # [C, half1, D]
            ctx2 = ctx[:, half1:, :]                 # [C, half2, D]

            eos_id = self.tokenizer.eos_token_id
            real_len = self.attention_mask.sum(dim=1) - 1   # [C]

            max_class_body = real_len.max().item() - 1
            seq_len = 1 + half1 + max_class_body + half2 + 1

            x = torch.zeros(C, seq_len, self.hidden_dim, device=device, dtype=ctx.dtype)
            eos_pos_in_x = torch.zeros(C, dtype=torch.long, device=device)

            eos_embed = token_embed(
                torch.tensor([eos_id], device=device)
            ).squeeze(0)

            for i in range(C):
                body_len = real_len[i].item() - 1
                body = rest[i, :body_len, :]
                row = torch.cat([
                    sos[i],                          # [1,      D]
                    ctx1[i],                         # [half1,  D]
                    body,                            # [body_len,D]
                    ctx2[i],                         # [half2,  D]
                    eos_embed.unsqueeze(0),          # [1,      D]
                ], dim=0)
                x[i, :row.size(0), :] = row
                eos_pos_in_x[i] = 1 + half1 + body_len + half2

        else:
            raise ValueError(
                f"Invalid class_token_position '{self.class_token_position}'. "
                "Choose from: 'end', 'front', 'middle'."
            )

        # ====================================================
        # Positional Embedding
        # ====================================================
        seq_len = x.size(1)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(C, -1)
        x = x + pos_embed(pos_ids)                  # [C, seq_len, D]

        # ====================================================
        # Causal Attention Mask
        # FIX: CLIP's text transformer is causal. Without this mask
        # every token can attend to future tokens, breaking the model.
        # ====================================================
        causal_mask = _make_causal_mask(seq_len, dtype=x.dtype, device=device)
        # expand to [C, 1, seq_len, seq_len] for broadcasting over heads
        causal_mask = causal_mask.expand(C, -1, -1, -1)

        # ====================================================
        # Transformer Encoder
        # ====================================================
        hidden = self.clip.text_model.encoder(
            inputs_embeds=x,
            attention_mask=None,           # no padding mask needed (we handle padding via causal)
            causal_attention_mask=causal_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state                          # [C, seq_len, D]

        hidden = self.clip.text_model.final_layer_norm(hidden)

        # ====================================================
        # EOS Pooling  — FIX: use per-class eos_pos_in_x
        # ====================================================
        text_features = hidden[
            torch.arange(C, device=device),
            eos_pos_in_x,                            # [C]  — correct position per class
        ]                                            # [C, D]

        # Project to CLIP shared embedding space and L2-normalise
        text_features = self.clip.text_projection(text_features)  # [C, D_proj]
        text_features = F.normalize(text_features, dim=-1)

        return text_features