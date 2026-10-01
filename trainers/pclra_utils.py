


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ================================================================
#  Fixed-rank vanilla LoRA (unchanged from original)
# ================================================================

class LoRALinear(nn.Module):
    """
    LoRA wrapper for nn.Linear.
    Replaces W with W + (B @ A) * (alpha / rank).

    Args:
        linear   : original nn.Linear layer
        rank     : LoRA rank r
        alpha    : LoRA scaling factor (default = rank → scale = 1.0)
        dropout  : dropout on the LoRA path
    """

    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = None, dropout: float = 0.0):
        super().__init__()

        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.rank         = rank
        self.alpha        = alpha if alpha is not None else float(rank)
        self.scaling      = self.alpha / self.rank

        # Freeze the original weight
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        # LoRA matrices  A: (r, in)   B: (out, r)
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # Kaiming init for A (same as in the original paper)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base path (frozen)
        base = F.linear(x, self.weight, self.bias)
        # LoRA path
        lora = F.linear(self.dropout(x), self.lora_A)   # (..., r)
        lora = F.linear(lora, self.lora_B)               # (..., out)
        return base + lora * self.scaling

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}")


# ---------------------------------------------------------------
# Helpers (vision / SAM injection — unchanged)
# ---------------------------------------------------------------

def _apply_lora_to_attention(attn_module, rank, alpha, dropout, target_keys):
    """
    Replace nn.Linear children whose name is in target_keys with LoRALinear.
    Works generically for any transformer attention block.
    """
    for name, module in list(attn_module.named_children()):
        if isinstance(module, nn.Linear) and name in target_keys:
            setattr(attn_module, name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))


def inject_lora_clip_vision(
    clip_vision_model,
    lora_rank: int          = 4,
    lora_alpha: float       = None,
    lora_dropout: float     = 0.0,
    last_n_layers: int      = -1,
    target_keys             = ("q_proj", "v_proj", "k_proj", "out_proj"),
):
    """Inject fixed-rank LoRA into CLIP vision encoder."""
    layers = clip_vision_model.encoder.layers
    total  = len(layers)
    if last_n_layers == 0:
        print(f"[LoRA] CLIP vision: skipped (last_n_layers=0)")
        return clip_vision_model
    start = 0 if last_n_layers == -1 else max(0, total - last_n_layers)

    injected = 0
    for idx, layer in enumerate(layers):
        if idx < start:
            continue
        attn = layer.self_attn
        _apply_lora_to_attention(attn, rank=lora_rank, alpha=lora_alpha,
                                 dropout=lora_dropout, target_keys=set(target_keys))
        injected += 1

    print(f"[LoRA] CLIP vision: injected into {injected}/{total} layers "
          f"(last_n_layers={last_n_layers}, rank={lora_rank})")
    return clip_vision_model


def inject_lora_clip_text(
    clip_text_model,
    lora_rank: int          = 4,
    lora_alpha: float       = None,
    lora_dropout: float     = 0.0,
    last_n_layers: int      = -1,
    target_keys             = ("q_proj", "v_proj", "k_proj", "out_proj"),
):
    """Inject fixed-rank LoRA into CLIP text encoder layers."""
    layers = clip_text_model.encoder.layers
    total  = len(layers)
    if last_n_layers == 0:
        print(f"[LoRA] CLIP text: skipped (last_n_layers=0)")
        return clip_text_model
    start  = 0 if last_n_layers == -1 else max(0, total - last_n_layers)

    injected = 0
    for idx, layer in enumerate(layers):
        if idx < start:
            continue
        attn = layer.self_attn
        _apply_lora_to_attention(attn, rank=lora_rank, alpha=lora_alpha,
                                 dropout=lora_dropout, target_keys=set(target_keys))
        injected += 1

    print(f"[LoRA] CLIP text: injected into {injected}/{total} layers "
          f"(last_n_layers={last_n_layers}, rank={lora_rank})")
    return clip_text_model


def inject_lora_sam_vision(
    sam_vision_encoder,
    lora_rank: int          = 4,
    lora_alpha: float       = None,
    lora_dropout: float     = 0.0,
    last_n_layers: int      = -1,
    target_keys             = ("qkv", "proj", "q", "k", "v"),
):
    """Inject fixed-rank LoRA into SAM-2 vision encoder backbone blocks."""
    blocks = sam_vision_encoder.backbone.blocks
    total  = len(blocks)
    if last_n_layers == 0:
        print(f"[LoRA] SAM vision: skipped (last_n_layers=0)")
        return sam_vision_encoder
    start  = 0 if last_n_layers == -1 else max(0, total - last_n_layers)

    injected = 0
    for idx, block in enumerate(blocks):
        if idx < start:
            continue
        _inject_lora_recursive(block, lora_rank, lora_alpha, lora_dropout, set(target_keys))
        injected += 1

    print(f"[LoRA] SAM vision: injected into {injected}/{total} blocks "
          f"(last_n_layers={last_n_layers}, rank={lora_rank})")
    return sam_vision_encoder


def _inject_lora_recursive(module, rank, alpha, dropout, target_keys):
    """Recursively replace nn.Linear layers whose attribute name is in target_keys."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in target_keys:
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
        else:
            _inject_lora_recursive(child, rank, alpha, dropout, target_keys)


def get_lora_params(model):
    """Return only LoRA parameters (lora_A and lora_B) for optimisation."""
    return [p for name, p in model.named_parameters() if "lora_A" in name or "lora_B" in name]


def lora_param_count(model):
    """
    Print a full breakdown of trainable parameters covering both
    fixed-rank LoRA (lora_A / lora_B) and all Self-Guided PCLRA
    components (lora_A per-encoder, B_gen, prompt_agg, prompts).

    Parameter groups counted
    ────────────────────────
    Fixed LoRA
      lora_A / lora_B          — standard LoRALinear matrices

    PCLRA (self-guided, fixed rank)
      lora_A (enc1 + enc2)     — independent per-encoder A matrices
      B_gen  (MLP weights)     — dynamic B matrix generator (self-prompted)
      prompt_agg               — prompt aggregation LayerNorm + Linear
      prompts                  — learnable prompt tokens

    Other trainable
      everything else with requires_grad=True
    """

    # ── fixed-rank LoRA (LoRALinear, NOT inside pclra_layers) ────────────
    n_fixed_lora_A = sum(p.numel() for n, p in model.named_parameters()
                         if "lora_A" in n and "pclra_layers" not in n and p.requires_grad)
    n_lora_B       = sum(p.numel() for n, p in model.named_parameters()
                         if "lora_B" in n and p.requires_grad)

    # ── PCLRA: independent A matrices (enc1 + enc2) ─────────────────
    n_pclra_A = sum(p.numel() for n, p in model.named_parameters()
                      if "lora_A" in n and "pclra_layers" in n and p.requires_grad)

    # ── PCLRA: B generator MLPs (self-prompted, no gate_gen) ────────
    n_B_gen = sum(p.numel() for n, p in model.named_parameters()
                  if "B_gen" in n and p.requires_grad)

    # ── PCLRA: prompt aggregation ───────────────────────────────────
    n_agg = sum(p.numel() for n, p in model.named_parameters()
                if "prompt_agg" in n and p.requires_grad)

    # ── PCLRA: learned prompt tokens ───────────────────────────────
    n_prompts = sum(p.numel() for n, p in model.named_parameters()
                    if "prompts" in n and p.requires_grad)

    # ── aggregates ────────────────────────────────────────────────────
    pclra_total    = n_pclra_A + n_B_gen + n_agg + n_prompts
    fixed_lora_total = n_fixed_lora_A + n_lora_B

    seen_train = set(); seen_all = set()
    total_trainable = 0; total_model = 0
    for p in model.parameters():
        if id(p) not in seen_all:
            total_model += p.numel(); seen_all.add(id(p))
        if p.requires_grad and id(p) not in seen_train:
            total_trainable += p.numel(); seen_train.add(id(p))

    total_frozen = total_model - total_trainable
    other        = total_trainable - fixed_lora_total - pclra_total

    def pt(n): return 100.0 * n / max(total_trainable, 1)
    def pm(n): return 100.0 * n / max(total_model, 1)

    print("=" * 72)
    print("  Trainable Parameter Breakdown")
    print("=" * 72)
    print(f"  {'Component':<26}  {'Params':>12}   {'% trainable':>11}   {'% of model':>10}")
    print("-" * 72)
    print(f"  Fixed-rank LoRA")
    print(f"    {'lora_A matrices':<24}  {n_fixed_lora_A:>12,}   {pt(n_fixed_lora_A):>10.2f}%   {pm(n_fixed_lora_A):>9.2f}%")
    print(f"    {'lora_B matrices':<24}  {n_lora_B:>12,}   {pt(n_lora_B):>10.2f}%   {pm(n_lora_B):>9.2f}%")
    print(f"    {'subtotal':<24}  {fixed_lora_total:>12,}   {pt(fixed_lora_total):>10.2f}%   {pm(fixed_lora_total):>9.2f}%")
    print(f"  Self-Guided PCLRA (text encoders, fixed rank)")
    print(f"    {'lora_A enc1+enc2':<24}  {n_pclra_A:>12,}   {pt(n_pclra_A):>10.2f}%   {pm(n_pclra_A):>9.2f}%")
    print(f"    {'B_gen MLPs':<24}  {n_B_gen:>12,}   {pt(n_B_gen):>10.2f}%   {pm(n_B_gen):>9.2f}%")
    print(f"    {'prompt_agg':<24}  {n_agg:>12,}   {pt(n_agg):>10.2f}%   {pm(n_agg):>9.2f}%")
    print(f"    {'prompt tokens':<24}  {n_prompts:>12,}   {pt(n_prompts):>10.2f}%   {pm(n_prompts):>9.2f}%")
    print(f"    {'subtotal':<24}  {pclra_total:>12,}   {pt(pclra_total):>10.2f}%   {pm(pclra_total):>9.2f}%")
    print(f"  {'Other trainable':<26}  {other:>12,}   {pt(other):>10.2f}%   {pm(other):>9.2f}%")
    print("-" * 72)
    print(f"  {'TOTAL trainable':<26}  {total_trainable:>12,}   {'100.00%':>11}   {pm(total_trainable):>9.2f}%")
    print(f"  {'Frozen':<26}  {total_frozen:>12,}   {' ---':>11}   {pm(total_frozen):>9.2f}%")
    print(f"  {'TOTAL model':<26}  {total_model:>12,}   {' ---':>11}   {'100.00%':>10}")
    print("=" * 72)

    return {
        "lora_A_fixed":    n_fixed_lora_A,
        "lora_B":          n_lora_B,
        "fixed_lora":      fixed_lora_total,
        "lora_A_pg":       n_pclra_A,
        "B_gen":           n_B_gen,
        "prompt_agg":      n_agg,
        "prompts":         n_prompts,
        "pclra":         pclra_total,
        "other":           other,
        "total_trainable": total_trainable,
        "total_frozen":    total_frozen,
        "total_model":     total_model,
    }


# ================================================================
#  Self-Guided Prompt-Conditioned Low-Rank Adapter — Fixed Rank
#  ── each encoder uses ONLY its own prompt vector
#  ── no cross-connection, independently deployable per encoder
# ================================================================

class _PromptToBMatrix(nn.Module):
    """
    Lightweight MLP: prompt_vec [d_p] → B matrix [d_out, r].

    SELF-GUIDED: conditioned on this encoder's OWN prompt vector,
    not the other encoder's. This makes the layer fully self-contained.

    The final linear is zero-initialised so the LoRA delta starts at 0,
    matching the standard LoRA initialisation convention (B=0 at t=0).

    LayerNorm on the input stabilises conditioning against prompt vector
    magnitude drift during training.
    """
    def __init__(self, prompt_dim: int, d_out: int, r: int, hidden: int = 256):
        super().__init__()
        self.r     = r
        self.d_out = d_out
        self.mlp   = nn.Sequential(
            nn.LayerNorm(prompt_dim),          # stabilise prompt vec magnitude
            nn.Linear(prompt_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_out * r),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """v: [d_p]  →  B: [d_out, r]"""
        return self.mlp(v).view(self.d_out, self.r)


class PCLRALayer(nn.Module):
    """
    A single attention projection (q/k/v/out) wrapped with
    Self-Guided Prompt-Conditioned Low-Rank Adapter at fixed rank r.

    Mathematics
    ───────────
    Forward pass:

        h = W x + ΔW(v_self) x
        ΔW = (α/r) · B(v_self) · A

    where:
        x        ∈ R^{d_in}            input activation
        W        ∈ R^{d_out × d_in}    frozen base weight
        A        ∈ R^{r × d_in}        learnable (Kaiming init), rank r fixed
        B(·)     : R^{d_p} → R^{d_out × r}  self-guided MLP (zero-init output)
        v_self   ∈ R^{d_p}             prompt vector from THIS encoder only

    Key differences from cross-guided version:
      - v_self replaces v_cross: no dependency on the other encoder
      - gate removed: rank is fixed at r throughout training
      - layer is independently saveable / loadable onto any model

    Args
    ────
    linear       : original frozen nn.Linear
    r            : LoRA rank (fixed, no gating)
    alpha        : LoRA scaling  (α/r is the effective multiplier)
    prompt_dim   : dimension d_p of the aggregated prompt vector
    dropout      : dropout on the LoRA input path
    """

    def __init__(
        self,
        linear:     nn.Linear,
        r:          int   = 8,
        alpha:      float = 16.0,
        prompt_dim: int   = 256,
        dropout:    float = 0.0,
    ):
        super().__init__()
        d_in  = linear.in_features
        d_out = linear.out_features
        self.r       = r
        self.scaling = alpha / r

        # ── frozen base weights ──────────────────────────────────────
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        self.bias   = (nn.Parameter(linear.bias.data.clone(), requires_grad=False)
                       if linear.bias is not None else None)

        # ── independent A matrix (per-encoder, fixed rank) ──────────
        # Kaiming init; NOT shared between encoders so each can develop
        # a complementary input subspace.
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # ── self-guided B generator ──────────────────────────────────
        # Conditioned on THIS encoder's own prompt vector v_self.
        # Zero-init output → ΔW = 0 at step 0 (standard LoRA convention).
        self.B_gen = _PromptToBMatrix(prompt_dim, d_out, r)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, self_v: torch.Tensor) -> torch.Tensor:
        """
        Args
        ────
        x       : [..., d_in]   input activation
        self_v  : [d_p]         aggregated prompt vector from THIS encoder

        Returns
        ───────
        [..., d_out]  =  Wx + (α/r) · B(self_v) · A · x
        """
        base = F.linear(x, self.weight, self.bias)

        # dynamic B from this encoder's own prompt vector
        B = self.B_gen(self_v)                          # [d_out, r]

        # low-rank update: x → A → B → scale
        lora_out = F.linear(self.dropout(x), self.lora_A)  # [..., r]
        lora_out = F.linear(lora_out, B)                   # [..., d_out]

        return base + lora_out * self.scaling

    def extra_repr(self):
        return (f"r={self.r}, scaling={self.scaling:.4f}, "
                f"in={self.weight.shape[1]}, out={self.weight.shape[0]}")


class PCLRATextEncoder(nn.Module):
    """
    Generic wrapper that injects PCLRALayer (self-guided, fixed rank)
    into any transformer text encoder's attention projections.

    SELF-GUIDED: each encoder conditions its LoRA entirely on its own
    prompt vector. No cross-encoder dependency. This means:
      - enc1 and enc2 can be saved/loaded/applied to any model independently
      - forward() requires no input from the other encoder
      - regularisation is computed in DualTextEncoderPCLRA using both
        self-prompt vectors to compare and penalise redundancy

    Usage (standalone, single encoder)
    ───────────────────────────────────
        pg_enc = PCLRATextEncoder(
            encoder=text_encoder,
            get_layers=lambda enc: enc.clip.text_model.encoder.layers,
            get_prompt_tokens=lambda enc: enc.prompts,
            hidden_dim=512, r=8, alpha=16.0, prompt_dim=256,
        )

        # training step (no other encoder needed)
        out = pg_enc(*args, **kwargs)

    Saving / loading independently
    ───────────────────────────────
        # Save enc1's PCLRA state
        torch.save(dual.enc1.state_dict(), "enc1_pclra.pt")

        # Load onto a fresh PCLRATextEncoder of same config
        new_enc1.load_state_dict(torch.load("enc1_pclra.pt"))

    Args
    ────
    encoder
        The text encoder module to wrap.
    get_layers
        ``encoder_module → list[layer]`` — returns transformer attention layers.
        Example: ``lambda enc: enc.clip.text_model.encoder.layers``
    get_prompt_tokens
        ``encoder_module → Tensor[..., D]`` — returns raw prompt tokens.
        For ProDA: ``lambda enc: enc.prompts``  (shape [K, n_ctx, D])
        For CoOp:  ``lambda enc: enc.ctx``      (shape [n_ctx, D])
    hidden_dim
        Token embedding size of the text backbone (e.g. 512 for CLIP ViT-B/16).
    r
        Fixed LoRA rank (no gating). Must match between save and load.
    alpha
        LoRA scaling. Effective scale = alpha / r.
    prompt_dim
        Dimension d_p of the aggregated prompt vector fed to B_gen.
    last_n_layers
        How many transformer layers from the end to inject LoRA. -1 = all.
    target_keys
        Attention projection names to wrap with PCLRALayer.
    dropout
        Dropout on the LoRA input path.
   
    """

    def __init__(
        self,
        encoder,
        get_layers,           # callable: encoder → list[transformer_layer]
        get_prompt_tokens,    # callable: encoder → Tensor[..., hidden_dim]
        hidden_dim:    int,
        r:             int   = 8,
        alpha:         float = 16.0,
        prompt_dim:    int   = 256,
        last_n_layers: int   = -1,
        target_keys          = ("q_proj", "v_proj", "k_proj", "out_proj"),
        dropout:       float = 0.0,
    ):
        super().__init__()
        self.encoder            = encoder
        self._get_prompt_tokens = get_prompt_tokens
        self.r                  = r
        self.prompt_dim         = prompt_dim

       
        self._prompt_weights = None

        # ── prompt aggregator: (weighted) pool → d_p ─────────────────
        self.prompt_agg = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, prompt_dim),
        )

        # ── inject PCLRALayer into target attention layers ─
        layers = get_layers(encoder)
        total  = len(layers)
        start  = 0 if last_n_layers == -1 else max(0, total - last_n_layers)

        self._pclra_layers: list[PCLRALayer] = []
        n_layers_injected = 0

        for idx, layer in enumerate(layers):
            if idx < start:
                continue
            attn = layer.self_attn
            for name, child in list(attn.named_children()):
                if isinstance(child, nn.Linear) and name in set(target_keys):
                    pg_layer = PCLRALayer(
                        linear=child,
                        r=r,
                        alpha=alpha,
                        prompt_dim=prompt_dim,
                        dropout=dropout,
                    )
                    setattr(attn, name, pg_layer)
                    self._pclra_layers.append(pg_layer)
            n_layers_injected += 1

        # register as submodule so parameters() picks them up
        self.pclra_layers = nn.ModuleList(self._pclra_layers)

        # ── EMA of self prompt vector (for stable self-conditioning) ──
        # updated during training; used at eval time for test-time stability
        self._ema_prompt_vec: torch.Tensor | None = None
        self._ema_momentum: float = 0.99

        # install forward hooks so transformers call pg_layer(x) normally
        self._install_self_v_hooks()

    # ── prompt vector ──────────────────────────────────────────────────

    def current_prompt_vec(self) -> torch.Tensor:
        """
        Aggregate this encoder's own prompt tokens into a single [prompt_dim]
        vector and maintain a momentum EMA for training-time stability.


        EMA
        ───
            ema ← momentum * ema + (1 − momentum) * v_current

        Training : returns live differentiable v_current (gradient flows to
                   prompt_agg + prompt tokens); EMA updated in background.
        Eval     : returns frozen EMA for test-time stability.
        """
        P = self._get_prompt_tokens(self.encoder)   # [C, ..., D]  or  [..., D]

        if self._prompt_weights is not None:
            C   = P.shape[0]
            D   = P.shape[-1]
            Pf  = P.reshape(C, -1, D).mean(dim=1)          # [C, D]
            w   = self._prompt_weights[:C].to(Pf.device)   # [C]
            pooled = (Pf * w.unsqueeze(-1)).sum(dim=0)      # [D]
        else:
            pooled = P.reshape(-1, P.shape[-1]).mean(dim=0) # [D]

        v_current = self.prompt_agg(pooled)   # [prompt_dim], differentiable

        # ── EMA update (training only) ────────────────────────────────
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                if self._ema_prompt_vec is None:
                    self._ema_prompt_vec = v_current.detach().clone()
                else:
                    self._ema_prompt_vec.mul_(self._ema_momentum).add_(
                        v_current.detach(), alpha=1.0 - self._ema_momentum
                    )
            return v_current   # live: gradients flow

        # eval / no_grad: return frozen EMA
        if self._ema_prompt_vec is not None:
            return self._ema_prompt_vec.to(v_current.device)
        return v_current

    # ── hook installation ──────────────────────────────────────────────

    def _install_self_v_hooks(self):
        """
        Wrap every PCLRALayer's forward so it automatically
        reads self._self_v (this encoder's own prompt vector) from a
        module-level side-channel.

        This keeps the internal transformer call signature unchanged:
        the transformer calls ``pg_layer(x)`` without knowing about self_v.
        The side-channel is set in forward() before the encoder is called.
        """
        wrapper = self

        def make_hook(pg_layer):
            original_forward = pg_layer.forward

            def hooked_forward(x, self_v=None):
                sv = wrapper._self_v
                if sv is None:
                    raise RuntimeError(
                        "PCLRATextEncoder: self_v not set. "
                        "Call forward() which sets _self_v before the encoder."
                    )
                return original_forward(x, sv)

            return hooked_forward

        for pg_layer in self._pclra_layers:
            pg_layer.forward = make_hook(pg_layer)

    # ── forward ────────────────────────────────────────────────────────

    def forward(self, *args, **kwargs):
        """
        Self-guided forward — no input from the other encoder required.

        1. Compute this encoder's own prompt vector v_self (EMA-updated).
        2. Inject v_self into all PCLRALayer hooks via _self_v.
        3. Call the underlying encoder and return its output transparently.

        Args / Returns
        ──────────────
        *args, **kwargs : forwarded verbatim to self.encoder
        return          : whatever self.encoder returns
                          (e.g. (mu, sigma) for ProDA; class embeddings for CoOp)
        """
        # compute and cache self prompt vector; EMA updated inside
        self._self_v = self.current_prompt_vec()   # [d_p], differentiable

        return self.encoder(*args, **kwargs)


# ================================================================
#  Dual container — holds both encoders, exposes regulariser
# ================================================================

class DualTextEncoderPCLRA(nn.Module):
    """
    Container holding two independently self-guided PCLRATextEncoders.

    Each encoder (enc1, enc2) is fully self-contained:
      - Conditions its LoRA on its OWN prompt vector only
      - Can be saved/loaded and applied to any compatible model independently
      - No runtime dependency on the other encoder

    Regularisation
    ──────────────
    Complementarity and diversity are enforced at training time through
    loss_hor(), which implements the Hierarchical Orthogonality Regularization
    (HOR) loss — a single differentiable objective combining three cosine
    redundancy measures via a log-sum-exp (soft-maximum) operator.

    Three cosine measures:

        c_prompt   — |cos(v1, v2)|               prompt-vector alignment    ∈ [0, 1]
        c_deltaW   — mean_i |cos(m1_i, m2_i)|   mean-row weight-delta align ∈ [0, 1]
        c_A        — mean_i |cos(a1_i, a2_i)|   mean-row A-matrix align     ∈ [0, 1]

    Combined via LSE (temperature τ):

        L_HOR = τ · log( exp(c_prompt/τ) + exp(c_deltaW/τ) + exp(c_A/τ) )
              ∈ [τ·log(3), 1 + τ·log(3)]  ≈ [0.077, 1.077]  at τ = 0.07

    Key properties vs. the original three-loss formulation:
      • Non-vanishing gradient at init — uses |cos| not cos², so gradient
        does not collapse to zero when cosines are small
      • Adaptive gradient routing — softmax weights automatically concentrate
        gradient on whichever level is currently most redundant
      • No warmup schedule required — meaningful signal from step 0
      • Single hyperparameter τ instead of three λs + warmup steps

    Independent deployment
    ──────────────────────
        # Save enc1's trained PCLRA weights
        torch.save(dual.enc1.state_dict(), "enc1_pclra.pt")

        # Later: build a fresh enc1 with same config and load
        fresh_enc1 = PCLRATextEncoder(encoder=..., ...)
        fresh_enc1.load_state_dict(torch.load("enc1_pclra.pt"))

    Args
    ────
    encoder1, encoder2       : text encoder modules to wrap
    get_layers               : ``encoder → list[layer]``
    get_prompt_tokens        : ``encoder → Tensor[..., D]``
    hidden_dim               : token embedding size
    r                        : fixed LoRA rank
    alpha                    : LoRA scaling (α/r = effective scale)
    prompt_dim               : aggregated prompt vector dimension d_p
    last_n_layers            : layers to inject (-1 = all)
    target_keys              : attention projection names to wrap
    dropout                  : dropout on LoRA input path
    tau                      : LSE temperature (default 0.07 — matches CLIP contrastive);
                               lower → sharper gradient focus on the worst-redundancy level
    """

    def __init__(
        self,
        encoder1,
        encoder2,
        get_layers,
        get_prompt_tokens,
        hidden_dim:    int,
        r:             int   = 8,
        alpha:         float = 16.0,
        prompt_dim:    int   = 256,
        last_n_layers: int   = -1,
        target_keys          = ("q_proj", "v_proj", "k_proj", "out_proj"),
        dropout:       float = 0.0,
        tau:           float = 0.07,
    ):
        super().__init__()
        self.tau        = tau
        self.r          = r

        common_kwargs = dict(
            get_layers=get_layers,
            get_prompt_tokens=get_prompt_tokens,
            hidden_dim=hidden_dim,
            r=r, alpha=alpha, prompt_dim=prompt_dim,
            last_n_layers=last_n_layers,
            target_keys=target_keys,
            dropout=dropout,
        )

        self.enc1 = PCLRATextEncoder(encoder=encoder1, **common_kwargs)
        self.enc2 = PCLRATextEncoder(encoder=encoder2, **common_kwargs)

        print(f"[PCLRA] Dual self-guided encoders built: "
              f"{len(self.enc1._pclra_layers)} PCLRA layers each, "
              f"fixed rank r={r}, tau={tau}"
              )

    # ── forward ─────────────────────────────────────────────────────

    def _normalize_encoder_kwargs(self, enc1_kwargs, enc2_kwargs):
        """
        Normalize flexible caller inputs into two kwargs dicts.

        Backward compatibility:
        - If a Tensor is passed as the first positional argument, interpret it
          as `image_features` for both encoders (used by CoCoOp/KgCoOp paths).
        """
        if torch.is_tensor(enc1_kwargs):
            if enc2_kwargs is not None:
                raise TypeError(
                    "When first argument is a Tensor shorthand, second argument "
                    "must be None. Pass dict kwargs explicitly otherwise."
                )
            shared = enc1_kwargs
            return {"image_features": shared}, {"image_features": shared}

        enc1_kwargs = {} if enc1_kwargs is None else enc1_kwargs
        enc2_kwargs = {} if enc2_kwargs is None else enc2_kwargs

        if not isinstance(enc1_kwargs, dict):
            raise TypeError(f"enc1_kwargs must be dict | None, got {type(enc1_kwargs).__name__}")
        if not isinstance(enc2_kwargs, dict):
            raise TypeError(f"enc2_kwargs must be dict | None, got {type(enc2_kwargs).__name__}")

        return enc1_kwargs, enc2_kwargs

    def forward(self, enc1_kwargs: dict | None = None,
                enc2_kwargs: dict | None = None):
        """
        Self-guided forward for both encoders.

        Each encoder independently:
          1. Computes its own prompt vector v_k from its own prompt tokens.
          2. Injects v_k into its own PCLRALayers.
          3. Runs its underlying encoder forward.

        No cross-injection. No dependency between enc1 and enc2.

        Returns
        ───────
        (out1, out2) — whatever each underlying encoder returns.
        """
        enc1_kwargs, enc2_kwargs = self._normalize_encoder_kwargs(enc1_kwargs, enc2_kwargs)

        out1 = self.enc1(**enc1_kwargs)
        out2 = self.enc2(**enc2_kwargs)

        return out1, out2

    def encode_mean_prompt(self, enc1_kwargs: dict | None = None,
                           enc2_kwargs: dict | None = None):
        """
        Inference forward (eval mode).
        Each encoder uses its frozen EMA prompt vector (via current_prompt_vec).
        Returns (out1, out2).
        """
        enc1_kwargs, enc2_kwargs = self._normalize_encoder_kwargs(enc1_kwargs, enc2_kwargs)
        return self.enc1(**enc1_kwargs), self.enc2(**enc2_kwargs)

    # ── regularisation losses ────────────────────────────────────────

    def loss_hor(self) -> tuple[torch.Tensor, dict]:
        """
        Hierarchical Orthogonality Regularizer (HOR), denoted L_HOR in the paper.

        Measures redundancy between the two PCLRA encoders at three levels
        of the information pathway and combines them via a log-sum-exp operator
        that automatically concentrates gradient on whichever level is most
        redundant at each training step.

        ── Level 1: Prompt space ────────────────────────────────────────────
            c_prompt = |cos(v1, v2)|                                 ∈ [0, 1]
            Uses absolute cosine (not squared) so gradient is non-zero even
            when v1 ⊥ v2.  Gradient → prompt_agg weights of both encoders.

        ── Level 2: Weight-delta space ──────────────────────────────────────
            For each layer i:
              m_k^(i) = mean over output rows of ΔW_k^(i)           ∈ R^{d_in}
            c_deltaW = (1/N) Σ_i |cos(m1^(i), m2^(i))|             ∈ [0, 1]
            Row-mean aggregation is more stable than row-wise averaging.
            Gradient → B_gen weights + lora_A (v is detached here).

        ── Level 3: Input-projection space ──────────────────────────────────
            a_k^(i) = mean over rank-dim rows of A_k^(i)            ∈ R^{d_in}
            c_A = (1/N) Σ_i |cos(a1^(i), a2^(i))|                  ∈ [0, 1]
            Gradient → lora_A only (no B_gen, no prompt_agg).

        ── LSE combination ──────────────────────────────────────────────────
            L_HOR = τ · log( exp(c_prompt/τ) + exp(c_deltaW/τ) + exp(c_A/τ) )

            Range: [τ·log(3), 1 + τ·log(3)] ≈ [0.077, 1.077] at τ=0.07
            ∂L/∂c_k = softmax(c/τ)_k  — always non-zero; adapts automatically.

        Returns
        ───────
        (L_HOR, info_dict)
          c_prompt    : ∈ [0,1]; 0 = diverse prompts
          c_deltaW    : ∈ [0,1]; 0 = complementary ΔW directions
          c_A         : ∈ [0,1]; 0 = orthogonal A row-mean directions
          loss_hor    : raw HOR value ∈ [τ·log(3), 1+τ·log(3)]
        """
        tau = self.tau

        # ── Level 1: Prompt space cosine ─────────────────────────────────
        # Live prompt vectors so gradient flows to prompt_agg weights only
        def _live_v(enc):
            P = enc._get_prompt_tokens(enc.encoder)
            if enc._prompt_weights is not None:
                C = P.shape[0]; D = P.shape[-1]
                Pf = P.reshape(C, -1, D).mean(dim=1)
                w  = enc._prompt_weights[:C].to(Pf.device)
                pooled = (Pf * w.unsqueeze(-1)).sum(dim=0)
            else:
                pooled = P.reshape(-1, P.shape[-1]).mean(dim=0)
            return enc.prompt_agg(pooled.detach())   # grad → prompt_agg only

        v1 = _live_v(self.enc1)
        v2 = _live_v(self.enc2)
        c_prompt = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).abs().squeeze()

        # ── Level 2: Weight-delta space cosine ───────────────────────────
        # v is detached → grad flows only to B_gen and lora_A, not prompt_agg
        if self.enc1._ema_prompt_vec is not None:
            v1_det = self.enc1._ema_prompt_vec.detach()
            v2_det = self.enc2._ema_prompt_vec.detach()
        else:
            v1_det = self.enc1.current_prompt_vec().detach()
            v2_det = self.enc2.current_prompt_vec().detach()

        dw_cosines = []
        for pg1, pg2 in zip(self.enc1._pclra_layers, self.enc2._pclra_layers):
            dW1 = pg1.B_gen(v1_det) @ pg1.lora_A   # [d_out, d_in]
            dW2 = pg2.B_gen(v2_det) @ pg2.lora_A   # [d_out, d_in]
            m1  = dW1.mean(dim=0)                   # [d_in]  mean row direction
            m2  = dW2.mean(dim=0)                   # [d_in]
            cos = F.cosine_similarity(m1.unsqueeze(0), m2.unsqueeze(0)).abs().squeeze()
            dw_cosines.append(cos)
        c_deltaW = torch.stack(dw_cosines).mean()

        # ── Level 3: Input-projection space cosine ────────────────────────
        # grad → lora_A only (no B_gen, no prompt_agg involved)
        a_cosines = []
        for pg1, pg2 in zip(self.enc1._pclra_layers, self.enc2._pclra_layers):
            a1 = pg1.lora_A.mean(dim=0)             # [d_in]  mean row direction
            a2 = pg2.lora_A.mean(dim=0)             # [d_in]
            cos = F.cosine_similarity(a1.unsqueeze(0), a2.unsqueeze(0)).abs().squeeze()
            a_cosines.append(cos)
        c_A = torch.stack(a_cosines).mean()

        # ── LSE combination ───────────────────────────────────────────────
        # L = τ · log( exp(c_prompt/τ) + exp(c_deltaW/τ) + exp(c_A/τ) )
        # Equivalent to: τ · logsumexp(stack / τ)
        c_stack = torch.stack([c_prompt, c_deltaW, c_A])   # [3]
        loss_hor = tau * torch.logsumexp(c_stack / tau, dim=0)

        return loss_hor, {
            "c_prompt":  c_prompt.item(),
            "c_deltaW":  c_deltaW.item(),
            "c_A":       c_A.item(),
            "loss_hor":  loss_hor.item(),
        }

    @torch.no_grad()
    def log_effective_rank(self) -> dict:
        """
        Return stable rank for each encoder's A matrices (for logging).
        Stable rank ρ = ||A||_F² / ||A||_op² ∈ [1, r].
        """
        def mean_stable_rank(enc):
            ranks = []
            for pg in enc._pclra_layers:
                A = pg.lora_A
                fro_sq = A.pow(2).sum()
                op_sq  = torch.linalg.matrix_norm(A, ord=2) ** 2
                ranks.append((fro_sq / (op_sq + 1e-8)).item())
            return sum(ranks) / max(len(ranks), 1)

        return {
            "stable_rank_enc1": mean_stable_rank(self.enc1),
            "stable_rank_enc2": mean_stable_rank(self.enc2),
        }


# ================================================================
#  Factory function
# ================================================================

def inject_pclra_text(
    encoder1,
    encoder2,
    get_layers,           # callable: encoder → list[transformer_layer]
    get_prompt_tokens,    # callable: encoder → Tensor[..., hidden_dim]
    hidden_dim:    int,
    r:             int   = 8,
    alpha:         float = 16.0,
    prompt_dim:    int   = 256,
    last_n_layers: int   = -1,
    target_keys          = ("q_proj", "v_proj", "k_proj", "out_proj"),
    dropout:       float = 0.0,
    tau:           float = 0.07,
) -> "DualTextEncoderPCLRA":
    """
    Factory: inject Self-Guided Prompt-Conditioned Low-Rank Adapter (fixed rank) into both
    text encoders and return a DualTextEncoderPCLRA container.

    Each encoder is independently saveable and applicable. The returned
    container's enc1 / enc2 can each be:
      - serialised via enc.state_dict() and torch.save()
      - loaded into any compatible PCLRATextEncoder
      - applied to a model without the other encoder being present

    Args
    ────
    encoder1, encoder2
        The two text encoder modules to wrap.
    get_layers
        ``encoder → list[layer]`` — locates transformer attention layers.
        Example for HuggingFace CLIP::

            get_layers = lambda enc: enc.clip.text_model.encoder.layers

    get_prompt_tokens
        ``encoder → Tensor[..., D]`` — returns raw prompt token tensors.
        Example for ProDA (multi-prompt [K, n_ctx, D])::

            get_prompt_tokens = lambda enc: enc.prompts

        Example for CoOp (single context [n_ctx, D])::

            get_prompt_tokens = lambda enc: enc.ctx

    hidden_dim
        Token embedding size of the text backbone (e.g. 512 for CLIP
        ViT-B/16, 768 for ViT-L/14).
    r                  : fixed LoRA rank (no gating)
    alpha              : LoRA scaling (alpha/r = effective multiplier)
    prompt_dim         : dimension of aggregated self-prompt vector d_p
    last_n_layers      : transformer layers to inject (-1 = all, 0 = none)
    target_keys        : attention projection names to wrap
    dropout            : dropout on LoRA input path
    tau                : LSE temperature (default 0.07); lower = sharper
                         gradient focus on the most redundant level

    Returns
    ───────
    DualTextEncoderPCLRA — holds enc1, enc2, forward(), loss_hor()
    """
    return DualTextEncoderPCLRA(
        encoder1=encoder1,
        encoder2=encoder2,
        get_layers=get_layers,
        get_prompt_tokens=get_prompt_tokens,
        hidden_dim=hidden_dim,
        r=r, alpha=alpha, prompt_dim=prompt_dim,
        last_n_layers=last_n_layers,
        target_keys=target_keys,
        dropout=dropout,
        tau=tau,
    )
