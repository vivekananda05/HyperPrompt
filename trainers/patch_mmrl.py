"""
MMRL: Multi-Modal Representation Learning for CLIP
=====================================================

MMRL follows the same prompt-learning structure as the repository's
CoOp/MaPLe implementations, but uses learnable compound representation
tokens projected into both text and vision branches.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_causal_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def _extract_hidden(layer_output):
    if hasattr(layer_output, "last_hidden_state"):
        return layer_output.last_hidden_state
    if isinstance(layer_output, tuple):
        return layer_output[0]
    if torch.is_tensor(layer_output):
        return layer_output
    raise TypeError(f"Unsupported encoder layer output type: {type(layer_output)}")


class MultiModalRepresentationLearner(nn.Module):
    """Learns compound tokens projected to both text and vision spaces."""

    def __init__(self, tokenizer, classnames, clip_model, n_rep_tokens=8, rep_dim=512, n_layers=12, ctx_init=None, class_token_position="end", csc=False):
        super().__init__()

        self.dtype = clip_model.dtype
        self.classnames = classnames
        self.n_rep_tokens = n_rep_tokens
        self.rep_layers_length = n_layers
        self.rep_dim = rep_dim
        self.class_token_position = class_token_position
        self.csc = csc

        text_dim = clip_model.text_model.config.hidden_size
        visual_dim = clip_model.vision_model.config.hidden_size

        self.compound_rep_tokens = nn.Parameter(torch.empty(self.n_rep_tokens, self.rep_dim))
        if ctx_init:
            init_tokens = tokenizer(ctx_init.replace('_', ' '), return_tensors="pt")
            with torch.no_grad():
                init_embeddings = clip_model.text_model.embeddings.token_embedding(init_tokens.input_ids)
            init_embeddings = init_embeddings[0, 1:-1, :]
            if init_embeddings.size(0) >= self.n_rep_tokens:
                self.compound_rep_tokens.data.copy_(init_embeddings[:self.n_rep_tokens].type_as(self.compound_rep_tokens))
            else:
                nn.init.normal_(self.compound_rep_tokens, std=0.02)
                self.compound_rep_tokens.data[:init_embeddings.size(0)] = init_embeddings.type_as(self.compound_rep_tokens)
        else:
            nn.init.normal_(self.compound_rep_tokens, std=0.02)

        self.compound_rep_tokens_r2vproj = _get_clones(nn.Linear(self.rep_dim, visual_dim), self.rep_layers_length)
        self.compound_rep_tokens_r2tproj = _get_clones(nn.Linear(self.rep_dim, text_dim), self.rep_layers_length)

    def forward(self):
        compound_rep_tokens_text = []
        compound_rep_tokens_visual = []

        for layer_idx in range(self.rep_layers_length):
            rep_tokens = self.compound_rep_tokens
            compound_rep_tokens_text.append(
                self.compound_rep_tokens_r2tproj[layer_idx](rep_tokens).type(self.dtype)
            )
            compound_rep_tokens_visual.append(
                self.compound_rep_tokens_r2vproj[layer_idx](rep_tokens).type(self.dtype)
            )

        return compound_rep_tokens_text, compound_rep_tokens_visual


class TextEncoder_MMRL(nn.Module):
    """CLIP text encoder with MMRL compound-token injection."""

    def __init__(self, clip_model):
        super().__init__()
        self.text_model = clip_model.text_model
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_rep_tokens_text=None):
        if compound_rep_tokens_text is None:
            text_outputs = self.text_model(
                input_ids=tokenized_prompts,
                attention_mask=None,
                return_dict=True,
            )
            pooled_output = text_outputs.pooler_output
            text_features = self.text_projection(pooled_output)
            return F.normalize(text_features, dim=-1)

        n_rep_tokens = compound_rep_tokens_text[0].shape[0]
        pos_embed = self.text_model.embeddings.position_embedding.weight.type(self.dtype)
        x = prompts + pos_embed[: prompts.size(1)]
        eot_index = tokenized_prompts.argmax(dim=-1)

        hidden = x
        causal_mask_orig = _make_causal_mask(x.size(1), x.dtype, x.device).expand(x.size(0), -1, -1, -1)

        for layer_idx, encoder_layer in enumerate(self.text_model.encoder.layers):
            if layer_idx < len(compound_rep_tokens_text):
                rep = compound_rep_tokens_text[layer_idx].to(hidden.dtype)
                rep = rep.unsqueeze(0).expand(hidden.size(0), -1, -1)
                hidden_with_prompt = torch.cat([hidden[:, :1, :], rep, hidden[:, 1:, :]], dim=1)
                causal_mask = _make_causal_mask(hidden_with_prompt.size(1), hidden.dtype, hidden.device).expand(hidden.size(0), -1, -1, -1)
                layer_output = encoder_layer(
                    hidden_with_prompt,
                    attention_mask=None,
                    causal_attention_mask=causal_mask,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                hidden_out = _extract_hidden(layer_output)
                hidden = torch.cat([hidden_out[:, :1, :], hidden_out[:, 1 + n_rep_tokens:, :]], dim=1)
            else:
                layer_output = encoder_layer(
                    hidden,
                    attention_mask=None,
                    causal_attention_mask=causal_mask_orig,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                hidden = _extract_hidden(layer_output)

        hidden = self.text_model.final_layer_norm(hidden)
        text_features = hidden[torch.arange(hidden.size(0), device=hidden.device), eot_index + n_rep_tokens]
        text_features = self.text_projection(text_features)
        return F.normalize(text_features, dim=-1)


class VisualEncoder_MMRL(nn.Module):
    """CLIP vision encoder with MMRL compound-token injection."""

    def __init__(self, clip_model):
        super().__init__()
        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection
        self.dtype = clip_model.dtype

    def forward(self, pixel_values, compound_rep_tokens_visual=None):
        vision_outputs = self.vision_model(pixel_values=pixel_values, return_dict=True)
        pooled_output = vision_outputs.pooler_output
        base_features = F.normalize(self.visual_projection(pooled_output), dim=-1)

        if compound_rep_tokens_visual is None:
            return base_features, base_features

        hidden = self.vision_model.embeddings(pixel_values=pixel_values)
        hidden = self.vision_model.pre_layrnorm(hidden)
        batch_size = hidden.size(0)
        n_rep_tokens = compound_rep_tokens_visual[0].shape[0]

        for layer_idx, encoder_layer in enumerate(self.vision_model.encoder.layers):
            if layer_idx < len(compound_rep_tokens_visual):
                rep = compound_rep_tokens_visual[layer_idx].to(hidden.dtype)
                rep = rep.unsqueeze(0).expand(batch_size, -1, -1)
                hidden_with_prompt = torch.cat([hidden[:, :1, :], rep, hidden[:, 1:, :]], dim=1)
                layer_output = encoder_layer(
                    hidden_with_prompt,
                    attention_mask=None,
                    causal_attention_mask=None,
                    output_attentions=False,
                    return_dict=True,
                )
                hidden_out = _extract_hidden(layer_output)
                hidden = torch.cat([hidden_out[:, :1, :], hidden_out[:, 1 + n_rep_tokens:, :]], dim=1)
            else:
                layer_output = encoder_layer(
                    hidden,
                    attention_mask=None,
                    causal_attention_mask=None,
                    output_attentions=False,
                    return_dict=True,
                )
                hidden = _extract_hidden(layer_output)

        pooled = self.vision_model.post_layernorm(hidden[:, 0, :])
        prompted_features = F.normalize(self.visual_projection(pooled), dim=-1)
        return base_features, prompted_features


class MMRL_Loss(nn.Module):
    def __init__(self, reg_weight=1.0, alpha=0.7):
        super().__init__()
        self.reg_weight = reg_weight
        self.alpha = alpha

    def forward(
        self,
        logits_main,
        logits_token_enhanced,
        image_features,
        text_features,
        image_features_frozen,
        text_features_frozen,
        labels,
    ):
        xe_loss_main = F.cross_entropy(logits_main, labels)
        xe_loss_token = F.cross_entropy(logits_token_enhanced, labels)

        img_norm = F.normalize(image_features, dim=-1)
        frozen_img_norm = F.normalize(image_features_frozen, dim=-1)
        cossim_reg_img = 1.0 - torch.mean(F.cosine_similarity(img_norm, frozen_img_norm, dim=-1))

        text_norm = F.normalize(text_features, dim=-1)
        frozen_text_norm = F.normalize(text_features_frozen, dim=-1)
        cossim_reg_text = 1.0 - torch.mean(F.cosine_similarity(text_norm, frozen_text_norm, dim=-1))

        total_loss = (
            self.alpha * xe_loss_main
            + (1.0 - self.alpha) * xe_loss_token
            + self.reg_weight * cossim_reg_img
            + self.reg_weight * cossim_reg_text
        )

        return total_loss
