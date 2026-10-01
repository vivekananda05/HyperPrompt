# train.py
"""
Universal Training Script for HSI Classification
Supports the paper's CoOp, KgCoOp, MaPLe, PromptSRC, PromptKD, and MMRL variants.
"""

import os
import sys
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils import compute_imf_weights_torch, compute_metrics, extract_fused_logits_from_model_output
from config_loader import get_config_from_args
from dataset import build_dataloader, CUDAPrefetcher


def get_model(model_name, cfg, in_channels):
    """
    Load the appropriate model based on model_name.
    
    Args:
        model_name: 'patch_coop', or 
        cfg: Config object with all required parameters
        in_channels: Input spectral channels
    
    Returns:
        Instantiated model
    """
    # Helper function for PCLRA enablement
    def is_pclra_enabled(cfg):
        """Check if PCLRA should be enabled based on CLI override or config"""
        if hasattr(cfg, '_pclra_override'):
            return cfg._pclra_override
        # Fall back to checking PCLRA_RANK from config
        return cfg.PCLRA_RANK > 0
    
    if model_name.lower() == 'patch_coop':
        from trainers.models.patch_coop_model import HSIPatchCoOp
        model = HSIPatchCoOp(
            in_channels          = in_channels,
            num_classes          = cfg.NUM_CLASSES,
            classnames           = cfg.CLASS_NAMES,
            clip_name            = cfg.CLIP_MODEL_NAME,
            ctx_len              = cfg.CTX_LEN,
            class_token_position = cfg.CLASS_TOKEN_POSITION,
            csc                  = cfg.CSC,
            ctx_init             = cfg.CTX_INIT,
        )

    elif model_name.lower() == 'pixel_coop':
        from trainers.models.pixel_coop_model import HSIPixelCoOp
        model = HSIPixelCoOp(
            in_channels          = in_channels,
            num_classes          = cfg.NUM_CLASSES,
            classnames           = cfg.CLASS_NAMES,
            clip_name            = cfg.CLIP_MODEL_NAME,
            sam_model            = cfg.SAM_MODEL_NAME,
            ctx_len              = cfg.CTX_LEN,
            class_token_position = cfg.CLASS_TOKEN_POSITION,
            csc                  = cfg.CSC,
            ctx_init             = cfg.CTX_INIT,
            upsampler_type       = cfg.UPSAMPLER_TYPE if hasattr(cfg, 'UPSAMPLER_TYPE') else 'bilinear',
            upsampler_weights    = cfg.UPSAMPLER_WEIGHTS if hasattr(cfg, 'UPSAMPLER_WEIGHTS') else None,
        )    

    elif model_name.lower() == 'pixel_kgcoop':
        from trainers.models.pixel_kgcoop_model import HSIPixelKgCoOp
        model = HSIPixelKgCoOp(
            in_channels          = in_channels,
            num_classes          = cfg.NUM_CLASSES,
            classnames           = cfg.CLASS_NAMES,
            clip_name            = cfg.CLIP_MODEL_NAME,
            sam_model            = cfg.SAM_MODEL_NAME,
            ctx_len              = cfg.CTX_LEN,
            class_token_position = cfg.CLASS_TOKEN_POSITION,
            csc                  = cfg.CSC,
            ctx_init             = cfg.CTX_INIT,
            dataset_name         = cfg.DATASET_NAME,
            knowledge_weight     = cfg.LAMBDA_KNOWLEDGE if hasattr(cfg, 'LAMBDA_KNOWLEDGE') else 0.1,
            upsampler_type       = cfg.UPSAMPLER_TYPE if hasattr(cfg, 'UPSAMPLER_TYPE') else 'bilinear',
            upsampler_weights    = cfg.UPSAMPLER_WEIGHTS if hasattr(cfg, 'UPSAMPLER_WEIGHTS') else None,
        )
    
    elif model_name.lower() == 'pixel_maple':
        from trainers.models.pixel_maple_model import HSIPixelMaPLe
        model = HSIPixelMaPLe(
            in_channels          = in_channels,
            num_classes          = cfg.NUM_CLASSES,
            classnames           = cfg.CLASS_NAMES,
            clip_name            = cfg.CLIP_MODEL_NAME,
            sam_model            = cfg.SAM_MODEL_NAME,
            ctx_len              = cfg.CTX_LEN,
            prompt_depth         = cfg.PROMPT_DEPTH if hasattr(cfg, 'PROMPT_DEPTH') else 1,
            class_token_position = cfg.CLASS_TOKEN_POSITION,
            csc                  = cfg.CSC,
            ctx_init             = cfg.CTX_INIT,
            upsampler_type       = cfg.UPSAMPLER_TYPE if hasattr(cfg, 'UPSAMPLER_TYPE') else 'bilinear',
            upsampler_weights    = cfg.UPSAMPLER_WEIGHTS if hasattr(cfg, 'UPSAMPLER_WEIGHTS') else None,
        )
    
    elif model_name.lower() == 'pixel_promptkd':
        from trainers.models.pixel_promptkd_model import HSIPixelPromptKD
        model = HSIPixelPromptKD(
            in_channels          = in_channels,
            num_classes          = cfg.NUM_CLASSES,
            classnames           = cfg.CLASS_NAMES,
            clip_name            = cfg.CLIP_MODEL_NAME,
            sam_model            = cfg.SAM_MODEL_NAME,
            ctx_len              = cfg.CTX_LEN,
            class_token_position = cfg.CLASS_TOKEN_POSITION,
            csc                  = cfg.CSC,
            ctx_init             = cfg.CTX_INIT,
            temperature          = cfg.PROMPTKD_TEMPERATURE if hasattr(cfg, 'PROMPTKD_TEMPERATURE') else 4.0,
            kd_weight            = cfg.PROMPTKD_KD_WEIGHT if hasattr(cfg, 'PROMPTKD_KD_WEIGHT') else 1.0,
            upsampler_type       = cfg.UPSAMPLER_TYPE if hasattr(cfg, 'UPSAMPLER_TYPE') else 'bilinear',
            upsampler_weights    = cfg.UPSAMPLER_WEIGHTS if hasattr(cfg, 'UPSAMPLER_WEIGHTS') else None,
        )
    
    elif model_name.lower() == 'pixel_promptsrc':
        from trainers.models.pixel_promptsrc_model import HSIPixelPromptSRC
        model = HSIPixelPromptSRC(
            in_channels          = in_channels,
            num_classes          = cfg.NUM_CLASSES,
            classnames           = cfg.CLASS_NAMES,
            clip_name            = cfg.CLIP_MODEL_NAME,
            sam_model            = cfg.SAM_MODEL_NAME,
            ctx_len              = cfg.CTX_LEN,
            class_token_position = cfg.CLASS_TOKEN_POSITION,
            csc                  = cfg.CSC,
            ctx_init             = cfg.CTX_INIT,
            temperature          = cfg.PROMPTSRC_TEMPERATURE if hasattr(cfg, 'PROMPTSRC_TEMPERATURE') else 1.0,
            upsampler_type       = cfg.UPSAMPLER_TYPE if hasattr(cfg, 'UPSAMPLER_TYPE') else 'bilinear',
            upsampler_weights    = cfg.UPSAMPLER_WEIGHTS if hasattr(cfg, 'UPSAMPLER_WEIGHTS') else None,
        )

    elif model_name.lower() == 'pixel_mmrl':
        from trainers.models.pixel_mmrl_model import HSIPixelMMRL
        model = HSIPixelMMRL(
            in_channels          = in_channels,
            num_classes          = cfg.NUM_CLASSES,
            classnames           = cfg.CLASS_NAMES,
            clip_name            = cfg.CLIP_MODEL_NAME,
            sam_model            = cfg.SAM_MODEL_NAME,
            ctx_len              = cfg.CTX_LEN if hasattr(cfg, 'CTX_LEN') else 4,
            class_token_position = cfg.CLASS_TOKEN_POSITION if hasattr(cfg, 'CLASS_TOKEN_POSITION') else 'end',
            csc                  = cfg.CSC if hasattr(cfg, 'CSC') else False,
            ctx_init             = cfg.CTX_INIT if hasattr(cfg, 'CTX_INIT') else 'hyperspectral image of a',
            n_rep_tokens         = cfg.N_REP_TOKENS if hasattr(cfg, 'N_REP_TOKENS') else 8,
            rep_dim              = cfg.REP_DIM if hasattr(cfg, 'REP_DIM') else 512,
            n_layers             = cfg.N_LAYERS if hasattr(cfg, 'N_LAYERS') else 12,
            alpha                = cfg.MMRL_ALPHA if hasattr(cfg, 'MMRL_ALPHA') else 0.7,
            reg_weight           = cfg.REG_WEIGHT if hasattr(cfg, 'REG_WEIGHT') else 1.0,
            upsampler_type       = cfg.UPSAMPLER_TYPE if hasattr(cfg, 'UPSAMPLER_TYPE') else 'bilinear',
            upsampler_weights    = cfg.UPSAMPLER_WEIGHTS if hasattr(cfg, 'UPSAMPLER_WEIGHTS') else None,
        )
    
    elif model_name.lower() == 'patch_promptkd':
        from trainers.models.patch_promptkd_model import HSIPatchPromptKD
        model = HSIPatchPromptKD(
            in_channels          = in_channels,
            num_classes          = cfg.NUM_CLASSES,
            classnames           = cfg.CLASS_NAMES,
            clip_name            = cfg.CLIP_MODEL_NAME,
            ctx_len              = cfg.CTX_LEN,
            class_token_position = cfg.CLASS_TOKEN_POSITION,
            csc                  = cfg.CSC,
            ctx_init             = cfg.CTX_INIT,
            temperature          = cfg.PROMPTKD_TEMPERATURE if hasattr(cfg, 'PROMPTKD_TEMPERATURE') else (cfg.TEMPERATURE if hasattr(cfg, 'TEMPERATURE') else 4.0),
            kd_weight            = cfg.KD_WEIGHT if hasattr(cfg, 'KD_WEIGHT') else (cfg.LAMBDA_KL if hasattr(cfg, 'LAMBDA_KL') else 1.0),
        )
    
    elif model_name.lower() == 'patch_promptsrc':
        from trainers.models.patch_promptsrc_model import HSIPatchPromptSRC
        model = HSIPatchPromptSRC(
            in_channels          = in_channels,
            num_classes          = cfg.NUM_CLASSES,
            classnames           = cfg.CLASS_NAMES,
            clip_name            = cfg.CLIP_MODEL_NAME,
            ctx_len_text         = cfg.CTX_LEN_TEXT if hasattr(cfg, 'CTX_LEN_TEXT') else 4,
            ctx_len_vision       = cfg.CTX_LEN_VISION if hasattr(cfg, 'CTX_LEN_VISION') else 4,
            ctx_init             = cfg.CTX_INIT,
            temperature          = cfg.PROMPTSRC_TEMPERATURE if hasattr(cfg, 'PROMPTSRC_TEMPERATURE') else (cfg.TEMPERATURE if hasattr(cfg, 'TEMPERATURE') else 1.0),
        )
    
    elif model_name.lower() == 'patch_maple':
        from trainers.models.patch_maple_model import HSIPatchMaPLe
        model = HSIPatchMaPLe(
            in_channels          = in_channels,
            num_classes          = cfg.NUM_CLASSES,
            classnames           = cfg.CLASS_NAMES,
            clip_name            = cfg.CLIP_MODEL_NAME,
            ctx_len              = cfg.CTX_LEN,
            prompt_depth         = cfg.PROMPT_DEPTH if hasattr(cfg, 'PROMPT_DEPTH') else 1,
            class_token_position = cfg.CLASS_TOKEN_POSITION,
            csc                  = cfg.CSC,
            ctx_init             = cfg.CTX_INIT,
        )
    
    elif model_name.lower() == 'patch_kgcoop':
        from trainers.models.patch_kgcoop_model import HSIPatchKgCoOp
        model = HSIPatchKgCoOp(
            in_channels          = in_channels,
            num_classes          = cfg.NUM_CLASSES,
            classnames           = cfg.CLASS_NAMES,
            clip_name            = cfg.CLIP_MODEL_NAME,
            ctx_len              = cfg.CTX_LEN,
            class_token_position = cfg.CLASS_TOKEN_POSITION,
            csc                  = cfg.CSC,
            ctx_init             = cfg.CTX_INIT,
            dataset_name         = cfg.DATASET_NAME,
            knowledge_weight     = cfg.LAMBDA_KNOWLEDGE if hasattr(cfg, 'LAMBDA_KNOWLEDGE') else 0.1,
        )
    
    elif model_name.lower() == 'hyperprompt_coop':
        from trainers.models.hyperprompt_coop_model import HSIHyperPromptCoOp
        model = HSIHyperPromptCoOp(
            in_channels          = in_channels,
            num_classes          = cfg.NUM_CLASSES,
            classnames           = cfg.CLASS_NAMES,
            clip_name            = cfg.CLIP_MODEL_NAME,
            sam_model            = cfg.SAM_MODEL_NAME,
            ctx_len_patch        = cfg.CTX_LEN_PATCH,
            ctx_len_pixel        = cfg.CTX_LEN_PIXEL,
            class_token_position = cfg.CLASS_TOKEN_POSITION,
            csc                  = cfg.CSC,
            ctx_init             = cfg.CTX_INIT,
            upsampler_type       = cfg.UPSAMPLER_TYPE if hasattr(cfg, 'UPSAMPLER_TYPE') else 'bilinear',
            upsampler_weights    = cfg.UPSAMPLER_WEIGHTS if hasattr(cfg, 'UPSAMPLER_WEIGHTS') else None,
            tcdm_last_k_layers   = cfg.TCDM_LAST_K_LAYERS if hasattr(cfg, 'TCDM_LAST_K_LAYERS') else 4,
            use_cls_only         = cfg.TCDM_USE_CLS_ONLY if hasattr(cfg, 'TCDM_USE_CLS_ONLY') else False,
            pclra_enabled      = is_pclra_enabled(cfg),
            pclra_rank         = cfg.PCLRA_RANK if hasattr(cfg, 'PCLRA_RANK') else 0,
            pclra_alpha        = cfg.PCLRA_ALPHA if hasattr(cfg, 'PCLRA_ALPHA') else None,
            pclra_prompt_dim   = cfg.PCLRA_PROMPT_DIM if hasattr(cfg, 'PCLRA_PROMPT_DIM') else 256,
            pclra_last_n_layers= cfg.PCLRA_LAST_N if hasattr(cfg, 'PCLRA_LAST_N') else -1,
            pclra_target_keys  = cfg.PCLRA_TARGET_KEY if hasattr(cfg, 'PCLRA_TARGET_KEY') else ("q_proj", "v_proj"),
            pclra_dropout      = cfg.PCLRA_DROPOUT if hasattr(cfg, 'PCLRA_DROPOUT') else 0.0,
            pclra_tau          = cfg.PCLRA_TAU if hasattr(cfg, 'PCLRA_TAU') else 0.07,
        )

    elif model_name.lower() == 'hyperprompt_kgcoop':
        from trainers.models.hyperprompt_kgcoop_model import HSIHyperPromptKgCoOp
        model = HSIHyperPromptKgCoOp(
            in_channels          = in_channels,
            num_classes          = cfg.NUM_CLASSES,
            classnames           = cfg.CLASS_NAMES,
            clip_name            = cfg.CLIP_MODEL_NAME,
            sam_model            = cfg.SAM_MODEL_NAME,
            ctx_len_patch        = cfg.CTX_LEN_PATCH,
            ctx_len_pixel        = cfg.CTX_LEN_PIXEL,
            class_token_position = cfg.CLASS_TOKEN_POSITION,
            csc                  = cfg.CSC,
            ctx_init             = cfg.CTX_INIT,
            dataset_name         = cfg.DATASET_NAME,
            upsampler_type       = cfg.UPSAMPLER_TYPE if hasattr(cfg, 'UPSAMPLER_TYPE') else 'bilinear',
            upsampler_weights    = cfg.UPSAMPLER_WEIGHTS if hasattr(cfg, 'UPSAMPLER_WEIGHTS') else None,
            tcdm_last_k_layers   = cfg.TCDM_LAST_K_LAYERS if hasattr(cfg, 'TCDM_LAST_K_LAYERS') else 4,
            pclra_enabled      = is_pclra_enabled(cfg),
            pclra_rank         = cfg.PCLRA_RANK if hasattr(cfg, 'PCLRA_RANK') else 0,
            pclra_alpha        = cfg.PCLRA_ALPHA if hasattr(cfg, 'PCLRA_ALPHA') else None,
            pclra_prompt_dim   = cfg.PCLRA_PROMPT_DIM if hasattr(cfg, 'PCLRA_PROMPT_DIM') else 256,
            pclra_last_n_layers= cfg.PCLRA_LAST_N if hasattr(cfg, 'PCLRA_LAST_N') else -1,
            pclra_target_keys  = cfg.PCLRA_TARGET_KEY if hasattr(cfg, 'PCLRA_TARGET_KEY') else ("q_proj", "v_proj"),
            pclra_dropout      = cfg.PCLRA_DROPOUT if hasattr(cfg, 'PCLRA_DROPOUT') else 0.0,
            pclra_tau          = cfg.PCLRA_TAU if hasattr(cfg, 'PCLRA_TAU') else 0.07,
        )
    
    elif model_name.lower() == 'hyperprompt_maple':
        from trainers.models.hyperprompt_maple_model import HSIHyperPromptMaPLe
        model = HSIHyperPromptMaPLe(
            in_channels          = in_channels,
            num_classes          = cfg.NUM_CLASSES,
            classnames           = cfg.CLASS_NAMES,
            clip_name            = cfg.CLIP_MODEL_NAME,
            sam_model            = cfg.SAM_MODEL_NAME,
            ctx_len_patch        = cfg.CTX_LEN_PATCH,
            ctx_len_pixel        = cfg.CTX_LEN_PIXEL,
            prompt_depth         = cfg.PROMPT_DEPTH if hasattr(cfg, 'PROMPT_DEPTH') else 1,
            class_token_position = cfg.CLASS_TOKEN_POSITION,
            csc                  = cfg.CSC,
            ctx_init             = cfg.CTX_INIT,
            upsampler_type       = cfg.UPSAMPLER_TYPE if hasattr(cfg, 'UPSAMPLER_TYPE') else 'bilinear',
            upsampler_weights    = cfg.UPSAMPLER_WEIGHTS if hasattr(cfg, 'UPSAMPLER_WEIGHTS') else None,
            tcdm_last_k_layers   = cfg.TCDM_LAST_K_LAYERS if hasattr(cfg, 'TCDM_LAST_K_LAYERS') else 4,
            pclra_enabled      = is_pclra_enabled(cfg),
            pclra_rank         = cfg.PCLRA_RANK if hasattr(cfg, 'PCLRA_RANK') else 0,
            pclra_alpha        = cfg.PCLRA_ALPHA if hasattr(cfg, 'PCLRA_ALPHA') else None,
            pclra_prompt_dim   = cfg.PCLRA_PROMPT_DIM if hasattr(cfg, 'PCLRA_PROMPT_DIM') else 256,
            pclra_last_n_layers= cfg.PCLRA_LAST_N if hasattr(cfg, 'PCLRA_LAST_N') else -1,
            pclra_target_keys  = cfg.PCLRA_TARGET_KEY if hasattr(cfg, 'PCLRA_TARGET_KEY') else ("q_proj", "v_proj"),
            pclra_dropout      = cfg.PCLRA_DROPOUT if hasattr(cfg, 'PCLRA_DROPOUT') else 0.0,
            pclra_tau          = cfg.PCLRA_TAU if hasattr(cfg, 'PCLRA_TAU') else 0.07,
        )

    elif model_name.lower() == 'hyperprompt_mmrl':
        from trainers.models.hyperprompt_mmrl_model import HSIHyperPromptMMRL
        model = HSIHyperPromptMMRL(
            in_channels          = in_channels,
            num_classes          = cfg.NUM_CLASSES,
            classnames           = cfg.CLASS_NAMES,
            clip_name            = cfg.CLIP_MODEL_NAME,
            sam_model            = cfg.SAM_MODEL_NAME,
            ctx_len_patch        = cfg.CTX_LEN_PATCH if hasattr(cfg, 'CTX_LEN_PATCH') else cfg.CTX_LEN if hasattr(cfg, 'CTX_LEN') else 4,
            ctx_len_pixel        = cfg.CTX_LEN_PIXEL if hasattr(cfg, 'CTX_LEN_PIXEL') else cfg.CTX_LEN if hasattr(cfg, 'CTX_LEN') else 4,
            class_token_position = cfg.CLASS_TOKEN_POSITION if hasattr(cfg, 'CLASS_TOKEN_POSITION') else 'end',
            csc                  = cfg.CSC if hasattr(cfg, 'CSC') else False,
            ctx_init             = cfg.CTX_INIT if hasattr(cfg, 'CTX_INIT') else 'hyperspectral image of a',
            n_rep_tokens         = cfg.N_REP_TOKENS if hasattr(cfg, 'N_REP_TOKENS') else 8,
            rep_dim              = cfg.REP_DIM if hasattr(cfg, 'REP_DIM') else 512,
            n_layers             = cfg.N_LAYERS if hasattr(cfg, 'N_LAYERS') else 12,
            upsampler_type       = cfg.UPSAMPLER_TYPE if hasattr(cfg, 'UPSAMPLER_TYPE') else 'bilinear',
            upsampler_weights    = cfg.UPSAMPLER_WEIGHTS if hasattr(cfg, 'UPSAMPLER_WEIGHTS') else None,
            tcdm_last_k_layers   = cfg.TCDM_LAST_K_LAYERS if hasattr(cfg, 'TCDM_LAST_K_LAYERS') else 4,
            pclra_enabled      = is_pclra_enabled(cfg),
            pclra_rank         = cfg.PCLRA_RANK if hasattr(cfg, 'PCLRA_RANK') else 0,
            pclra_alpha        = cfg.PCLRA_ALPHA if hasattr(cfg, 'PCLRA_ALPHA') else None,
            pclra_prompt_dim   = cfg.PCLRA_PROMPT_DIM if hasattr(cfg, 'PCLRA_PROMPT_DIM') else 256,
            pclra_last_n_layers= cfg.PCLRA_LAST_N if hasattr(cfg, 'PCLRA_LAST_N') else -1,
            pclra_target_keys  = cfg.PCLRA_TARGET_KEY if hasattr(cfg, 'PCLRA_TARGET_KEY') else ("q_proj", "v_proj"),
            pclra_dropout      = cfg.PCLRA_DROPOUT if hasattr(cfg, 'PCLRA_DROPOUT') else 0.0,
            pclra_tau          = cfg.PCLRA_TAU if hasattr(cfg, 'PCLRA_TAU') else 0.07,
        )
    
    elif model_name.lower() == 'hyperprompt_promptkd':
        from trainers.models.hyperprompt_promptkd_model import HSIHyperPromptPromptKD
        model = HSIHyperPromptPromptKD(
            in_channels          = in_channels,
            num_classes          = cfg.NUM_CLASSES,
            classnames           = cfg.CLASS_NAMES,
            clip_name            = cfg.CLIP_MODEL_NAME,
            sam_model            = cfg.SAM_MODEL_NAME,
            ctx_len_patch        = cfg.CTX_LEN_PATCH,
            ctx_len_pixel        = cfg.CTX_LEN_PIXEL,
            class_token_position = cfg.CLASS_TOKEN_POSITION,
            csc                  = cfg.CSC,
            ctx_init             = cfg.CTX_INIT,
            temperature          = cfg.PROMPTKD_TEMPERATURE if hasattr(cfg, 'PROMPTKD_TEMPERATURE') else (cfg.TEMPERATURE if hasattr(cfg, 'TEMPERATURE') else 4.0),
            upsampler_type       = cfg.UPSAMPLER_TYPE if hasattr(cfg, 'UPSAMPLER_TYPE') else 'bilinear',
            upsampler_weights    = cfg.UPSAMPLER_WEIGHTS if hasattr(cfg, 'UPSAMPLER_WEIGHTS') else None,
            tcdm_last_k_layers   = cfg.TCDM_LAST_K_LAYERS if hasattr(cfg, 'TCDM_LAST_K_LAYERS') else 4,
            pclra_enabled      = is_pclra_enabled(cfg),
            pclra_rank         = cfg.PCLRA_RANK if hasattr(cfg, 'PCLRA_RANK') else 0,
            pclra_alpha        = cfg.PCLRA_ALPHA if hasattr(cfg, 'PCLRA_ALPHA') else None,
            pclra_prompt_dim   = cfg.PCLRA_PROMPT_DIM if hasattr(cfg, 'PCLRA_PROMPT_DIM') else 256,
            pclra_last_n_layers= cfg.PCLRA_LAST_N if hasattr(cfg, 'PCLRA_LAST_N') else -1,
            pclra_target_keys  = cfg.PCLRA_TARGET_KEY if hasattr(cfg, 'PCLRA_TARGET_KEY') else ("q_proj", "v_proj"),
            pclra_dropout      = cfg.PCLRA_DROPOUT if hasattr(cfg, 'PCLRA_DROPOUT') else 0.0,
            pclra_tau          = cfg.PCLRA_TAU if hasattr(cfg, 'PCLRA_TAU') else 0.07,
        )

    elif model_name.lower() == 'hyperprompt_promptsrc':
        from trainers.models.hyperprompt_promptsrc_model import HSIHyperPromptPromptSRC
        model = HSIHyperPromptPromptSRC(
            in_channels=in_channels,
            num_classes=cfg.NUM_CLASSES,
            classnames=cfg.CLASS_NAMES,
            clip_name=cfg.CLIP_MODEL_NAME,
            sam_model=cfg.SAM_MODEL_NAME,
            ctx_len_patch=cfg.CTX_LEN_PATCH,
            ctx_len_pixel=cfg.CTX_LEN_PIXEL,
            class_token_position=cfg.CLASS_TOKEN_POSITION,
            csc=cfg.CSC,
            ctx_init=cfg.CTX_INIT,
            temperature=cfg.PROMPTSRC_TEMPERATURE if hasattr(cfg, 'PROMPTSRC_TEMPERATURE') else 1.0,
            upsampler_type=cfg.UPSAMPLER_TYPE if hasattr(cfg, 'UPSAMPLER_TYPE') else 'bilinear',
            upsampler_weights=cfg.UPSAMPLER_WEIGHTS if hasattr(cfg, 'UPSAMPLER_WEIGHTS') else None,
            tcdm_last_k_layers=cfg.TCDM_LAST_K_LAYERS if hasattr(cfg, 'TCDM_LAST_K_LAYERS') else 4,
            use_cls_only=cfg.TCDM_USE_CLS_ONLY if hasattr(cfg, 'TCDM_USE_CLS_ONLY') else False,
            pclra_enabled=is_pclra_enabled(cfg),
            pclra_rank=cfg.PCLRA_RANK if hasattr(cfg, 'PCLRA_RANK') else 0,
            pclra_alpha=cfg.PCLRA_ALPHA if hasattr(cfg, 'PCLRA_ALPHA') else None,
            pclra_prompt_dim=cfg.PCLRA_PROMPT_DIM if hasattr(cfg, 'PCLRA_PROMPT_DIM') else 256,
            pclra_last_n_layers=cfg.PCLRA_LAST_N if hasattr(cfg, 'PCLRA_LAST_N') else -1,
            pclra_target_keys=cfg.PCLRA_TARGET_KEY if hasattr(cfg, 'PCLRA_TARGET_KEY') else ("q_proj", "v_proj"),
            pclra_dropout=cfg.PCLRA_DROPOUT if hasattr(cfg, 'PCLRA_DROPOUT') else 0.0,
            pclra_tau=cfg.PCLRA_TAU if hasattr(cfg, 'PCLRA_TAU') else 0.07,
        )

    elif model_name.lower() == 'patch_mmrl':
        from trainers.models.patch_mmrl_model import HSIPatchMMRL
        model = HSIPatchMMRL(
            in_channels          = in_channels,
            num_classes          = cfg.NUM_CLASSES,
            classnames           = cfg.CLASS_NAMES,
            clip_name            = cfg.CLIP_MODEL_NAME,
            ctx_len              = cfg.CTX_LEN if hasattr(cfg, 'CTX_LEN') else 4,
            ctx_init             = cfg.CTX_INIT if hasattr(cfg, 'CTX_INIT') else 'hyperspectral image of a',
            class_token_position = cfg.CLASS_TOKEN_POSITION if hasattr(cfg, 'CLASS_TOKEN_POSITION') else 'end',
            csc                  = cfg.CSC if hasattr(cfg, 'CSC') else False,
            n_rep_tokens         = cfg.N_REP_TOKENS if hasattr(cfg, 'N_REP_TOKENS') else 8,
            rep_dim              = cfg.REP_DIM if hasattr(cfg, 'REP_DIM') else 512,
            n_layers             = cfg.N_LAYERS if hasattr(cfg, 'N_LAYERS') else 12,
            alpha                = cfg.MMRL_ALPHA if hasattr(cfg, 'MMRL_ALPHA') else 0.7,
            reg_weight           = cfg.REG_WEIGHT if hasattr(cfg, 'REG_WEIGHT') else 1.0,
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Available models: patch_coop, patch_kgcoop, patch_maple, patch_promptsrc, patch_promptkd, patch_mmrl, pixel_coop, pixel_kgcoop, pixel_maple, pixel_promptsrc, pixel_promptkd, pixel_mmrl, hyperprompt_coop, hyperprompt_kgcoop, hyperprompt_maple, hyperprompt_promptsrc, hyperprompt_promptkd, hyperprompt_mmrl")
    
    return model


def get_loss_function(model_name, cfg):
    """
    Get the loss computation function for the given model.
    
    This function routes to the appropriate loss function module based on model name.

    Args:
        model_name: Model name (e.g., 'patch_coop', etc.)
        cfg: Config object

    Returns:
        loss_fn: A function that computes loss given model outputs and labels
    """
    from trainers.losses import (
        get_coop_loss_fn,
        get_pixel_coop_loss_fn,
        get_mmrl_loss_fn,
        get_promptkd_loss_fn,
        get_promptsrc_loss_fn,
        get_maple_loss_fn,
        get_kgcoop_loss_fn,
        get_pixel_kgcoop_loss_fn,
        get_pixel_maple_loss_fn,
        get_pixel_promptkd_loss_fn,
        get_pixel_promptsrc_loss_fn,
        get_pixel_mmrl_loss_fn,
        get_hyperprompt_coop_loss_fn,
        get_hyperprompt_kgcoop_loss_fn,
        get_hyperprompt_maple_loss_fn,
        get_hyperprompt_promptsrc_loss_fn,
        get_hyperprompt_mmrl_loss_fn,
        get_hyperprompt_promptkd_loss_fn,
    )
    
    normalized_model_name = model_name.lower()
    
    # Single-branch models
    if normalized_model_name == 'patch_coop':
        return get_coop_loss_fn()
    elif normalized_model_name == 'pixel_coop':
        return get_pixel_coop_loss_fn()    
    elif normalized_model_name == 'pixel_kgcoop':
        return get_pixel_kgcoop_loss_fn()
    elif normalized_model_name == 'pixel_maple':
        return get_pixel_maple_loss_fn()
    elif normalized_model_name == 'pixel_promptkd':
        return get_pixel_promptkd_loss_fn()
    elif normalized_model_name == 'pixel_promptsrc':
        return get_pixel_promptsrc_loss_fn()
    elif normalized_model_name == 'pixel_mmrl':
        return get_pixel_mmrl_loss_fn()
    elif normalized_model_name == 'patch_mmrl':
        return get_mmrl_loss_fn()
    elif normalized_model_name == 'patch_promptkd':
        return get_promptkd_loss_fn()
    elif normalized_model_name == 'patch_promptsrc':
        return get_promptsrc_loss_fn()
    elif normalized_model_name == 'patch_maple':
        return get_maple_loss_fn()
    elif normalized_model_name == 'patch_kgcoop':
        return get_kgcoop_loss_fn()
    
    # HyperPrompt models
    elif normalized_model_name == 'hyperprompt_coop':
        return get_hyperprompt_coop_loss_fn()
    elif normalized_model_name == 'hyperprompt_kgcoop':
        return get_hyperprompt_kgcoop_loss_fn()
    elif normalized_model_name == 'hyperprompt_maple':
        return get_hyperprompt_maple_loss_fn()
    elif normalized_model_name == 'hyperprompt_promptsrc':
        return get_hyperprompt_promptsrc_loss_fn()
    elif normalized_model_name == 'hyperprompt_mmrl':
        return get_hyperprompt_mmrl_loss_fn()
    elif normalized_model_name == 'hyperprompt_promptkd':
        return get_hyperprompt_promptkd_loss_fn()
    else:
        raise ValueError(f"Unknown model: {model_name}. Available models: patch_coop, patch_kgcoop, patch_maple, patch_promptsrc, patch_promptkd, patch_mmrl, pixel_coop, pixel_kgcoop, pixel_maple, pixel_promptsrc, pixel_promptkd, pixel_mmrl, hyperprompt_coop, hyperprompt_kgcoop, hyperprompt_maple, hyperprompt_promptsrc, hyperprompt_promptkd, hyperprompt_mmrl")


def evaluate_model(model, test_loader, device, model_name):
    """Evaluate model on test set."""
    model.eval()
    y_true, y_pred = [], []
    normalized_model_name = model_name.lower()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", ncols=100):
            imgs = batch["image"].to(device)
            labels = batch["label"]

            model_outputs = model(imgs)

            # Extract logits based on model type using unified utility functions
            try:
                # Try to use utility for dual models
                if normalized_model_name.startswith('hyperprompt_'):
                    logits = extract_fused_logits_from_model_output(model_outputs, normalized_model_name)
                # Single-branch models - extract first element or handle specially
                elif normalized_model_name in ['patch_maple', 'progard', 'progradmodel', 
                                                 'patch_promptkd', 'patch_promptsrc', 'patch_kgcoop', 'pixel_coop', 'pixel_kgcoop', 'pixel_maple']:
                    logits = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs
                elif normalized_model_name in ['patch_coop']:
                    logits = model_outputs
                elif normalized_model_name in ['patch_mmrl', 'pixel_mmrl']:
                    logits = model_outputs[2]   # logits_fused    
                else:
                    # Fallback: assume first element is logits for unknown models
                    logits = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs
            except Exception as e:
                print(f"Error extracting logits for {normalized_model_name}: {e}")
                # Fallback for any errors
                logits = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs

            # Safety: if a tuple/list accidentally leaks through, use first element as logits
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            # Ensure logits is a tensor
            if not isinstance(logits, torch.Tensor):
                raise TypeError(f"logits should be torch.Tensor, got {type(logits)}")

            preds = logits.argmax(dim=1).cpu().numpy() + 1  # back to 1-indexed
            labels_np = labels.cpu().numpy()

            # Flatten arrays to ensure proper shape matching
            y_pred.extend(preds.flatten())
            y_true.extend(labels_np.flatten())

    results = compute_metrics(
        np.array(y_pred),
        np.array(y_true),
        ignored_labels=[0],
        n_classes=model.num_classes,
    )

    # Optional debug: show prediction / label distribution to detect
    # prediction collapse (e.g., always predicting one class).
    if os.environ.get("DPPGL_DEBUG_EVAL", "0") == "1":
        y_pred_arr = np.array(y_pred, dtype=np.int64)
        y_true_arr = np.array(y_true, dtype=np.int64)

        # Labels are 1-indexed in this codebase; ignore 0 (background).
        pred_counts = np.bincount(y_pred_arr, minlength=model.num_classes + 1)[1:]
        true_counts = np.bincount(y_true_arr, minlength=model.num_classes + 1)[1:]

        def _topk(counts, k=3):
            idx = np.argsort(-counts)[:k]
            return [(int(i + 1), int(counts[i])) for i in idx]

        print("\n[DEBUG EVAL] Top predicted classes (class_id, count):", _topk(pred_counts))
        print("[DEBUG EVAL] Top true classes      (class_id, count):", _topk(true_counts))

    model.train()

    return {
        "OA": results["OA"],
        "AA": results["AA"],
        "Kappa": results["Kappa"],
    }


def count_model_parameters(model):
    """Return total and trainable parameter counts."""
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params, trainable_params


def _cfg_to_printable_dict(cfg):
    """Convert config object to a printable dictionary."""
    if isinstance(cfg, dict):
        return dict(sorted(cfg.items(), key=lambda x: x[0]))

    # Try common config APIs first
    for method_name in ["to_dict", "as_dict", "dump"]:
        if hasattr(cfg, method_name):
            try:
                value = getattr(cfg, method_name)()
                if isinstance(value, dict):
                    return dict(sorted(value.items(), key=lambda x: x[0]))
            except Exception:
                pass

    # Fallback: collect public attributes
    out = {}
    for key in dir(cfg):
        if key.startswith("_"):
            continue
        try:
            value = getattr(cfg, key)
        except Exception:
            continue
        if callable(value):
            continue
        out[key] = value

    return dict(sorted(out.items(), key=lambda x: x[0]))


def _print_cfg_keys(cfg, keys, title):
    """Print selected config keys if present."""
    present_keys = [k for k in keys if hasattr(cfg, k)]
    if not present_keys:
        return
    print(f"\n{'-'*80}")
    print(title)
    print(f"{'-'*80}")
    for key in present_keys:
        print(f"{key}: {getattr(cfg, key)}")


def print_run_settings_for_model(cfg, model, optimizer=None):
    """Print model-specific parameters/hyperparameters used in this run."""
    model_name = cfg.MODEL_NAME.lower()

    # Shared keys used across all models
    common_keys = [
        "MODEL_NAME", "DATASET_NAME", "RUN_ID", "SEED", "DEVICE",
        "EPOCHS", "LR", "WEIGHT_DECAY", "BATCH_SIZE_TRAIN", "NUM_WORKERS",
        "PATCH_SIZE", "STRIDE", "TRAIN_SPLIT_RATIO", "RE_RATIO",
        "CLASS_TOKEN_POSITION", "CTX_INIT", "CSC",
    ]

    # Prompt/context keys based on model family
    if model_name.startswith("hyperprompt_"):
        prompt_keys = ["CTX_LEN_PATCH", "CTX_LEN_PIXEL"]
    else:
        prompt_keys = ["CTX_LEN"]

    # SAM backbone settings (for dual models)
    sam_keys = []
    if model_name.startswith("hyperprompt_"):
        sam_keys.append("TCDM_LAST_K_LAYERS")
        sam_keys.append("TCDM_USE_CLS_ONLY")

    # Model-specific hyperparameters
    specific_keys = []
    if "patch_maple" in model_name:
        specific_keys.append("PROMPT_DEPTH")
    if "patch_promptkd" in model_name:
        specific_keys.extend(["PROMPTKD_TEMPERATURE", "TEMPERATURE", "KD_WEIGHT", "LAMBDA_KL"])
    if "patch_kgcoop" in model_name:
        specific_keys.append("LAMBDA_KNOWLEDGE")

    # Loss keys by model family
    loss_keys = ["LAMBDA_CLS", "LAMBDA_MSE"]

    # Standard LoRA keys for '*_lora' models.
    lora_keys = []
    if "lora" in model_name and not model_name.startswith("hyperprompt_"):
        lora_keys.extend([
            "LORA_ENABLED", "LORA_RANK", "LORA_ALPHA", "LORA_LAST_N_LAYERS", "LORA_DROPOUT", "LORA_TARGET_KEYS",
        ])

    # PCLRA keys only for HyperPrompt variants.
    pclra_keys = []
    pclra_condition = model_name.startswith("hyperprompt_") or getattr(cfg, "PCLRA_RANK", 0) > 0
    if pclra_condition:
        pclra_keys.extend([
            "PCLRA_RANK", "PCLRA_ALPHA", "PCLRA_PROMPT_DIM", "PCLRA_LAST_N",
            "PCLRA_DROPOUT", "PCLRA_TAU", "PCLRA_TARGET_KEY", "LAMBDA_HOR",
        ])

    print(f"\n{'='*80}")
    print("RUN CONFIGURATION (MODEL-SPECIFIC)")
    print(f"{'='*80}")

    _print_cfg_keys(cfg, common_keys, "COMMON SETTINGS")
    _print_cfg_keys(cfg, prompt_keys, "PROMPT SETTINGS")
    _print_cfg_keys(cfg, sam_keys, "SAM BACKBONE SETTINGS")
    _print_cfg_keys(cfg, specific_keys, "MODEL-SPECIFIC SETTINGS")
    _print_cfg_keys(cfg, loss_keys, "LOSS WEIGHTS")
    _print_cfg_keys(cfg, lora_keys, "LORA SETTINGS")
    _print_cfg_keys(cfg, pclra_keys, "PCLRA SETTINGS")

    print(f"\n{'-'*80}")
    print("MODEL PARAMETER DETAILS")
    print(f"{'-'*80}")
    for name, param in model.named_parameters():
        status = "trainable" if param.requires_grad else "frozen"
        print(f"{name}: shape={tuple(param.shape)}, numel={param.numel()}, {status}")

    if optimizer is not None:
        print(f"\n{'-'*80}")
        print("OPTIMIZER PARAMETER GROUPS")
        print(f"{'-'*80}")
        for idx, group in enumerate(optimizer.param_groups):
            group_info = {k: v for k, v in group.items() if k != 'params'}
            print(f"Group {idx}: {group_info}")

    print(f"{'='*80}\n")


def train():
    # Load config
    cfg = get_config_from_args()
    
    # Set global random seeds for reproducibility
    import random
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)
    
    # Helper function for PCLRA status
    def is_pclra_enabled(cfg):
        """Check if PCLRA should be enabled based on CLI override or config"""
        if hasattr(cfg, '_pclra_override'):
            return cfg._pclra_override
        return cfg.PCLRA_RANK > 0
    
    pclra_enabled = is_pclra_enabled(cfg)
    
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Model: {cfg.MODEL_NAME.upper()}")
    print(f"Dataset: {cfg.DATASET_NAME}")
    print(f"Seed: {cfg.SEED}")
    print(f"Device: {cfg.DEVICE}")
    print(f"PCLRA Enabled: {'TRUE' if pclra_enabled else 'FALSE'}")
    print(f"{'='*60}\n")

    # -------------------------------------------------------
    # Build Dataloader
    # -------------------------------------------------------
    train_loader = build_dataloader(
        hsi_path          = cfg.SOURCE_HSI_PATH,
        gt_path           = cfg.SOURCE_GT_PATH,
        batch_size        = cfg.BATCH_SIZE_TRAIN,
        shuffle           = True,
        num_workers       = cfg.NUM_WORKERS,
        patch_size        = cfg.PATCH_SIZE,
        stride            = cfg.STRIDE,
        flip_aug          = cfg.FLIP_AUG,
        radiation_aug     = cfg.RADIATION_AUG,
        mixture_aug       = cfg.MIXTURE_AUG,
        ignored_labels    = [0],
        re_ratio          = cfg.RE_RATIO,
        split_ratio       = cfg.TRAIN_SPLIT_RATIO,
        split             = 'train',
        seed              = cfg.SEED,
    )

    test_loader = build_dataloader(
        hsi_path          = cfg.SOURCE_HSI_PATH,
        gt_path           = cfg.SOURCE_GT_PATH,
        batch_size        = 32,
        shuffle           = False,
        num_workers       = cfg.NUM_WORKERS,
        patch_size        = cfg.PATCH_SIZE,
        stride            = cfg.STRIDE,
        flip_aug          = False,
        radiation_aug     = False,
        mixture_aug       = False,
        ignored_labels    = [0],
        re_ratio          = 1,
        split_ratio       = cfg.TRAIN_SPLIT_RATIO,
        split             = 'test',
        seed              = cfg.SEED
    )

    sample_batch = next(iter(train_loader))
    in_channels = sample_batch["image"].shape[1]
    print(f"Input channels: {in_channels}")
    
    # -------------------------------------------------------
    # Build Model
    # -------------------------------------------------------
    model = get_model(cfg.MODEL_NAME, cfg, in_channels).to(cfg.DEVICE)
    total_params, trainable_params = count_model_parameters(model)
    trainable_ratio = (100.0 * trainable_params / total_params) if total_params > 0 else 0.0
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_ratio:.2f}%)")

    # -------------------------------------------------------
    # Setup Optimizer
    # -------------------------------------------------------
    WARMUP_CONS_LR = 1e-5
    WARMUP_EPOCH = 1
    MAX_EPOCH = cfg.EPOCHS

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = cfg.LR,
        momentum     = 0.9,
        weight_decay = cfg.WEIGHT_DECAY,
        nesterov     = True,
    )

    for pg in optimizer.param_groups:
        pg["initial_lr"] = cfg.LR

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max      = float(MAX_EPOCH),
        eta_min    = 0.0,
        last_epoch = WARMUP_EPOCH,
    )

    class ConstantWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
        def __init__(self, optimizer, successor, warmup_epoch, cons_lr, last_epoch=-1):
            self.successor    = successor
            self.warmup_epoch = warmup_epoch
            self.cons_lr      = cons_lr
            super().__init__(optimizer, last_epoch)

        def get_lr(self):
            if self.last_epoch >= self.warmup_epoch:
                return self.successor.get_last_lr()
            return [self.cons_lr for _ in self.base_lrs]

        def step(self, epoch=None):
            if self.last_epoch >= self.warmup_epoch:
                self.successor.step(epoch)
                self._last_lr = self.successor.get_last_lr()
            else:
                super().step(epoch)

    scheduler = ConstantWarmupScheduler(
        optimizer,
        successor    = cosine_scheduler,
        warmup_epoch = WARMUP_EPOCH,
        cons_lr      = WARMUP_CONS_LR,
    )

    # Print model-specific run settings for this specific model instance
    print_run_settings_for_model(cfg, model, optimizer)

    # -------------------------------------------------------
    # Setup Logging
    # -------------------------------------------------------
    loss_fn = get_loss_function(cfg.MODEL_NAME, cfg)
    log_file = cfg.LOG_PATH
    log_columns = None  # Will be initialized after first batch

    # Write run meta (includes LoRA config) to companion .meta file for traceability
    try:
        meta_path = cfg.LOG_PATH + ".meta"
        meta = {
            "MODEL_NAME": cfg.MODEL_NAME,
            "DATASET_NAME": cfg.DATASET_NAME,
            "RUN_ID": cfg.RUN_ID,
            "SEED": cfg.SEED,
            "LORA_ENABLED": getattr(cfg, 'LORA_ENABLED', getattr(cfg, 'PCLRA_ENABLED', False)),
            "LORA_RANK": getattr(cfg, 'LORA_RANK', getattr(cfg, 'PCLRA_RANK', 0)),
            "LORA_ALPHA": getattr(cfg, 'LORA_ALPHA', getattr(cfg, 'PCLRA_ALPHA', None)),
            "LORA_LAST_N_LAYERS": getattr(cfg, 'LORA_LAST_N_LAYERS', getattr(cfg, 'PCLRA_LAST_N', -1)),
            "LORA_DROPOUT": getattr(cfg, 'LORA_DROPOUT', getattr(cfg, 'PCLRA_DROPOUT', 0.0)),
            "LORA_TARGET_KEYS": getattr(cfg, 'LORA_TARGET_KEYS', getattr(cfg, 'PCLRA_TARGET_KEY', None)),
        }
        with open(meta_path, 'w') as mf:
            json.dump(meta, mf, indent=2)
    except Exception as e:
        print(f"Warning: failed to write run meta file: {e}")

    model.train()

    # -------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------
    for epoch in range(cfg.EPOCHS):
        loss_total_accum = 0.0
        loss_dict_accum = None  # Will be initialized from first loss_dict

        print(f"\nEpoch {epoch + 1}/{cfg.EPOCHS}")

        # GPU path
        if torch.cuda.is_available():
            prefetcher = CUDAPrefetcher(train_loader, cfg.DEVICE)
            batch = prefetcher.next()
            pbar = tqdm(total=len(train_loader))

            while batch is not None:
                images = batch["image"]
                labels = batch["label"] - 1  # 0-indexed

                # Pass labels to model for training (TAP and some other models need them)
                if cfg.MODEL_NAME.lower() in []:
                    model_outputs = model(images, labels=labels)
                else:
                    model_outputs = model(images)
                loss_total, loss_dict = loss_fn(model_outputs, labels, model, cfg)

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                loss_total_accum += loss_dict['loss_total']
                if loss_dict_accum is None:
                    loss_dict_accum = {k: 0.0 for k in loss_dict.keys()}
                    # Initialize log columns from first batch
                    if log_columns is None:
                        log_columns = ["epoch", "loss_total"]
                        # Add all loss components except 'loss_total'
                        for k in sorted(loss_dict.keys()):
                            if k != 'loss_total':
                                log_columns.append(f"{k}_loss")
                        log_columns.extend(["test_OA", "test_AA", "test_Kappa"])
                        if not os.path.exists(log_file):
                            with open(log_file, mode="w", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow(log_columns)
                for k, v in loss_dict.items():
                    loss_dict_accum[k] += v

                pbar.set_postfix({
                    k: (f"{v:.8f}" if k == "reg" else f"{v:.4f}")
                    for k, v in loss_dict.items()
                })
                pbar.update(1)
                batch = prefetcher.next()

            pbar.close()

        # CPU fallback
        else:
            pbar = tqdm(train_loader)

            for batch in pbar:
                images = batch["image"].to(cfg.DEVICE)
                labels = (batch["label"] - 1).to(cfg.DEVICE)

                model_outputs = model(images)
                loss_total, loss_dict = loss_fn(model_outputs, labels, model, cfg)

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                loss_total_accum += loss_dict['loss_total']
                if loss_dict_accum is None:
                    loss_dict_accum = {k: 0.0 for k in loss_dict.keys()}
                    # Initialize log columns from first batch
                    if log_columns is None:
                        log_columns = ["epoch", "loss_total"]
                        # Add all loss components except 'loss_total'
                        for k in sorted(loss_dict.keys()):
                            if k != 'loss_total':
                                log_columns.append(f"{k}_loss")
                        log_columns.extend(["test_OA", "test_AA", "test_Kappa"])
                        if not os.path.exists(log_file):
                            with open(log_file, mode="w", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow(log_columns)
                for k, v in loss_dict.items():
                    loss_dict_accum[k] += v

                pbar.set_postfix({
                    k: (f"{v:.8f}" if k == "reg" else f"{v:.4f}")
                    for k, v in loss_dict.items()
                })

            pbar.close()

        # Epoch summary
        n = len(train_loader)
        avg_loss = loss_total_accum / n
        for k in loss_dict_accum:
            loss_dict_accum[k] /= n

        print(f"Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f} | " +
              " | ".join([
                  f"{k}: {v:.8f}" if k == "reg" else f"{k}: {v:.4f}"
                  for k, v in loss_dict_accum.items()
              ]))

        # Evaluate
        eval_metrics = evaluate_model(model, test_loader, cfg.DEVICE, cfg.MODEL_NAME)
        test_oa = eval_metrics["OA"]
        test_aa = eval_metrics["AA"]
        test_kappa = eval_metrics["Kappa"]

        print(f"  Test Metrics | OA: {test_oa * 100:.2f}% | AA: {test_aa * 100:.2f}% | Kappa: {test_kappa:.4f}")

        # Log results - build row dynamically based on available losses
        log_row = [epoch + 1, avg_loss]

        # Add all loss components except 'loss_total' in sorted order
        for k in sorted(loss_dict_accum.keys()):
            if k != 'loss_total':
                log_row.append(loss_dict_accum[k])
        
        log_row.extend([test_oa, test_aa, test_kappa])

        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(log_row)

        # LR schedule
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"  LR -> {current_lr:.2e}")

        # Save checkpoint
        torch.save(model.state_dict(), cfg.CKPT_PATH)

    print(f"\nTraining complete. Model saved to: {cfg.CKPT_PATH}")
    print(f"Logs saved to: {cfg.LOG_PATH}")


if __name__ == "__main__":
    train()
