# test.py
"""
Universal Testing Script for HSI Classification
Evaluates trained models on target scene (domain adaptation)
Evaluates the paper's CoOp, KgCoOp, MaPLe, PromptSRC, PromptKD, and MMRL variants.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from config_loader import get_config_from_args
from dataset import build_dataloader
from utils import compute_metrics, visualize_dataset_and_results, extract_fused_logits_from_model_output
from hsi_preprocessing import load_hsi_data


def get_model(model_name, cfg, in_channels, device):
    """
    Load the appropriate model based on model_name.

    Args:
        model_name: One of the configured patch, pixel, or HyperPrompt variants.
        cfg: Config object with all required parameters
        in_channels: Input spectral channels
        device: torch device

    Returns:
        Instantiated model on device
    """
    # Helper function for PCLRA enablement
    def is_pclra_enabled(cfg):
        """Check if PCLRA should be enabled based on CLI override or config"""
        if hasattr(cfg, '_pclra_override'):
            return cfg._pclra_override
        # Fall back to checking PCLRA_RANK from config
        return cfg.PCLRA_RANK > 0

    # ─────────────────────────────────────────────────────────────
    # SINGLE-BRANCH MODELS
    # ─────────────────────────────────────────────────────────────
    
    if model_name.lower() == 'patch_coop':
        from trainers.models.patch_coop_model import HSIPatchCoOp
        model = HSIPatchCoOp(
            in_channels=in_channels,
            num_classes=cfg.NUM_CLASSES,
            classnames=cfg.CLASS_NAMES,
            clip_name=cfg.CLIP_MODEL_NAME,
            ctx_len=cfg.CTX_LEN,
            class_token_position=cfg.CLASS_TOKEN_POSITION,
            csc=cfg.CSC,
            ctx_init=cfg.CTX_INIT,
        )

    elif model_name.lower() == 'pixel_coop':
        from trainers.models.pixel_coop_model import HSIPixelCoOp
        model = HSIPixelCoOp(
            in_channels=in_channels,
            num_classes=cfg.NUM_CLASSES,
            classnames=cfg.CLASS_NAMES,
            clip_name=cfg.CLIP_MODEL_NAME,
            sam_model=cfg.SAM_MODEL_NAME,
            ctx_len=cfg.CTX_LEN,
            class_token_position=cfg.CLASS_TOKEN_POSITION,
            csc=cfg.CSC,
            ctx_init=cfg.CTX_INIT,
            upsampler_type=cfg.UPSAMPLER_TYPE if hasattr(cfg, 'UPSAMPLER_TYPE') else 'bilinear',
            upsampler_weights=cfg.UPSAMPLER_WEIGHTS if hasattr(cfg, 'UPSAMPLER_WEIGHTS') else None,
        )    

    elif model_name.lower() == 'pixel_kgcoop':
        from trainers.models.pixel_kgcoop_model import HSIPixelKgCoOp
        model = HSIPixelKgCoOp(
            in_channels=in_channels,
            num_classes=cfg.NUM_CLASSES,
            classnames=cfg.CLASS_NAMES,
            clip_name=cfg.CLIP_MODEL_NAME,
            sam_model=cfg.SAM_MODEL_NAME,
            ctx_len=cfg.CTX_LEN,
            class_token_position=cfg.CLASS_TOKEN_POSITION,
            csc=cfg.CSC,
            ctx_init=cfg.CTX_INIT,
            dataset_name=cfg.DATASET_NAME,
            knowledge_weight=cfg.LAMBDA_KNOWLEDGE if hasattr(cfg, 'LAMBDA_KNOWLEDGE') else 0.1,
            upsampler_type=cfg.UPSAMPLER_TYPE if hasattr(cfg, 'UPSAMPLER_TYPE') else 'bilinear',
            upsampler_weights=cfg.UPSAMPLER_WEIGHTS if hasattr(cfg, 'UPSAMPLER_WEIGHTS') else None,
        )

    elif model_name.lower() == 'pixel_maple':
        from trainers.models.pixel_maple_model import HSIPixelMaPLe
        model = HSIPixelMaPLe(
            in_channels=in_channels,
            num_classes=cfg.NUM_CLASSES,
            classnames=cfg.CLASS_NAMES,
            clip_name=cfg.CLIP_MODEL_NAME,
            sam_model=cfg.SAM_MODEL_NAME,
            ctx_len=cfg.CTX_LEN,
            prompt_depth=cfg.PROMPT_DEPTH if hasattr(cfg, 'PROMPT_DEPTH') else 1,
            class_token_position=cfg.CLASS_TOKEN_POSITION,
            csc=cfg.CSC,
            ctx_init=cfg.CTX_INIT,
            upsampler_type=cfg.UPSAMPLER_TYPE if hasattr(cfg, 'UPSAMPLER_TYPE') else 'bilinear',
            upsampler_weights=cfg.UPSAMPLER_WEIGHTS if hasattr(cfg, 'UPSAMPLER_WEIGHTS') else None,
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

    elif model_name.lower() == 'patch_promptkd':
        from trainers.models.patch_promptkd_model import HSIPatchPromptKD
        model = HSIPatchPromptKD(
            in_channels=in_channels,
            num_classes=cfg.NUM_CLASSES,
            classnames=cfg.CLASS_NAMES,
            clip_name=cfg.CLIP_MODEL_NAME,
            ctx_len=cfg.CTX_LEN,
            class_token_position=cfg.CLASS_TOKEN_POSITION,
            csc=cfg.CSC,
            ctx_init=cfg.CTX_INIT,
            temperature=cfg.PROMPTKD_TEMPERATURE if hasattr(cfg, 'PROMPTKD_TEMPERATURE') else (cfg.TEMPERATURE if hasattr(cfg, 'TEMPERATURE') else 4.0),
            kd_weight=cfg.KD_WEIGHT if hasattr(cfg, 'KD_WEIGHT') else (cfg.LAMBDA_KL if hasattr(cfg, 'LAMBDA_KL') else 1.0),
        )

    elif model_name.lower() == 'patch_promptsrc':
        from trainers.models.patch_promptsrc_model import HSIPatchPromptSRC
        model = HSIPatchPromptSRC(
            in_channels=in_channels,
            num_classes=cfg.NUM_CLASSES,
            classnames=cfg.CLASS_NAMES,
            clip_name=cfg.CLIP_MODEL_NAME,
            ctx_len_text=cfg.CTX_LEN_TEXT if hasattr(cfg, 'CTX_LEN_TEXT') else 4,
            ctx_len_vision=cfg.CTX_LEN_VISION if hasattr(cfg, 'CTX_LEN_VISION') else 4,
            ctx_init=cfg.CTX_INIT,
            temperature=cfg.PROMPTSRC_TEMPERATURE if hasattr(cfg, 'PROMPTSRC_TEMPERATURE') else (cfg.TEMPERATURE if hasattr(cfg, 'TEMPERATURE') else 1.0),
        )

    elif model_name.lower() == 'patch_maple':
        from trainers.models.patch_maple_model import HSIPatchMaPLe
        model = HSIPatchMaPLe(
            in_channels=in_channels,
            num_classes=cfg.NUM_CLASSES,
            classnames=cfg.CLASS_NAMES,
            clip_name=cfg.CLIP_MODEL_NAME,
            ctx_len=cfg.CTX_LEN,
            prompt_depth=cfg.PROMPT_DEPTH if hasattr(cfg, 'PROMPT_DEPTH') else 1,
            class_token_position=cfg.CLASS_TOKEN_POSITION,
            csc=cfg.CSC,
            ctx_init=cfg.CTX_INIT,
        )

    elif model_name.lower() == 'patch_kgcoop':
        from trainers.models.patch_kgcoop_model import HSIPatchKgCoOp
        model = HSIPatchKgCoOp(
            in_channels=in_channels,
            num_classes=cfg.NUM_CLASSES,
            classnames=cfg.CLASS_NAMES,
            clip_name=cfg.CLIP_MODEL_NAME,
            ctx_len=cfg.CTX_LEN,
            class_token_position=cfg.CLASS_TOKEN_POSITION,
            csc=cfg.CSC,
            ctx_init=cfg.CTX_INIT,
            dataset_name=cfg.DATASET_NAME,
            knowledge_weight=cfg.LAMBDA_KNOWLEDGE if hasattr(cfg, 'LAMBDA_KNOWLEDGE') else 0.1,
        )

    elif model_name.lower() == 'hyperprompt_coop':
        from trainers.models.hyperprompt_coop_model import HSIHyperPromptCoOp
        model = HSIHyperPromptCoOp(
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
            upsampler_type=cfg.UPSAMPLER_TYPE if hasattr(cfg, 'UPSAMPLER_TYPE') else 'bilinear',
            upsampler_weights=cfg.UPSAMPLER_WEIGHTS if hasattr(cfg, 'UPSAMPLER_WEIGHTS') else None,
            tcdm_last_k_layers=cfg.TCDM_LAST_K_LAYERS if hasattr(cfg, 'TCDM_LAST_K_LAYERS') else 4,
            use_cls_only=cfg.TCDM_USE_CLS_ONLY if hasattr(cfg, 'TCDM_USE_CLS_ONLY') else False,
            pclra_enabled=is_pclra_enabled(cfg),
            pclra_rank=cfg.PCLRA_RANK if hasattr(cfg, 'PCLRA_RANK') else 0,
            pclra_alpha=cfg.PCLRA_ALPHA if hasattr(cfg, 'PCLRA_ALPHA') else None,
            pclra_prompt_dim=cfg.PCLRA_PROMPT_DIM if hasattr(cfg, 'PCLRA_PROMPT_DIM') else 256,
            pclra_last_n_layers=cfg.PCLRA_LAST_N if hasattr(cfg, 'PCLRA_LAST_N') else -1,
            pclra_dropout=cfg.PCLRA_DROPOUT if hasattr(cfg, 'PCLRA_DROPOUT') else 0.0,
            pclra_tau=cfg.PCLRA_TAU if hasattr(cfg, 'PCLRA_TAU') else 0.07,
        )

    elif model_name.lower() == 'hyperprompt_kgcoop':
        from trainers.models.hyperprompt_kgcoop_model import HSIHyperPromptKgCoOp
        model = HSIHyperPromptKgCoOp(
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
            dataset_name=cfg.DATASET_NAME,
            upsampler_type=cfg.UPSAMPLER_TYPE if hasattr(cfg, 'UPSAMPLER_TYPE') else 'bilinear',
            upsampler_weights=cfg.UPSAMPLER_WEIGHTS if hasattr(cfg, 'UPSAMPLER_WEIGHTS') else None,
            pclra_enabled=is_pclra_enabled(cfg),
            pclra_rank=cfg.PCLRA_RANK if hasattr(cfg, 'PCLRA_RANK') else 0,
            pclra_alpha=cfg.PCLRA_ALPHA if hasattr(cfg, 'PCLRA_ALPHA') else None,
            pclra_prompt_dim=cfg.PCLRA_PROMPT_DIM if hasattr(cfg, 'PCLRA_PROMPT_DIM') else 256,
            pclra_last_n_layers=cfg.PCLRA_LAST_N if hasattr(cfg, 'PCLRA_LAST_N') else -1,
            pclra_dropout=cfg.PCLRA_DROPOUT if hasattr(cfg, 'PCLRA_DROPOUT') else 0.0,
            pclra_tau=cfg.PCLRA_TAU if hasattr(cfg, 'PCLRA_TAU') else 0.07,
        )

    elif model_name.lower() == 'hyperprompt_maple':
        from trainers.models.hyperprompt_maple_model import HSIHyperPromptMaPLe
        model = HSIHyperPromptMaPLe(
            in_channels=in_channels,
            num_classes=cfg.NUM_CLASSES,
            classnames=cfg.CLASS_NAMES,
            clip_name=cfg.CLIP_MODEL_NAME,
            sam_model=cfg.SAM_MODEL_NAME,
            ctx_len_patch=cfg.CTX_LEN_PATCH,
            ctx_len_pixel=cfg.CTX_LEN_PIXEL,
            prompt_depth=cfg.PROMPT_DEPTH if hasattr(cfg, 'PROMPT_DEPTH') else 1,
            class_token_position=cfg.CLASS_TOKEN_POSITION,
            csc=cfg.CSC,
            ctx_init=cfg.CTX_INIT,
            upsampler_type=cfg.UPSAMPLER_TYPE if hasattr(cfg, 'UPSAMPLER_TYPE') else 'bilinear',
            upsampler_weights=cfg.UPSAMPLER_WEIGHTS if hasattr(cfg, 'UPSAMPLER_WEIGHTS') else None,
            pclra_enabled=is_pclra_enabled(cfg),
            pclra_rank=cfg.PCLRA_RANK if hasattr(cfg, 'PCLRA_RANK') else 0,
            pclra_alpha=cfg.PCLRA_ALPHA if hasattr(cfg, 'PCLRA_ALPHA') else None,
            pclra_prompt_dim=cfg.PCLRA_PROMPT_DIM if hasattr(cfg, 'PCLRA_PROMPT_DIM') else 256,
            pclra_last_n_layers=cfg.PCLRA_LAST_N if hasattr(cfg, 'PCLRA_LAST_N') else -1,
            pclra_dropout=cfg.PCLRA_DROPOUT if hasattr(cfg, 'PCLRA_DROPOUT') else 0.0,
            pclra_tau=cfg.PCLRA_TAU if hasattr(cfg, 'PCLRA_TAU') else 0.07,
        )

    elif model_name.lower() == 'hyperprompt_promptkd':
        from trainers.models.hyperprompt_promptkd_model import HSIHyperPromptPromptKD
        model = HSIHyperPromptPromptKD(
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
            temperature=cfg.PROMPTKD_TEMPERATURE if hasattr(cfg, 'PROMPTKD_TEMPERATURE') else (cfg.TEMPERATURE if hasattr(cfg, 'TEMPERATURE') else 4.0),
            upsampler_type=cfg.UPSAMPLER_TYPE if hasattr(cfg, 'UPSAMPLER_TYPE') else 'bilinear',
            upsampler_weights=cfg.UPSAMPLER_WEIGHTS if hasattr(cfg, 'UPSAMPLER_WEIGHTS') else None,
            pclra_enabled=is_pclra_enabled(cfg),
            pclra_rank=cfg.PCLRA_RANK if hasattr(cfg, 'PCLRA_RANK') else 0,
            pclra_alpha=cfg.PCLRA_ALPHA if hasattr(cfg, 'PCLRA_ALPHA') else None,
            pclra_prompt_dim=cfg.PCLRA_PROMPT_DIM if hasattr(cfg, 'PCLRA_PROMPT_DIM') else 256,
            pclra_last_n_layers=cfg.PCLRA_LAST_N if hasattr(cfg, 'PCLRA_LAST_N') else -1,
            pclra_dropout=cfg.PCLRA_DROPOUT if hasattr(cfg, 'PCLRA_DROPOUT') else 0.0,
            pclra_tau=cfg.PCLRA_TAU if hasattr(cfg, 'PCLRA_TAU') else 0.07,
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
            in_channels=in_channels,
            num_classes=cfg.NUM_CLASSES,
            classnames=cfg.CLASS_NAMES,
            clip_name=cfg.CLIP_MODEL_NAME,
            n_rep_tokens=cfg.N_REP_TOKENS if hasattr(cfg, 'N_REP_TOKENS') else 5,
            rep_dim=cfg.REP_DIM if hasattr(cfg, 'REP_DIM') else 512,
            n_layers=cfg.N_LAYERS if hasattr(cfg, 'N_LAYERS') else 12,
            alpha=cfg.MMRL_ALPHA if hasattr(cfg, 'MMRL_ALPHA') else 0.7,
            reg_weight=cfg.REG_WEIGHT if hasattr(cfg, 'REG_WEIGHT') else 0.2,
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

    else:
        raise ValueError(f"Unknown model: {model_name}. Available models include hyperprompt_promptsrc and the other configured patch, pixel, and HyperPrompt variants")

    return model.to(device)


def load_checkpoint(model, ckpt_path, device):
    """
    Load checkpoint with multiple format support.
    
    Args:
        model: Model to load checkpoint into
        ckpt_path: Path to checkpoint
        device: torch device
    """
    # print(f"Loading checkpoint: {ckpt_path}")
    
    # if not os.path.exists(ckpt_path):
    #     raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # checkpoint = torch.load(ckpt_path, map_location=device)
    
    # # Handle multiple checkpoint formats
    # if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    #     model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    # elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    #     model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    # else:
    #     # Assume it's the state dict directly
    #     model.load_state_dict(checkpoint, strict=False)
        
    
    # print("✓ Checkpoint loaded successfully")
    print(f"Loading checkpoint: {ckpt_path}")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Handle multiple checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # Assume it's the state dict directly
        state_dict = checkpoint
    
    # because they may have shape mismatches during initialization
    pca_related_keys = ['rgb_proj.pca_basis', 'rgb_proj.running_cov', 'rgb_proj.fit_batches_seen']
    pca_state = {}
    remaining_state = {}
    
    for key, value in state_dict.items():
        if key in pca_related_keys:
            pca_state[key] = value
        else:
            remaining_state[key] = value
    
    # Load non-PCA parts first
    try:
        model.load_state_dict(remaining_state, strict=False)
    except Exception as e:
        print(f"Warning loading non-PCA state: {e}")
    
    # Manually load PCA-related buffers to bypass shape mismatch issues
    if hasattr(model, 'rgb_proj'):
        if 'rgb_proj.pca_basis' in pca_state:
            print(f"Loading rgb_proj.pca_basis: {pca_state['rgb_proj.pca_basis'].shape}")
            model.rgb_proj.pca_basis = pca_state['rgb_proj.pca_basis'].to(device)
        
        if 'rgb_proj.running_cov' in pca_state:
            print(f"Loading rgb_proj.running_cov: {pca_state['rgb_proj.running_cov'].shape}")
            model.rgb_proj.running_cov = pca_state['rgb_proj.running_cov'].to(device)
        
        if 'rgb_proj.fit_batches_seen' in pca_state:
            fit_batches_val = pca_state['rgb_proj.fit_batches_seen'].item()
            print(f"Loading rgb_proj.fit_batches_seen: {fit_batches_val}")
            model.rgb_proj.fit_batches_seen = pca_state['rgb_proj.fit_batches_seen'].to(device)
    
    print("✓ Checkpoint loaded successfully")


def evaluate_on_target(model, test_loader, cfg, device, model_name):
    """
    Evaluate model on target domain data.

    Args:
        model: Trained model in eval mode
        test_loader: DataLoader for target scene
        cfg: Config object
        device: torch device
        model_name: One of: patch_coop, patch_kgcoop, patch_maple, patch_promptsrc, patch_promptkd, patch_mmrl, pixel_coop, pixel_kgcoop, pixel_maple, pixel_promptsrc, pixel_promptkd, pixel_mmrl, hyperprompt_coop, hyperprompt_kgcoop, hyperprompt_maple, hyperprompt_promptkd, hyperprompt_mmrl

    Returns:
        Dictionary with metrics and predictions
    """
    y_true, y_pred, locations = [], [], []

    # Canonicalize known aliases/typos so downstream extraction utilities match.
    normalized_name = model_name.lower()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing on Target", ncols=100):

            images = batch["image"].to(device)
            labels = batch["label"]
            locs = batch["location"]

            # Forward pass - get model outputs
            model_outputs = model(images)

            # Extract logits based on model type using unified utility functions
            try:
                # Try to use utility for dual models
                if normalized_name.startswith('hyperprompt_'):
                    logits = extract_fused_logits_from_model_output(model_outputs, normalized_name)
                # Single-branch models - extract first element or handle specially
                elif normalized_name in ['patch_maple', 'patch_promptkd', 'patch_promptsrc', 'patch_kgcoop', 'pixel_coop', 'pixel_kgcoop', 'pixel_maple', 'pixel_promptkd', 'pixel_promptsrc']:
                    logits = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs
                elif normalized_name in ['patch_coop']:
                    logits = model_outputs
                elif normalized_name in ['patch_mmrl', 'pixel_mmrl']:
                    logits = model_outputs[2]   # logits_fused    
                else:
                    # Fallback: assume first element is logits for unknown models
                    logits = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs
            except Exception as e:
                print(f"Error extracting logits for {model_name}: {e}")
                # Fallback for any errors
                logits = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs

            # Safety: if a tuple/list accidentally leaks through, use first element as logits
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            # Get predictions (1-indexed)
            preds = logits.argmax(dim=1).cpu().numpy() + 1
            labels_np = labels.cpu().numpy()

            # Flatten arrays to ensure proper shape matching
            y_pred.extend(preds.flatten())
            y_true.extend(labels_np.flatten())
            locations.extend(locs.numpy())

    # Compute metrics
    results = compute_metrics(
        np.array(y_pred),
        np.array(y_true),
        ignored_labels=[0],
        n_classes=cfg.NUM_CLASSES,
    )

    return {
        'y_true': np.array(y_true),
        'y_pred': np.array(y_pred),
        'locations': np.array(locations),
        'metrics': results,
    }


def print_results(results, cfg, dataset_name="Target"):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results: Dictionary from evaluate_on_target()
        cfg: Config object
        dataset_name: Name of dataset being evaluated
    """
    metrics = results['metrics']
    
    OA = metrics["OA"]
    AA = metrics["AA"]
    kappa = metrics["Kappa"]
    class_acc = metrics["class_acc"]
    F1_scores = metrics["F1_scores"]
    cm = metrics["Confusion_matrix"]
    
    print(f"\n{'='*60}")
    print(f"  {dataset_name} Domain Evaluation Results")
    print(f"{'='*60}")
    print(f"Overall Accuracy (OA)   : {OA * 100:.2f}%")
    print(f"Average Accuracy (AA)   : {AA * 100:.2f}%")
    print(f"Kappa Coefficient       : {kappa:.4f}")
    print(f"{'-'*60}")
    print("Per-Class Accuracy:")
    for i, acc in enumerate(class_acc):
        if i < len(cfg.CLASS_NAMES):
            name = cfg.CLASS_NAMES[i]
        else:
            name = f"Class {i+1}"
        print(f"  {i+1:2d}. {name:<40s}: {acc * 100:.2f}%")
    print(f"{'-'*60}")
    print("F1 Scores:")
    for i, f1 in enumerate(F1_scores):
        if i < len(cfg.CLASS_NAMES):
            name = cfg.CLASS_NAMES[i]
        else:
            name = f"Class {i+1}"
        print(f"  {i+1:2d}. {name:<40s}: {f1:.4f}")
    print(f"{'-'*60}")
    print("Confusion Matrix (Classes 1..C):")
    print(cm)
    print(f"{'='*60}\n")


def build_classification_map(locations, predictions, hsi_shape):
    """
    Build classification map from predictions and their locations.
    
    Args:
        locations: Array of [H, W] location coordinates
        predictions: Array of predicted class labels (1-indexed)
        hsi_shape: Shape of full HSI (H, W, C)
    
    Returns:
        2D classification map
    """
    H, W, _ = hsi_shape
    cls_map = np.zeros((H, W), dtype=np.int32)
    
    for (r, c), pred in zip(locations, predictions):
        r, c = int(r), int(c)
        if 0 <= r < H and 0 <= c < W:
            cls_map[r, c] = pred
    
    return cls_map


def test():
    """Main testing function."""
    
    # Load configuration
    cfg = get_config_from_args()
    device = cfg.DEVICE
    
    print(f"\n{'='*60}")
    print(f"Testing Configuration")
    print(f"{'='*60}")
    print(f"Model: {cfg.MODEL_NAME}")
    print(f"Dataset: {cfg.DATASET_NAME}")
    print(f"Checkpoint: {cfg.CKPT_PATH}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # -------------------------------------------------------
    # Build Target Domain Dataloader (full dataset, no split)
    # -------------------------------------------------------
    print("Building target domain dataloader...")
    test_loader = build_dataloader(
        hsi_path=cfg.TARGET_HSI_PATH,
        gt_path=cfg.TARGET_GT_PATH,
        batch_size=32,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        patch_size=cfg.PATCH_SIZE,
        stride=cfg.STRIDE,
        flip_aug=False,
        radiation_aug=False,
        mixture_aug=False,
        ignored_labels=[0],
        re_ratio=1,
        split_ratio=1.0,  # Use entire target dataset
        seed=cfg.SEED,
    )
    
    # Get input channels
    sample_batch = next(iter(test_loader))
    in_channels = sample_batch["image"].shape[1]
    print(f"Input channels: {in_channels}\n")
    
    # -------------------------------------------------------
    # Build Model
    # -------------------------------------------------------
    print("Building model...")
    model = get_model(cfg.MODEL_NAME, cfg, in_channels, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # -------------------------------------------------------
    # Load Checkpoint
    # -------------------------------------------------------
    
    load_checkpoint(model, cfg.CKPT_PATH, device)
    model.eval()
    
    # -------------------------------------------------------
    # Evaluate on Target Domain
    # -------------------------------------------------------
    print("Starting evaluation on target domain...\n")
    results = evaluate_on_target(model, test_loader, cfg, device, cfg.MODEL_NAME)
    
    # Print results
    print_results(results, cfg, dataset_name="Target")
    
    # -------------------------------------------------------
    # Build and Save Classification Map
    # -------------------------------------------------------
    print("Building classification map...")
    
    # Load full HSI for shape
    hsi_full, gt_full = load_hsi_data(
        cfg.TARGET_HSI_PATH,
        cfg.TARGET_GT_PATH,
    )
    
    # Build classification map
    cls_map = build_classification_map(
        results['locations'],
        results['y_pred'],
        hsi_full.shape,
    )
    
    # -------------------------------------------------------
    # Visualization (if available)
    # -------------------------------------------------------
    try:
        print("Generating visualization...")
        visualize_dataset_and_results(
            img=hsi_full,
            gt=gt_full,
            pred=cls_map,
            bands=cfg.RGB_BANDS if hasattr(cfg, 'RGB_BANDS') else None,
            palette=cfg.PALETTE if hasattr(cfg, 'PALETTE') else None,
            class_names=cfg.CLASS_NAMES,
            save_path=cfg.FIG_PATH,
        )
        print(f"✓ Classification map saved to: {cfg.FIG_PATH}\n")
    except Exception as e:
        print(f"⚠ Visualization failed: {e}\n")
    
    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    metrics = results['metrics']
    print(f"{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Model: {cfg.MODEL_NAME.upper()}")
    print(f"Dataset: {cfg.DATASET_NAME}")
    print(f"Overall Accuracy: {metrics['OA']*100:.2f}%")
    print(f"Average Accuracy: {metrics['AA']*100:.2f}%")
    print(f"Kappa: {metrics['Kappa']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test()
