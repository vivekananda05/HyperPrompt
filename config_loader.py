# config_loader.py

import os
import yaml
import argparse
from types import SimpleNamespace

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ConfigManager:
    """Load and merge configurations from YAML files in configs folder."""

    def __init__(self, project_root=None):
        if project_root is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
        self.project_root = project_root
        self.configs_dir = os.path.join(project_root, "configs")
        self.data_root = os.path.join(project_root, "data")

    def load_yaml(self, filepath):
        """Load YAML config file."""
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)

    def load_dataset_config(self, dataset_name):
        """Load dataset config from configs/datasets/{dataset_name}.yaml"""
        config_path = os.path.join(self.configs_dir, "datasets", f"{dataset_name.lower()}.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Dataset config not found: {config_path}")
        return self.load_yaml(config_path)

    def load_model_config(self, model_name):
        """Load model config from configs/models/{model_name}.yaml"""
        config_path = os.path.join(self.configs_dir, "models", f"{model_name.lower()}.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config not found: {config_path}")
        return self.load_yaml(config_path)

    def load_trainer_config(self, trainer_name):
        """Load trainer config from configs/trainers/{trainer_name}.yaml"""
        config_path = os.path.join(self.configs_dir, "trainers", f"{trainer_name.lower()}.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Trainer config not found: {config_path}")
        return self.load_yaml(config_path)

    def merge_configs(self, *configs):
        """Recursively merge multiple config dicts."""
        merged = {}
        for config in configs:
            if config:
                self._merge_dict(merged, config)
        return merged

    def _merge_dict(self, target, source):
        """Recursively merge source dict into target dict."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_dict(target[key], value)
            else:
                target[key] = value

    def build_config(self, dataset_name, trainer_name, model_configs=None, pclra_config=None):
        """
        Build complete config from dataset, trainer, and optional model/pclra configs.

        Args:
            dataset_name: Name of dataset config (e.g., 'houston', 'pavia')
            trainer_name: Name of trainer config (e.g., 'patch_coop', 'hyperprompt_coop')
            model_configs: List of model config names (e.g., ['clip', 'sam'])
            pclra_config: Whether to include pclra config (True/False or 'pclra')

        Returns:
            SimpleNamespace with flattened config attributes
        """
        configs_to_merge = []

        # Load dataset config
        dataset_cfg = self.load_dataset_config(dataset_name)
        configs_to_merge.append(dataset_cfg)

        # Load trainer config
        trainer_cfg = self.load_trainer_config(trainer_name)
        configs_to_merge.append(trainer_cfg)

        # Load model configs
        if model_configs:
            if isinstance(model_configs, str):
                model_configs = [model_configs]
            for model_name in model_configs:
                model_cfg = self.load_model_config(model_name)
                configs_to_merge.append(model_cfg)

        # Load pclra config if requested
        if pclra_config:
            if pclra_config is True:
                pclra_config = 'pclra'
            pclra_cfg = self.load_trainer_config(pclra_config)
            configs_to_merge.append(pclra_cfg)

        # Merge all configs
        merged = self.merge_configs(*configs_to_merge)

        # Add runtime paths (only if not already defined in configs)
        if 'project_root' not in merged:
            merged['project_root'] = self.project_root
        if 'data_root' not in merged:
            merged['data_root'] = self.data_root
        if 'configs_dir' not in merged:
            merged['configs_dir'] = self.configs_dir

        # Flatten nested dicts to flat namespace (for backward compatibility)
        flat_config = self._flatten_dict(merged)

        # Ensure dataset name is available for programmatic callers
        if 'DATASET_NAME' not in flat_config:
            dataset_display_name = dataset_cfg.get('name', dataset_name.capitalize())
            flat_config['DATASET_NAME'] = dataset_display_name

        # Map config keys to training-friendly names
        flat_config = self._create_training_config(flat_config)

        # Make DATA_ROOT absolute if it's relative (from dataset config)
        if 'DATA_ROOT' in flat_config and not os.path.isabs(flat_config['DATA_ROOT']):
            flat_config['DATA_ROOT'] = os.path.join(self.project_root, flat_config['DATA_ROOT'])

        # Convert relative data paths to absolute paths (before creating SimpleNamespace)
        if 'DATA_ROOT' in flat_config and 'SOURCE_HSI_PATH' in flat_config:
            if not os.path.isabs(flat_config['SOURCE_HSI_PATH']):
                flat_config['SOURCE_HSI_PATH'] = os.path.join(flat_config['DATA_ROOT'], flat_config['SOURCE_HSI_PATH'])
            if not os.path.isabs(flat_config['SOURCE_GT_PATH']):
                flat_config['SOURCE_GT_PATH'] = os.path.join(flat_config['DATA_ROOT'], flat_config['SOURCE_GT_PATH'])
            if 'TARGET_HSI_PATH' in flat_config and not os.path.isabs(flat_config['TARGET_HSI_PATH']):
                flat_config['TARGET_HSI_PATH'] = os.path.join(flat_config['DATA_ROOT'], flat_config['TARGET_HSI_PATH'])
            if 'TARGET_GT_PATH' in flat_config and not os.path.isabs(flat_config['TARGET_GT_PATH']):
                flat_config['TARGET_GT_PATH'] = os.path.join(flat_config['DATA_ROOT'], flat_config['TARGET_GT_PATH'])

        # Map model paths (CLIP/SAM) from model configs if present
        if 'MODEL_PATH' in flat_config and not hasattr(flat_config, 'CLIP_MODEL_NAME'):
            # This is a model config - store it appropriately
            pass  # Will be handled in get_config_from_args

        # Ensure default model paths for programmatic callers (build_config users)
        if 'CLIP_MODEL_NAME' not in flat_config or not flat_config.get('CLIP_MODEL_NAME'):
            flat_config['CLIP_MODEL_NAME'] = os.path.join(self.project_root, "models/clip-vit-base-patch16")
        if 'SAM_MODEL_NAME' not in flat_config or not flat_config.get('SAM_MODEL_NAME'):
            flat_config['SAM_MODEL_NAME'] = os.path.join(self.project_root, "models/sam2.1-hiera-small")

        # Add device default (only if torch is available)
        if TORCH_AVAILABLE:
            flat_config['DEVICE'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            flat_config['PIN_MEMORY'] = torch.cuda.is_available()
        else:
            flat_config['DEVICE'] = "cuda"  # Will be properly set when torch is available
            flat_config['PIN_MEMORY'] = False

        return SimpleNamespace(**flat_config)

    def _flatten_dict(self, d, parent_key='', sep='_'):
        """Flatten nested dict with underscore-separated keys in UPPERCASE."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}".upper() if parent_key else k.upper()
            # Keep special dicts intact (palette, position_diversity_weights)
            if k in ['palette', 'position_diversity_weights'] and isinstance(v, dict):
                items.append((new_key, v))
            elif isinstance(v, dict):
                items.extend(self._flatten_dict(v, k, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, v))
            else:
                items.append((new_key, v))
        return dict(items)

    def _create_training_config(self, flat_config):
        """Map flattened config keys to training-friendly names."""
        # Map preprocessing keys
        if 'PREPROCESSING_PATCH_SIZE' in flat_config:
            flat_config['PATCH_SIZE'] = flat_config['PREPROCESSING_PATCH_SIZE']
        if 'PREPROCESSING_STRIDE' in flat_config:
            flat_config['STRIDE'] = flat_config['PREPROCESSING_STRIDE']
        if 'PREPROCESSING_RE_RATIO' in flat_config:
            flat_config['RE_RATIO'] = flat_config['PREPROCESSING_RE_RATIO']

        # Map augmentation keys
        if 'AUGMENTATION_FLIP' in flat_config:
            flat_config['FLIP_AUG'] = flat_config['AUGMENTATION_FLIP']
        if 'AUGMENTATION_RADIATION' in flat_config:
            flat_config['RADIATION_AUG'] = flat_config['AUGMENTATION_RADIATION']
        if 'AUGMENTATION_MIXTURE' in flat_config:
            flat_config['MIXTURE_AUG'] = flat_config['AUGMENTATION_MIXTURE']

        # Map prompt keys
        if 'PROMPT_CLASS_TOKEN_POSITION' in flat_config:
            flat_config['CLASS_TOKEN_POSITION'] = flat_config['PROMPT_CLASS_TOKEN_POSITION']
        if 'PROMPT_CTX_INIT' in flat_config:
            flat_config['CTX_INIT'] = flat_config['PROMPT_CTX_INIT']
        if 'PROMPT_CTX_LEN_PATCH' in flat_config:
            flat_config['CTX_LEN_PATCH'] = flat_config['PROMPT_CTX_LEN_PATCH']
        if 'PROMPT_CTX_LEN_TEXT_PATCH' in flat_config:
            flat_config['CTX_LEN_TEXT_PATCH'] = flat_config['PROMPT_CTX_LEN_TEXT_PATCH']
        if 'PROMPT_CTX_LEN_VISION_PATCH' in flat_config:
            flat_config['CTX_LEN_VISION_PATCH'] = flat_config['PROMPT_CTX_LEN_VISION_PATCH']
        if 'PROMPT_CTX_LEN_PIXEL' in flat_config:
            flat_config['CTX_LEN_PIXEL'] = flat_config['PROMPT_CTX_LEN_PIXEL']
        if 'PROMPT_CTX_LEN_TEXT_PIXEL' in flat_config:
            flat_config['CTX_LEN_TEXT_PIXEL'] = flat_config['PROMPT_CTX_LEN_TEXT_PIXEL']
        if 'PROMPT_CTX_LEN_VISION_PIXEL' in flat_config:
            flat_config['CTX_LEN_VISION_PIXEL'] = flat_config['PROMPT_CTX_LEN_VISION_PIXEL']
        # Backward-compatible aliases for summary/log code paths that still read
        # CTX_LEN_PATCH/CTX_LEN_PIXEL. Prefer text lengths, then vision lengths.
        if 'CTX_LEN_PATCH' not in flat_config:
            if 'CTX_LEN_TEXT_PATCH' in flat_config:
                flat_config['CTX_LEN_PATCH'] = flat_config['CTX_LEN_TEXT_PATCH']
            elif 'CTX_LEN_VISION_PATCH' in flat_config:
                flat_config['CTX_LEN_PATCH'] = flat_config['CTX_LEN_VISION_PATCH']
        if 'CTX_LEN_PIXEL' not in flat_config:
            if 'CTX_LEN_TEXT_PIXEL' in flat_config:
                flat_config['CTX_LEN_PIXEL'] = flat_config['CTX_LEN_TEXT_PIXEL']
            elif 'CTX_LEN_VISION_PIXEL' in flat_config:
                flat_config['CTX_LEN_PIXEL'] = flat_config['CTX_LEN_VISION_PIXEL']
        if 'PROMPT_CTX_LEN' in flat_config:
            flat_config['CTX_LEN'] = flat_config['PROMPT_CTX_LEN']
            if 'CTX_LEN_PATCH' not in flat_config:
                flat_config['CTX_LEN_PATCH'] = flat_config['PROMPT_CTX_LEN']
            if 'CTX_LEN_PIXEL' not in flat_config:
                flat_config['CTX_LEN_PIXEL'] = flat_config['PROMPT_CTX_LEN']
        if 'PROMPT_CSC' in flat_config:
            flat_config['CSC'] = flat_config['PROMPT_CSC']
        if 'PROMPT_NUM_PROMPTS' in flat_config:
            flat_config['PROMPT_K'] = flat_config['PROMPT_NUM_PROMPTS']
        if 'PROMPT_BATCH_SIZE' in flat_config:
            flat_config['PROMPT_BSZ'] = flat_config['PROMPT_BATCH_SIZE']
        if 'PROMPT_PROMPT_DEPTH' in flat_config:
            flat_config['PROMPT_DEPTH'] = flat_config['PROMPT_PROMPT_DEPTH']

        # Map projection keys for baseline RGB adapters
        if 'PROJECTION_BAND_INDICES' in flat_config:
            flat_config['BAND_INDICES'] = flat_config['PROJECTION_BAND_INDICES']
        if 'PROJECTION_NORMALIZE_RGB' in flat_config:
            flat_config['NORMALIZE_RGB'] = flat_config['PROJECTION_NORMALIZE_RGB']

        # Map TAP-specific prompt keys
        if 'PROMPT_TAP_DEEP' in flat_config:
            flat_config['TAP_DEEP'] = flat_config['PROMPT_TAP_DEEP']
        if 'PROMPT_TAP_DROPOUT' in flat_config:
            flat_config['TAP_DROPOUT'] = flat_config['PROMPT_TAP_DROPOUT']
        if 'PROMPT_TAP_U1' in flat_config:
            flat_config['TAP_U1'] = flat_config['PROMPT_TAP_U1']
        if 'PROMPT_TAP_U2' in flat_config:
            flat_config['TAP_U2'] = flat_config['PROMPT_TAP_U2']
        if 'PROMPT_TAP_U3' in flat_config:
            flat_config['TAP_U3'] = flat_config['PROMPT_TAP_U3']

        # Note: standard LORA configs are kept under LORA_* keys. Do not map
        # them to PCLRA_* aliases to avoid accidentally enabling PCLRA when
        # using the standard LoRA trainer config.

        # Map training keys
        if 'TRAINING_EPOCHS' in flat_config:
            flat_config['EPOCHS'] = flat_config['TRAINING_EPOCHS']
        if 'TRAINING_BATCH_SIZE_TRAIN' in flat_config:
            flat_config['BATCH_SIZE_TRAIN'] = flat_config['TRAINING_BATCH_SIZE_TRAIN']
        if 'TRAINING_BATCH_SIZE_TEST' in flat_config:
            flat_config['BATCH_SIZE_TEST'] = flat_config['TRAINING_BATCH_SIZE_TEST']
        if 'TRAINING_NUM_WORKERS' in flat_config:
            flat_config['NUM_WORKERS'] = flat_config['TRAINING_NUM_WORKERS']
        if 'TRAINING_LEARNING_RATE' in flat_config:
            flat_config['LR'] = flat_config['TRAINING_LEARNING_RATE']
        if 'TRAINING_WEIGHT_DECAY' in flat_config:
            flat_config['WEIGHT_DECAY'] = flat_config['TRAINING_WEIGHT_DECAY']
        if 'TRAINING_SEED' in flat_config:
            flat_config['SEED'] = flat_config['TRAINING_SEED']
        if 'TRAINING_TRAIN_SPLIT_RATIO' in flat_config:
            flat_config['TRAIN_SPLIT_RATIO'] = flat_config['TRAINING_TRAIN_SPLIT_RATIO']

        # Map loss weights
        if 'LOSS_WEIGHTS_MSE' in flat_config:
            flat_config['LAMBDA_MSE'] = flat_config['LOSS_WEIGHTS_MSE']
        if 'LOSS_WEIGHTS_CLS' in flat_config:
            flat_config['LAMBDA_CLS'] = flat_config['LOSS_WEIGHTS_CLS']
        if 'LOSS_WEIGHTS_PRODA' in flat_config:
            flat_config['LAMBDA_PRODA'] = flat_config['LOSS_WEIGHTS_PRODA']
        if 'LOSS_WEIGHTS_PROMPT_ORTHO' in flat_config:
            flat_config['LAMBDA_PROMPT_ORTHO'] = flat_config['LOSS_WEIGHTS_PROMPT_ORTHO']
        if 'LOSS_WEIGHTS_KL' in flat_config:
            flat_config['LAMBDA_KL'] = flat_config['LOSS_WEIGHTS_KL']
        if 'LOSS_WEIGHTS_CONSISTENCY' in flat_config:
            flat_config['LAMBDA_CONSISTENCY'] = flat_config['LOSS_WEIGHTS_CONSISTENCY']
        if 'LOSS_WEIGHTS_KNOWLEDGE' in flat_config:
            flat_config['LAMBDA_KNOWLEDGE'] = flat_config['LOSS_WEIGHTS_KNOWLEDGE']
        if 'LOSS_WEIGHTS_HOR' in flat_config:
            flat_config['LAMBDA_HOR'] = flat_config['LOSS_WEIGHTS_HOR']

        # Also map PCLRA_LAMBDA_HOR (from pclra.yaml) if not already set
        if 'LAMBDA_HOR' not in flat_config and 'PCLRA_LAMBDA_HOR' in flat_config:
            flat_config['LAMBDA_HOR'] = flat_config['PCLRA_LAMBDA_HOR']
        if 'LOSS_WEIGHTS_KL_DIVERGENCE' in flat_config:
            flat_config['LAMBDA_KL'] = flat_config['LOSS_WEIGHTS_KL_DIVERGENCE']
        if 'LOSS_WEIGHTS_SRC_TEXT' in flat_config:
            flat_config['LAMBDA_SRC_TEXT'] = flat_config['LOSS_WEIGHTS_SRC_TEXT']
        if 'LOSS_WEIGHTS_SRC_IMAGE' in flat_config:
            flat_config['LAMBDA_SRC_IMAGE'] = flat_config['LOSS_WEIGHTS_SRC_IMAGE']
        if 'LOSS_WEIGHTS_SRC_LOGIT' in flat_config:
            flat_config['LAMBDA_SRC_LOGIT'] = flat_config['LOSS_WEIGHTS_SRC_LOGIT']

        # Map ProGrad-specific loss weights
        if 'LOSS_CONFIG_WEIGHT_XE' in flat_config:
            flat_config['WEIGHT_XE'] = flat_config['LOSS_CONFIG_WEIGHT_XE']
        if 'LOSS_CONFIG_WEIGHT_KL' in flat_config:
            flat_config['WEIGHT_KL'] = flat_config['LOSS_CONFIG_WEIGHT_KL']

        # Map ProGrad teacher-student settings
        if 'TEACHER_STUDENT_TEMPERATURE' in flat_config:
            flat_config['PROGRAD_TEMPERATURE'] = flat_config['TEACHER_STUDENT_TEMPERATURE']
            flat_config['PROMPTKD_TEMPERATURE'] = flat_config['TEACHER_STUDENT_TEMPERATURE']
            flat_config['TEMPERATURE'] = flat_config['TEACHER_STUDENT_TEMPERATURE']
        if 'SELF_REINFORCING_TEMPERATURE' in flat_config:
            flat_config['PROMPTSRC_TEMPERATURE'] = flat_config['SELF_REINFORCING_TEMPERATURE']
            flat_config['TEMPERATURE'] = flat_config['SELF_REINFORCING_TEMPERATURE']

        # Map position diversity weights
        if 'POSITION_DIVERSITY_WEIGHTS' in flat_config:
            flat_config['POSITION_WEIGHTS'] = flat_config['POSITION_DIVERSITY_WEIGHTS']

        # Map model paths
        if 'MODEL_NAME' in flat_config:
            # This could be CLIP or SAM, store it appropriately
            pass  # Will be handled in get_config_from_args

        # Map upsampler keys (from configs/models/upsamplers.yaml)
        if 'UPSAMPLERS_TYPE' in flat_config:
            flat_config['UPSAMPLER_TYPE'] = flat_config['UPSAMPLERS_TYPE']
        if 'UPSAMPLERS_WEIGHTS' in flat_config:
            flat_config['UPSAMPLER_WEIGHTS'] = flat_config['UPSAMPLERS_WEIGHTS']

        # Map MMRL-specific keys
        if 'MMRL_N_REP_TOKENS' in flat_config:
            flat_config['N_REP_TOKENS'] = flat_config['MMRL_N_REP_TOKENS']
        if 'MMRL_REP_DIM' in flat_config:
            flat_config['REP_DIM'] = flat_config['MMRL_REP_DIM']
        if 'MMRL_N_LAYERS' in flat_config:
            flat_config['N_LAYERS'] = flat_config['MMRL_N_LAYERS']
        if 'MMRL_ALPHA' in flat_config:
            flat_config['MMRL_ALPHA'] = flat_config['MMRL_ALPHA']
        if 'MMRL_REG_WEIGHT' in flat_config:
            flat_config['REG_WEIGHT'] = flat_config['MMRL_REG_WEIGHT']
        if 'LOSS_WEIGHTS_REG' in flat_config:
            flat_config['REG_WEIGHT'] = flat_config['LOSS_WEIGHTS_REG']
            flat_config['LAMBDA_REG'] = flat_config['LOSS_WEIGHTS_REG']

        # TCDM settings are automatically flattened from nested tcdm.last_k_layers 
        # into TCDM_LAST_K_LAYERS, which is read by HyperPrompt training code
        # Also flatten use_cls_only from nested tcdm.use_cls_only into TCDM_USE_CLS_ONLY
        # (already flattened by _flatten_dict, but set default if not present)
        if 'TCDM_USE_CLS_ONLY' not in flat_config:
            flat_config['TCDM_USE_CLS_ONLY'] = False  # Default to False if not in config

        # Add position diversity option mapping
        if 'POSITION_DIVERSITY_WEIGHTS' in flat_config and isinstance(flat_config['POSITION_DIVERSITY_WEIGHTS'], dict):
            flat_config['POSITION_WEIGHTS'] = flat_config['POSITION_DIVERSITY_WEIGHTS']

        return flat_config


def get_config_from_args(args=None):
    """
    Parse command-line arguments and build config.
    
    Args:
        args: Optional pre-parsed arguments (for programmatic use)
    """
    parser = argparse.ArgumentParser(description='HSI Classification Training')
    parser.add_argument('--dataset', type=str, default='houston',
                        help='Dataset config name (houston, pavia, hyrank)')
    parser.add_argument('--model', type=str, default='hyperprompt_coop',
                        help='Model config name: patch_coop, patch_kgcoop, patch_maple, patch_promptsrc, patch_promptkd, patch_mmrl, pixel_coop, pixel_kgcoop, pixel_maple, pixel_promptsrc, pixel_promptkd, pixel_mmrl, hyperprompt_coop, hyperprompt_kgcoop, hyperprompt_maple, hyperprompt_promptsrc, hyperprompt_promptkd, hyperprompt_mmrl')
    parser.add_argument('--pclra', type=str, choices=['true', 'false'], default=None,
                        help='Override Prompt-Conditioned Low-Rank Adapter enabled (true/false); defaults to enabled for hyperprompt_* models')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--run-id', type=str, default='run_0',
                        help='Run identifier for checkpoint and logs')

    parsed_args = parser.parse_args(args)

    # Build config
    manager = ConfigManager()
    model_configs = ['clip', 'sam', 'upsamplers']


    # Map model names to trainer configs
    model_to_trainer_map = {
        # Single-branch models
        'patch_coop': 'patch_coop',
        'patch_promptkd': 'patch_promptkd',
        'patch_promptsrc': 'patch_promptsrc',
        'patch_maple': 'patch_maple',
        'patch_kgcoop': 'patch_kgcoop',
        'patch_mmrl': 'patch_mmrl',
        'pixel_coop': 'pixel_coop',
        'pixel_kgcoop': 'pixel_kgcoop',
        'pixel_maple': 'pixel_maple',
        'pixel_promptkd': 'pixel_promptkd',
        'pixel_promptsrc': 'pixel_promptsrc',
        'pixel_mmrl': 'pixel_mmrl',
        # HyperPrompt models
        'hyperprompt_coop': 'hyperprompt_coop',
        'hyperprompt_promptkd': 'hyperprompt_promptkd',
        'hyperprompt_maple': 'hyperprompt_maple',
        'hyperprompt_kgcoop': 'hyperprompt_kgcoop',
        'hyperprompt_promptsrc': 'hyperprompt_promptsrc',
        'hyperprompt_mmrl': 'hyperprompt_mmrl',
    }
    
    # Map model name to trainer config
    trainer_name = model_to_trainer_map.get(parsed_args.model, parsed_args.model)

    # Load dataset config first to get the proper dataset name before it gets overwritten by model configs
    dataset_cfg = manager.load_dataset_config(parsed_args.dataset)
    dataset_display_name = dataset_cfg.get('name', parsed_args.dataset.capitalize())

    def _parse_optional_bool(value):
        if isinstance(value, str):
            return value.lower() == 'true'
        return None

    model_name_lower = parsed_args.model.lower()
    pclra_override = _parse_optional_bool(parsed_args.pclra)
    if pclra_override is None:
        pclra_enabled = model_name_lower.startswith('hyperprompt_')
    else:
        pclra_enabled = pclra_override

    pclra_config = pclra_enabled and 'pclra'

    # Keep the PCLRA prompt aggregation vector aligned with the trainer
    # config used for these PCLRA models.
    if model_name_lower.startswith('hyperprompt_') and pclra_enabled:
        if pclra_config is None:
            pclra_config = 'pclra'

    cfg = manager.build_config(
        dataset_name=parsed_args.dataset,
        trainer_name=trainer_name,
        model_configs=model_configs,
        pclra_config=pclra_config
    )

    # Add run_id and model name
    cfg.RUN_ID = parsed_args.run_id
    cfg.MODEL_NAME = parsed_args.model
    cfg.DATASET_NAME = dataset_display_name  # Use the name from dataset config
    
    # Set seed (from CLI or default to 42)
    if parsed_args.seed is not None:
        cfg.SEED = parsed_args.seed
    else:
        cfg.SEED = 42  # Default seed if not provided via CLI

    if pclra_override is not None:
        cfg._pclra_override = pclra_override

    if model_name_lower.startswith('hyperprompt_') and pclra_enabled:
        cfg.PCLRA_PROMPT_DIM = 512

    # --- Ensure standard LoRA convenience attributes exist ---
    # Keep PCLRA settings untouched when using standard LoRA trainer configs.
    if not hasattr(cfg, 'LORA_ENABLED'):
        cfg.LORA_ENABLED = False
    if not hasattr(cfg, 'LORA_RANK'):
        cfg.LORA_RANK = 0
    if not hasattr(cfg, 'LORA_ALPHA'):
        cfg.LORA_ALPHA = None
    if not hasattr(cfg, 'LORA_LAST_N_LAYERS'):
        cfg.LORA_LAST_N_LAYERS = -1
    if not hasattr(cfg, 'LORA_DROPOUT'):
        cfg.LORA_DROPOUT = 0.0
    if not hasattr(cfg, 'LORA_TARGET_KEYS'):
        cfg.LORA_TARGET_KEYS = None

    # Setup output directories based on dataset
    dataset_name = cfg.DATASET_NAME  # Use the explicitly set dataset name
    output_root = os.path.join(cfg.PROJECT_ROOT, 'outputs')
    dataset_output_root = os.path.join(output_root, dataset_name)

    cfg.CKPT_DIR = os.path.join(dataset_output_root, 'checkpoints')
    cfg.FIG_DIR = os.path.join(dataset_output_root, 'figures')
    cfg.LOG_DIR = os.path.join(dataset_output_root, 'logs')
    cfg.PLOT_DIR = os.path.join(dataset_output_root, 'plots')

    # Create output directories
    for directory in [cfg.CKPT_DIR, cfg.FIG_DIR, cfg.LOG_DIR, cfg.PLOT_DIR]:
        os.makedirs(directory, exist_ok=True)

    # Setup checkpoint and log paths (with seed before run_id)
    checkpoint_prefix = "hsi_prompt"

    cfg.CKPT_PATH = os.path.join(
        cfg.CKPT_DIR,
        f"{checkpoint_prefix}_{parsed_args.model}_{dataset_name}_{cfg.SEED}_{cfg.RUN_ID}.pth"
    )
    cfg.FIG_PATH = os.path.join(
        cfg.FIG_DIR,
        f"classification_map_{parsed_args.model}_{dataset_name}_{cfg.SEED}_{cfg.RUN_ID}.png"
    )
    cfg.LOG_PATH = os.path.join(
        cfg.LOG_DIR,
        f"train_{parsed_args.model}_{dataset_name}_{cfg.SEED}_{cfg.RUN_ID}.csv"
    )

    # Convert relative data paths to absolute paths
    if hasattr(cfg, 'DATA_ROOT') and hasattr(cfg, 'SOURCE_HSI_PATH'):
        if not os.path.isabs(cfg.SOURCE_HSI_PATH):
            cfg.SOURCE_HSI_PATH = os.path.join(cfg.DATA_ROOT, cfg.SOURCE_HSI_PATH)
        if not os.path.isabs(cfg.SOURCE_GT_PATH):
            cfg.SOURCE_GT_PATH = os.path.join(cfg.DATA_ROOT, cfg.SOURCE_GT_PATH)
        if hasattr(cfg, 'TARGET_HSI_PATH') and not os.path.isabs(cfg.TARGET_HSI_PATH):
            cfg.TARGET_HSI_PATH = os.path.join(cfg.DATA_ROOT, cfg.TARGET_HSI_PATH)
        if hasattr(cfg, 'TARGET_GT_PATH') and not os.path.isabs(cfg.TARGET_GT_PATH):
            cfg.TARGET_GT_PATH = os.path.join(cfg.DATA_ROOT, cfg.TARGET_GT_PATH)

    # Add model paths from model configs
    if hasattr(cfg, 'MODEL_PATH'):
        # This is a CLIP model config
        if 'clip' in dataset_name.lower():
            cfg.CLIP_MODEL_NAME = cfg.MODEL_PATH

    # Ensure default model paths if not in config
    if not hasattr(cfg, 'CLIP_MODEL_NAME'):
        cfg.CLIP_MODEL_NAME = os.path.join(cfg.PROJECT_ROOT, "models/clip-vit-base-patch16")
    if not hasattr(cfg, 'SAM_MODEL_NAME'):
        cfg.SAM_MODEL_NAME = os.path.join(cfg.PROJECT_ROOT, "models/sam2.1-hiera-small")

    # Set default upsampler if not configured
    if not hasattr(cfg, 'UPSAMPLER_TYPE'):
        cfg.UPSAMPLER_TYPE = "resize_conv"
    if not hasattr(cfg, 'UPSAMPLER_WEIGHTS'):
        cfg.UPSAMPLER_WEIGHTS = os.path.join(cfg.PROJECT_ROOT, "models/weights/clip_jbu_stack_cocostuff.ckpt")
    elif cfg.UPSAMPLER_WEIGHTS and not os.path.isabs(cfg.UPSAMPLER_WEIGHTS):
        cfg.UPSAMPLER_WEIGHTS = os.path.join(cfg.PROJECT_ROOT, cfg.UPSAMPLER_WEIGHTS)

    # Set default PCLRA settings if not configured
    if not hasattr(cfg, 'PCLRA_RANK'):
        cfg.PCLRA_RANK = 0  # 0 = disabled
    if not hasattr(cfg, 'PCLRA_ALPHA'):
        cfg.PCLRA_ALPHA = 0
    if not hasattr(cfg, 'PCLRA_PROMPT_DIM'):
        cfg.PCLRA_PROMPT_DIM = 512
    if not hasattr(cfg, 'PCLRA_LAST_N'):
        cfg.PCLRA_LAST_N = -1
    if not hasattr(cfg, 'PCLRA_DROPOUT'):
        cfg.PCLRA_DROPOUT = 0.1
    if not hasattr(cfg, 'PCLRA_LAMBDA_COUPLE'):
        cfg.PCLRA_LAMBDA_COUPLE = 0.1
    if not hasattr(cfg, 'PCLRA_LAMBDA_RANK'):
        cfg.PCLRA_LAMBDA_RANK = 0.01
    if not hasattr(cfg, 'PCLRA_LAMBDA_SUBSPACE'):
        cfg.PCLRA_LAMBDA_SUBSPACE = 0.01
    if not hasattr(cfg, 'PCLRA_RANK_WARMUP'):
        cfg.PCLRA_RANK_WARMUP = 200

    # Set default loss weights if not configured
    if not hasattr(cfg, 'LAMBDA_CLS'):
        cfg.LAMBDA_CLS = 1.0
    if not hasattr(cfg, 'LAMBDA_MSE'):
        cfg.LAMBDA_MSE = 0.1
    if not hasattr(cfg, 'LAMBDA_PRODA'):
        cfg.LAMBDA_PRODA = 1.0
    if not hasattr(cfg, 'LAMBDA_PROMPT_ORTHO'):
        cfg.LAMBDA_PROMPT_ORTHO = 0.1
    if not hasattr(cfg, 'LAMBDA_KL'):
        cfg.LAMBDA_KL = 1.0
    if not hasattr(cfg, 'LAMBDA_CONSISTENCY'):
        cfg.LAMBDA_CONSISTENCY = 8.0
    if not hasattr(cfg, 'LAMBDA_KNOWLEDGE'):
        cfg.LAMBDA_KNOWLEDGE = 8.0
    if not hasattr(cfg, 'LAMBDA_HOR'):
        cfg.LAMBDA_HOR = 1.0


    # Set default prompt/context settings if not configured (for prompt learning)
    if not hasattr(cfg, 'CTX_LEN'):
        cfg.CTX_LEN = 4
    if not hasattr(cfg, 'CTX_LEN_PATCH'):
        cfg.CTX_LEN_PATCH = cfg.CTX_LEN
    if not hasattr(cfg, 'CTX_LEN_PIXEL'):
        cfg.CTX_LEN_PIXEL = cfg.CTX_LEN
    if not hasattr(cfg, 'CLASS_TOKEN_POSITION'):
        cfg.CLASS_TOKEN_POSITION = 'end'
    if not hasattr(cfg, 'CSC'):
        cfg.CSC = False
    if not hasattr(cfg, 'CTX_INIT'):
        cfg.CTX_INIT = None
    if not hasattr(cfg, 'PROMPT_K'):
        cfg.PROMPT_K = 32
    if not hasattr(cfg, 'PROMPT_BSZ'):
        cfg.PROMPT_BSZ = 4

    # Set default position diversity weights if not configured
    if not hasattr(cfg, 'POSITION_WEIGHTS'):
        cfg.POSITION_WEIGHTS = None

    # Set default ProGrad parameters if not configured
    if not hasattr(cfg, 'TEMPERATURE'):
        cfg.TEMPERATURE = 0.07
    if not hasattr(cfg, 'WEIGHT_XE'):
        cfg.WEIGHT_XE = 1.0
    if not hasattr(cfg, 'WEIGHT_KL'):
        cfg.WEIGHT_KL = 1.0

    # Map learning rate if available
    if not hasattr(cfg, 'LEARNING_RATE') and hasattr(cfg, 'LR'):
        cfg.LEARNING_RATE = cfg.LR
    elif not hasattr(cfg, 'LEARNING_RATE'):
        cfg.LEARNING_RATE = 2.0e-3

    return cfg
