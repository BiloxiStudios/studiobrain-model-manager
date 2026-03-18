"""
YAML-based configuration loader for StudioBrain Model Manager.

Reads models.yaml to configure available models, tiers, VRAM budgets,
and server settings. Falls back to sensible defaults when no config is provided.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    """Runtime settings populated from models.yaml + environment."""

    # Server
    host: str = "0.0.0.0"
    port: int = 7070

    # GPU
    gpu_available: bool = field(default_factory=lambda: torch.cuda.is_available())
    gpu_device: str = "cuda:0"

    # VRAM management
    max_vram_usage: float = 24.0  # GB — total VRAM budget
    vram_budget_gb: float = 0.0   # 0 = auto-detect from GPU
    tier_1_vram_limit: float = 4.0
    tier_2_vram_limit: float = 8.0
    tier_3_vram_limit: float = 16.0

    # Model loading behaviour
    preload_models_at_startup: bool = True
    always_loaded_models: List[str] = field(default_factory=lambda: ["ram", "blip2"])
    tier1_models_always_loaded: bool = True
    model_cache_timeout: int = 300
    model_auto_unload_timeout: int = 300
    auto_load_models: bool = False  # Let ModelManager handle loading

    # Image models
    enable_image_models: bool = True
    local_image_models: List[str] = field(default_factory=lambda: ["florence2", "ram", "blip2"])

    # Audio models
    enable_audio_models: bool = False
    audio_models: List[str] = field(default_factory=list)

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    max_batch_size: int = 64

    # Paths
    model_cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "studiobrain-models")

    # Extra model configs from YAML
    model_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def _auto_detect_vram() -> float:
    """Return total GPU VRAM in GB, or 24.0 as fallback."""
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            pass
    return 24.0


def load_config(config_path: Optional[str] = None) -> Settings:
    """
    Load configuration from a YAML file + environment overrides.

    Priority: env vars > YAML values > defaults.

    Args:
        config_path: Path to models.yaml. If None, looks at MODEL_MANAGER_CONFIG env var,
                     then ./models.yaml, then uses pure defaults.
    """
    settings = Settings()

    # Resolve config file
    if config_path is None:
        config_path = os.environ.get("MODEL_MANAGER_CONFIG")
    if config_path is None:
        candidate = Path("models.yaml")
        if candidate.exists():
            config_path = str(candidate)

    # Load YAML
    cfg: Dict[str, Any] = {}
    if config_path and Path(config_path).exists():
        logger.info(f"Loading config from {config_path}")
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        logger.info("No config file found; using defaults + env overrides")

    # --- Server ---
    server = cfg.get("server", {})
    settings.host = server.get("host", settings.host)
    settings.port = int(server.get("port", settings.port))

    # --- VRAM ---
    vram = cfg.get("vram", {})
    settings.vram_budget_gb = float(os.environ.get(
        "VRAM_BUDGET_GB",
        vram.get("budget_gb", settings.vram_budget_gb),
    ))
    settings.max_vram_usage = float(vram.get("max_usage_gb", settings.max_vram_usage))
    settings.tier_1_vram_limit = float(vram.get("tier_1_limit_gb", settings.tier_1_vram_limit))
    settings.tier_2_vram_limit = float(vram.get("tier_2_limit_gb", settings.tier_2_vram_limit))
    settings.tier_3_vram_limit = float(vram.get("tier_3_limit_gb", settings.tier_3_vram_limit))

    # Auto-detect if no budget set
    if settings.vram_budget_gb <= 0 and settings.max_vram_usage <= 0:
        settings.max_vram_usage = _auto_detect_vram()

    # --- Loading ---
    loading = cfg.get("loading", {})
    settings.preload_models_at_startup = loading.get("preload_at_startup", settings.preload_models_at_startup)
    settings.always_loaded_models = loading.get("always_loaded", settings.always_loaded_models)
    settings.tier1_models_always_loaded = loading.get("tier1_always_loaded", settings.tier1_models_always_loaded)
    settings.model_cache_timeout = int(loading.get("cache_timeout_seconds", settings.model_cache_timeout))
    settings.model_auto_unload_timeout = int(loading.get("auto_unload_timeout_seconds", settings.model_auto_unload_timeout))

    # --- Embedding ---
    emb = cfg.get("embedding", {})
    settings.embedding_model = emb.get("model", settings.embedding_model)
    settings.embedding_dimension = int(emb.get("dimension", settings.embedding_dimension))
    settings.max_batch_size = int(emb.get("max_batch_size", settings.max_batch_size))

    # --- Paths ---
    cache_dir = os.environ.get("MODEL_CACHE_DIR", cfg.get("cache_dir"))
    if cache_dir:
        settings.model_cache_dir = Path(cache_dir)

    # --- Per-model configs ---
    settings.model_configs = cfg.get("models", {})

    # --- Env overrides ---
    if os.environ.get("MODEL_MANAGER_PORT"):
        settings.port = int(os.environ["MODEL_MANAGER_PORT"])
    if os.environ.get("MODEL_MANAGER_HOST"):
        settings.host = os.environ["MODEL_MANAGER_HOST"]

    return settings
