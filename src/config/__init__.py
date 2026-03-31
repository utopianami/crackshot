"""Configuration package — 기존 import 호환 유지."""

from src.config.common import (
    Config,
    DataConfig,
    DatasetConfig,
    ModelConfig,
    PathsConfig,
)
from src.config.data import Kospi260306Config
from src.config.models import (
    AEMLPConfig,
    TabMConfig,
)

from omegaconf import OmegaConf, DictConfig


def register_configs() -> None:
    """No-op. Kept for backward compatibility."""
    pass


def load_config(
    config_path: str | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Build config from dataclass defaults + YAML + CLI overrides.

    Usage:
        cfg = load_config("configs/tab_m.yaml", ["tab_m.d_block=256"])
        cfg = load_config(overrides=["model.name=tab_m", "tab_m.npy_dir=/path"])
    """
    from src.training.registry import TRAINER_META

    # 1. Defaults from Config dataclass
    cfg = OmegaConf.structured(Config())
    OmegaConf.set_struct(cfg, False)

    # 2. Merge YAML
    if config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_path))

    # 3. Merge CLI overrides
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    # 4. For models with registry defaults — inject default_config then merge overrides
    model_name = cfg.model.name
    meta = TRAINER_META.get(model_name, {})
    default_config = meta.get("default_config")
    if default_config:
        config_key = meta.get("config_key", model_name)
        existing_raw = cfg.get(config_key)
        if existing_raw is not None and OmegaConf.is_config(existing_raw):
            existing = OmegaConf.to_container(existing_raw, resolve=True)
        else:
            existing = existing_raw if isinstance(existing_raw, dict) else {}
        defaults = OmegaConf.create(default_config)
        merged = OmegaConf.merge(defaults, OmegaConf.create(existing or {}))
        cfg[config_key] = merged

    return cfg


__all__ = [
    "Config",
    "DataConfig",
    "DatasetConfig",
    "Kospi260306Config",
    "ModelConfig",
    "PathsConfig",
    "AEMLPConfig",
    "TabMConfig",
    "load_config",
    "register_configs",
]
