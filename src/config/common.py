"""공통 configuration dataclasses."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetConfig:
    """NPY dataset 공통 기본값 설정 (30-label 기준)."""
    horizon_indices: list = field(default_factory=lambda: [17, 21, 25])
    primary_horizon: int = 1
    vol_idx: int = -1
    class_bins: list = field(default_factory=lambda: [-0.01, -0.003, 0.003, 0.01])


@dataclass
class DataConfig:
    """어떤 데이터셋이든 공통으로 필요한 설정."""
    data_path: str = ""
    symbols: list[str] = field(default_factory=list)
    exclude_symbols: list[str] = field(default_factory=list)
    target_labels: list[str] = field(default_factory=list)
    feature_selection: str = "all"
    mode: str = "tabular"  # "tabular" | "sequence"
    seq_len: int = 1
    train_start: str = ""
    train_end: str = ""
    val_start: str = ""
    val_end: str = ""
    test_start: str = ""
    test_end: str = ""
    sample_ratio: float = 1.0
    max_symbols: int = 0  # 0 = all symbols


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = ""


@dataclass
class PathsConfig:
    """Output paths configuration."""
    saved_model_path: str = "./outputs/models"
    inference_path: str = "./outputs/inference"
    log_path: str = "./outputs/logs"


@dataclass
class Config:
    """Root configuration."""
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
