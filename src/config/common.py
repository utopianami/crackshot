"""공통 configuration dataclasses."""

from dataclasses import dataclass, field

from src.config.data import Kospi260306Config  # noqa: F401


@dataclass
class DataConfig:
    """어떤 데이터셋이든 공통으로 필요한 설정."""
    data_path: str = ""
    symbols: list[str] = field(default_factory=list)
    exclude_symbols: list[str] = field(default_factory=list)
    target_labels: list[str] = field(default_factory=lambda: ["log_diff_m2m_30"])
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
    dataset: Kospi260306Config = field(default_factory=Kospi260306Config)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
