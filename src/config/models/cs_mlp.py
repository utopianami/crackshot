"""Cross-sectional MLP configuration."""

from dataclasses import dataclass, field


@dataclass
class CSMLPConfig:
    hidden_units: list[int] = field(default_factory=lambda: [512, 512, 256])
    dropout: float = 0.3
    lr: float = 0.001
    weight_decay: float = 0.01
    batch_size: int = 2048  # timestamps per batch (not rows)
    epochs: int = 100
    patience: int = 15
    label_smoothing: float = 0.05
    max_stocks: int = 200
    # ListNet
    listnet_weight: float = 0.3
    listnet_tau_true: float = 0.1
    listnet_tau_pred: float = 1.0
    n_timestamps_per_batch: int = 32  # how many timestamps per training step
    horizon_weight_120: float = 1.0
    # Feature modes: "raw_mean_zscore" (default), "zscore", "raw_zscore", "rank"
    feature_mode: str = "raw_mean_zscore"
    use_layernorm: bool = False
    activation: str = "gelu"
    warmup_epochs: int = 0  # 0 = no warmup, >0 = linear warmup then cosine decay
    freeze_body: bool = False  # True = freeze hidden layers, train head only
    feature_indices: list[int] = field(default_factory=list)  # empty = all features
    residual: bool = False  # True = add skip connections between same-dim layers
