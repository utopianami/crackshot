"""KOSPI 260306 dataset-specific configuration."""

from dataclasses import dataclass, field


@dataclass
class Kospi260306Config:
    """KOSPI 260306 npy 데이터셋 고유 설정."""
    horizon_indices: list[int] = field(default_factory=lambda: [17, 21, 25])
    primary_horizon: int = 1  # index into horizon_indices (c2c_120)
    vol_idx: int = -1  # last column
    class_bins: list[float] = field(default_factory=lambda: [-0.01, -0.003, 0.003, 0.01])
