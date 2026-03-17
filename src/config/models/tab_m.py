"""TabM model configuration."""

from dataclasses import dataclass, field


@dataclass
class TabMConfig:
    # TabM architecture
    variant: str = "tabm"  # "tabm" (BatchEnsemble) or "mini" (MiniEnsemble)
    num_members: int = 32
    d_block: int = 512
    d_embedding: int = 16
    n_blocks: int = 3
    n_bins: int = 48
    input_dropout: float = 0.0
    block_dropout: float = 0.1
    use_skip: bool = False  # residual connections in blocks 2+
    # Training
    lr: float = 0.002
    weight_decay: float = 3e-4
    lr_schedule: str = "constant"
    batch_size: int = 4096
    epochs: int = 100
    patience: int = 15
    label_smoothing: float = 0.05
    horizon_weight_120: float = 1.0
    class_weights: list[float] = field(default_factory=list)  # empty=uniform, e.g. [3,1.5,0.3,1.5,3]
    focal_gamma: float = 0.0  # 0=standard CE, >0=focal loss
    ordinal_weight: float = 0.0  # 0=off, >0=add EMD ordinal loss (e.g. 0.5)
    reg_weight: float = 0.0  # 0=off, >0=add regression auxiliary loss on c2c_120
    # ListNet cross-sectional ranking loss
    listnet_weight: float = 0.0  # 0=off, >0=add listwise ranking loss
    listnet_tau_true: float = 0.1  # temperature for true rank distribution
    listnet_tau_pred: float = 1.0  # temperature for predicted score distribution
    n_timestamps_per_batch: int = 5  # timestamps per batch for ListNet
    max_stocks: int = 820  # max stocks per timestamp (for padding)
