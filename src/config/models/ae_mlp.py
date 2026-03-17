"""AE-MLP model configuration."""

from dataclasses import dataclass, field


@dataclass
class AEMLPConfig:
    # Autoencoder
    bottleneck_dim: int = 64
    ae_hidden: int = 32
    ae_dropout: float = 0.2
    decoder_dropout: float = 0.2
    noise_sigma: float = 0.1
    ae_lr: float = 0.005
    ae_epochs: int = 200
    ae_patience: int = 10
    # MLP
    mlp_hidden_units: list[int] = field(default_factory=lambda: [384, 896, 896, 394])
    mlp_input_dropout: float = 0.13
    mlp_dropout: float = 0.25
    mlp_lr: float = 0.0005
    mlp_epochs: int = 100
    mlp_patience: int = 10
    label_smoothing: float = 0.05
    mlp_weight_decay: float = 0.0
    horizon_weight_120: float = 1.0  # c2c_120 loss weight (1.0=equal, 2.0=2x)
    reg_weight: float = 0.0  # regression auxiliary loss weight (0=cls only)
    mlp_lr_schedule: str = "constant"  # "constant" | "cosine"
    class_weights: list[float] = field(default_factory=list)  # empty=uniform, e.g. [3,1,0.5,1,3]
    focal_gamma: float = 0.0  # 0=standard CE, >0=focal loss (1-p_t)^gamma
    # Shared
    batch_size: int = 4096
    sae_checkpoint: str = ""  # path to pre-trained SAE dir (skip Stage 1)
