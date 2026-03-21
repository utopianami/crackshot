"""FT-Transformer model configuration."""

from dataclasses import dataclass, field


@dataclass
class FTTransformerConfig:
    # Architecture
    d_token: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ffn_multiplier: float = 4 / 3
    attention_dropout: float = 0.1
    ffn_dropout: float = 0.1
    residual_dropout: float = 0.0
    # PLE tokenization
    n_bins: int = 48
    d_embedding: int = 16      # PLE embedding dim before projection to d_token
    # Training
    lr: float = 1e-4
    lr_schedule: str = "cosine"
    weight_decay: float = 1e-5
    batch_size: int = 2048
    epochs: int = 100
    patience: int = 15
    label_smoothing: float = 0.05
    # Optional
    class_weights: list[float] = field(default_factory=list)
