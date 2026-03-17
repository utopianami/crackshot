"""SAE+MLP: gogo827jz 1st place solution 기반의 Supervised Autoencoder + MLP 모델.

Architecture overview:
  Phase 1 - SupervisedAutoencoder:
    Input(F) → LayerNorm → GaussianNoise (train only)
    → Linear(F, bottleneck_dim) → ReLU              [Encoder]
    → Dropout → Linear(bottleneck_dim, F)            [Decoder]
    → Linear(F, ae_hidden) → ReLU → Dropout
    → Linear(ae_hidden, ae_hidden) → ReLU → Dropout
    → Linear(ae_hidden, num_horizons * num_classes)  [AE classification head]

  Phase 2 - FrozenEncoderMLP:
    Input(F) → frozen_encoder → encoder_out(bottleneck_dim)
    Concat([encoder_out, x]) = F + bottleneck_dim
    → LayerNorm → Dropout(mlp_input_dropout)
    → [Linear → LayerNorm → ReLU → Dropout] × N layers
    → Linear(last_hidden, num_horizons * num_classes) [MLP classification head]

Note: LayerNorm used instead of BatchNorm — stateless and vmap-friendly in Equinox.
Weight init: Equinox default (Lecun normal).
"""

import equinox as eqx
import jax
import jax.numpy as jnp


class SupervisedAutoencoder(eqx.Module):
    """Supervised Autoencoder (SAE) — Phase 1 학습 대상.

    입력 특성을 저차원 bottleneck으로 압축하면서,
    reconstruction과 분류를 동시에 학습한다.
    """

    # Normalization
    input_norm: eqx.nn.LayerNorm

    # Encoder
    encoder: eqx.nn.Linear

    # Decoder
    decoder_dropout: eqx.nn.Dropout
    decoder: eqx.nn.Linear

    # Classification head (2 hidden layers)
    ae_fc1: eqx.nn.Linear
    ae_norm1: eqx.nn.LayerNorm
    ae_drop1: eqx.nn.Dropout
    ae_fc2: eqx.nn.Linear
    ae_norm2: eqx.nn.LayerNorm
    ae_drop2: eqx.nn.Dropout
    ae_head: eqx.nn.Linear

    # Config (static)
    noise_sigma: float
    num_horizons: int
    num_classes: int
    bottleneck_dim: int

    def __init__(
        self,
        num_features: int,
        bottleneck_dim: int = 64,
        ae_hidden: int = 32,
        ae_dropout: float = 0.2,
        decoder_dropout: float = 0.2,
        noise_sigma: float = 0.1,
        num_classes: int = 5,
        num_horizons: int = 3,
        *,
        key: jax.Array,
    ):
        self.noise_sigma = noise_sigma
        self.num_horizons = num_horizons
        self.num_classes = num_classes
        self.bottleneck_dim = bottleneck_dim

        keys = jax.random.split(key, 14)  # enough keys for all layers + init

        # Normalization
        self.input_norm = eqx.nn.LayerNorm(num_features)

        # Encoder: F → bottleneck_dim
        self.encoder = eqx.nn.Linear(num_features, bottleneck_dim, key=keys[0])

        # Decoder: bottleneck_dim → F
        self.decoder_dropout = eqx.nn.Dropout(p=decoder_dropout)
        self.decoder = eqx.nn.Linear(bottleneck_dim, num_features, key=keys[1])

        # Classification head: F → ae_hidden → ae_hidden → (num_horizons * num_classes)
        self.ae_fc1 = eqx.nn.Linear(num_features, ae_hidden, key=keys[2])
        self.ae_norm1 = eqx.nn.LayerNorm(ae_hidden)
        self.ae_drop1 = eqx.nn.Dropout(p=ae_dropout)

        self.ae_fc2 = eqx.nn.Linear(ae_hidden, ae_hidden, key=keys[3])
        self.ae_norm2 = eqx.nn.LayerNorm(ae_hidden)
        self.ae_drop2 = eqx.nn.Dropout(p=ae_dropout)

        self.ae_head = eqx.nn.Linear(ae_hidden, num_horizons * num_classes, key=keys[4])

    def __call__(
        self, x: jax.Array, *, key: jax.Array, inference: bool = False
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Forward pass.

        Args:
            x: Input features [F].
            key: PRNG key for dropout and noise.
            inference: If True, disable dropout and Gaussian noise.

        Returns:
            decoder_out: Reconstructed features [F].
            ae_pred:     Per-horizon classification logits [num_horizons, num_classes].
            encoder_out: Bottleneck representation [bottleneck_dim] (for MLP).
        """
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # --- Input normalization ---
        h = self.input_norm(x)

        # --- Gaussian noise (training only) ---
        if not inference:
            noise = self.noise_sigma * jax.random.normal(k1, h.shape)
            h = h + noise

        # --- Encoder: F → bottleneck_dim → ReLU ---
        encoder_out = jax.nn.relu(self.encoder(h))

        # --- Decoder: bottleneck_dim → Dropout → F ---
        dec_h = self.decoder_dropout(encoder_out, key=k2, inference=inference)
        decoder_out = self.decoder(dec_h)

        # --- AE classification head (uses decoder output as input) ---
        # Layer 1
        cl = self.ae_fc1(decoder_out)
        cl = self.ae_norm1(cl)
        cl = jax.nn.relu(cl)
        cl = self.ae_drop1(cl, key=k3, inference=inference)

        # Layer 2
        cl = self.ae_fc2(cl)
        cl = self.ae_norm2(cl)
        cl = jax.nn.relu(cl)
        cl = self.ae_drop2(cl, key=k4, inference=inference)

        # Head → reshape to (num_horizons, num_classes)
        ae_logits = self.ae_head(cl)
        ae_pred = ae_logits.reshape(self.num_horizons, self.num_classes)

        return decoder_out, ae_pred, encoder_out


class FrozenEncoderMLP(eqx.Module):
    """MLP using frozen SAE encoder bottleneck — Phase 2 학습 대상.

    encoder_out과 원본 입력을 concat해서 MLP 입력으로 사용.
    Encoder weights는 Phase 2에서 freeze된다 (trainer가 pytree partition으로 처리).
    """

    # Frozen encoder reference
    encoder: eqx.nn.Linear         # SAE encoder (weights frozen by trainer)
    input_norm: eqx.nn.LayerNorm   # SAE input norm (shared, frozen)

    # MLP input processing
    mlp_input_norm: eqx.nn.LayerNorm
    mlp_input_drop: eqx.nn.Dropout

    # MLP hidden layers
    mlp_layers: list[eqx.nn.Linear]
    mlp_norms: list[eqx.nn.LayerNorm]
    mlp_drops: list[eqx.nn.Dropout]

    # Output heads
    mlp_head: eqx.nn.Linear      # classification: [num_horizons * num_classes]
    reg_head: eqx.nn.Linear      # regression: [num_horizons] scalar predictions

    # Config (static)
    num_horizons: int
    num_classes: int

    def __init__(
        self,
        num_features: int,
        bottleneck_dim: int = 64,
        mlp_hidden_units: list[int] | None = None,
        mlp_input_dropout: float = 0.13,
        mlp_dropout: float = 0.25,
        num_classes: int = 5,
        num_horizons: int = 3,
        *,
        sae: SupervisedAutoencoder,
        key: jax.Array,
    ):
        if mlp_hidden_units is None:
            mlp_hidden_units = [384, 896, 896, 394]

        self.num_horizons = num_horizons
        self.num_classes = num_classes

        # Freeze SAE encoder by holding a reference — trainer uses eqx.partition
        # to keep these weights out of the gradient update.
        self.encoder = sae.encoder
        self.input_norm = sae.input_norm

        # MLP concatenated input size: bottleneck_dim + num_features
        concat_dim = bottleneck_dim + num_features

        self.mlp_input_norm = eqx.nn.LayerNorm(concat_dim)
        self.mlp_input_drop = eqx.nn.Dropout(p=mlp_input_dropout)

        # Build hidden layers
        num_layers = len(mlp_hidden_units)
        keys = jax.random.split(key, num_layers + 2)  # layers + 2 heads

        layer_sizes = [concat_dim] + mlp_hidden_units
        self.mlp_layers = []
        self.mlp_norms = []
        self.mlp_drops = []

        for i in range(num_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            self.mlp_layers.append(eqx.nn.Linear(in_dim, out_dim, key=keys[i]))
            self.mlp_norms.append(eqx.nn.LayerNorm(out_dim))
            self.mlp_drops.append(eqx.nn.Dropout(p=mlp_dropout))

        # Output heads
        self.mlp_head = eqx.nn.Linear(
            mlp_hidden_units[-1], num_horizons * num_classes, key=keys[num_layers]
        )
        self.reg_head = eqx.nn.Linear(
            mlp_hidden_units[-1], num_horizons, key=keys[num_layers + 1]
        )

    def __call__(
        self, x: jax.Array, *, key: jax.Array, inference: bool = False
    ) -> tuple[jax.Array, jax.Array]:
        """Forward pass.

        Args:
            x: Input features [F].
            key: PRNG key for dropout.
            inference: If True, disable dropout.

        Returns:
            cls_logits: Per-horizon classification logits [num_horizons, num_classes].
            reg_preds:  Per-horizon regression predictions [num_horizons].
        """
        keys = jax.random.split(key, len(self.mlp_layers) + 1)
        drop_keys = keys[:-1]
        input_drop_key = keys[-1]

        # --- Frozen encoder pass (stop_gradient blocks grads into encoder/input_norm) ---
        normed_x = jax.lax.stop_gradient(self.input_norm(x))
        encoder_out = jax.lax.stop_gradient(jax.nn.relu(self.encoder(normed_x)))  # [bottleneck_dim]

        # --- Concat encoder_out and original x ---
        concat = jnp.concatenate([encoder_out, x], axis=-1)  # [bottleneck_dim + F]

        # --- Input normalization + dropout ---
        h = self.mlp_input_norm(concat)
        h = self.mlp_input_drop(h, key=input_drop_key, inference=inference)

        # --- Hidden layers: Linear → LayerNorm → ReLU → Dropout ---
        for layer, norm, drop, dkey in zip(
            self.mlp_layers, self.mlp_norms, self.mlp_drops, drop_keys
        ):
            h = layer(h)
            h = norm(h)
            h = jax.nn.relu(h)
            h = drop(h, key=dkey, inference=inference)

        # --- Classification head → reshape to (num_horizons, num_classes) ---
        logits = self.mlp_head(h)
        cls_logits = logits.reshape(self.num_horizons, self.num_classes)

        # --- Regression head → (num_horizons,) ---
        reg_preds = self.reg_head(h)

        return cls_logits, reg_preds
