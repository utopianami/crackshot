"""TabM variants with PLE version B (ICLR 2025).

Two ensemble strategies, both with PLE-B:

TabM† (BatchEnsemble):
  All layers are BatchEnsembleLinear (shared W + per-member r,s scaling).
  tabm_init: first layer r=random-signs, s=ones. Rest layers r=s=ones.

TabM-mini† (MiniEnsemble):
  K independent affine transforms at input → shared backbone MLP.

PLE version B:
  output = linear0(x) + NLinear(PLE(x))
  - linear0: per-feature scalar→d_embedding linear (residual path)
  - NLinear: per-feature n_bins→d_embedding projection (starts at zeros)
  - First/last bins are NOT clamped (extrapolation preserved)

Training: all K members forward, loss on all K predictions.
Inference: softmax each member → average probabilities.
No normalization layers in MLP (no LayerNorm/BatchNorm).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


# ── PLE version B ───────────────────────────────────────────────


class PiecewiseLinearEncoding(eqx.Module):
    """PLE: raw piecewise-linear encoding (before projection).

    boundaries: [F, n_bins+1] per-feature bin edges.
    First bin: no lower clamp. Last bin: no upper clamp.
    Middle bins: clamp to [0, 1].
    Output: [F, n_bins].
    """

    boundaries: jax.Array  # [F, n_bins+1] static
    num_features: int
    n_bins: int

    def __init__(self, boundaries: np.ndarray):
        self.boundaries = jnp.array(boundaries, dtype=jnp.float32)
        self.num_features = boundaries.shape[0]
        self.n_bins = boundaries.shape[1] - 1

    def __call__(self, x: jax.Array) -> jax.Array:
        """x: [F] → [F, n_bins]."""
        x_expanded = x[:, None]  # [F, 1]
        left = self.boundaries[:, :-1]   # [F, T]
        right = self.boundaries[:, 1:]   # [F, T]
        width = jnp.maximum(right - left, 1e-8)
        frac = (x_expanded - left) / width  # [F, T]

        # First bin: clamp_max(1.0) only. Last bin: clamp_min(0.0) only. Middle: [0,1].
        T = self.n_bins
        lower = jnp.where(jnp.arange(T) == 0, -jnp.inf, 0.0)       # [T]
        upper = jnp.where(jnp.arange(T) == T - 1, jnp.inf, 1.0)    # [T]
        encoding = jnp.clip(frac, lower[None, :], upper[None, :])    # [F, T]
        return encoding


class NLinear(eqx.Module):
    """Per-feature independent linear projection: [F, n_bins] → [F, d_embedding].

    Each feature has its own weight matrix. Initialized to zeros so PLE
    contribution starts as residual. No bias (version B: linear0 provides bias).
    """

    weights: jax.Array  # [F, d_embedding, n_bins]

    def __init__(self, num_features: int, n_bins: int, d_embedding: int):
        self.weights = jnp.zeros((num_features, d_embedding, n_bins))

    def __call__(self, x: jax.Array) -> jax.Array:
        """x: [F, n_bins] → [F, d_embedding]."""
        return jnp.einsum('fd k, f k -> f d', self.weights, x)


class LinearEmbeddings(eqx.Module):
    """Per-feature scalar → d_embedding linear: linear0 in PLE-B.

    Each feature x_j is independently projected to d_embedding dimensions.
    """

    weights: jax.Array  # [F, d_embedding]
    biases: jax.Array   # [F, d_embedding]

    def __init__(self, num_features: int, d_embedding: int, *, key: jax.Array):
        # Official init: uniform(±1/sqrt(d_embedding))
        d_rsqrt = 1.0 / jnp.sqrt(jnp.array(d_embedding, dtype=jnp.float32))
        k1, k2 = jax.random.split(key)
        self.weights = jax.random.uniform(k1, (num_features, d_embedding), minval=-d_rsqrt, maxval=d_rsqrt)
        self.biases = jax.random.uniform(k2, (num_features, d_embedding), minval=-d_rsqrt, maxval=d_rsqrt)

    def __call__(self, x: jax.Array) -> jax.Array:
        """x: [F] → [F, d_embedding]."""
        return x[:, None] * self.weights + self.biases


class PLEv2(eqx.Module):
    """PLE version B: output = linear0(x) + NLinear(PLE(x)).

    Output shape: [F * d_embedding] (flattened).
    """

    ple: PiecewiseLinearEncoding
    linear0: LinearEmbeddings  # residual path
    nlinear: NLinear           # PLE projection (starts at zeros)
    d_embedding: int

    def __init__(self, boundaries: np.ndarray, d_embedding: int = 16, *, key: jax.Array):
        num_features = boundaries.shape[0]
        n_bins = boundaries.shape[1] - 1
        self.d_embedding = d_embedding

        self.ple = PiecewiseLinearEncoding(boundaries)
        self.linear0 = LinearEmbeddings(num_features, d_embedding, key=key)
        self.nlinear = NLinear(num_features, n_bins, d_embedding)

    def __call__(self, x: jax.Array) -> jax.Array:
        """x: [F] → [F * d_embedding]."""
        ple_out = self.ple(x)                    # [F, n_bins]
        linear0_out = self.linear0(x)            # [F, d_embedding]
        nlinear_out = self.nlinear(ple_out)      # [F, d_embedding]
        combined = linear0_out + nlinear_out     # [F, d_embedding]
        return combined.reshape(-1)              # [F * d_embedding]


# ── BatchEnsemble ───────────────────────────────────────────────


class BatchEnsembleLinear(eqx.Module):
    """BatchEnsemble linear: K members sharing W, per-member r, s, bias."""

    weight: jax.Array      # [out_features, in_features] shared
    r: jax.Array           # [K, in_features]
    s: jax.Array           # [K, out_features]
    bias: jax.Array        # [K, out_features]
    num_members: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_members: int,
        *,
        scaling_init: str = "ones",
        key: jax.Array,
    ):
        self.num_members = num_members
        k1, k2, k3 = jax.random.split(key, 3)

        # Lecun normal init for shared weight
        std = jnp.sqrt(1.0 / in_features)
        self.weight = jax.random.truncated_normal(k1, -2.0, 2.0, shape=(out_features, in_features)) * std

        # Per-member bias (initialized to same zeros, then split across K)
        self.bias = jnp.zeros((num_members, out_features))

        # Scaling init
        if scaling_init == "ones":
            self.r = jnp.ones((num_members, in_features))
            self.s = jnp.ones((num_members, out_features))
        elif scaling_init == "random-signs":
            # {-1, +1} random
            self.r = (jax.random.bernoulli(k2, 0.5, (num_members, in_features)).astype(jnp.float32) * 2 - 1)
            self.s = jnp.ones((num_members, out_features))
        else:
            self.r = jnp.ones((num_members, in_features))
            self.s = jnp.ones((num_members, out_features))

    def __call__(self, x: jax.Array, member_idx: int) -> jax.Array:
        """x: [in_features] → [out_features]."""
        scaled_input = self.r[member_idx] * x
        out = self.weight @ scaled_input
        return self.s[member_idx] * out + self.bias[member_idx]


class TabMBlock(eqx.Module):
    """BatchEnsembleLinear → ReLU → Dropout. Optional skip connection."""

    linear: BatchEnsembleLinear
    dropout: eqx.nn.Dropout
    use_skip: bool

    def __init__(self, in_features: int, out_features: int, num_members: int,
                 dropout: float = 0.1, *, scaling_init: str = "ones",
                 use_skip: bool = False, key: jax.Array):
        self.linear = BatchEnsembleLinear(
            in_features, out_features, num_members,
            scaling_init=scaling_init, key=key,
        )
        self.dropout = eqx.nn.Dropout(p=dropout)
        self.use_skip = use_skip and (in_features == out_features)

    def __call__(self, x: jax.Array, member_idx: int, *,
                 key: jax.Array, inference: bool = False) -> jax.Array:
        h = self.linear(x, member_idx)
        h = jax.nn.relu(h)
        h = self.dropout(h, key=key, inference=inference)
        if self.use_skip:
            h = h + x
        return h


class TabM(eqx.Module):
    """TabM†: PLE-B + BatchEnsemble MLP.

    tabm_init: first block uses random-signs for r, rest use ones.
    Training: all K members forward, loss on all K predictions.
    Inference: softmax each member → average probabilities.
    """

    ple_b: PLEv2
    input_drop: eqx.nn.Dropout
    blocks: list[TabMBlock]
    head: BatchEnsembleLinear
    reg_head: BatchEnsembleLinear | None  # optional regression head

    num_members: int
    num_horizons: int
    num_classes: int

    def __init__(
        self,
        num_features: int,
        num_members: int = 32,
        d_block: int = 512,
        d_embedding: int = 16,
        n_blocks: int = 3,
        input_dropout: float = 0.0,
        block_dropout: float = 0.1,
        num_classes: int = 5,
        num_horizons: int = 3,
        use_reg_head: bool = False,
        use_skip: bool = False,
        *,
        ple_boundaries: np.ndarray,
        key: jax.Array,
    ):
        self.num_members = num_members
        self.num_horizons = num_horizons
        self.num_classes = num_classes

        k_ple, k_blocks = jax.random.split(key)

        self.ple_b = PLEv2(ple_boundaries, d_embedding, key=k_ple)
        input_dim = num_features * d_embedding

        self.input_drop = eqx.nn.Dropout(p=input_dropout)

        # tabm_init: first block random-signs, rest ones
        keys = jax.random.split(k_blocks, n_blocks + 2)
        self.blocks = []
        in_dim = input_dim
        for i in range(n_blocks):
            si = "random-signs" if i == 0 else "ones"
            # skip connection only for blocks where in_dim == d_block (not first block)
            skip_i = use_skip and (i > 0)
            self.blocks.append(
                TabMBlock(in_dim, d_block, num_members, block_dropout,
                          scaling_init=si, use_skip=skip_i, key=keys[i])
            )
            in_dim = d_block

        # Output head: ones init (no random scaling)
        self.head = BatchEnsembleLinear(
            d_block, num_horizons * num_classes, num_members,
            scaling_init="ones", key=keys[n_blocks],
        )

        # Optional regression head: predicts continuous return per horizon
        if use_reg_head:
            self.reg_head = BatchEnsembleLinear(
                d_block, num_horizons, num_members,
                scaling_init="ones", key=keys[n_blocks + 1],
            )
        else:
            self.reg_head = None

    def _forward_member(
        self, x: jax.Array, member_idx: int, *, key: jax.Array, inference: bool = False
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        keys = jax.random.split(key, len(self.blocks) + 1)

        h = self.ple_b(x)
        h = self.input_drop(h, key=keys[0], inference=inference)

        for i, block in enumerate(self.blocks):
            h = block(h, member_idx, key=keys[i + 1], inference=inference)

        logits = self.head(h, member_idx)
        logits = logits.reshape(self.num_horizons, self.num_classes)

        if self.reg_head is not None and not inference:
            reg = self.reg_head(h, member_idx)  # [num_horizons]
            return logits, reg
        return logits

    def __call__(
        self, x: jax.Array, *, key: jax.Array, inference: bool = False
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Forward pass.

        Training: [K, H, C] logits (+ [K, H] reg if reg_head).
        Inference: [H, C] averaged probabilities.
        """
        member_keys = jax.random.split(key, self.num_members)

        def forward_k(k_idx):
            return self._forward_member(x, k_idx, key=member_keys[k_idx], inference=inference)

        if self.reg_head is not None and not inference:
            all_logits, all_reg = jax.vmap(forward_k)(jnp.arange(self.num_members))
            return all_logits, all_reg  # [K, H, C], [K, H]

        all_logits = jax.vmap(forward_k)(jnp.arange(self.num_members))  # [K, H, C]

        if inference:
            all_probs = jax.nn.softmax(all_logits, axis=-1)  # [K, H, C]
            return jnp.mean(all_probs, axis=0)  # [H, C]
        return all_logits  # [K, H, C]


# ── TabM-mini† ──────────────────────────────────────────────────




class TabMmini(eqx.Module):
    """TabM-mini†: K independent affine transforms → shared backbone MLP.

    Training: all K members forward, loss on all K predictions.
    Inference: softmax each → average probabilities.
    No normalization in backbone.
    """

    ple_b: PLEv2

    # K input projections (the only per-member params)
    input_projs: list[eqx.nn.Linear]

    # Shared backbone (no normalization)
    backbone_layers: list[eqx.nn.Linear]
    backbone_drops: list[eqx.nn.Dropout]

    # Shared output head
    head: eqx.nn.Linear

    num_members: int
    num_horizons: int
    num_classes: int

    def __init__(
        self,
        num_features: int,
        num_members: int = 32,
        d_block: int = 512,
        d_embedding: int = 16,
        n_blocks: int = 3,
        input_dropout: float = 0.0,
        block_dropout: float = 0.1,
        num_classes: int = 5,
        num_horizons: int = 3,
        *,
        ple_boundaries: np.ndarray,
        key: jax.Array,
    ):
        self.num_members = num_members
        self.num_horizons = num_horizons
        self.num_classes = num_classes

        k_ple, k_rest = jax.random.split(key)

        self.ple_b = PLEv2(ple_boundaries, d_embedding, key=k_ple)
        input_dim = num_features * d_embedding

        # K independent input projections with separate keys
        proj_keys = jax.random.split(k_rest, num_members + (n_blocks - 1) + 1)
        self.input_projs = []
        for k in range(num_members):
            self.input_projs.append(eqx.nn.Linear(input_dim, d_block, key=proj_keys[k]))

        ki = num_members

        # Shared backbone (n_blocks - 1 layers, no normalization)
        self.backbone_layers = []
        self.backbone_drops = []
        for i in range(n_blocks - 1):
            self.backbone_layers.append(eqx.nn.Linear(d_block, d_block, key=proj_keys[ki]))
            self.backbone_drops.append(eqx.nn.Dropout(p=block_dropout))
            ki += 1

        self.head = eqx.nn.Linear(d_block, num_horizons * num_classes, key=proj_keys[ki])

    def _forward_member(
        self, x: jax.Array, member_idx: int, *, key: jax.Array, inference: bool = False
    ) -> jax.Array:
        n_backbone = len(self.backbone_layers)
        keys = jax.random.split(key, n_backbone + 1)

        h = self.ple_b(x)

        # Per-member input projection via lax.switch
        def make_proj_fn(proj):
            return lambda c: proj(c)
        branches = [make_proj_fn(p) for p in self.input_projs]
        h = jax.lax.switch(member_idx, branches, h)
        h = jax.nn.relu(h)

        # Shared backbone (no normalization)
        for i, (layer, drop) in enumerate(
            zip(self.backbone_layers, self.backbone_drops)
        ):
            h = layer(h)
            h = jax.nn.relu(h)
            h = drop(h, key=keys[i + 1], inference=inference)

        logits = self.head(h)
        return logits.reshape(self.num_horizons, self.num_classes)

    def __call__(
        self, x: jax.Array, *, key: jax.Array, inference: bool = False
    ) -> jax.Array:
        member_keys = jax.random.split(key, self.num_members)

        def forward_k(k_idx):
            return self._forward_member(x, k_idx, key=member_keys[k_idx], inference=inference)
        all_logits = jax.vmap(forward_k)(jnp.arange(self.num_members))  # [K, H, C]

        if inference:
            all_probs = jax.nn.softmax(all_logits, axis=-1)
            return jnp.mean(all_probs, axis=0)  # [H, C]
        else:
            return all_logits  # [K, H, C]
