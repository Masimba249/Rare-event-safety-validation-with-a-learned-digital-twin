"""The disturbance space: a standard-normal latent vector and its decoding.

The whole validation problem is posed on a latent vector ``z in R^d`` with the
nominal (operational) law ``p(z) = N(0, I_d)``.  Physical disturbances --
range-finder noise and dropouts, actuation latency, floor traction, obstacle
placement, sensor bias, and the dynamics residual drawn from the twin's own
predictive distribution -- are deterministic transforms of ``z``:

* Gaussian quantities are affine in ``z``;
* discrete quantities (dropout, latency) use an inverse CDF on ``Phi(z)``.

Why bother:

1. ``log p(z)`` is analytic, so importance-sampling weights ``p(z)/q(z)`` are
   exact rather than estimated.
2. Adaptive stress testing maximises ``sum_t log p(a_t)`` over actions; in this
   space that is simply ``-||z||^2 / 2`` plus a constant, so "most likely
   failure" has an unambiguous meaning.
3. The cross-entropy method and Gaussian-mixture proposals live naturally in an
   unbounded, isotropic space.
4. A scenario replayed on hardware and in the twin shares the same ``z``, which
   gives paired (common-random-number) sim-to-real comparisons.

Latent layout (``n_static = 6``, then ``n_per_step = 4`` values per control step)::

    z[0]                       floor traction coefficient mu
    z[1]                       obstacle longitudinal placement
    z[2]                       obstacle lateral placement
    z[3]                       range-finder calibration bias
    z[4]                       actuation latency (categorical)
    z[5]                       twin ensemble member (categorical; ignored by hardware)
    z[6 + 4t + 0]              range-finder noise at step t
    z[6 + 4t + 1]              range dropout indicator at step t
    z[6 + 4t + 2]              forward-velocity dynamics residual at step t
    z[6 + 4t + 3]              yaw-rate dynamics residual at step t
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.special import ndtr, ndtri

from .config import ScenarioCfg

LOG2PI = float(np.log(2.0 * np.pi))

# Channel offsets inside a per-step block.
CH_RANGE_NOISE = 0
CH_DROPOUT = 1
CH_RESID_V = 2
CH_RESID_W = 3

# Static channel indices.
IDX_MU = 0
IDX_OBS_X = 1
IDX_OBS_Y = 2
IDX_BIAS = 3
IDX_LATENCY = 4
IDX_MODEL = 5

STATIC_NAMES = (
    "floor_traction_mu",
    "obstacle_x_offset",
    "obstacle_y_offset",
    "range_bias",
    "actuation_latency",
    "twin_member",
)
CHANNEL_NAMES = ("range_noise", "range_dropout", "residual_v", "residual_w")


@dataclass
class Disturbances:
    """Decoded, physically meaningful disturbances for a batch of scenarios."""

    mu: np.ndarray  # (B,) floor traction coefficient
    obstacle_x: np.ndarray  # (B,) obstacle centre, metres
    obstacle_y: np.ndarray  # (B,)
    range_bias: np.ndarray  # (B,) persistent range-finder bias, metres
    latency: np.ndarray  # (B,) int, actuation delay in control steps
    model_index: np.ndarray  # (B,) int, ensemble member used for this rollout
    range_noise: np.ndarray  # (B, T) additive range noise, metres
    dropout: np.ndarray  # (B, T) bool, True -> stale range reading
    residual: np.ndarray  # (B, T, 2) standardised dynamics residual draws

    @property
    def batch(self) -> int:
        return int(self.mu.shape[0])


class ScenarioSpace:
    """Maps latent vectors to disturbances and scores them under the nominal law."""

    def __init__(self, cfg: ScenarioCfg, n_models: int = 1) -> None:
        self.cfg = cfg
        self.n_models = int(max(1, n_models))
        self.horizon = int(cfg.horizon)
        self.n_static = int(cfg.n_static)
        self.n_per_step = int(cfg.n_per_step)
        self.dim = self.n_static + self.n_per_step * self.horizon

        # Inverse-CDF thresholds for the discrete channels.
        self._dropout_threshold = float(ndtri(1.0 - cfg.dropout_prob))
        self._latency_edges = np.cumsum(np.asarray(cfg.latency_probs, dtype=float))
        self._latency_edges[-1] = 1.0

    # ------------------------------------------------------------------ #
    # Indexing helpers
    # ------------------------------------------------------------------ #
    def step_index(self, t: int, channel: int) -> int:
        """Latent index of ``channel`` at control step ``t``."""
        if not 0 <= t < self.horizon:
            raise IndexError("step %d out of range" % t)
        return self.n_static + self.n_per_step * int(t) + int(channel)

    def channel_indices(self, channel: int) -> np.ndarray:
        """All latent indices belonging to one per-step channel."""
        return self.n_static + self.n_per_step * np.arange(self.horizon) + int(channel)

    def index_label(self, i: int) -> str:
        """Human-readable name for latent dimension ``i`` (used in reports)."""
        if i < self.n_static:
            return STATIC_NAMES[i]
        j = i - self.n_static
        return "%s[t=%d]" % (CHANNEL_NAMES[j % self.n_per_step], j // self.n_per_step)

    @property
    def max_latency(self) -> int:
        return len(self.cfg.latency_probs)

    # ------------------------------------------------------------------ #
    # Nominal law
    # ------------------------------------------------------------------ #
    def sample_nominal(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.standard_normal((int(n), self.dim))

    def log_p(self, z: np.ndarray) -> np.ndarray:
        """Log nominal density ``log N(z; 0, I)``, shape ``(B,)``."""
        z = np.atleast_2d(z)
        return -0.5 * np.sum(z * z, axis=1) - 0.5 * self.dim * LOG2PI

    def log_p_diag_gaussian(
        self, z: np.ndarray, mean: np.ndarray, sd: np.ndarray
    ) -> np.ndarray:
        """Log density of a diagonal Gaussian, shape ``(B,)``."""
        z = np.atleast_2d(z)
        mean = np.atleast_1d(mean)
        sd = np.atleast_1d(sd)
        r = (z - mean) / sd
        return -0.5 * np.sum(r * r, axis=1) - np.sum(np.log(sd)) - 0.5 * self.dim * LOG2PI

    # ------------------------------------------------------------------ #
    # Decoding
    # ------------------------------------------------------------------ #
    def decode(self, z: np.ndarray) -> Disturbances:
        """Decode a batch of latent vectors into physical disturbances."""
        z = np.atleast_2d(np.asarray(z, dtype=float))
        if z.shape[1] != self.dim:
            raise ValueError(
                "latent dimension %d does not match scenario dim %d"
                % (z.shape[1], self.dim)
            )
        c = self.cfg
        b = z.shape[0]

        static = z[:, : self.n_static]
        mu = np.clip(c.mu_mean + c.mu_sd * static[:, IDX_MU], c.mu_min, c.mu_max)
        obstacle_x = c.obstacle_x_mean + c.obstacle_x_sd * static[:, IDX_OBS_X]
        obstacle_y = c.obstacle_y_sd * static[:, IDX_OBS_Y]
        range_bias = c.sensor_bias_sd * static[:, IDX_BIAS]

        u_lat = ndtr(static[:, IDX_LATENCY])
        latency = 1 + np.searchsorted(self._latency_edges, u_lat, side="right")
        latency = np.clip(latency, 1, self.max_latency).astype(np.int64)

        u_model = ndtr(static[:, IDX_MODEL])
        model_index = np.clip(
            (u_model * self.n_models).astype(np.int64), 0, self.n_models - 1
        )

        steps = z[:, self.n_static :].reshape(b, self.horizon, self.n_per_step)
        range_noise = c.range_noise_sd * steps[:, :, CH_RANGE_NOISE]
        dropout = steps[:, :, CH_DROPOUT] > self._dropout_threshold
        residual = np.stack(
            (steps[:, :, CH_RESID_V], steps[:, :, CH_RESID_W]), axis=2
        )

        return Disturbances(
            mu=mu,
            obstacle_x=obstacle_x,
            obstacle_y=obstacle_y,
            range_bias=range_bias,
            latency=latency,
            model_index=model_index,
            range_noise=range_noise,
            dropout=dropout,
            residual=residual,
        )

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def describe(self, z: np.ndarray, top_k: int = 6) -> Dict[str, object]:
        """Summarise one scenario: physical values plus the dimensions that
        carry most of its improbability (largest ``z_i^2``)."""
        z = np.asarray(z, dtype=float).reshape(-1)
        d = self.decode(z[None, :])
        order = np.argsort(-np.abs(z))[: int(top_k)]
        drivers: List[Dict[str, float]] = [
            {
                "dimension": self.index_label(int(i)),
                "z": float(z[i]),
                "log_likelihood_cost": float(0.5 * z[i] ** 2),
            }
            for i in order
        ]
        return {
            "log_p": float(self.log_p(z[None, :])[0]),
            "z_norm": float(np.linalg.norm(z)),
            "floor_traction_mu": float(d.mu[0]),
            "obstacle_x": float(d.obstacle_x[0]),
            "obstacle_y": float(d.obstacle_y[0]),
            "range_bias_m": float(d.range_bias[0]),
            "actuation_latency_steps": int(d.latency[0]),
            "twin_member": int(d.model_index[0]),
            "n_dropouts": int(d.dropout[0].sum()),
            "max_dropout_run": int(_max_run(d.dropout[0])),
            "max_abs_residual": float(np.max(np.abs(d.residual[0]))),
            "top_drivers": drivers,
        }

    def nominal_like(self, z: np.ndarray) -> np.ndarray:
        """Number of nominal standard deviations of the whole latent vector.

        ``||z|| / sqrt(d)`` is 1 for a typical nominal draw; stress-test
        scenarios sit above it.
        """
        z = np.atleast_2d(z)
        return np.linalg.norm(z, axis=1) / np.sqrt(self.dim)


def _max_run(mask: np.ndarray) -> int:
    """Length of the longest run of ``True`` in a 1-D boolean array."""
    best = run = 0
    for v in np.asarray(mask).reshape(-1):
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def make_space(cfg: ScenarioCfg, n_models: int = 1) -> ScenarioSpace:
    return ScenarioSpace(cfg, n_models=n_models)


def latent_from_disturbance_overrides(
    space: ScenarioSpace,
    base: Optional[np.ndarray] = None,
    mu: Optional[float] = None,
    latency_steps: Optional[int] = None,
    dropout_window: Optional[range] = None,
) -> np.ndarray:
    """Build a latent vector with specific physical disturbances forced on.

    Used by tests and by the hand-built sanity scenarios in the report; not part
    of the estimation path.
    """
    z = np.zeros(space.dim) if base is None else np.array(base, dtype=float).reshape(-1)
    c = space.cfg
    if mu is not None:
        z[IDX_MU] = (float(mu) - c.mu_mean) / c.mu_sd
    if latency_steps is not None:
        k = int(np.clip(latency_steps, 1, space.max_latency))
        lo = 0.0 if k == 1 else float(space._latency_edges[k - 2])
        hi = float(space._latency_edges[k - 1])
        z[IDX_LATENCY] = float(ndtri(np.clip(0.5 * (lo + hi), 1e-9, 1 - 1e-9)))
    if dropout_window is not None:
        for t in dropout_window:
            if 0 <= t < space.horizon:
                z[space.step_index(t, CH_DROPOUT)] = space._dropout_threshold + 0.5
    return z
