"""The learned digital twin.

Wraps a probabilistic backend (deep ensemble by default, GP optionally) with the
feature normalisation, the :class:`~rsv.dynamics.Dynamics` interface used by the
rollout, persistence, and the diagnostics that make its *uncertainty* legible.

The contrast with a hand-tuned simulator is the point of the project: this model
is fitted to logged driving and reports, for any state-command pair, both how
noisy the robot is there (aleatoric) and how little it knows about that region
(epistemic).  Both numbers are used downstream -- the first is sampled as part
of the probability space, the second decides which stress-test findings are
worth spending hardware time on.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .config import Config, TwinCfg
from .dynamics import FEATURE_NAMES, N_FEATURES, N_TARGETS, TARGET_NAMES, Dynamics, build_features
from .models import calibration
from .models.ensemble import Ensemble
from .models.gp import GPTwin
from .utils import load_json, save_json, save_npz, load_npz


def _make_backend(cfg: TwinCfg, rng: np.random.Generator):
    if cfg.backend == "ensemble":
        return Ensemble(cfg, N_FEATURES, N_TARGETS, rng)
    if cfg.backend == "gp":
        return GPTwin(cfg, N_FEATURES, N_TARGETS, rng)
    raise ValueError("unknown twin backend %r" % cfg.backend)


class Twin(Dynamics):
    """Probabilistic dynamics model fitted to logged driving data."""

    name = "twin"
    probabilistic = True

    def __init__(self, cfg: TwinCfg, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.backend = _make_backend(cfg, rng)
        self.x_mean = np.zeros(N_FEATURES)
        self.x_std = np.ones(N_FEATURES)
        self.y_mean = np.zeros(N_TARGETS)
        self.y_std = np.ones(N_TARGETS)
        self.fitted = False
        self.train_size = 0
        # Variance the measurement chain adds to the regression targets.  It is
        # subtracted from the learned predictive variance when the twin is used
        # as a simulator, so rollouts carry the robot's process noise and not
        # the overhead camera's pixel noise.  See rsv.data.estimate_target_noise.
        self.measurement_var = np.zeros(N_TARGETS)

    # ------------------------------------------------------------------ #
    @property
    def n_models(self) -> int:
        return int(self.backend.n_members)

    def _normalise_x(self, x: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(np.asarray(x, dtype=float)) - self.x_mean) / self.x_std

    def _denormalise_y(self, mean_n: np.ndarray, sd_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return mean_n * self.y_std + self.y_mean, sd_n * np.abs(self.y_std)

    def set_measurement_noise(self, variance: Optional[np.ndarray]) -> None:
        """Register the target-noise variance of the state estimator."""
        self.measurement_var = (
            np.zeros(N_TARGETS) if variance is None else np.asarray(variance, dtype=float).reshape(-1)
        )

    def _deconvolve(self, sd: np.ndarray) -> np.ndarray:
        """Remove the measurement-noise floor from a predictive sd.

        The learned variance is the sum of the robot's process variance and the
        estimator's noise variance.  Only the former belongs in a simulation.
        The residue is floored rather than clipped to zero so that a region where
        the correction would over-subtract still keeps a sane amount of noise.
        """
        if not np.any(self.measurement_var > 0.0):
            return sd
        var = sd ** 2 - self.measurement_var[None, :]
        return np.sqrt(np.maximum(var, (0.25 * sd) ** 2))

    # ------------------------------------------------------------------ #
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
        sample_weight: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
    ) -> None:
        """Fit on a design matrix of features and velocity-increment targets."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        y = np.atleast_2d(np.asarray(y, dtype=float))
        self.x_mean = x.mean(axis=0)
        self.x_std = np.maximum(x.std(axis=0), 1e-6)
        self.y_mean = y.mean(axis=0)
        self.y_std = np.maximum(y.std(axis=0), 1e-9)
        self.backend.fit(
            self._normalise_x(x),
            (y - self.y_mean) / self.y_std,
            rng=rng,
            epochs=epochs,
            sample_weight=sample_weight,
        )
        self.fitted = True
        self.train_size = int(x.shape[0])

    # ------------------------------------------------------------------ #
    def predict(
        self, features: np.ndarray, deconvolve: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(mean, aleatoric_sd, epistemic_sd)`` in physical units.

        ``deconvolve=False`` reports the predictive distribution of the
        *measured* target, which is what calibration must be judged against;
        ``deconvolve=True`` reports the robot's own process noise, which is what
        a rollout should sample.
        """
        xn = self._normalise_x(features)
        mean_n, alea_n, epi_n = self.backend.predict(xn)
        mean = mean_n * self.y_std + self.y_mean
        alea = alea_n * np.abs(self.y_std)
        if deconvolve:
            alea = self._deconvolve(alea)
        return mean, alea, epi_n * np.abs(self.y_std)

    def step(
        self,
        v: np.ndarray,
        omega: np.ndarray,
        v_cmd: np.ndarray,
        omega_cmd: np.ndarray,
        mu: np.ndarray,
        eps: np.ndarray,
        model_index: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One stochastic step: predicted mean plus a sampled residual.

        ``eps`` is the standardised residual draw taken from the scenario's
        latent vector, so the same disturbance realisation can be replayed on
        the hardware surrogate for a paired comparison.
        """
        feats = build_features(v, omega, v_cmd, omega_cmd, mu)
        xn = self._normalise_x(feats)
        if model_index is None:
            model_index = np.zeros(xn.shape[0], dtype=np.int64)
        mean_n, sd_n = self.backend.predict_assigned(xn, model_index)
        mean, sd = self._denormalise_y(mean_n, sd_n)
        sd = self._deconvolve(sd)
        eps = np.atleast_2d(np.asarray(eps, dtype=float))
        out = mean + sd * eps
        return out[:, 0], out[:, 1]

    # ------------------------------------------------------------------ #
    def calibration_report(
        self, x: np.ndarray, y: np.ndarray, label: str = "validation"
    ) -> Dict[str, object]:
        mean, alea, epi = self.predict(x)
        rep = calibration.report(y, mean, alea, epi, target_names=TARGET_NAMES)
        rep["label"] = label
        return rep

    def reliability(self, x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        mean, alea, epi = self.predict(x)
        return calibration.reliability_curve(y, mean, np.sqrt(alea ** 2 + epi ** 2))

    # ------------------------------------------------------------------ #
    def braking_curve(
        self, mu_grid: np.ndarray, v0: float, dt: float
    ) -> Dict[str, np.ndarray]:
        """Predicted deceleration during an emergency stop, as a function of mu.

        This is the single most decision-relevant slice of the model: the
        obstacle-stop function commands ``v_cmd = 0`` from cruise speed, and
        whether the robot stops in time is decided by the achieved deceleration
        on the floor it happens to be driving on.
        """
        mu_grid = np.atleast_1d(np.asarray(mu_grid, dtype=float))
        n = mu_grid.shape[0]
        feats = build_features(
            np.full(n, v0), np.zeros(n), np.zeros(n), np.zeros(n), mu_grid
        )
        mean, alea, epi = self.predict(feats, deconvolve=True)
        return {
            "mu": mu_grid,
            "accel_mean": mean[:, 0] / dt,
            "accel_aleatoric": alea[:, 0] / dt,
            "accel_epistemic": epi[:, 0] / dt,
        }

    def uncertainty_map(
        self, v_grid: np.ndarray, mu_grid: np.ndarray, v_cmd: float = 0.0
    ) -> Dict[str, np.ndarray]:
        """Epistemic standard deviation over a ``(speed, traction)`` grid."""
        vv, mm = np.meshgrid(np.asarray(v_grid), np.asarray(mu_grid), indexing="ij")
        flat_v, flat_mu = vv.reshape(-1), mm.reshape(-1)
        feats = build_features(
            flat_v,
            np.zeros_like(flat_v),
            np.full_like(flat_v, v_cmd),
            np.zeros_like(flat_v),
            flat_mu,
        )
        _, alea, epi = self.predict(feats, deconvolve=True)
        return {
            "v": np.asarray(v_grid),
            "mu": np.asarray(mu_grid),
            "epistemic": epi[:, 0].reshape(vv.shape),
            "aleatoric": alea[:, 0].reshape(vv.shape),
        }

    def epistemic_at(self, features: np.ndarray) -> np.ndarray:
        """Scalar epistemic uncertainty (velocity channel) for a feature batch."""
        _, _, epi = self.predict(features)
        return epi[:, 0]

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        path = str(path)
        state = {k: np.asarray(v) for k, v in self.backend.state_dict().items()}
        state.update(
            {
                "measurement_var": self.measurement_var,
                "x_mean": self.x_mean,
                "x_std": self.x_std,
                "y_mean": self.y_mean,
                "y_std": self.y_std,
            }
        )
        save_npz(path, **state)
        save_json(
            str(Path(path).with_suffix(".meta.json")),
            {
                "backend": self.cfg.backend,
                "n_members": self.n_models,
                "hidden": list(self.cfg.hidden),
                "train_size": self.train_size,
                "feature_names": list(FEATURE_NAMES),
                "target_names": list(TARGET_NAMES),
            },
        )

    @classmethod
    def load(cls, path: str, cfg: TwinCfg, rng: np.random.Generator) -> "Twin":
        """Rebuild a saved twin.

        The architecture comes from the file's own metadata, not from the config
        in hand: a checkpoint's backend, ensemble size and layer widths are
        properties of the saved weights, and silently rebuilding them from a
        config that has since changed would either crash or, worse, load
        mismatched weights.
        """
        meta_path = Path(path).with_suffix(".meta.json")
        meta = load_json(str(meta_path)) if meta_path.exists() else {}
        if meta:
            cfg = dataclasses.replace(
                cfg,
                backend=meta.get("backend", cfg.backend),
                n_members=int(meta.get("n_members", cfg.n_members)),
                hidden=list(meta.get("hidden", cfg.hidden)),
            )

        twin = cls(cfg, rng)
        state = load_npz(path)
        twin.x_mean = state.pop("x_mean")
        twin.x_std = state.pop("x_std")
        twin.y_mean = state.pop("y_mean")
        twin.y_std = state.pop("y_std")
        twin.measurement_var = state.pop("measurement_var", np.zeros(N_TARGETS))
        twin.backend.load_state_dict(state)
        twin.fitted = True
        twin.train_size = int(meta.get("train_size", 0))
        return twin


def build_twin(cfg: Config, rng: np.random.Generator) -> Twin:
    return Twin(cfg.twin, rng)
