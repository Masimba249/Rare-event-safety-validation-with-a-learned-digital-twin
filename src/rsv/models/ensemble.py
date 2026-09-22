"""Deep ensemble of heteroscedastic networks.

The ensemble separates the two kinds of uncertainty the validation argument
needs to keep apart:

* **aleatoric** -- irreducible randomness of the real robot (wheel slip, motor
  jitter), read off each member's predicted variance.  This is a genuine part
  of the probability space being integrated, so the rollout *samples* it.
* **epistemic** -- the model's ignorance, read off the disagreement between
  members.  This is not a property of the robot; it is a warning that the
  estimate in this region of state space is not trustworthy.  The pipeline
  reports it alongside every failure mode and uses it to choose which scenarios
  are worth replaying on hardware.

Sampling a member index per episode (from the latent vector, see
:mod:`rsv.scenario`) turns the ensemble into a proper mixture model, so the
failure probability estimated by the rollout marginalises over model
uncertainty instead of conditioning on one arbitrary fit.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import TwinCfg
from .mlp import MLP, train_mlp


class Ensemble:
    """Bootstrap ensemble of :class:`~rsv.models.mlp.MLP` predictors."""

    name = "ensemble"

    def __init__(self, cfg: TwinCfg, n_in: int, n_out: int, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.members: List[MLP] = [
            MLP(
                n_in,
                n_out,
                cfg.hidden,
                rng=np.random.default_rng(rng.integers(1 << 62)),
                activation=cfg.activation,
                min_sigma=cfg.min_sigma,
                max_sigma=cfg.max_sigma,
                beta_nll=cfg.beta_nll,
            )
            for _ in range(int(cfg.n_members))
        ]
        self.history: List[List[float]] = []

    @property
    def n_members(self) -> int:
        return len(self.members)

    # ------------------------------------------------------------------ #
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
        epochs: Optional[int] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Fit every member on its own bootstrap resample of the data."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        y = np.atleast_2d(np.asarray(y, dtype=float))
        n = x.shape[0]
        self.history = []
        for m, member in enumerate(self.members):
            sub = np.random.default_rng(rng.integers(1 << 62))
            if self.cfg.bootstrap and n > 1:
                sel = sub.integers(0, n, size=n)
            else:
                sel = np.arange(n)
            w = None if sample_weight is None else np.asarray(sample_weight)[sel]
            hist = train_mlp(
                member,
                x[sel],
                y[sel],
                rng=sub,
                epochs=int(self.cfg.epochs if epochs is None else epochs),
                batch_size=self.cfg.batch_size,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                sample_weight=w,
            )
            self.history.append(hist)

    # ------------------------------------------------------------------ #
    def predict_members(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Per-member predictions: ``(means, sigmas)`` of shape ``(M, B, K)``."""
        outs = [member.predict(x) for member in self.members]
        means = np.stack([o[0] for o in outs], axis=0)
        sigmas = np.stack([o[1] for o in outs], axis=0)
        return means, sigmas

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mixture statistics: ``(mean, aleatoric_sd, epistemic_sd)``."""
        means, sigmas = self.predict_members(x)
        mean = means.mean(axis=0)
        aleatoric = np.sqrt(np.mean(sigmas ** 2, axis=0))
        epistemic = means.std(axis=0)
        return mean, aleatoric, epistemic

    def predict_assigned(
        self, x: np.ndarray, member_index: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predictions from a per-row choice of member.

        Evaluating each member on only its own rows keeps the cost of an
        ensemble rollout the same as a single-network rollout, which is what
        makes a 200k-episode Monte Carlo affordable.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        member_index = np.asarray(member_index, dtype=np.int64).reshape(-1)
        mean = np.empty((x.shape[0], self.n_out))
        sigma = np.empty((x.shape[0], self.n_out))
        for m, member in enumerate(self.members):
            rows = member_index == m
            if not rows.any():
                continue
            mu, sd = member.predict(x[rows])
            mean[rows] = mu
            sigma[rows] = sd
        return mean, sigma

    # ------------------------------------------------------------------ #
    def state_dict(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for m, member in enumerate(self.members):
            for key, val in member.state_dict().items():
                out["m%d_%s" % (m, key)] = val
        return out

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        for m, member in enumerate(self.members):
            member.load_state_dict(
                {
                    key[len("m%d_" % m) :]: val
                    for key, val in state.items()
                    if key.startswith("m%d_" % m)
                }
            )
