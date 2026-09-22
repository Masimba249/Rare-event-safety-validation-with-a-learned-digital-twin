"""Exact Gaussian-process backend for the digital twin.

Offered as the alternative to the deep ensemble named in the project plan.  The
GP gives a textbook-clean separation of the two uncertainties -- the posterior
variance is epistemic and shrinks where data is dense, the learned noise term is
aleatoric -- at the cost of an O(n^3) fit, so it runs on a subsampled inducing
set.  It is a useful cross-check on the ensemble's calibration; the ensemble is
the default because it stays cheap inside a 200k-episode Monte Carlo.

One independent ARD-RBF GP is fitted per output dimension, with hyperparameters
from type-II maximum likelihood (analytic gradients, L-BFGS, multiple restarts).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize

from ..config import TwinCfg


class _ARDGP:
    """Single-output exact GP with an ARD squared-exponential kernel."""

    def __init__(self, n_in: int) -> None:
        self.n_in = int(n_in)
        self.theta = np.zeros(self.n_in + 2)  # [log l_1..l_D, log sf, log sn]
        self.x: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        self.chol: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    def _unpack(self, theta: np.ndarray):
        ls = np.exp(theta[: self.n_in])
        sf2 = float(np.exp(2.0 * theta[self.n_in]))
        sn2 = float(np.exp(2.0 * theta[self.n_in + 1]))
        return ls, sf2, sn2

    def _sqdist(self, a: np.ndarray, b: np.ndarray, ls: np.ndarray) -> np.ndarray:
        aa = a / ls
        bb = b / ls
        return np.maximum(
            np.sum(aa * aa, axis=1)[:, None]
            + np.sum(bb * bb, axis=1)[None, :]
            - 2.0 * aa @ bb.T,
            0.0,
        )

    def _nll_and_grad(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray):
        n = x.shape[0]
        ls, sf2, sn2 = self._unpack(theta)
        d2 = self._sqdist(x, x, ls)
        kf = sf2 * np.exp(-0.5 * d2)
        k = kf + (sn2 + 1e-8) * np.eye(n)
        try:
            c, low = cho_factor(k, lower=True)
        except np.linalg.LinAlgError:
            return 1e12, np.zeros_like(theta)
        alpha = cho_solve((c, low), y)
        logdet = 2.0 * float(np.sum(np.log(np.diag(c))))
        nll = 0.5 * float(y @ alpha) + 0.5 * logdet + 0.5 * n * np.log(2.0 * np.pi)

        kinv = cho_solve((c, low), np.eye(n))
        w = np.outer(alpha, alpha) - kinv  # 0.5 * tr(W dK/dtheta) is the gradient
        grad = np.zeros_like(theta)
        xs = x / ls
        for d in range(self.n_in):
            diff = xs[:, d][:, None] - xs[:, d][None, :]
            grad[d] = -0.5 * float(np.sum(w * (kf * diff * diff)))
        grad[self.n_in] = -0.5 * float(np.sum(w * (2.0 * kf)))
        grad[self.n_in + 1] = -0.5 * float(np.trace(w) * 2.0 * sn2)
        return nll, grad

    # ------------------------------------------------------------------ #
    def fit(self, x: np.ndarray, y: np.ndarray, rng: np.random.Generator, restarts: int = 2) -> None:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        n = x.shape[0]

        # Median heuristic for the initial lengthscales.
        sub = x[rng.permutation(n)[: min(n, 200)]]
        diff = sub[:, None, :] - sub[None, :, :]
        med = np.sqrt(np.maximum(np.median(diff ** 2, axis=(0, 1)), 1e-6))
        base = np.concatenate(
            [np.log(np.maximum(med, 1e-2)), [np.log(np.std(y) + 1e-6)], [np.log(0.3 * np.std(y) + 1e-6)]]
        )

        best = (np.inf, base)
        for r in range(int(max(1, restarts))):
            start = base if r == 0 else base + rng.normal(0.0, 0.4, size=base.shape)
            res = minimize(
                lambda t: self._nll_and_grad(t, x, y),
                start,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 120},
            )
            if res.fun < best[0]:
                best = (float(res.fun), res.x)
        self.theta = np.asarray(best[1], dtype=float)

        ls, sf2, sn2 = self._unpack(self.theta)
        k = sf2 * np.exp(-0.5 * self._sqdist(x, x, ls)) + (sn2 + 1e-8) * np.eye(n)
        self.chol = cholesky(k, lower=True)
        self.alpha = cho_solve((self.chol, True), y)
        self.x = x

    def predict(self, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return ``(mean, epistemic_sd, aleatoric_sd)``."""
        if self.x is None or self.alpha is None or self.chol is None:
            raise RuntimeError("GP has not been fitted")
        ls, sf2, sn2 = self._unpack(self.theta)
        ks = sf2 * np.exp(-0.5 * self._sqdist(np.atleast_2d(xs), self.x, ls))
        mean = ks @ self.alpha
        v = solve_triangular(self.chol, ks.T, lower=True)
        var = np.maximum(sf2 - np.sum(v * v, axis=0), 1e-12)
        return mean, np.sqrt(var), float(np.sqrt(sn2))


class GPTwin:
    """Multi-output GP exposing the same surface as :class:`~rsv.models.ensemble.Ensemble`."""

    name = "gp"

    def __init__(self, cfg: TwinCfg, n_in: int, n_out: int, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.gps: List[_ARDGP] = [_ARDGP(n_in) for _ in range(n_out)]
        self.history: List[List[float]] = []

    @property
    def n_members(self) -> int:
        return 1

    # ------------------------------------------------------------------ #
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
        epochs: Optional[int] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        y = np.atleast_2d(np.asarray(y, dtype=float))
        n = x.shape[0]
        m = int(min(n, self.cfg.gp_inducing))
        if sample_weight is None:
            sel = rng.permutation(n)[:m]
        else:  # keep the re-weighted (replayed) points preferentially
            p = np.asarray(sample_weight, dtype=float)
            p = p / p.sum()
            sel = rng.choice(n, size=m, replace=False, p=p)
        for k, gp in enumerate(self.gps):
            gp.fit(x[sel], y[sel, k], rng=rng, restarts=self.cfg.gp_restarts)

    # ------------------------------------------------------------------ #
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        means, alea, epi = [], [], []
        for gp in self.gps:
            mu, sd_e, sd_a = gp.predict(x)
            means.append(mu)
            epi.append(sd_e)
            alea.append(np.full_like(mu, sd_a))
        return (
            np.stack(means, axis=1),
            np.stack(alea, axis=1),
            np.stack(epi, axis=1),
        )

    def predict_members(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean, alea, _ = self.predict(x)
        return mean[None, ...], alea[None, ...]

    def predict_assigned(
        self, x: np.ndarray, member_index: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Total predictive sd: a GP rollout samples epistemic *and* aleatoric."""
        mean, alea, epi = self.predict(x)
        return mean, np.sqrt(alea ** 2 + epi ** 2)

    # ------------------------------------------------------------------ #
    def state_dict(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for k, gp in enumerate(self.gps):
            out["gp%d_theta" % k] = gp.theta
            out["gp%d_x" % k] = gp.x if gp.x is not None else np.zeros((0, self.n_in))
            out["gp%d_alpha" % k] = gp.alpha if gp.alpha is not None else np.zeros(0)
            out["gp%d_chol" % k] = gp.chol if gp.chol is not None else np.zeros((0, 0))
        return out

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        for k, gp in enumerate(self.gps):
            gp.theta = np.asarray(state["gp%d_theta" % k], dtype=float)
            gp.x = np.asarray(state["gp%d_x" % k], dtype=float)
            gp.alpha = np.asarray(state["gp%d_alpha" % k], dtype=float)
            gp.chol = np.asarray(state["gp%d_chol" % k], dtype=float)
