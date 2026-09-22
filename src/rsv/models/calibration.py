"""Calibration diagnostics for a probabilistic dynamics model.

A rare-event estimate produced from a learned twin is only as trustworthy as
the twin's *uncertainty*, not just its mean.  These metrics answer three
questions:

* is the mean accurate (RMSE / MAE);
* are the error bars the right size (NLL, PICP, reliability, miscalibration
  area);
* are they the right size *everywhere*, or only where the data was dense
  (the same metrics restricted to a held-out tail subset).

The last one matters most here: the failure region is, by construction, the
part of the state space nominal logging barely visited.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.special import ndtr

LOG2PI = float(np.log(2.0 * np.pi))
DEFAULT_LEVELS = (0.5, 0.68, 0.8, 0.9, 0.95, 0.99)


def gaussian_nll(y: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> float:
    """Mean Gaussian negative log-likelihood (nats per sample, summed over outputs)."""
    sd = np.maximum(np.asarray(sd, dtype=float), 1e-12)
    z = (np.asarray(y, dtype=float) - np.asarray(mean, dtype=float)) / sd
    return float(np.mean(np.sum(0.5 * (z * z) + np.log(sd) + 0.5 * LOG2PI, axis=1)))


def gaussian_crps(y: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> float:
    """Mean continuous ranked probability score for a Gaussian forecast."""
    sd = np.maximum(np.asarray(sd, dtype=float), 1e-12)
    z = (np.asarray(y, dtype=float) - np.asarray(mean, dtype=float)) / sd
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    crps = sd * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * pdf - 1.0 / np.sqrt(np.pi))
    return float(np.mean(np.sum(crps, axis=1)))


def coverage(
    y: np.ndarray, mean: np.ndarray, sd: np.ndarray, levels: Sequence[float] = DEFAULT_LEVELS
) -> Dict[str, float]:
    """Empirical coverage of central predictive intervals, per nominal level."""
    from scipy.special import ndtri

    sd = np.maximum(np.asarray(sd, dtype=float), 1e-12)
    z = np.abs((np.asarray(y, dtype=float) - np.asarray(mean, dtype=float)) / sd)
    out: Dict[str, float] = {}
    for level in levels:
        crit = float(ndtri(0.5 + 0.5 * level))
        out["%.2f" % level] = float(np.mean(np.all(z <= crit, axis=1)))
    return out


def marginal_coverage(
    y: np.ndarray, mean: np.ndarray, sd: np.ndarray, levels: Sequence[float] = DEFAULT_LEVELS
) -> Dict[str, List[float]]:
    """Per-output-dimension coverage (the reliability diagram's data)."""
    from scipy.special import ndtri

    sd = np.maximum(np.asarray(sd, dtype=float), 1e-12)
    z = np.abs((np.asarray(y, dtype=float) - np.asarray(mean, dtype=float)) / sd)
    out: Dict[str, List[float]] = {}
    for level in levels:
        crit = float(ndtri(0.5 + 0.5 * level))
        out["%.2f" % level] = [float(np.mean(z[:, k] <= crit)) for k in range(z.shape[1])]
    return out


def miscalibration_area(
    y: np.ndarray, mean: np.ndarray, sd: np.ndarray, n_levels: int = 41
) -> float:
    """Area between the reliability curve and the diagonal (0 is perfect)."""
    from scipy.special import ndtri

    sd = np.maximum(np.asarray(sd, dtype=float), 1e-12)
    z = np.abs((np.asarray(y, dtype=float) - np.asarray(mean, dtype=float)) / sd).reshape(-1)
    levels = np.linspace(0.0, 0.999, int(n_levels))
    crit = ndtri(0.5 + 0.5 * levels)
    emp = np.array([float(np.mean(z <= c)) for c in crit])
    return float(np.trapezoid(np.abs(emp - levels), levels))


def report(
    y: np.ndarray,
    mean: np.ndarray,
    aleatoric: np.ndarray,
    epistemic: Optional[np.ndarray] = None,
    target_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Full calibration report for one evaluation set."""
    y = np.atleast_2d(np.asarray(y, dtype=float))
    mean = np.atleast_2d(np.asarray(mean, dtype=float))
    aleatoric = np.atleast_2d(np.asarray(aleatoric, dtype=float))
    total = aleatoric if epistemic is None else np.sqrt(aleatoric ** 2 + np.atleast_2d(epistemic) ** 2)

    err = y - mean
    names = list(target_names or ["y%d" % k for k in range(y.shape[1])])
    out: Dict[str, object] = {
        "n": int(y.shape[0]),
        "rmse": {n: float(np.sqrt(np.mean(err[:, k] ** 2))) for k, n in enumerate(names)},
        "mae": {n: float(np.mean(np.abs(err[:, k]))) for k, n in enumerate(names)},
        "nll_total": gaussian_nll(y, mean, total),
        "nll_aleatoric_only": gaussian_nll(y, mean, aleatoric),
        "crps": gaussian_crps(y, mean, total),
        "sharpness": {n: float(np.mean(total[:, k])) for k, n in enumerate(names)},
        "coverage_joint": coverage(y, mean, total),
        "coverage_marginal": marginal_coverage(y, mean, total),
        "miscalibration_area": miscalibration_area(y, mean, total),
        "mean_epistemic": (
            {n: float(np.mean(np.atleast_2d(epistemic)[:, k])) for k, n in enumerate(names)}
            if epistemic is not None
            else None
        ),
        "mean_aleatoric": {n: float(np.mean(aleatoric[:, k])) for k, n in enumerate(names)},
    }
    return out


def reliability_curve(
    y: np.ndarray, mean: np.ndarray, sd: np.ndarray, n_levels: int = 21
) -> Dict[str, np.ndarray]:
    """Nominal-versus-empirical coverage, for the calibration figure."""
    from scipy.special import ndtri

    sd = np.maximum(np.asarray(sd, dtype=float), 1e-12)
    z = np.abs((np.asarray(y, dtype=float) - np.asarray(mean, dtype=float)) / sd).reshape(-1)
    levels = np.linspace(0.01, 0.99, int(n_levels))
    crit = ndtri(0.5 + 0.5 * levels)
    emp = np.array([float(np.mean(z <= c)) for c in crit])
    return {"nominal": levels, "empirical": emp}
