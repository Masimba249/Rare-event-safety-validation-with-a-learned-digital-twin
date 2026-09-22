"""Dimension screening for high-dimensional importance-sampling proposals.

The disturbance space here has 566 dimensions, but a failure is caused by a
handful of them: the floor traction, where the obstacle ended up, and the few
range readings that went stale while the robot was closing on it.  The other
~540 are along for the ride.

That distinction is not cosmetic -- it decides whether importance sampling works
at all.  Fitting a full diagonal Gaussian to a set of failures shifts and
shrinks *every* coordinate by whatever sampling noise happens to be in it, and
in 566 dimensions those individually negligible errors compound in the
normalising constant.  Shrinking the standard deviation to 0.45 in every
coordinate, for instance, moves ``log q`` by ``566 * log(1/0.45) ~ 450`` nats;
the likelihood ratio ``p/q`` then underflows and the estimator degenerates to
whatever its defensive component alone can do -- measurably *worse* than naive
Monte Carlo, which is exactly what this project would otherwise claim to beat.

The fix is to tilt only the coordinates whose shift is larger than its own
sampling error.  Rather than a hard test, which would stall a cross-entropy
search before the shifts have grown, each coordinate is soft-thresholded:

    shrink_j = max(0, 1 - c * se_j / |mean_j|)

so a coordinate carrying a real signal keeps almost all of its shift, one
carrying only noise is returned exactly to the nominal law, and the transition
between them is continuous.  Standard deviations are shrunk toward 1 by the same
factor, so an untilted coordinate contributes nothing at all to ``log(p/q)``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def soft_screen(
    mean: np.ndarray,
    sd: np.ndarray,
    n_effective: float,
    c: float = 3.0,
    abs_floor: float = 0.05,
    sd_floor: float = 0.45,
    sd_ceiling: float = 1.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shrink a fitted diagonal Gaussian back toward ``N(0, I)`` coordinatewise.

    ``n_effective`` is the effective number of samples behind the fit (use the
    weights' effective sample size when the fit was weighted).  Returns the
    screened ``(mean, sd)``.
    """
    mean = np.asarray(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)
    n_eff = float(max(n_effective, 1.0))

    se = np.maximum(sd, 1e-6) / np.sqrt(n_eff)
    magnitude = np.maximum(np.abs(mean), 1e-12)
    shrink = np.clip(1.0 - float(c) * se / magnitude, 0.0, 1.0)
    shrink = np.where(np.abs(mean) < float(abs_floor), 0.0, shrink)

    screened_mean = shrink * mean
    screened_sd = 1.0 + shrink * (np.clip(sd, sd_floor, sd_ceiling) - 1.0)
    return screened_mean, screened_sd


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size of a set of non-negative weights."""
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0:
        return 0.0
    return float(total ** 2 / np.sum(w ** 2))


def tilted_dimensions(mean: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Indices the screening left tilted -- the interpretable failure signature."""
    return np.flatnonzero(np.abs(np.asarray(mean, dtype=float)) > tol)


def relevant_coordinates(
    samples: np.ndarray,
    weights: np.ndarray,
    n_keep: int = 24,
) -> np.ndarray:
    """Coordinates along which a set of failures departs most from the nominal law.

    Scores each coordinate by how far its failure-conditional first *and* second
    moment sit from ``N(0, 1)``::

        score_j = mean_j^2 + (var_j - 1)^2

    The variance term is not optional.  A mode such as "the obstacle was offset
    far enough to fall outside the beam" is symmetric in the sign of the offset,
    so its mean shift is zero while its variance is large; scoring on the mean
    alone would discard exactly the coordinate that defines it.

    This is what makes clustering viable.  In 566 dimensions every pair of
    failures sits at almost the same Euclidean distance -- the ~540 irrelevant
    coordinates contribute a common ``sqrt(2 d)`` that swamps the handful of
    coordinates the modes actually differ in -- so k-means on the raw latent
    vectors returns essentially arbitrary groups.  Restricted to the coordinates
    below, the modes separate cleanly.
    """
    x = np.atleast_2d(np.asarray(samples, dtype=float))
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = np.maximum(w, 0.0)
    if w.sum() <= 0:
        w = np.ones(x.shape[0])
    p = w / w.sum()

    mean = (p[:, None] * x).sum(axis=0)
    var = (p[:, None] * (x - mean) ** 2).sum(axis=0)
    score = mean ** 2 + (var - 1.0) ** 2
    n_keep = int(max(1, min(n_keep, x.shape[1])))
    return np.sort(np.argsort(-score)[:n_keep])


def temper_weights(
    log_weights: np.ndarray,
    min_ess_fraction: float = 0.25,
    iterations: int = 40,
) -> Tuple[np.ndarray, float]:
    """Flatten importance weights until they support a usable fit.

    Returns normalised ``w ~ exp(beta * log_w)`` and the exponent ``beta`` used.

    Weights on the failures returned by a stress-test search span hundreds of
    nats, because the search distribution ends up far from the nominal law.
    Raw, they concentrate everything on a handful of points -- in one run here,
    4 of 17836 -- and the proposal is then fitted to those 4.  That is how a
    whole failure mode goes missing: the search found the obstacle offset in
    both directions, but the four surviving points happened to be mostly one
    sign, so the proposal only covered one side and quietly lost half that
    mode's probability.

    ``beta`` is chosen as the largest exponent whose effective sample size still
    reaches ``min_ess_fraction`` of the sample, by bisection.  ``beta = 1`` is
    the exact weighting and ``beta = 0`` is uniform, which maximises coverage at
    the cost of fidelity.  Tempering biases the *fit*, never the estimate: the
    proposal is only a sampling distribution, and the importance weights used in
    the final estimator are always the exact ones.
    """
    lw = np.asarray(log_weights, dtype=float).reshape(-1)
    n = lw.size
    if n == 0:
        return np.zeros(0), 1.0
    lw = lw - lw.max()

    def ess_fraction(beta: float) -> float:
        w = np.exp(beta * lw)
        total = float(w.sum())
        if total <= 0:
            return 0.0
        return float(total ** 2 / np.sum(w ** 2)) / n

    if ess_fraction(1.0) >= float(min_ess_fraction):
        w = np.exp(lw)
        return w / w.sum(), 1.0

    lo, hi = 0.0, 1.0
    for _ in range(int(iterations)):
        mid = 0.5 * (lo + hi)
        if ess_fraction(mid) >= float(min_ess_fraction):
            lo = mid
        else:
            hi = mid
    w = np.exp(lo * lw)
    return w / w.sum(), float(lo)
