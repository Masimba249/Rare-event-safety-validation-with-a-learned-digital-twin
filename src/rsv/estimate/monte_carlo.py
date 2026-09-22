"""Naive Monte Carlo baseline, with honest confidence intervals.

This is the estimator the project exists to beat, and it is also the reference
that keeps everything else honest: importance sampling is only trustworthy if
it agrees with brute force wherever brute force can still say something.

Interval choice matters at these rates.  With a handful of observed failures the
normal approximation is badly wrong -- it can produce a lower bound below zero,
or a zero-width interval when nothing failed at all.  So the headline interval
is Clopper-Pearson (exact, conservative), with Wilson reported alongside, and
the zero-failure case is reported as the one-sided bound ``p < 1 - alpha^(1/n)``
that is all the data supports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import norm

from ..config import Config
from ..dynamics import Dynamics
from ..rollout import simulate
from ..scenario import ScenarioSpace
from ..utils import batched


@dataclass
class MCEstimate:
    """A naive Monte Carlo failure-probability estimate."""

    p_hat: float
    n: int
    n_failures: int
    standard_error: float
    ci_low: float
    ci_high: float
    ci_method: str
    confidence: float
    relative_error: float
    seconds: float
    wilson: Optional[List[float]] = None
    convergence: Dict[str, List[float]] = field(default_factory=dict)
    failures: Optional[np.ndarray] = None  # latent vectors, if collected

    def as_dict(self) -> Dict[str, object]:
        return {
            "estimator": "naive_monte_carlo",
            "p_hat": self.p_hat,
            "n": self.n,
            "n_failures": self.n_failures,
            "standard_error": self.standard_error,
            "ci": [self.ci_low, self.ci_high],
            "ci_method": self.ci_method,
            "confidence": self.confidence,
            "relative_error": self.relative_error,
            "wilson_ci": self.wilson,
            "seconds": self.seconds,
        }


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> List[float]:
    """Exact binomial interval.  Conservative, and correct at ``k = 0``."""
    alpha = 1.0 - float(confidence)
    lo = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return [lo, hi]


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> List[float]:
    """Wilson score interval: better behaved than Wald, still closed form."""
    if n == 0:
        return [0.0, 1.0]
    z = float(norm.ppf(0.5 + 0.5 * float(confidence)))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def zero_failure_bound(n: int, confidence: float = 0.95) -> float:
    """Upper bound on ``p`` when no failure was observed in ``n`` trials.

    The "rule of three" generalised: ``P(no failure) = (1-p)^n = 1 - conf``.
    """
    if n <= 0:
        return 1.0
    return float(1.0 - (1.0 - confidence) ** (1.0 / n))


def naive_monte_carlo(
    dynamics: Dynamics,
    cfg: Config,
    space: ScenarioSpace,
    rng: np.random.Generator,
    n: Optional[int] = None,
    batch: Optional[int] = None,
    collect_failures: bool = False,
    convergence_points: Optional[int] = None,
) -> MCEstimate:
    """Sample from the nominal law and count collisions."""
    import time

    ec = cfg.estimate
    n = int(ec.n_mc if n is None else n)
    size = int(ec.batch if batch is None else batch)
    n_points = int(ec.convergence_points if convergence_points is None else convergence_points)
    checkpoints = np.unique(np.geomspace(max(size, 10), n, num=max(n_points, 2)).astype(int))

    t0 = time.perf_counter()
    seen = 0
    hits = 0
    failure_rows: List[np.ndarray] = []
    trace_n: List[float] = []
    trace_p: List[float] = []
    trace_lo: List[float] = []
    trace_hi: List[float] = []
    next_cp = 0

    for lo, hi in batched(n, size):
        z = space.sample_nominal(rng, hi - lo)
        result = simulate(z, dynamics, cfg, space)
        failed = result.failure
        hits += int(failed.sum())
        seen += hi - lo
        if collect_failures and failed.any():
            failure_rows.append(z[failed])
        while next_cp < len(checkpoints) and seen >= checkpoints[next_cp]:
            ci = clopper_pearson(hits, seen, ec.confidence)
            trace_n.append(float(seen))
            trace_p.append(hits / seen)
            trace_lo.append(ci[0])
            trace_hi.append(ci[1])
            next_cp += 1

    seconds = time.perf_counter() - t0
    p_hat = hits / max(seen, 1)
    se = float(np.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / max(seen, 1)))
    ci = clopper_pearson(hits, seen, ec.confidence)
    if hits == 0:
        ci = [0.0, zero_failure_bound(seen, ec.confidence)]
        method = "zero-failure upper bound"
    else:
        method = "Clopper-Pearson"

    return MCEstimate(
        p_hat=float(p_hat),
        n=int(seen),
        n_failures=int(hits),
        standard_error=se,
        ci_low=float(ci[0]),
        ci_high=float(ci[1]),
        ci_method=method,
        confidence=float(ec.confidence),
        relative_error=float(se / p_hat) if p_hat > 0 else float("inf"),
        seconds=float(seconds),
        wilson=wilson_interval(hits, seen, ec.confidence),
        convergence={"n": trace_n, "p": trace_p, "ci_low": trace_lo, "ci_high": trace_hi},
        failures=np.concatenate(failure_rows, axis=0) if failure_rows else None,
    )


def samples_for_relative_error(p: float, rel_error: float) -> float:
    """Episodes a naive Monte Carlo needs to reach a given relative error.

    ``n = (1 - p) / (p * rel^2)`` -- the reason rare-event validation by direct
    testing is hopeless: it scales as ``1/p``.
    """
    if p <= 0.0:
        return float("inf")
    return float((1.0 - p) / (p * rel_error ** 2))
