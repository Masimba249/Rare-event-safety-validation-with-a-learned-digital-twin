"""Comparing the two estimators: how much does importance sampling buy?

The fair currency is *variance per episode*, not wall-clock: both estimators run
the same simulator, so an episode costs the same either way.  For naive Monte
Carlo the per-episode variance of the indicator is ``p(1-p)``; for importance
sampling it is the sample variance of ``w * 1[failure]``.  Their ratio is the
variance-reduction factor, and it converts directly into "episodes saved":
reaching a given relative error needs ``Var / (rel^2 * p^2)`` episodes either
way.

Both per-episode variances are evaluated at the *same* ``p`` -- the importance
sampling estimate, being the more precise one -- so the comparison is not
flattered by the two estimators disagreeing about the probability itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .importance import ISEstimate
from .monte_carlo import MCEstimate


@dataclass
class Comparison:
    """Variance-reduction summary for the report."""

    p_reference: float
    variance_per_sample_mc: float
    variance_per_sample_is: float
    variance_reduction_factor: float
    equivalent_mc_episodes: float
    target_relative_error: float
    episodes_for_target_mc: float
    episodes_for_target_is: float
    episode_saving_factor: float
    seconds_mc: float
    seconds_is: float
    agreement_z: Optional[float]
    intervals_overlap: Optional[bool]

    def as_dict(self) -> Dict[str, object]:
        return {
            "p_reference": self.p_reference,
            "variance_per_episode": {
                "naive_monte_carlo": self.variance_per_sample_mc,
                "importance_sampling": self.variance_per_sample_is,
            },
            "variance_reduction_factor": self.variance_reduction_factor,
            "equivalent_mc_episodes": self.equivalent_mc_episodes,
            "target_relative_error": self.target_relative_error,
            "episodes_for_target": {
                "naive_monte_carlo": self.episodes_for_target_mc,
                "importance_sampling": self.episodes_for_target_is,
            },
            "episode_saving_factor": self.episode_saving_factor,
            "seconds": {"naive_monte_carlo": self.seconds_mc, "importance_sampling": self.seconds_is},
            "agreement_z": self.agreement_z,
            "intervals_overlap": self.intervals_overlap,
        }


def episodes_for_relative_error(variance_per_sample: float, p: float, rel: float) -> float:
    """``n = Var / (rel^2 * p^2)`` -- episodes to hit a target relative error."""
    if p <= 0.0 or rel <= 0.0:
        return float("inf")
    return float(variance_per_sample / (rel ** 2 * p ** 2))


def compare(
    mc: MCEstimate,
    is_est: ISEstimate,
    target_relative_error: float = 0.10,
) -> Comparison:
    """Variance reduction of importance sampling against naive Monte Carlo."""
    p = is_est.p_hat if is_est.p_hat > 0 else mc.p_hat
    var_mc = float(max(p * (1.0 - p), 0.0))
    var_is = float(is_est.variance_per_sample)

    vrf = float(var_mc / var_is) if var_is > 0 else float("inf")
    n_mc = episodes_for_relative_error(var_mc, p, target_relative_error)
    n_is = episodes_for_relative_error(var_is, p, target_relative_error)

    agreement_z: Optional[float] = None
    overlap: Optional[bool] = None
    if mc.n_failures > 0:
        se = float(np.sqrt(mc.standard_error ** 2 + is_est.standard_error ** 2))
        if se > 0:
            agreement_z = float((mc.p_hat - is_est.p_hat) / se)
        overlap = bool(mc.ci_low <= is_est.ci_high and is_est.ci_low <= mc.ci_high)

    return Comparison(
        p_reference=float(p),
        variance_per_sample_mc=var_mc,
        variance_per_sample_is=var_is,
        variance_reduction_factor=vrf,
        equivalent_mc_episodes=float(is_est.n * vrf),
        target_relative_error=float(target_relative_error),
        episodes_for_target_mc=n_mc,
        episodes_for_target_is=n_is,
        episode_saving_factor=float(n_mc / n_is) if n_is > 0 else float("inf"),
        seconds_mc=float(mc.seconds),
        seconds_is=float(is_est.seconds),
        agreement_z=agreement_z,
        intervals_overlap=overlap,
    )


def covers(estimate_low: float, estimate_high: float, reference: float) -> bool:
    """Does an interval contain the reference value?"""
    return bool(estimate_low <= reference <= estimate_high)


def accuracy_report(
    label: str,
    p_hat: float,
    ci_low: float,
    ci_high: float,
    reference: float,
) -> Dict[str, object]:
    """How an estimate stands against the hardware-oracle reference."""
    return {
        "estimator": label,
        "p_hat": float(p_hat),
        "ci": [float(ci_low), float(ci_high)],
        "reference": float(reference),
        "ratio_to_reference": float(p_hat / reference) if reference > 0 else float("nan"),
        "relative_bias": float((p_hat - reference) / reference) if reference > 0 else float("nan"),
        "covers_reference": covers(ci_low, ci_high, reference),
    }
