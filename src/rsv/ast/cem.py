"""Cross-entropy stress testing and proposal fitting.

The cross-entropy method for rare events (Rubinstein; de Boer et al. 2005) walks
a parametric sampling distribution toward the failure region through a sequence
of shrinking robustness levels.  Two things make it the right partner for the
MCTS searcher in :mod:`rsv.ast.mcts`:

1. It is cheap and batched -- thousands of episodes per iteration -- so it maps
   the shape of a failure mode rather than finding one path into it.
2. The distribution it returns is already the object importance sampling needs.
   Fitting the elite set *weighted by the likelihood ratio* ``p(z)/q(z)`` makes
   the fitted Gaussian an approximation of the zero-variance optimal proposal
   ``p(z | failure)`` rather than merely "wherever the search drifted", which is
   what keeps the downstream weights well behaved.

Several restarts are run from different seeds; because the failure surface here
has genuinely distinct modes (stop too late on a slippery floor, versus never
see the obstacle at all), the restarts land in different places and their union
is what the mixture proposal is built from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config import Config
from ..dynamics import Dynamics
from ..estimate.screening import effective_sample_size, soft_screen, tilted_dimensions
from ..rollout import robustness_of
from ..scenario import ScenarioSpace


@dataclass
class CEMRun:
    """Outcome of one cross-entropy restart."""

    mean: np.ndarray  # (d,) fitted proposal mean
    sd: np.ndarray  # (d,) fitted proposal standard deviations
    failures: np.ndarray  # (n_fail, d) latent vectors that produced a collision
    failure_logp: np.ndarray  # (n_fail,) nominal log-density of each
    failure_logq: np.ndarray  # (n_fail,) density of the search distribution that drew it
    best_robustness: float
    hit_rate: float  # failure fraction in the final population
    evaluations: int
    n_tilted: int = 0  # coordinates the screening left shifted
    history: List[Dict[str, float]] = field(default_factory=list)

    @property
    def found_failure(self) -> bool:
        return self.failures.shape[0] > 0


def cem_search(
    dynamics: Dynamics,
    cfg: Config,
    space: ScenarioSpace,
    rng: np.random.Generator,
    init_mean: Optional[np.ndarray] = None,
    init_sd: Optional[float] = None,
) -> CEMRun:
    """Run one cross-entropy search for the most likely failures."""
    ac, ec = cfg.ast, cfg.estimate
    d = space.dim
    pop = int(ac.cem_population)
    n_elite = max(4, int(round(ac.cem_elite_frac * pop)))

    mean = np.zeros(d) if init_mean is None else np.array(init_mean, dtype=float)
    sd = np.full(d, float(ac.cem_sd_init if init_sd is None else init_sd))

    failures: List[np.ndarray] = []
    failure_logq: List[np.ndarray] = []
    history: List[Dict[str, float]] = []
    best = np.inf
    evaluations = 0

    for it in range(int(ac.cem_iterations)):
        z = mean + sd * rng.standard_normal((pop, d))
        rob = robustness_of(z, dynamics, cfg, space, batch=ec.batch)
        evaluations += pop
        failed = rob < 0.0
        if failed.any():
            failures.append(z[failed])
            # Remember which distribution drew each failure, so the proposal fit
            # can reweight them into an estimate of p(z | failure) instead of an
            # estimate of wherever this search happened to be looking.
            failure_logq.append(space.log_p_diag_gaussian(z[failed], mean, sd))

        # Elites: the failures if we have enough of them, otherwise the closest
        # calls.  This is the CE level sequence, with the level implied by the
        # elite quantile of the robustness.
        order = np.argsort(rob)
        elite_idx = np.flatnonzero(failed) if int(failed.sum()) >= n_elite else order[:n_elite]
        elites = z[elite_idx]

        # Likelihood-ratio weights, so the fit targets p(z | failure) rather
        # than the search distribution's own drift.  Computed in log space.
        log_w = space.log_p(elites) - space.log_p_diag_gaussian(elites, mean, sd)
        w = np.exp(log_w - log_w.max())
        w_sum = float(w.sum())
        if not np.isfinite(w_sum) or w_sum <= 0:
            w = np.ones(elites.shape[0])
            w_sum = float(w.sum())

        new_mean = (w[:, None] * elites).sum(axis=0) / w_sum
        var = (w[:, None] * (elites - new_mean) ** 2).sum(axis=0) / w_sum
        new_sd = np.sqrt(np.maximum(var, 0.0))

        # Screen out coordinates whose apparent shift is within sampling noise.
        # Without this the fit tilts all 566 dimensions and the proposal becomes
        # useless as an importance-sampling distribution -- see
        # rsv.estimate.screening for why that is fatal rather than merely untidy.
        new_mean, new_sd = soft_screen(
            new_mean,
            new_sd,
            effective_sample_size(w),
            c=float(ac.screen_c_search),
            sd_floor=float(ac.cem_sd_floor),
        )

        a = float(ac.cem_smoothing)
        mean = a * new_mean + (1.0 - a) * mean
        sd = a * new_sd + (1.0 - a) * sd

        best = min(best, float(rob.min()))
        history.append(
            {
                "iteration": it,
                "elite_robustness": float(rob[elite_idx].mean()),
                "best_robustness": best,
                "failure_rate": float(failed.mean()),
                "mean_norm": float(np.linalg.norm(mean)),
                "tilted_dimensions": int(tilted_dimensions(mean).size),
            }
        )

    fails = (
        np.concatenate(failures, axis=0) if failures else np.zeros((0, d), dtype=float)
    )
    logq = np.concatenate(failure_logq) if failure_logq else np.zeros(0)
    return CEMRun(
        mean=mean,
        sd=sd,
        failures=fails,
        failure_logp=space.log_p(fails) if len(fails) else np.zeros(0),
        failure_logq=logq,
        best_robustness=best,
        hit_rate=float(history[-1]["failure_rate"]) if history else 0.0,
        evaluations=evaluations,
        n_tilted=int(tilted_dimensions(mean).size),
        history=history,
    )


def multi_start_cem(
    dynamics: Dynamics,
    cfg: Config,
    space: ScenarioSpace,
    rng: np.random.Generator,
) -> List[CEMRun]:
    """Independent restarts, so distinct failure modes get their own proposal.

    The first restart starts at the nominal mean and finds whichever mode is
    easiest to reach.  Each later restart is seeded by tilting a single *static*
    disturbance channel -- floor traction, obstacle placement, range bias,
    actuation latency -- because those channels are what separate the modes:
    pushing the lateral obstacle offset finds "never saw it", pushing traction
    finds "stopped too late".  Seeding one interpretable coordinate at a time is
    both a better spread than a random high-dimensional start and a statement
    about which hypotheses the search covered.

    Each channel is seeded in both directions.  An obstacle offset to the left
    and the same offset to the right are separate modes carrying equal
    probability, and a search seeded only one way finds only one of them --
    which silently halves that mode's contribution to the final estimate.
    """
    runs: List[CEMRun] = []
    d = space.dim
    n_static = space.n_static
    for r in range(int(cfg.ast.cem_restarts)):
        init_mean = None
        if r > 0:
            init_mean = np.zeros(d)
            channel = (r - 1) % n_static
            sign = 1.0 if ((r - 1) // n_static) % 2 == 0 else -1.0
            init_mean[channel] = sign * float(cfg.ast.cem_seed_shift)
        runs.append(
            cem_search(
                dynamics,
                cfg,
                space,
                np.random.default_rng(rng.integers(1 << 62)),
                init_mean=init_mean,
            )
        )
    return runs
