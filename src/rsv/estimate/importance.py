"""Importance sampling with a defensive mixture proposal.

The estimator is

    p_hat = (1/N) * sum_i  w_i * 1[failure_i],      w_i = p(z_i) / q(z_i),
    z_i ~ q,

which is unbiased for any ``q`` whose support covers the failure region.  Two
design choices make it trustworthy rather than merely fast:

**Mixture, not unimodal.**  The failure region here has separate modes -- "the
stop started too late on a slippery floor" and "the obstacle was never inside
the beam" -- and they are far apart in latent space.  A single Gaussian centred
between them would be a bad fit to both, so the proposal is a mixture with one
component per mode discovered by the stress testers.

**Defensive.**  A fraction ``alpha`` of the proposal is the nominal law itself
(Hesterberg's defensive importance sampling).  Then ``q >= alpha * p``
everywhere, so

    w = p / q <= 1 / alpha

is *bounded*.  A bounded weight times a bounded indicator has finite variance by
construction, which is what licenses the central-limit interval below.  Without
it, an importance-sampling estimate of a rare event can have infinite variance
and produce confidence intervals that are confidently wrong -- the classic
failure mode of the technique, and the reason a bare variance-reduction factor
is not on its own evidence of correctness.

Reported alongside every estimate: the effective sample size, the largest
single-sample weight share, and the number of distinct failures hit.  Those are
the diagnostics that reveal a proposal that has collapsed onto one mode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from ..config import Config
from ..dynamics import Dynamics
from ..rollout import simulate
from ..scenario import IDX_MODEL, STATIC_NAMES, ScenarioSpace
from ..utils import batched, kmeans
from .screening import (
    effective_sample_size,
    relevant_coordinates,
    soft_screen,
    temper_weights,
    tilted_dimensions,
)


# --------------------------------------------------------------------------- #
# Proposal
# --------------------------------------------------------------------------- #


@dataclass
class MixtureProposal:
    """Diagonal-Gaussian mixture whose first component is the nominal law."""

    weights: np.ndarray  # (K,) mixture weights, weights[0] is the defensive part
    means: np.ndarray  # (K, d)
    sds: np.ndarray  # (K, d)
    labels: List[str] = field(default_factory=list)
    fit_temperature: float = 1.0  # exponent the fitting weights were tempered by

    @property
    def n_components(self) -> int:
        return int(self.weights.shape[0])

    @property
    def dim(self) -> int:
        return int(self.means.shape[1])

    @property
    def defensive_alpha(self) -> float:
        return float(self.weights[0])

    @property
    def max_weight(self) -> float:
        """Hard upper bound on the likelihood ratio ``p/q``."""
        return 1.0 / max(self.defensive_alpha, 1e-12)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        comp = rng.choice(self.n_components, size=int(n), p=self.weights)
        z = rng.standard_normal((int(n), self.dim))
        return self.means[comp] + self.sds[comp] * z

    def log_q(self, z: np.ndarray) -> np.ndarray:
        """Log density of the mixture, shape ``(n,)``."""
        z = np.atleast_2d(np.asarray(z, dtype=float))
        d = self.dim
        parts = np.empty((self.n_components, z.shape[0]))
        for k in range(self.n_components):
            r = (z - self.means[k]) / self.sds[k]
            parts[k] = (
                -0.5 * np.sum(r * r, axis=1)
                - np.sum(np.log(self.sds[k]))
                - 0.5 * d * np.log(2.0 * np.pi)
                + np.log(self.weights[k])
            )
        return logsumexp(parts, axis=0)

    def describe(self) -> List[Dict[str, object]]:
        return [
            {
                "component": self.labels[k] if k < len(self.labels) else "component_%d" % k,
                "weight": float(self.weights[k]),
                "shift_norm": float(np.linalg.norm(self.means[k])),
                "tilted_dimensions": int(tilted_dimensions(self.means[k]).size),
                "mean_sd": float(np.mean(self.sds[k])),
                "static_shift": [round(float(v), 2) for v in self.means[k][:6]],
            }
            for k in range(self.n_components)
        ]


def fit_modes(
    failures: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
    weights: Optional[np.ndarray] = None,
    n_modes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster discovered failures into mixture components.

    Returns ``(means, sds, mass)``, where ``mass`` is each cluster's share of the
    failure probability.  ``weights`` are the importance weights of the failures
    themselves: a stress-test search does not sample from ``p(z | failure)``, it
    samples from whatever distribution it drifted to, so its failures must be
    reweighted by ``p(z) / q_search(z)`` before they describe the failure region
    under the *nominal* law.  Without that the proposal inherits the search's own
    bias, and the mode it believes dominant may not be the one that does.

    Clustering keeps the component count small while still covering distinct
    modes; each cluster is screened (see :mod:`rsv.estimate.screening`) so that
    only coordinates carrying real signal end up tilted.
    """
    ac, ec = cfg.ast, cfg.estimate
    failures = np.atleast_2d(np.asarray(failures, dtype=float))
    n = failures.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    w = np.maximum(w, 0.0)
    if w.sum() <= 0:
        w = np.ones(n)

    k = int(min(n_modes or ac.n_modes, max(1, n)))
    # Cluster in the subspace the failures actually differ in, on a sample
    # drawn in proportion to the weights -- so component budget is spent on
    # the modes that carry failure probability, not on whichever mode the
    # search happened to visit most.
    coords = relevant_coordinates(failures, w, n_keep=int(ac.cluster_dimensions))
    sub_x = failures[:, coords]
    if n > k:
        pick = rng.choice(n, size=int(min(n, 4000)), replace=True, p=w / w.sum())
        centres, _ = kmeans(sub_x[pick], k, rng, iters=ac.kmeans_iters)
        d2 = np.sum((sub_x[:, None, :] - centres[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1)
    else:
        centres, labels = kmeans(sub_x, k, rng, iters=ac.kmeans_iters)

    means, sds, mass = [], [], []
    for j in range(centres.shape[0]):
        rows = labels == j
        members = failures[rows]
        mw = w[rows]
        if members.shape[0] == 0 or mw.sum() <= 0:
            continue
        pw = mw / mw.sum()
        mean = (pw[:, None] * members).sum(axis=0)
        var = (pw[:, None] * (members - mean) ** 2).sum(axis=0)
        sd = np.sqrt(np.maximum(var, 0.0)) if members.shape[0] > 1 else np.ones(failures.shape[1])
        mean, sd = soft_screen(
            mean,
            sd,
            effective_sample_size(mw),
            c=float(ac.screen_c_proposal),
            sd_floor=float(ec.mode_sd_floor),
            sd_ceiling=float(ec.mode_sd_ceiling),
        )
        means.append(mean)
        sds.append(sd)
        mass.append(float(mw.sum()))
    return np.asarray(means), np.asarray(sds), np.asarray(mass, dtype=float)


def hypothesis_components(
    space: ScenarioSpace, shift: float = 3.5
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """One component per static disturbance channel, pushed into each tail.

    A safety engineer asked to enumerate ways the stop could fail would start by
    taking each thing that can vary -- floor traction, where the obstacle ended
    up, how far out the range finder reads, how long the command takes to land --
    and pushing it to its extreme, in both directions.  These components encode
    exactly that, and they stay in the proposal for every round.

    They exist because searching for modes is not the same as covering them.  A
    cross-entropy restart seeded toward one tail does not have to stay there: if
    an easier failure mechanism is reachable from where it starts, the elite
    selection pulls it away, and the mode it was sent to look for goes
    unrepresented.  That happened here with the lateral obstacle offset -- the
    restart seeded toward a positive offset drifted back to the traction mode,
    leaving the proposal covering only negative offsets and silently losing half
    of that mode's probability.  Keeping a standing component per channel per
    direction makes coverage a property of the construction rather than an
    outcome of the search; the adaptive rounds then measure how much probability
    each one really carries, and the useless ones cost only their small fixed
    share of the samples.

    The ensemble-member channel is excluded: it selects which member of the twin
    governs an episode rather than describing a disturbance, and pushing it to a
    tail just pins one member.
    """
    means: List[np.ndarray] = []
    labels: List[str] = []
    for j in range(space.n_static):
        if j == IDX_MODEL:
            continue
        for sign, word in ((1.0, "high"), (-1.0, "low")):
            m = np.zeros(space.dim)
            m[j] = sign * float(shift)
            means.append(m)
            labels.append("single factor: %s %s" % (STATIC_NAMES[j], word))
    mean_arr = np.asarray(means)
    return mean_arr, np.ones_like(mean_arr), labels


def build_proposal(
    failures: np.ndarray,
    cfg: Config,
    space: ScenarioSpace,
    rng: np.random.Generator,
    log_weights: Optional[np.ndarray] = None,
    labels: Optional[Sequence[str]] = None,
) -> MixtureProposal:
    """Assemble the defensive mixture from the stress testers' findings.

    Component weights follow each mode's share of the failure probability, so
    the modes that dominate get the most samples.  If no failure was ever found
    the proposal degenerates to the nominal law and the estimator reduces to
    naive Monte Carlo -- the right behaviour, since there is nothing to aim at.
    """
    alpha = float(cfg.estimate.defensive_alpha)
    eta = float(cfg.estimate.hypothesis_weight)
    d = space.dim
    nominal_mean = np.zeros((1, d))
    nominal_sd = np.ones((1, d))
    hyp_mean, hyp_sd, hyp_labels = hypothesis_components(
        space, shift=float(cfg.estimate.hypothesis_shift)
    )
    n_hyp = hyp_mean.shape[0]
    if eta <= 0 or n_hyp == 0:
        hyp_mean, hyp_sd, hyp_labels, n_hyp, eta = (
            np.zeros((0, d)),
            np.zeros((0, d)),
            [],
            0,
            0.0,
        )

    failures = (
        np.atleast_2d(np.asarray(failures, dtype=float))
        if failures is not None and np.size(failures)
        else np.zeros((0, d))
    )
    if failures.shape[0] == 0:
        # Nothing found yet: fall back to the nominal law plus the standing
        # single-factor hypotheses, which is still a usable pilot.
        w0 = 1.0 - eta
        return MixtureProposal(
            weights=np.concatenate(([w0], np.full(n_hyp, eta / max(n_hyp, 1)))),
            means=np.concatenate((nominal_mean, hyp_mean), axis=0),
            sds=np.concatenate((nominal_sd, hyp_sd), axis=0),
            labels=["nominal (no fitted failure mode)"] + hyp_labels,
        )

    weights = None
    beta = 1.0
    if log_weights is not None and np.size(log_weights):
        weights, beta = temper_weights(log_weights, cfg.estimate.min_fit_ess_fraction)
    means, sds, mass = fit_modes(failures, cfg, rng, weights=weights)
    # The ensemble-member coordinate indexes *model* uncertainty rather than a
    # disturbance, so the proposal is held at the nominal law there.  Tilting it
    # would quietly reweight the model average -- and keeping it nominal is what
    # lets the per-member estimates below share one set of weights.
    if means.size:
        means[:, IDX_MODEL] = 0.0
        sds[:, IDX_MODEL] = 1.0
    share = mass / mass.sum()
    # Components carrying a negligible share of the failure probability are
    # dropped: they cost samples and, if they were fitted to a stray cluster
    # far out in the tail, they would widen the weights for nothing.
    keep = share >= float(cfg.estimate.min_component_mass)
    if not keep.any():
        keep = share >= share.max()
    means, sds, share = means[keep], sds[keep], share[keep] / share[keep].sum()
    mix = np.concatenate(
        ([alpha], np.full(n_hyp, eta / max(n_hyp, 1)), (1.0 - alpha - eta) * share)
    )
    all_means = np.concatenate((nominal_mean, hyp_mean, means), axis=0)
    all_sds = np.concatenate((nominal_sd, hyp_sd, sds), axis=0)

    names = ["nominal (defensive)"] + list(hyp_labels)
    for j in range(means.shape[0]):
        name = labels[j] if labels is not None and j < len(labels) else "failure mode %d" % (j + 1)
        names.append("%s (%.0f%% of failure mass)" % (name, 100.0 * share[j]))
    return MixtureProposal(
        weights=mix, means=all_means, sds=all_sds, labels=names, fit_temperature=beta
    )


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #


@dataclass
class ISEstimate:
    """An importance-sampling failure-probability estimate."""

    p_hat: float
    n: int
    n_failures: int
    standard_error: float
    ci_low: float
    ci_high: float
    ci_boot_low: float
    ci_boot_high: float
    confidence: float
    relative_error: float
    variance_per_sample: float
    effective_sample_size: float
    max_weight_share: float
    weight_bound: float
    seconds: float
    convergence: Dict[str, List[float]] = field(default_factory=dict)
    proposal: Optional[List[Dict[str, object]]] = None
    failures: Optional[np.ndarray] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "estimator": "importance_sampling",
            "p_hat": self.p_hat,
            "n": self.n,
            "n_failures": self.n_failures,
            "standard_error": self.standard_error,
            "ci": [self.ci_low, self.ci_high],
            "ci_bootstrap": [self.ci_boot_low, self.ci_boot_high],
            "ci_method": "central limit theorem (bounded weights)",
            "confidence": self.confidence,
            "relative_error": self.relative_error,
            "variance_per_sample": self.variance_per_sample,
            "effective_sample_size": self.effective_sample_size,
            "max_weight_share": self.max_weight_share,
            "weight_bound": self.weight_bound,
            "seconds": self.seconds,
            "proposal": self.proposal,
        }


def importance_sample(
    dynamics: Dynamics,
    cfg: Config,
    space: ScenarioSpace,
    proposal: MixtureProposal,
    rng: np.random.Generator,
    n: Optional[int] = None,
    batch: Optional[int] = None,
    collect_failures: bool = False,
    convergence_points: Optional[int] = None,
) -> ISEstimate:
    """Estimate the failure probability under ``q`` and reweight to ``p``."""
    ec = cfg.estimate
    n = int(ec.n_is if n is None else n)
    size = int(ec.batch if batch is None else batch)
    n_points = int(ec.convergence_points if convergence_points is None else convergence_points)
    checkpoints = np.unique(np.geomspace(max(size, 10), n, num=max(n_points, 2)).astype(int))

    t0 = time.perf_counter()
    contributions: List[np.ndarray] = []
    weights_all: List[np.ndarray] = []
    failure_rows: List[np.ndarray] = []
    n_fail = 0

    for lo, hi in batched(n, size):
        z = proposal.sample(rng, hi - lo)
        log_w = space.log_p(z) - proposal.log_q(z)
        w = np.exp(log_w)
        failed = simulate(z, dynamics, cfg, space).failure
        contributions.append(w * failed)
        weights_all.append(w)
        n_fail += int(failed.sum())
        if collect_failures and failed.any():
            failure_rows.append(z[failed])

    h = np.concatenate(contributions)
    w_all = np.concatenate(weights_all)
    seconds = time.perf_counter() - t0

    p_hat = float(h.mean())
    var_h = float(h.var(ddof=1)) if h.size > 1 else 0.0
    se = float(np.sqrt(var_h / h.size))
    crit = float(norm.ppf(0.5 + 0.5 * ec.confidence))
    ci = [max(0.0, p_hat - crit * se), p_hat + crit * se]

    boot_lo, boot_hi = _bootstrap_ci(h, ec.bootstrap, ec.confidence, rng)

    # Convergence trace: the running estimate with its own interval.
    trace = {"n": [], "p": [], "ci_low": [], "ci_high": []}
    for cp in checkpoints:
        sub = h[: int(cp)]
        if sub.size < 2:
            continue
        m = float(sub.mean())
        s = float(np.sqrt(sub.var(ddof=1) / sub.size))
        trace["n"].append(float(cp))
        trace["p"].append(m)
        trace["ci_low"].append(max(0.0, m - crit * s))
        trace["ci_high"].append(m + crit * s)

    ess = float(w_all.sum() ** 2 / max(np.sum(w_all ** 2), 1e-300))
    max_share = float(h.max() / h.sum()) if h.sum() > 0 else 0.0

    return ISEstimate(
        p_hat=p_hat,
        n=int(h.size),
        n_failures=int(n_fail),
        standard_error=se,
        ci_low=float(ci[0]),
        ci_high=float(ci[1]),
        ci_boot_low=float(boot_lo),
        ci_boot_high=float(boot_hi),
        confidence=float(ec.confidence),
        relative_error=float(se / p_hat) if p_hat > 0 else float("inf"),
        variance_per_sample=var_h,
        effective_sample_size=ess,
        max_weight_share=max_share,
        weight_bound=proposal.max_weight,
        seconds=float(seconds),
        convergence=trace,
        proposal=proposal.describe(),
        failures=np.concatenate(failure_rows, axis=0) if failure_rows else None,
    )


def _bootstrap_ci(
    h: np.ndarray, n_boot: int, confidence: float, rng: np.random.Generator
) -> Tuple[float, float]:
    """Percentile bootstrap, which does not assume the mean is near-normal.

    With a rare event most contributions are exactly zero, so the sampling
    distribution of the mean is skewed even at large ``N``; comparing this
    interval against the CLT one is a cheap check that ``N`` is large enough for
    the normal approximation to be usable.
    """
    n_boot = int(max(0, n_boot))
    if n_boot == 0 or h.size == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, h.size, size=(n_boot, h.size)) if h.size <= 20000 else None
    if idx is not None:
        means = h[idx].mean(axis=1)
    else:  # avoid materialising a huge index matrix
        means = np.empty(n_boot)
        for b in range(n_boot):
            means[b] = h[rng.integers(0, h.size, size=h.size)].mean()
    lo = float(np.quantile(means, 0.5 - 0.5 * confidence))
    hi = float(np.quantile(means, 0.5 + 0.5 * confidence))
    return lo, hi


# --------------------------------------------------------------------------- #
# Adaptive importance sampling
# --------------------------------------------------------------------------- #


@dataclass
class AdaptiveISResult:
    """Final estimate plus the refinement history that produced its proposal."""

    estimate: ISEstimate
    proposal: MixtureProposal
    rounds: List[Dict[str, object]]
    adaptation_episodes: int

    def as_dict(self) -> Dict[str, object]:
        out = self.estimate.as_dict()
        out["adaptation_rounds"] = self.rounds
        out["adaptation_episodes"] = self.adaptation_episodes
        return out


def adaptive_importance_sample(
    dynamics: Dynamics,
    cfg: Config,
    space: ScenarioSpace,
    seed_failures: np.ndarray,
    rng: np.random.Generator,
    seed_log_weights: Optional[np.ndarray] = None,
    n_rounds: Optional[int] = None,
    n_per_round: Optional[int] = None,
    n_final: Optional[int] = None,
) -> "AdaptiveISResult":
    """Refine the proposal on properly weighted failures, then estimate.

    The failures handed over by adaptive stress testing are not a sample from
    ``p(z | failure)``: they come from whatever distribution the search drifted
    to, and reweighting them back is unreliable once that distribution has moved
    far -- the effective sample size collapses.  So they are used only to *aim*
    the first proposal.  Each subsequent round draws from a proposal whose
    density is known exactly, so the failures it finds carry exact weights and
    are a genuine weighted sample from the failure region.

    Failures are *accumulated* across rounds and reweighted against the
    deterministic mixture of every proposal used so far,

        w_i = p(z_i) / sum_r (n_r / N) q_r(z_i)

    -- the balance heuristic of Veach and Guibas, as used in adaptive multiple
    importance sampling.  Refitting on a single round's failures instead makes
    the scheme collapse: a round that happens to find few failures produces a
    fit with an effective sample size of one or two, the screening then shrinks
    the proposal back toward nominal, the next round finds fewer failures still,
    and the search dies.  Pooling every round keeps the fitting sample growing
    monotonically.

    The reported estimate comes from a final, independent batch drawn from the
    converged proposal, so no adaptation data enters it and the estimator stays
    unbiased.  Episodes spent adapting are reported separately and charged
    against importance sampling in the cost comparison.
    """
    ec = cfg.estimate
    n_rounds = int(ec.adapt_rounds if n_rounds is None else n_rounds)
    n_per_round = int(ec.adapt_samples if n_per_round is None else n_per_round)
    n_final = int(ec.n_is if n_final is None else n_final)

    proposal = build_proposal(seed_failures, cfg, space, rng, log_weights=seed_log_weights)
    rounds: List[Dict[str, object]] = []
    used: List[MixtureProposal] = []
    drawn: List[int] = []
    pool_z: List[np.ndarray] = []
    spent = 0

    for r in range(n_rounds):
        found_z: List[np.ndarray] = []
        contrib: List[np.ndarray] = []
        for lo, hi in batched(n_per_round, ec.batch):
            z = proposal.sample(rng, hi - lo)
            w = np.exp(space.log_p(z) - proposal.log_q(z))
            failed = simulate(z, dynamics, cfg, space).failure
            contrib.append(w * failed)
            if failed.any():
                found_z.append(z[failed])
        spent += n_per_round
        used.append(proposal)
        drawn.append(n_per_round)
        if found_z:
            pool_z.append(np.concatenate(found_z, axis=0))

        h = np.concatenate(contrib)
        n_found = int(sum(a.shape[0] for a in found_z))
        pooled = int(sum(a.shape[0] for a in pool_z))
        info: Dict[str, object] = {
            "round": r + 1,
            "episodes": n_per_round,
            "failures_found": n_found,
            "hit_rate": n_found / float(n_per_round),
            "p_hat_round": float(h.mean()),
            "pooled_failures": pooled,
            "components": proposal.n_components,
        }

        if pooled >= max(8, proposal.n_components):
            zf = np.concatenate(pool_z, axis=0)
            lwf = _balance_heuristic_log_weights(zf, space, used, drawn)
            wf = np.exp(lwf - lwf.max())
            info["weight_ess"] = float(wf.sum() ** 2 / max(np.sum(wf ** 2), 1e-300))
            proposal = build_proposal(zf, cfg, space, rng, log_weights=lwf)
            info["fit_temperature"] = float(proposal.fit_temperature)
        else:
            info["note"] = "too few failures pooled to refit; proposal kept"
        rounds.append(info)

    estimate = importance_sample(dynamics, cfg, space, proposal, rng, n=n_final)
    return AdaptiveISResult(
        estimate=estimate,
        proposal=proposal,
        rounds=rounds,
        adaptation_episodes=spent,
    )


def _balance_heuristic_log_weights(
    z: np.ndarray,
    space: ScenarioSpace,
    proposals: Sequence["MixtureProposal"],
    counts: Sequence[int],
) -> np.ndarray:
    """``log p(z) - log sum_r (n_r / N) q_r(z)`` for samples pooled across rounds."""
    total = float(sum(counts))
    parts = np.empty((len(proposals), z.shape[0]))
    for r, prop in enumerate(proposals):
        parts[r] = np.log(counts[r] / total) + prop.log_q(z)
    return space.log_p(z) - logsumexp(parts, axis=0)


# --------------------------------------------------------------------------- #
# Coverage diagnostics
# --------------------------------------------------------------------------- #


@dataclass
class CoverageReport:
    """How much of the failure region the proposal actually reaches.

    The dangerous way for importance sampling to fail is not a wide interval --
    it is a *narrow* one around the wrong number.  If the proposal misses a mode,
    the only samples that would reveal the omission are the rare draws from the
    defensive component that happen to land in it: a handful per run at best, and
    usually none.  The estimator then reports a tight confidence interval that
    excludes the truth, and the usual diagnostics (effective sample size, maximum
    weight share) all look healthy, because the pathological samples are exactly
    the ones that did not occur.
    """

    n_probe_failures: int
    uncovered_fraction: float
    uncovered_fraction_ci: List[float]
    median_coverage_ratio: float
    min_coverage_ratio: float
    threshold: float
    verdict: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_probe_failures": self.n_probe_failures,
            "uncovered_failure_mass_fraction": self.uncovered_fraction,
            "uncovered_fraction_ci": self.uncovered_fraction_ci,
            "median_coverage_ratio": self.median_coverage_ratio,
            "min_coverage_ratio": self.min_coverage_ratio,
            "threshold": self.threshold,
            "verdict": self.verdict,
        }


def coverage_diagnostic(
    probe_failures: Optional[np.ndarray],
    proposal: MixtureProposal,
    space: ScenarioSpace,
    threshold: float = 2.0,
    confidence: float = 0.95,
) -> CoverageReport:
    """Estimate the share of failure probability the proposal does not reach.

    ``probe_failures`` are failures collected by *naive* Monte Carlo, so they are
    an unbiased sample from the failure region under the nominal law -- exactly
    the sample needed to audit the proposal.  For each one, the coverage ratio

        r(z) = q(z) / (alpha * p(z))   >=  1

    says how much better the proposal is at producing that failure than its
    defensive component alone.  ``r ~ 1`` means the tilted components contribute
    nothing there: that failure is reached only by luck, and its probability
    mass enters the estimate only through rare, large-weight draws.  The fraction
    of probe failures with ``r < threshold`` estimates the share of failure mass
    in that condition, with a Clopper-Pearson interval for the small counts a
    naive run provides.

    A few percent is fine.  Tens of percent means the headline interval is not
    to be believed, however tight it looks, and the stress testing has to go back
    and find the missing mode.
    """
    from .monte_carlo import clopper_pearson

    if probe_failures is None or np.size(probe_failures) == 0:
        return CoverageReport(
            n_probe_failures=0,
            uncovered_fraction=float("nan"),
            uncovered_fraction_ci=[float("nan"), float("nan")],
            median_coverage_ratio=float("nan"),
            min_coverage_ratio=float("nan"),
            threshold=float(threshold),
            verdict="no probe failures: coverage could not be audited",
        )

    z = np.atleast_2d(np.asarray(probe_failures, dtype=float))
    alpha = max(proposal.defensive_alpha, 1e-12)
    ratio = np.exp(proposal.log_q(z) - space.log_p(z)) / alpha
    uncovered = ratio < float(threshold)
    k, n = int(uncovered.sum()), int(z.shape[0])
    frac = k / n
    ci = clopper_pearson(k, n, confidence)

    if frac <= 0.05:
        verdict = "good: the proposal reaches essentially all of the failure region"
    elif frac <= 0.20:
        verdict = "marginal: some failure mass is reached only by the defensive component"
    else:
        verdict = "poor: a failure mode is missing from the proposal; the interval understates the error"

    return CoverageReport(
        n_probe_failures=n,
        uncovered_fraction=float(frac),
        uncovered_fraction_ci=[float(ci[0]), float(ci[1])],
        median_coverage_ratio=float(np.median(ratio)),
        min_coverage_ratio=float(ratio.min()),
        threshold=float(threshold),
        verdict=verdict,
    )


@dataclass
class ModelUncertainty:
    """How much of the estimate's uncertainty is *not* sampling error.

    The rollout picks an ensemble member per episode, so the headline estimate
    already averages over model uncertainty -- but averaging hides it.  These
    are the same estimate computed conditional on each member in turn, which
    separates the two questions a safety case has to keep apart: how precisely
    has the integral been computed, and how much would the answer change if the
    dynamics were a little different.

    The spread here is routinely far wider than the confidence interval.  A
    rare-event probability is a steep function of the dynamics -- a few per cent
    on the achieved deceleration moves the stopping distance, the margin, and
    then the tail probability -- so a tight interval on a learned twin is a
    statement about the arithmetic, not about the robot.
    """

    per_member: List[float]
    mean: float
    spread_ratio: float  # max / min across members
    relative_spread: float  # (max - min) / mean
    n_per_member: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "per_member_p": self.per_member,
            "mean": self.mean,
            "max_over_min": self.spread_ratio,
            "relative_spread": self.relative_spread,
            "n_per_member": self.n_per_member,
        }


def per_member_estimates(
    dynamics: Dynamics,
    cfg: Config,
    space: ScenarioSpace,
    proposal: MixtureProposal,
    rng: np.random.Generator,
    n_models: int,
    n: Optional[int] = None,
) -> ModelUncertainty:
    """Importance-sampling estimate conditional on each ensemble member.

    One set of draws is reused for every member.  Because the proposal is held
    at the nominal law on the member coordinate, that coordinate contributes the
    same factor to ``p`` and to ``q``; overwriting it therefore leaves the
    weights exactly correct, and the conditional estimates average back to the
    unconditional one.
    """
    from scipy.special import ndtri

    n = int(n if n is not None else max(2000, cfg.estimate.n_is // 2))
    n_models = int(max(1, n_models))
    totals = np.zeros(n_models)

    for lo, hi in batched(n, cfg.estimate.batch):
        z = proposal.sample(rng, hi - lo)
        w = np.exp(space.log_p(z) - proposal.log_q(z))
        for m in range(n_models):
            zm = z.copy()
            zm[:, IDX_MODEL] = float(ndtri((m + 0.5) / n_models))
            totals[m] += float((w * simulate(zm, dynamics, cfg, space).failure).sum())

    per_member = totals / float(n)
    mean = float(per_member.mean())
    lo_p = float(max(per_member.min(), 1e-300))
    return ModelUncertainty(
        per_member=[float(v) for v in per_member],
        mean=mean,
        spread_ratio=float(per_member.max() / lo_p),
        relative_spread=float((per_member.max() - per_member.min()) / mean) if mean > 0 else float("nan"),
        n_per_member=n,
    )
