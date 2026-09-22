"""The statistical core.

These are the tests that matter most: a bug in the importance weights does not
crash anything, it just returns a confident wrong number.  So the estimator
identities are checked against cases whose answers are known in closed form,
independently of the robot simulator.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from rsv.estimate.compare import compare, episodes_for_relative_error
from rsv.estimate.importance import MixtureProposal, coverage_diagnostic, hypothesis_components
from rsv.estimate.monte_carlo import (
    MCEstimate,
    clopper_pearson,
    samples_for_relative_error,
    wilson_interval,
    zero_failure_bound,
)
from rsv.estimate.screening import (
    effective_sample_size,
    relevant_coordinates,
    soft_screen,
    temper_weights,
)
from rsv.scenario import ScenarioSpace
from rsv.utils import rng_for


@pytest.fixture
def small_space(default_cfg):
    cfg = default_cfg
    cfg.scenario.horizon = 10  # 46 latent dimensions: quick but not trivial
    return ScenarioSpace(cfg.scenario, n_models=1)


def _tilted_proposal(space, dim, shift, alpha=0.2):
    means = np.zeros((2, space.dim))
    means[1, dim] = shift
    return MixtureProposal(
        weights=np.array([alpha, 1.0 - alpha]),
        means=means,
        sds=np.ones((2, space.dim)),
        labels=["nominal", "tilted"],
    )


# --------------------------------------------------------------------------- #
# The identity every importance-sampling scheme must satisfy
# --------------------------------------------------------------------------- #


def test_weights_average_to_one(small_space):
    """``E_q[p/q] = 1`` -- the single identity that catches most weight bugs."""
    prop = _tilted_proposal(small_space, dim=0, shift=3.0)
    z = prop.sample(rng_for(21, "is"), 60000)
    w = np.exp(small_space.log_p(z) - prop.log_q(z))
    assert float(w.mean()) == pytest.approx(1.0, abs=0.02)


def test_mixture_density_matches_a_direct_computation(small_space):
    """``log_q`` must be the log of the actual mixture, not of one component."""
    prop = _tilted_proposal(small_space, dim=1, shift=2.0, alpha=0.3)
    z = prop.sample(rng_for(22, "is"), 16)
    direct = np.log(
        prop.weights[0] * np.exp(small_space.log_p_diag_gaussian(z, prop.means[0], prop.sds[0]))
        + prop.weights[1] * np.exp(small_space.log_p_diag_gaussian(z, prop.means[1], prop.sds[1]))
    )
    np.testing.assert_allclose(prop.log_q(z), direct, rtol=1e-9)


def test_importance_sampling_recovers_a_known_tail_probability(small_space):
    """Estimate ``P(z_0 > 4) = 3.17e-5`` -- a rare event with an exact answer.

    This exercises the whole estimator except the robot: sampling from the
    mixture, the likelihood ratio, and the mean of ``w * 1[event]``.
    """
    threshold = 4.0
    truth = float(norm.sf(threshold))
    prop = _tilted_proposal(small_space, dim=0, shift=threshold)

    z = prop.sample(rng_for(23, "is"), 40000)
    w = np.exp(small_space.log_p(z) - prop.log_q(z))
    h = w * (z[:, 0] > threshold)
    estimate = float(h.mean())
    se = float(np.sqrt(h.var(ddof=1) / h.size))

    assert estimate == pytest.approx(truth, rel=0.06)
    assert abs(estimate - truth) < 3 * se  # the reported interval is honest
    assert se / estimate < 0.03


def test_importance_sampling_beats_naive_sampling_on_that_problem(small_space):
    """...and does so by a margin that matches the variance ratio."""
    threshold = 4.0
    truth = float(norm.sf(threshold))
    prop = _tilted_proposal(small_space, dim=0, shift=threshold)

    z = prop.sample(rng_for(24, "is"), 40000)
    w = np.exp(small_space.log_p(z) - prop.log_q(z))
    var_is = float((w * (z[:, 0] > threshold)).var(ddof=1))
    var_mc = truth * (1.0 - truth)
    assert var_mc / var_is > 50.0


def test_defensive_component_bounds_the_weights(small_space):
    """The bound ``w <= 1/alpha`` is what makes the variance finite."""
    alpha = 0.25
    prop = _tilted_proposal(small_space, dim=0, shift=5.0, alpha=alpha)
    z = prop.sample(rng_for(25, "is"), 20000)
    w = np.exp(small_space.log_p(z) - prop.log_q(z))
    assert w.max() <= 1.0 / alpha + 1e-9
    assert prop.max_weight == pytest.approx(1.0 / alpha)


def test_unbiasedness_is_not_disturbed_by_useless_components(small_space):
    """Components aimed at nothing cost samples but must not bias the answer."""
    threshold = 4.0
    truth = float(norm.sf(threshold))
    means = np.zeros((4, small_space.dim))
    means[1, 0] = threshold
    means[2, 3] = 4.0  # points at a region the event never visits
    means[3, 5] = -4.0
    prop = MixtureProposal(
        weights=np.array([0.2, 0.4, 0.2, 0.2]),
        means=means,
        sds=np.ones((4, small_space.dim)),
        labels=["nominal", "useful", "useless", "useless"],
    )
    z = prop.sample(rng_for(26, "is"), 60000)
    w = np.exp(small_space.log_p(z) - prop.log_q(z))
    estimate = float((w * (z[:, 0] > threshold)).mean())
    assert estimate == pytest.approx(truth, rel=0.10)


# --------------------------------------------------------------------------- #
# Coverage auditing
# --------------------------------------------------------------------------- #


def test_coverage_diagnostic_flags_an_uncovered_mode(small_space):
    """A proposal that covers only one side of a symmetric event must be caught."""
    prop = _tilted_proposal(small_space, dim=0, shift=4.0)
    rng = rng_for(27, "is")

    covered = np.zeros((200, small_space.dim))
    covered[:, 0] = 4.2 + 0.1 * rng.standard_normal(200)
    mirrored = covered.copy()
    mirrored[:, 0] *= -1.0  # the mode the proposal does not reach

    good = coverage_diagnostic(covered, prop, small_space)
    bad = coverage_diagnostic(np.vstack([covered, mirrored]), prop, small_space)

    assert good.uncovered_fraction < 0.05
    assert good.verdict.startswith("good")
    assert bad.uncovered_fraction == pytest.approx(0.5, abs=0.05)
    assert bad.verdict.startswith("poor")


def test_coverage_diagnostic_handles_no_probe_failures(small_space):
    prop = _tilted_proposal(small_space, dim=0, shift=3.0)
    rep = coverage_diagnostic(None, prop, small_space)
    assert rep.n_probe_failures == 0
    assert "could not be audited" in rep.verdict


# --------------------------------------------------------------------------- #
# Screening and tempering
# --------------------------------------------------------------------------- #


def test_screening_zeroes_noise_and_keeps_signal():
    rng = rng_for(28, "misc")
    n = 400
    mean = np.zeros(50)
    mean[0], mean[1] = 3.0, -2.5  # real shifts
    sd = np.ones(50)
    noisy_mean = mean + rng.normal(0.0, 1.0 / np.sqrt(n), size=50)

    screened, screened_sd = soft_screen(noisy_mean, sd, n, c=3.0)
    assert abs(screened[0] - 3.0) < 0.3
    assert abs(screened[1] + 2.5) < 0.3
    assert np.all(screened[2:] == 0.0)
    assert np.all(screened_sd[2:] == 1.0)


def test_screening_keeps_the_likelihood_ratio_finite(small_space):
    """Tilting every coordinate is what makes p/q underflow; screening fixes it."""
    rng = rng_for(29, "misc")
    d = small_space.dim
    raw_mean = rng.normal(0.0, 0.15, size=d)  # pure noise in every coordinate
    raw_sd = np.full(d, 0.45)

    unscreened = MixtureProposal(
        weights=np.array([1.0]), means=raw_mean[None, :], sds=raw_sd[None, :]
    )
    mean_s, sd_s = soft_screen(raw_mean, raw_sd, 50, c=3.0)
    screened = MixtureProposal(weights=np.array([1.0]), means=mean_s[None, :], sds=sd_s[None, :])

    z = unscreened.sample(rng, 64)
    assert np.exp(small_space.log_p(z) - unscreened.log_q(z)).max() < 1e-3
    z2 = screened.sample(rng, 64)
    w2 = np.exp(small_space.log_p(z2) - screened.log_q(z2))
    assert 0.1 < float(w2.mean()) < 10.0


def test_tempering_restores_a_usable_effective_sample_size():
    log_w = np.concatenate([np.zeros(999), [200.0]])  # one point holds everything
    raw = np.exp(log_w - log_w.max())
    assert effective_sample_size(raw) < 1.1

    w, beta = temper_weights(log_w, min_ess_fraction=0.25)
    assert 0.0 <= beta < 1.0
    assert effective_sample_size(w) >= 0.25 * w.size - 1


def test_tempering_leaves_healthy_weights_alone():
    w, beta = temper_weights(np.zeros(500), min_ess_fraction=0.25)
    assert beta == 1.0
    assert effective_sample_size(w) == pytest.approx(500.0)


def test_relevant_coordinates_finds_a_variance_only_mode():
    """A symmetric mode has no mean shift; only the variance term reveals it."""
    rng = rng_for(30, "misc")
    x = rng.standard_normal((800, 40))
    sign = rng.choice([-1.0, 1.0], size=800)
    x[:, 7] = 4.0 * sign  # symmetric: mean 0, variance 16
    x[:, 19] += 2.5  # ordinary mean shift
    coords = relevant_coordinates(x, np.ones(800), n_keep=4)
    assert 7 in coords
    assert 19 in coords


# --------------------------------------------------------------------------- #
# Interval arithmetic
# --------------------------------------------------------------------------- #


def test_clopper_pearson_brackets_the_point_estimate():
    lo, hi = clopper_pearson(7, 10000, 0.95)
    assert lo < 7 / 10000 < hi
    assert clopper_pearson(0, 1000)[0] == 0.0


def test_zero_failure_bound_matches_the_rule_of_three():
    bound = zero_failure_bound(1000, 0.95)
    assert bound == pytest.approx(3.0 / 1000, rel=0.01)


def test_wilson_stays_inside_the_unit_interval():
    lo, hi = wilson_interval(0, 500)
    assert lo >= 0.0 and hi <= 1.0


def test_sample_size_scales_inversely_with_probability():
    assert samples_for_relative_error(1e-4, 0.1) == pytest.approx(
        10 * samples_for_relative_error(1e-3, 0.1), rel=0.01
    )


def test_comparison_reports_the_variance_ratio():
    mc = MCEstimate(
        p_hat=1e-4, n=100000, n_failures=10, standard_error=3.16e-5,
        ci_low=5e-5, ci_high=1.8e-4, ci_method="Clopper-Pearson", confidence=0.95,
        relative_error=0.32, seconds=1.0,
    )

    class FakeIS:
        p_hat = 1.0e-4
        n = 10000
        standard_error = 2.0e-6
        ci_low = 9.6e-5
        ci_high = 1.04e-4
        variance_per_sample = 4.0e-8
        seconds = 0.5

    cmp = compare(mc, FakeIS(), 0.1)
    assert cmp.variance_reduction_factor == pytest.approx(1e-4 * (1 - 1e-4) / 4e-8, rel=1e-6)
    assert cmp.episodes_for_target_is < cmp.episodes_for_target_mc
    assert cmp.intervals_overlap is True


def test_episodes_for_relative_error_is_consistent():
    assert episodes_for_relative_error(1e-4, 1e-2, 0.1) == pytest.approx(1e-4 / (0.01 * 1e-4), rel=1e-9)


def test_hypothesis_components_cover_both_tails(small_space):
    means, sds, labels = hypothesis_components(small_space, shift=3.5)
    assert means.shape[0] == 2 * (small_space.n_static - 1)  # both signs, no member channel
    assert np.allclose(sds, 1.0)
    for j in range(small_space.n_static - 1):
        column = means[:, j]
        assert column.max() == pytest.approx(3.5)
        assert column.min() == pytest.approx(-3.5)
    assert any("high" in s for s in labels) and any("low" in s for s in labels)


def test_per_member_estimates_average_to_the_unconditional_one(default_cfg):
    """Conditioning on each ensemble member must reproduce the overall estimate.

    The identity holds because members are equiprobable and the proposal is kept
    at the nominal law on the member coordinate, so overwriting that coordinate
    leaves the importance weights untouched.  A regression here would mean the
    reported model-uncertainty band describes a different quantity from the
    headline estimate.
    """
    from rsv.dynamics import Dynamics
    from rsv.estimate.importance import per_member_estimates

    n_models = 4

    class _MemberDependent(Dynamics):
        """A 'twin' whose members disagree about braking, and only about that.

        Acceleration is identical for all of them, so the robot always reaches
        the obstacle the same way; the weakest-braking member then collides and
        the strongest stops, which is exactly the disagreement the model
        uncertainty band is meant to expose.
        """

        def step(self, v, omega, v_cmd, omega_cmd, mu, eps, model_index=None):
            idx = np.zeros_like(v, dtype=int) if model_index is None else np.asarray(model_index)
            brake_rate = 0.020 + 0.020 * idx
            rate = np.where(v_cmd >= v, 0.30, brake_rate)
            return (v_cmd - v) * rate, np.zeros_like(v)

    cfg = default_cfg
    cfg.estimate.batch = 1200
    space = ScenarioSpace(cfg.scenario, n_models=n_models)
    prop = MixtureProposal(
        weights=np.array([1.0]),
        means=np.zeros((1, space.dim)),
        sds=np.ones((1, space.dim)),
    )

    mu = per_member_estimates(
        _MemberDependent(), cfg, space, prop, rng_for(31, "is"), n_models=n_models, n=2400
    )

    assert len(mu.per_member) == n_models
    assert mu.mean == pytest.approx(float(np.mean(mu.per_member)))
    # The weakest braker must fail at least as often as the strongest.
    assert mu.per_member[0] >= mu.per_member[-1]
    assert mu.per_member[0] > 0.5  # it cannot stop at all, so it always collides
    assert mu.spread_ratio > 1.0


def test_per_member_estimates_agree_with_the_unconditional_estimate(default_cfg):
    """The per-member mean must match a plain run over the same proposal."""
    from rsv.dynamics import Dynamics
    from rsv.estimate.importance import importance_sample, per_member_estimates

    n_models = 4

    class _Coin(Dynamics):
        """Members differ enough that the average is not a degenerate check."""

        def step(self, v, omega, v_cmd, omega_cmd, mu, eps, model_index=None):
            idx = np.zeros_like(v, dtype=int) if model_index is None else np.asarray(model_index)
            rate = np.where(v_cmd >= v, 0.30, 0.020 + 0.030 * idx)
            return (v_cmd - v) * rate, np.zeros_like(v)

    cfg = default_cfg
    cfg.estimate.batch = 1200
    cfg.estimate.bootstrap = 0
    space = ScenarioSpace(cfg.scenario, n_models=n_models)
    prop = MixtureProposal(
        weights=np.array([1.0]),
        means=np.zeros((1, space.dim)),
        sds=np.ones((1, space.dim)),
    )
    dyn = _Coin()

    direct = importance_sample(dyn, cfg, space, prop, rng_for(32, "is"), n=6000)
    banded = per_member_estimates(
        dyn, cfg, space, prop, rng_for(33, "is"), n_models=n_models, n=6000
    )
    # Independent draws, so allow sampling error but not a systematic gap.
    assert banded.mean == pytest.approx(direct.p_hat, rel=0.12)
