"""Adaptive stress testing: does the search actually find the rare failures?"""

from __future__ import annotations

import numpy as np
import pytest

from rsv.ast import AdaptiveStressTest, cem_search, multi_start_cem
from rsv.plant import Plant
from rsv.rollout import simulate
from rsv.scenario import IDX_OBS_Y, ScenarioSpace
from rsv.utils import rng_for


@pytest.fixture
def setup(default_cfg):
    cfg = default_cfg
    cfg.ast.mcts_iterations = 300
    cfg.ast.mcts_rollout_batch = 16
    cfg.ast.cem_iterations = 22
    cfg.ast.cem_population = 1200
    cfg.ast.cem_restarts = 3
    cfg.estimate.batch = 2000
    space = ScenarioSpace(cfg.scenario, n_models=1)
    return cfg, space, Plant(cfg.plant, cfg.scenario.dt)


def test_cross_entropy_finds_failures_naive_sampling_would_not(setup):
    cfg, space, plant = setup
    run = cem_search(plant, cfg, space, rng_for(81, "cem"))
    assert run.found_failure
    # The nominal failure rate is ~2e-4; the search should be orders of magnitude
    # above it, or it is not buying anything over sampling at random.
    assert run.hit_rate > 0.02
    assert run.best_robustness < 0.0


def test_cross_entropy_keeps_the_tilt_concentrated(setup):
    """Screening must keep the shift on the coordinates that matter.

    The soft threshold leaves a residual in many coordinates rather than exactly
    zeroing them, so counting non-zeros is not the test.  What matters is that
    the *energy* of the shift sits on a handful of interpretable coordinates --
    if it were spread evenly over 566, ``p/q`` would be unusable.
    """
    cfg, space, plant = setup
    run = cem_search(plant, cfg, space, rng_for(82, "cem"))
    energy = run.mean ** 2
    top = np.sort(energy)[::-1][:24].sum()
    assert top / max(energy.sum(), 1e-12) > 0.8
    assert run.failures.shape[1] == space.dim
    assert run.failure_logq.shape[0] == run.failures.shape[0]


def test_reported_failures_really_fail(setup):
    """A found scenario must reproduce when replayed through the simulator."""
    cfg, space, plant = setup
    run = cem_search(plant, cfg, space, rng_for(83, "cem"))
    replay = simulate(run.failures[:20], plant, cfg, space)
    np.testing.assert_array_equal(replay.failure, np.ones(replay.batch, dtype=bool))


def test_reported_failures_replay_exactly(setup):
    """The returned latent must reproduce the trajectory that produced it.

    This is the test that catches a tree whose level-to-timestep bookkeeping has
    drifted: such a search still returns collisions, but the disturbance vector
    it hands back describes a different episode from the one it simulated, and
    every downstream stage -- proposal fitting, hardware replay -- then works on
    scenarios that were never actually evaluated.
    """
    cfg, space, plant = setup
    result = AdaptiveStressTest(plant, cfg, space, rng_for(88, "mcts")).search()
    assert result.n_failures > 0
    replay = simulate(result.failures, plant, cfg, space)
    np.testing.assert_allclose(replay.robustness, result.robustness, atol=1e-12)


def test_early_retirement_does_not_change_reported_failures(setup):
    """Robustness is the margin at first contact, so retiring early is free."""
    cfg, space, plant = setup
    run = cem_search(plant, cfg, space, rng_for(89, "cem"))
    z = run.failures[:30]
    a = simulate(z, plant, cfg, space, compact=True)
    b = simulate(z, plant, cfg, space, compact=False)
    np.testing.assert_allclose(a.robustness, b.robustness, atol=0.0)


def test_restarts_cover_both_directions_of_each_channel(setup):
    """A mode can live in either tail; one-sided seeding would half-miss it."""
    cfg, space, plant = setup
    cfg.ast.cem_restarts = 13
    runs = multi_start_cem(plant, cfg, space, rng_for(84, "cem"))
    assert len(runs) == 13
    found = [r for r in runs if r.found_failure]
    assert len(found) >= 6
    offsets = np.concatenate([r.failures[:, IDX_OBS_Y] for r in found])
    assert (offsets > 2.0).any() and (offsets < -2.0).any()


def test_mcts_finds_likely_failures(setup):
    cfg, space, plant = setup
    ast = AdaptiveStressTest(plant, cfg, space, rng_for(85, "mcts"))
    result = ast.search()

    assert result.n_failures > 0
    assert result.episodes > 0
    assert result.tree_nodes > 1
    # The whole point of the AST reward: failures that are likely, not just any.
    typical = -0.5 * space.dim * (1.0 + np.log(2.0 * np.pi))
    assert result.best_log_p > typical
    np.testing.assert_array_equal(
        simulate(result.failures, plant, cfg, space).failure,
        np.ones(result.n_failures, dtype=bool),
    )


def test_mcts_failures_are_ranked_by_likelihood(setup):
    cfg, space, plant = setup
    result = AdaptiveStressTest(plant, cfg, space, rng_for(86, "mcts")).search()
    assert np.all(np.diff(result.log_p) <= 1e-9)
    assert result.best_log_p == pytest.approx(result.log_p[0])


def test_search_recovers_physically_sensible_scenarios(setup):
    """Found failures should look like the two mechanisms, not like noise."""
    cfg, space, plant = setup
    runs = multi_start_cem(plant, cfg, space, rng_for(87, "cem"))
    failures = np.concatenate([r.failures for r in runs if r.found_failure], axis=0)
    decoded = space.decode(failures)

    slippery = decoded.mu < cfg.scenario.mu_mean - cfg.scenario.mu_sd
    offset = np.abs(decoded.obstacle_y) > 2.0 * cfg.scenario.obstacle_y_sd
    # Essentially every failure is explained by one mechanism or the other.
    assert float((slippery | offset).mean()) > 0.9
