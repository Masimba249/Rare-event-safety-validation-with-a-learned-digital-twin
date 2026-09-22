"""The simulator: batching, early retirement, and the two access paths.

``simulate`` and ``EpisodeSimulator`` share a step body but are driven
differently, and the batch path retires finished episodes early for speed.  Both
optimisations must be invisible in the results, so they are pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from rsv.plant import Plant
from rsv.rollout import EpisodeSimulator, simulate
from rsv.scenario import IDX_MU, IDX_OBS_Y, ScenarioSpace
from rsv.utils import rng_for


@pytest.fixture
def setup(default_cfg):
    cfg = default_cfg
    space = ScenarioSpace(cfg.scenario, n_models=1)
    plant = Plant(cfg.plant, cfg.scenario.dt)
    return cfg, space, plant


def test_batching_does_not_change_results(setup):
    """One episode at a time must equal the whole batch in lock-step."""
    cfg, space, plant = setup
    z = space.sample_nominal(rng_for(11, "misc"), 12)
    batched = simulate(z, plant, cfg, space)
    singles = [simulate(z[i : i + 1], plant, cfg, space).robustness[0] for i in range(12)]
    np.testing.assert_allclose(batched.robustness, singles, rtol=1e-12, atol=1e-12)


def test_early_retirement_does_not_change_results(setup):
    """Retiring settled episodes is an optimisation, not a modelling choice."""
    cfg, space, plant = setup
    z = space.sample_nominal(rng_for(12, "misc"), 40)
    with_compaction = simulate(z, plant, cfg, space, compact=True)
    without = simulate(z, plant, cfg, space, compact=False)
    np.testing.assert_allclose(with_compaction.robustness, without.robustness, atol=1e-12)
    np.testing.assert_array_equal(with_compaction.failure, without.failure)
    np.testing.assert_array_equal(with_compaction.detect_step, without.detect_step)


def test_incremental_simulator_matches_the_batch_path(setup):
    """The MCTS stepper must agree with the batch driver step for step."""
    cfg, space, plant = setup
    z = space.sample_nominal(rng_for(13, "misc"), 6)
    reference = simulate(z, plant, cfg, space, compact=False)

    sim = EpisodeSimulator(cfg, plant)
    state = sim.init(space.decode(z))
    sim.run(state)
    np.testing.assert_allclose(state.robustness, reference.robustness, atol=1e-12)


def test_resuming_from_a_copied_state_matches_running_straight_through(setup):
    """A tree node resumes an episode; that must equal never having stopped."""
    cfg, space, plant = setup
    z = space.sample_nominal(rng_for(14, "misc"), 4)
    sim = EpisodeSimulator(cfg, plant)

    straight = sim.init(space.decode(z))
    sim.run(straight)

    paused = sim.init(space.decode(z))
    sim.run(paused, until=25)
    resumed = paused.copy()
    sim.run(resumed)
    np.testing.assert_allclose(resumed.robustness, straight.robustness, atol=1e-12)


def test_repeat_tiles_a_state_without_aliasing(setup):
    """Batched rollouts from one node must not share mutable state."""
    cfg, space, plant = setup
    z = space.sample_nominal(rng_for(15, "misc"), 1)
    sim = EpisodeSimulator(cfg, plant)
    state = sim.init(space.decode(z))
    sim.run(state, until=10)

    tiled = state.repeat(5)
    assert tiled.batch == 5
    tiled.v[0] = 99.0
    assert state.v[0] != 99.0
    np.testing.assert_allclose(tiled.v[1:], state.v[0])


def test_nominal_episodes_stop_safely(setup):
    """The safety function works in ordinary conditions -- otherwise the whole
    rare-event framing would be wrong."""
    cfg, space, plant = setup
    z = space.sample_nominal(rng_for(16, "misc"), 500)
    res = simulate(z, plant, cfg, space)
    assert res.failure.mean() < 0.01
    assert np.median(res.robustness) > 0.05
    assert (res.detect_step >= 0).mean() > 0.95  # the obstacle is normally seen


def test_slippery_floor_causes_a_late_stop(setup):
    """Forcing the traction disturbance into its tail must produce collisions."""
    cfg, space, plant = setup
    z = np.zeros((1, space.dim))
    z[0, IDX_MU] = -20.0  # clipped to mu_min: almost no grip at all
    res = simulate(z, plant, cfg, space)
    assert bool(res.failure[0])
    assert res.detect_step[0] >= 0  # it saw the obstacle; it could not stop


def test_lateral_offset_causes_a_missed_detection(setup):
    """The second failure mode: inside the body's reach, outside the beam."""
    cfg, space, plant = setup
    gap = cfg.scenario.robot_radius + cfg.scenario.obstacle_radius
    z = np.zeros((1, space.dim))
    z[0, IDX_OBS_Y] = (0.99 * gap) / cfg.scenario.obstacle_y_sd
    res = simulate(z, plant, cfg, space)
    assert bool(res.failure[0])
    assert res.detect_step[0] == -1  # never triggered the stop
    assert res.impact_speed[0] > 0.5 * cfg.controller.v_nominal


def test_robustness_is_the_minimum_margin(setup):
    """The reported robustness must equal the minimum of the recorded margin."""
    cfg, space, plant = setup
    z = space.sample_nominal(rng_for(17, "misc"), 5)
    res = simulate(z, plant, cfg, space, record=True, compact=False)
    np.testing.assert_allclose(res.robustness, res.traces.margin.min(axis=1), atol=1e-12)
