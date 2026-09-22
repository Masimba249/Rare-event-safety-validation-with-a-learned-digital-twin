"""Shared fixtures: small, fast configurations for the test suite."""

from __future__ import annotations

import pytest

from rsv.config import Config


@pytest.fixture
def tiny_cfg() -> Config:
    """A configuration small enough to run many times in a test session."""
    cfg = Config()
    cfg.scenario.horizon = 60
    cfg.data.n_missions = 8
    cfg.data.mission_steps = 90
    cfg.twin.n_members = 2
    cfg.twin.epochs = 12
    cfg.twin.hidden = [16, 16]
    cfg.estimate.batch = 500
    cfg.estimate.n_mc = 1000
    cfg.estimate.n_is = 1000
    cfg.estimate.bootstrap = 50
    cfg.estimate.convergence_points = 3
    cfg.estimate.adapt_rounds = 1
    cfg.estimate.adapt_samples = 500
    cfg.estimate.plant_reference_n = 2000
    cfg.ast.mcts_iterations = 20
    cfg.ast.mcts_rollout_batch = 8
    cfg.ast.cem_restarts = 2
    cfg.ast.cem_iterations = 3
    cfg.ast.cem_population = 120
    cfg.sim2real.n_replay = 4
    cfg.sim2real.n_repeats = 2
    cfg.sim2real.n_nominal_replay = 3
    cfg.sim2real.refit_epochs = 12
    return cfg


@pytest.fixture
def default_cfg() -> Config:
    return Config()
