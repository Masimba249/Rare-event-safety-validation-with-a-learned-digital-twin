"""The disturbance space: density, decoding, and the discrete channels.

Everything downstream is a statement about ``p(z)``, so if these transforms are
wrong the importance weights are wrong and no amount of sampling will notice.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from rsv.scenario import (
    CH_DROPOUT,
    CH_RESID_V,
    IDX_MU,
    ScenarioSpace,
    latent_from_disturbance_overrides,
)
from rsv.utils import rng_for


def test_log_p_matches_scipy(default_cfg):
    """``log_p`` must be the exact standard-normal log-density."""
    cfg = default_cfg
    cfg.scenario.horizon = 5  # keep the scipy reference cheap
    space = ScenarioSpace(cfg.scenario, n_models=3)
    z = rng_for(1, "misc").standard_normal((7, space.dim))
    expected = multivariate_normal(np.zeros(space.dim), np.eye(space.dim)).logpdf(z)
    np.testing.assert_allclose(space.log_p(z), expected, rtol=1e-10)


def test_diag_gaussian_density_matches_scipy(default_cfg):
    cfg = default_cfg
    cfg.scenario.horizon = 4
    space = ScenarioSpace(cfg.scenario, n_models=1)
    rng = rng_for(2, "misc")
    mean = rng.normal(size=space.dim)
    sd = np.exp(rng.normal(scale=0.3, size=space.dim))
    z = mean + sd * rng.standard_normal((5, space.dim))
    expected = multivariate_normal(mean, np.diag(sd ** 2)).logpdf(z)
    np.testing.assert_allclose(space.log_p_diag_gaussian(z, mean, sd), expected, rtol=1e-10)


def test_dropout_rate_matches_configuration(default_cfg):
    """The Bernoulli channel must realise the probability it was configured with."""
    space = ScenarioSpace(default_cfg.scenario, n_models=1)
    z = space.sample_nominal(rng_for(3, "misc"), 4000)
    rate = space.decode(z).dropout.mean()
    assert abs(rate - default_cfg.scenario.dropout_prob) < 0.004


def test_latency_distribution_matches_configuration(default_cfg):
    space = ScenarioSpace(default_cfg.scenario, n_models=1)
    z = space.sample_nominal(rng_for(4, "misc"), 20000)
    latency = space.decode(z).latency
    for k, p in enumerate(default_cfg.scenario.latency_probs, start=1):
        assert abs(float((latency == k).mean()) - p) < 0.015
    assert latency.min() >= 1
    assert latency.max() <= len(default_cfg.scenario.latency_probs)


def test_model_index_is_uniform_over_members(default_cfg):
    """Sampling the ensemble member from the latent makes the rollout a proper
    mixture over the twin's members, so the marginal must be uniform."""
    space = ScenarioSpace(default_cfg.scenario, n_models=5)
    idx = space.decode(space.sample_nominal(rng_for(5, "misc"), 20000)).model_index
    counts = np.bincount(idx, minlength=5) / idx.size
    assert np.all(np.abs(counts - 0.2) < 0.015)


def test_decode_is_deterministic(default_cfg):
    space = ScenarioSpace(default_cfg.scenario, n_models=4)
    z = space.sample_nominal(rng_for(6, "misc"), 12)
    a, b = space.decode(z), space.decode(z)
    np.testing.assert_array_equal(a.mu, b.mu)
    np.testing.assert_array_equal(a.dropout, b.dropout)
    np.testing.assert_array_equal(a.latency, b.latency)


def test_traction_decoding_is_affine_and_clipped(default_cfg):
    sc = default_cfg.scenario
    space = ScenarioSpace(sc, n_models=1)
    z = np.zeros((3, space.dim))
    z[0, IDX_MU], z[1, IDX_MU], z[2, IDX_MU] = 0.0, -1.0, -50.0
    mu = space.decode(z).mu
    assert mu[0] == pytest.approx(sc.mu_mean)
    assert mu[1] == pytest.approx(sc.mu_mean - sc.mu_sd)
    assert mu[2] == pytest.approx(sc.mu_min)  # clipped, not negative


def test_channel_indices_round_trip(default_cfg):
    space = ScenarioSpace(default_cfg.scenario, n_models=1)
    idx = space.channel_indices(CH_RESID_V)
    assert idx.size == space.horizon
    assert idx[0] == space.step_index(0, CH_RESID_V)
    assert idx[-1] == space.step_index(space.horizon - 1, CH_RESID_V)
    assert space.index_label(idx[3]).startswith("residual_v")


def test_overrides_produce_the_requested_disturbances(default_cfg):
    space = ScenarioSpace(default_cfg.scenario, n_models=1)
    z = latent_from_disturbance_overrides(
        space, mu=0.25, latency_steps=3, dropout_window=range(10, 14)
    )
    d = space.decode(z[None, :])
    assert d.mu[0] == pytest.approx(0.25)
    assert int(d.latency[0]) == 3
    assert bool(d.dropout[0, 10]) and bool(d.dropout[0, 13])
    assert not bool(d.dropout[0, 14])


def test_describe_reports_the_dominant_dimensions(default_cfg):
    space = ScenarioSpace(default_cfg.scenario, n_models=1)
    z = np.zeros(space.dim)
    z[IDX_MU] = -3.5
    z[space.step_index(7, CH_DROPOUT)] = 2.5
    info = space.describe(z, top_k=2)
    names = [d["dimension"] for d in info["top_drivers"]]
    assert "floor_traction_mu" in names
    assert any(n.startswith("range_dropout") for n in names)
    assert info["log_p"] == pytest.approx(float(space.log_p(z[None, :])[0]))
