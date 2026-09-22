"""The learned twin: does it fit, does it know what it does not know, does it save."""

from __future__ import annotations

import numpy as np
import pytest

from rsv.config import TwinCfg
from rsv.dynamics import N_FEATURES
from rsv.models.calibration import coverage, gaussian_nll, miscalibration_area
from rsv.models.mlp import MLP, train_mlp
from rsv.twin import Twin
from rsv.utils import rng_for


def _heteroscedastic_data(rng, n=3000):
    """A function whose noise depends on the input, which is the whole point."""
    x = rng.uniform(-2.0, 2.0, size=(n, N_FEATURES))
    signal = 0.4 * np.sin(1.5 * x[:, 0]) + 0.25 * x[:, 5]
    sd = 0.02 + 0.08 * np.abs(x[:, 0])
    y = np.stack(
        (signal + sd * rng.standard_normal(n), 0.1 * x[:, 1] + 0.02 * rng.standard_normal(n)),
        axis=1,
    )
    return x, y, sd


def test_gradients_match_finite_differences():
    """The hand-written backward pass is the thing most likely to be silently wrong.

    Checked at ``beta_nll = 0`` (plain Gaussian NLL), where the loss is exactly
    the function being differentiated.  Under beta-NLL the ``sigma^(2*beta)``
    factor is a *detached* constant by definition, so the analytic gradient is
    deliberately not the derivative of the printed loss and a finite difference
    of that loss would disagree by design.
    """
    rng = rng_for(41, "misc")
    model = MLP(3, 2, [6, 5], rng=rng, beta_nll=0.0)
    x = rng.standard_normal((9, 3))
    y = rng.standard_normal((9, 2))

    loss, gw, gb = model.loss_and_grads(x, y)
    eps = 1e-6
    for layer in range(len(model.weights)):
        for _ in range(4):
            i = int(rng.integers(model.weights[layer].shape[0]))
            j = int(rng.integers(model.weights[layer].shape[1]))
            original = model.weights[layer][i, j]
            model.weights[layer][i, j] = original + eps
            up, _, _ = model.loss_and_grads(x, y)
            model.weights[layer][i, j] = original - eps
            down, _, _ = model.loss_and_grads(x, y)
            model.weights[layer][i, j] = original
            numeric = (up - down) / (2 * eps)
            assert numeric == pytest.approx(gw[layer][i, j], abs=1e-6, rel=1e-4)


def test_beta_nll_keeps_fitting_the_mean_in_noisy_regions():
    """Plain NLL lets a network explain a hard region as noise; beta-NLL should not."""
    rng = rng_for(42, "misc")
    n = 4000
    x = rng.uniform(-2, 2, size=(n, 1))
    truth = np.sign(x[:, 0]) * np.minimum(np.abs(x[:, 0]), 1.0)
    sd = 0.02 + 0.30 * (x[:, 0] > 1.0)  # a noisy patch at one end
    y = (truth + sd * rng.standard_normal(n))[:, None]

    errors = {}
    for beta in (0.0, 1.0):
        model = MLP(1, 1, [24, 24], rng=rng_for(43, "misc"), beta_nll=beta)
        train_mlp(model, x, y, rng=rng_for(44, "misc"), epochs=120, batch_size=256, lr=5e-3)
        grid = np.linspace(1.1, 2.0, 40)[:, None]
        pred, _ = model.predict(grid)
        errors[beta] = float(np.mean(np.abs(pred[:, 0] - 1.0)))
    assert errors[1.0] <= errors[0.0] + 1e-3


def test_twin_learns_a_heteroscedastic_function_and_calibrates():
    rng = rng_for(45, "twin")
    x, y, sd = _heteroscedastic_data(rng)
    cfg = TwinCfg(n_members=3, epochs=90, hidden=[32, 32], batch_size=256)
    twin = Twin(cfg, rng_for(46, "twin"))
    twin.fit(x, y, rng_for(47, "twin"))

    xv, yv, sdv = _heteroscedastic_data(rng_for(48, "twin"), n=1500)
    mean, alea, epi = twin.predict(xv)
    total = np.sqrt(alea ** 2 + epi ** 2)

    # Compare against the irreducible noise, not an arbitrary constant: no model
    # can beat the standard deviation of the data-generating process.
    noise_floor = float(np.sqrt(np.mean(sdv ** 2)))
    rmse = float(np.sqrt(np.mean((mean[:, 0] - yv[:, 0]) ** 2)))
    assert rmse < 1.15 * noise_floor
    # The predicted noise tracks the true noise, not just its average.
    assert float(np.corrcoef(alea[:, 0], sdv)[0, 1]) > 0.7
    cov = coverage(yv, mean, total)
    assert 0.85 < cov["0.95"] < 1.0
    assert miscalibration_area(yv, mean, total) < 0.12


def test_epistemic_uncertainty_grows_away_from_the_data():
    """The claim the whole project leans on: the twin flags its own ignorance."""
    rng = rng_for(49, "twin")
    n = 2500
    x = np.zeros((n, N_FEATURES))
    x[:, 0] = rng.uniform(-1.0, 1.0, size=n)  # data only in a band
    y = np.stack((0.5 * np.sin(2.0 * x[:, 0]), np.zeros(n)), axis=1)
    y += 0.01 * rng.standard_normal(y.shape)

    twin = Twin(TwinCfg(n_members=5, epochs=90, hidden=[32, 32]), rng_for(50, "twin"))
    twin.fit(x, y, rng_for(51, "twin"))

    inside = np.zeros((40, N_FEATURES))
    inside[:, 0] = np.linspace(-0.8, 0.8, 40)
    outside = np.zeros((40, N_FEATURES))
    outside[:, 0] = np.linspace(2.5, 4.0, 40)

    _, _, epi_in = twin.predict(inside)
    _, _, epi_out = twin.predict(outside)
    assert float(epi_out[:, 0].mean()) > 3.0 * float(epi_in[:, 0].mean())


def test_measurement_noise_is_removed_when_simulating():
    """Learned variance includes the sensor's noise; a simulation must not."""
    rng = rng_for(52, "twin")
    x, y, _ = _heteroscedastic_data(rng, n=1500)
    twin = Twin(TwinCfg(n_members=2, epochs=40, hidden=[24, 24]), rng_for(53, "twin"))
    twin.fit(x, y, rng_for(54, "twin"))

    _, raw, _ = twin.predict(x[:200])
    twin.set_measurement_noise(np.array([0.004 ** 2, 0.004 ** 2]))
    _, corrected, _ = twin.predict(x[:200], deconvolve=True)

    assert np.all(corrected <= raw + 1e-12)
    assert np.all(corrected > 0.0)
    assert float(corrected[:, 0].mean()) < float(raw[:, 0].mean())


def test_save_and_load_round_trip(tmp_path):
    rng = rng_for(55, "twin")
    x, y, _ = _heteroscedastic_data(rng, n=800)
    cfg = TwinCfg(n_members=3, epochs=25, hidden=[16, 16])
    twin = Twin(cfg, rng_for(56, "twin"))
    twin.fit(x, y, rng_for(57, "twin"))
    twin.set_measurement_noise(np.array([1e-5, 2e-5]))

    path = str(tmp_path / "twin.npz")
    twin.save(path)

    # Deliberately mismatched config: the file's own metadata must win.
    reloaded = Twin.load(path, TwinCfg(n_members=7, hidden=[99]), rng_for(58, "twin"))
    assert reloaded.n_models == 3
    a, _, _ = twin.predict(x[:50])
    b, _, _ = reloaded.predict(x[:50])
    np.testing.assert_allclose(a, b, rtol=1e-12)
    np.testing.assert_allclose(reloaded.measurement_var, [1e-5, 2e-5])


def test_step_is_reproducible_and_uses_the_assigned_member():
    rng = rng_for(59, "twin")
    x, y, _ = _heteroscedastic_data(rng, n=600)
    twin = Twin(TwinCfg(n_members=3, epochs=20, hidden=[16, 16]), rng_for(60, "twin"))
    twin.fit(x, y, rng_for(61, "twin"))

    args = (np.full(6, 0.5), np.zeros(6), np.zeros(6), np.zeros(6), np.full(6, 0.4))
    eps = np.zeros((6, 2))
    members = np.array([0, 0, 1, 1, 2, 2])
    dv, _ = twin.step(*args, eps, model_index=members)
    np.testing.assert_allclose(dv[0], dv[1], rtol=1e-12)
    np.testing.assert_allclose(dv[2], dv[3], rtol=1e-12)
    assert not np.allclose(dv[0], dv[2])  # different members disagree


def test_gaussian_nll_matches_the_closed_form():
    y = np.array([[0.1, -0.2]])
    mean = np.array([[0.0, 0.0]])
    sd = np.array([[0.5, 0.25]])
    expected = sum(
        0.5 * (v / s) ** 2 + np.log(s) + 0.5 * np.log(2 * np.pi)
        for v, s in ((0.1, 0.5), (-0.2, 0.25))
    )
    assert gaussian_nll(y, mean, sd) == pytest.approx(expected)
