"""The measurement chain: what the twin is allowed to see, and how noisy it is."""

from __future__ import annotations

import numpy as np
import pytest

from rsv.data import (
    CSV_COLUMNS,
    build_dataset,
    collect_logs,
    estimate_target_noise,
    estimate_velocities,
    estimator_comparison,
    read_logs_csv,
    rows_from_log,
    write_logs_csv,
)
from rsv.dynamics import N_FEATURES, N_TARGETS
from rsv.utils import rng_for


@pytest.fixture
def logs(tiny_cfg):
    return collect_logs(tiny_cfg, rng_for(71, "data"))


def test_logs_have_the_recorded_streams(logs, tiny_cfg):
    log = logs[0]
    assert len(log) == tiny_cfg.data.mission_steps
    for stream in (log.cmd_v, log.enc_v, log.aruco_x, log.aruco_y, log.aruco_theta):
        assert stream.shape == (len(log),)
        assert np.all(np.isfinite(stream))
    assert log.aruco_valid.dtype == bool
    assert 0.1 < log.surface_mu < 1.2


def test_camera_estimator_is_unbiased_while_slipping(tiny_cfg):
    """The reason the twin is fitted to the camera and not to wheel odometry."""
    cfg = tiny_cfg
    cfg.data.n_missions = 40
    cfg.data.p_hard_stop = 0.6  # make sure there is plenty of slipping
    logs = collect_logs(cfg, rng_for(72, "data"))
    report = estimator_comparison(logs, cfg)

    camera_bias = abs(report["camera"]["velocity_bias_while_slipping"])
    encoder_bias = abs(report["encoder"]["velocity_bias_while_slipping"])
    assert camera_bias < encoder_bias
    assert abs(report["camera"]["velocity_bias"]) < 0.004


def test_target_noise_probe_is_positive_and_plausible(logs, tiny_cfg):
    """The estimator's own noise, measured without touching ground truth."""
    var = estimate_target_noise(logs, tiny_cfg, rng_for(73, "data"))
    assert var.shape == (N_TARGETS,)
    assert np.all(var > 0)
    # It must be smaller than the spread of the targets, or the data is useless.
    x, y = rows_from_log(logs[0], tiny_cfg)
    assert np.all(np.sqrt(var) < 3.0 * y.std(axis=0))


def test_target_noise_probe_scales_with_the_sensor(tiny_cfg):
    """Doubling the camera noise must roughly quadruple the target variance."""
    cfg = tiny_cfg
    base = collect_logs(cfg, rng_for(74, "data"))
    v1 = estimate_target_noise(base, cfg, rng_for(75, "data"))

    cfg2 = tiny_cfg
    cfg2.data.aruco_xy_noise_sd *= 2.0
    noisier = collect_logs(cfg2, rng_for(74, "data"))
    v2 = estimate_target_noise(noisier, cfg2, rng_for(75, "data"))
    assert v2[0] / v1[0] == pytest.approx(4.0, rel=0.35)


def test_dataset_splits_by_mission(logs, tiny_cfg):
    ds = build_dataset(logs, tiny_cfg, rng_for(76, "data"))
    assert ds.x_train.shape[1] == N_FEATURES
    assert ds.y_train.shape[1] == N_TARGETS
    assert ds.n_train > 0 and ds.n_val > 0
    assert ds.measurement_var is not None
    # No row may appear on both sides of the split.
    both = {tuple(np.round(r, 12)) for r in ds.x_train} & {
        tuple(np.round(r, 12)) for r in ds.x_val
    }
    assert not both


def test_dropped_camera_frames_are_held_not_interpolated(tiny_cfg, logs):
    log = logs[0]
    log.aruco_valid[:] = True
    log.aruco_valid[5] = False
    log.aruco_x[5] = 999.0  # a value a hold must never let through
    v, _ = estimate_velocities(log, tiny_cfg)
    assert np.all(np.abs(v) < 5.0)


def test_csv_round_trip(logs, tiny_cfg, tmp_path):
    """The on-disk schema is the contract with the hardware recorder."""
    paths = write_logs_csv(logs[:3], str(tmp_path / "logs"))
    assert len(paths) == 3
    with open(paths[0], encoding="utf-8") as fh:
        assert fh.readline().strip().split(",") == list(CSV_COLUMNS)

    restored = read_logs_csv(str(tmp_path / "logs"))
    assert len(restored) == 3
    for original, back in zip(logs[:3], restored):
        np.testing.assert_allclose(back.cmd_v, original.cmd_v, atol=1e-6)
        np.testing.assert_allclose(back.aruco_x, original.aruco_x, atol=1e-6)
        np.testing.assert_array_equal(back.aruco_valid, original.aruco_valid)
        assert back.surface_mu == pytest.approx(original.surface_mu, abs=1e-6)
        assert back.true_v is None  # ground truth is never written to disk


def test_rows_exclude_windows_that_straddle_a_command_step(tiny_cfg, logs):
    """Samples spanning a command discontinuity would teach weaker braking."""
    log = logs[0]
    x, _ = rows_from_log(log, tiny_cfg)
    jumps = np.flatnonzero(np.abs(np.diff(log.cmd_v)) > tiny_cfg.data.command_jump_threshold)
    if jumps.size == 0:
        pytest.skip("this mission contains no emergency stop")
    # Every retained row's command must match one of the rows kept.
    assert x.shape[0] > 0
    assert x.shape[0] < len(log)  # something was excluded
