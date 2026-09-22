"""Driving-data collection and dataset construction.

Stands in for a logging campaign on the real robot: the vehicle drives ordinary
missions while three streams are recorded at the control rate --

* the velocity commands sent to the base,
* wheel-odometry (encoder) velocities, which *over-read while the wheels spin
  and under-read while they skid*,
* the pose measured by an overhead ArUco camera, which is unbiased but noisy and
  occasionally drops a frame.

Neither stream alone is a good training target.  Encoders are precise but biased
exactly in the slip regime the safety case depends on; the camera is unbiased
but differentiating its position twice amplifies pixel noise.  So the dataset is
built from the camera track alone, differentiated with a window aligned to the
control period, and the encoder stream is kept as a diagnostic: it is precise
but wrong by an order of magnitude more, in the wrong direction, exactly while
the wheels are skidding.  A complementary filter is provided as an alternative
and quantified in the report.

The residual noise of that estimator is then measured, without using any ground
truth, by re-injecting an independent draw of the *calibrated* sensor noise and
re-running the estimator: the variance of the difference is exactly the
estimator's own variance.  :mod:`rsv.twin` subtracts it from the learned
predictive variance so that the twin simulates the robot's process noise rather
than the camera's.

Everything the twin sees comes from this module.  It never sees :mod:`rsv.plant`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config, DataCfg
from .controller import wrap_angle
from .dynamics import FEATURE_NAMES, TARGET_NAMES, build_features
from .plant import Plant

CSV_COLUMNS = (
    "mission",
    "step",
    "time_s",
    "cmd_v",
    "cmd_w",
    "enc_v",
    "enc_w",
    "aruco_x",
    "aruco_y",
    "aruco_theta",
    "aruco_valid",
    "surface_mu",
)


@dataclass
class DrivingLog:
    """One logged mission, in exactly the form the ROS 2 recorder writes."""

    mission: int
    time_s: np.ndarray
    cmd_v: np.ndarray
    cmd_w: np.ndarray
    enc_v: np.ndarray
    enc_w: np.ndarray
    aruco_x: np.ndarray
    aruco_y: np.ndarray
    aruco_theta: np.ndarray
    aruco_valid: np.ndarray
    surface_mu: float
    # Ground truth, recorded only so the report can quantify estimator error.
    # Nothing on the fitting path is allowed to read these.
    true_v: Optional[np.ndarray] = None
    true_w: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(self.time_s.shape[0])


# --------------------------------------------------------------------------- #
# Mission generation
# --------------------------------------------------------------------------- #


def _command_profiles(cfg: DataCfg, ctrl_accel: float, dt: float, rng: np.random.Generator):
    """Build ``(n_missions, steps)`` command arrays for the logging campaign.

    The campaign is what a team building this safety case would actually record:
    ordinary transport missions, plus a healthy fraction of deliberate
    emergency-brake tests from cruise speed.  So the twin does get to see
    traction-limited stops -- the gap it cannot close from this data is not
    "braking" but *braking on a floor slipperier than any that was tested*,
    because the surfaces available for logging (see ``DataCfg.mu_mean``) are
    better than the tail the stress tester will go looking for.

    Returns ``(cmd_v, cmd_w, kinds)``.
    """
    m, t_len = int(cfg.n_missions), int(cfg.mission_steps)
    cmd_v = np.zeros((m, t_len))
    cmd_w = np.zeros((m, t_len))
    kinds: List[str] = []
    step = ctrl_accel * dt

    for i in range(m):
        u = rng.random()
        if u < cfg.p_hard_stop:
            kind = "brake_test"
        elif u < cfg.p_hard_stop + cfg.p_gentle_stop:
            kind = "gentle_stop"
        else:
            kind = "cruise"
        kinds.append(kind)

        setpoints = np.zeros(t_len)
        braking = np.zeros(t_len, dtype=bool)

        if kind == "brake_test":
            # Structured protocol: accelerate to a target speed, hold, stop hard,
            # repeat.  Sweeping the target speed identifies the traction limit
            # across the whole operating range instead of at one speed.
            t = 0
            while t < t_len:
                target = float(rng.uniform(0.45, cfg.speed_hi + 0.05))
                hold = int(rng.integers(10, 20))
                accel_steps = int(np.ceil(target / max(ctrl_accel * dt, 1e-6)))
                brake = int(rng.integers(16, 24))
                setpoints[t : t + accel_steps + hold] = target
                braking[t + accel_steps + hold : t + accel_steps + hold + brake] = True
                t += accel_steps + hold + brake
        else:
            # Piecewise-constant speed set-points.
            t = 0
            while t < t_len:
                span = int(rng.integers(25, 70))
                setpoints[t : t + span] = float(rng.uniform(cfg.speed_lo, cfg.speed_hi))
                t += span
            if kind == "gentle_stop":
                s0 = int(rng.integers(t_len // 4, 3 * t_len // 4))
                setpoints[s0 : s0 + 40] = 0.0

        v = 0.0
        for t in range(t_len):
            v = 0.0 if braking[t] else float(np.clip(setpoints[t], v - step, v + step))
            cmd_v[i, t] = v

        # Slow yaw excitation so the turn-rate channel is identifiable too.
        amp = float(rng.uniform(0.0, 0.55))
        freq = float(rng.uniform(0.05, 0.30))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        cmd_w[i] = amp * np.sin(2.0 * np.pi * freq * np.arange(t_len) * dt + phase)

    return cmd_v, cmd_w, kinds


def collect_logs(cfg: Config, rng: np.random.Generator) -> List[DrivingLog]:
    """Drive the logging campaign on the plant and record the sensor streams."""
    dcfg, dt = cfg.data, cfg.scenario.dt
    plant = Plant(cfg.plant, dt)
    m, t_len = int(dcfg.n_missions), int(dcfg.mission_steps)

    cmd_v, cmd_w, kinds = _command_profiles(dcfg, cfg.controller.accel_limit, dt, rng)
    mu = np.clip(rng.normal(dcfg.mu_mean, dcfg.mu_sd, size=m), 0.12, 1.2)

    x = np.zeros((m, t_len))
    y = np.zeros((m, t_len))
    th = np.zeros((m, t_len))
    v = np.zeros((m, t_len))
    w = np.zeros((m, t_len))
    wheel_v = np.zeros((m, t_len))
    wheel_w = np.zeros((m, t_len))

    state_v = np.zeros(m)
    state_w = np.zeros(m)
    state_x = np.zeros(m)
    state_y = np.zeros(m)
    state_th = rng.uniform(-0.2, 0.2, size=m)

    for t in range(t_len):
        x[:, t], y[:, t], th[:, t] = state_x, state_y, state_th
        v[:, t], w[:, t] = state_v, state_w

        # Wheel speed differs from ground speed exactly when traction saturates.
        a_des = (cmd_v[:, t] - state_v) / np.where(
            cmd_v[:, t] < state_v, cfg.plant.tau_brake, cfg.plant.tau_accel
        )
        a_real, _ = plant.nominal_accel(state_v, cmd_v[:, t], mu)
        wheel_v[:, t] = state_v + dcfg.encoder_slip_gain * (a_des - a_real) * 0.08
        wheel_w[:, t] = state_w

        eps = rng.standard_normal((m, 2))
        dv, dw = plant.step(state_v, state_w, cmd_v[:, t], cmd_w[:, t], mu, eps)
        state_v = state_v + dv
        state_w = state_w + dw
        theta_mid = state_th + 0.5 * state_w * dt
        state_x = state_x + state_v * np.cos(theta_mid) * dt
        state_y = state_y + state_v * np.sin(theta_mid) * dt
        state_th = state_th + state_w * dt

    return apply_measurement_chain(
        x, y, th, v, w, cmd_v, cmd_w, mu, cfg, rng, wheel_v=wheel_v, wheel_w=wheel_w
    )


def apply_measurement_chain(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    v: np.ndarray,
    omega: np.ndarray,
    cmd_v: np.ndarray,
    cmd_w: np.ndarray,
    mu: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
    wheel_v: Optional[np.ndarray] = None,
    wheel_w: Optional[np.ndarray] = None,
    first_mission: int = 0,
) -> List[DrivingLog]:
    """Turn true trajectories into the three recorded sensor streams.

    Everything the twin is ever allowed to see passes through here: quantised,
    noisy wheel odometry that mis-reads while the tyres slip, and an overhead
    camera that is unbiased but noisy and drops the occasional frame.  Sharing
    one implementation between the logging campaign and the hardware-replay
    stage guarantees the twin cannot tell a replayed episode from an ordinary
    mission -- which is what makes the DAgger update fair.
    """
    dcfg, dt = cfg.data, cfg.scenario.dt
    plant = Plant(cfg.plant, dt)
    m, t_len = x.shape

    if wheel_v is None:
        # Reconstruct the slip-corrupted wheel speed from the recorded motion.
        wheel_v = np.empty_like(v)
        for t in range(t_len):
            tau = np.where(cmd_v[:, t] < v[:, t], cfg.plant.tau_brake, cfg.plant.tau_accel)
            a_des = (cmd_v[:, t] - v[:, t]) / tau
            a_real, _ = plant.nominal_accel(v[:, t], cmd_v[:, t], mu)
            wheel_v[:, t] = v[:, t] + dcfg.encoder_slip_gain * (a_des - a_real) * 0.08
    if wheel_w is None:
        wheel_w = omega

    logs: List[DrivingLog] = []
    for i in range(m):
        valid = rng.random(t_len) > dcfg.aruco_dropout
        valid[0] = True
        logs.append(
            DrivingLog(
                mission=first_mission + i,
                time_s=np.arange(t_len) * dt,
                cmd_v=cmd_v[i],
                cmd_w=cmd_w[i],
                enc_v=wheel_v[i] + rng.normal(0.0, dcfg.encoder_noise_sd, size=t_len),
                enc_w=wheel_w[i] + rng.normal(0.0, dcfg.encoder_noise_sd, size=t_len),
                aruco_x=x[i] + rng.normal(0.0, dcfg.aruco_xy_noise_sd, size=t_len),
                aruco_y=y[i] + rng.normal(0.0, dcfg.aruco_xy_noise_sd, size=t_len),
                aruco_theta=theta[i] + rng.normal(0.0, dcfg.aruco_theta_noise_sd, size=t_len),
                aruco_valid=valid,
                surface_mu=float(mu[i]),
                true_v=v[i],
                true_w=omega[i],
            )
        )
    return logs


def logs_from_trajectories(
    traces,
    mu: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
    first_mission: int = 10_000,
) -> List[DrivingLog]:
    """Log a batch of replayed episodes as if they had been recorded on the robot.

    ``traces`` is a :class:`rsv.rollout.Traces` record from a hardware replay.
    The mission ids start high so replayed runs stay distinguishable from the
    original campaign in the on-disk logs.
    """
    return apply_measurement_chain(
        traces.x,
        traces.y,
        traces.theta,
        traces.v,
        traces.omega,
        traces.v_cmd,
        traces.omega_cmd,
        np.asarray(mu, dtype=float),
        cfg,
        rng,
        first_mission=first_mission,
    )


# --------------------------------------------------------------------------- #
# State estimation from the logged streams
# --------------------------------------------------------------------------- #


def _hold_invalid(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Zero-order hold across dropped camera frames."""
    out = np.array(a, dtype=float, copy=True)
    last = out[0]
    for i in range(out.shape[0]):
        if valid[i]:
            last = out[i]
        else:
            out[i] = last
    return out


def _aligned_derivative(a: np.ndarray, half: int, dt: float, angular: bool = False) -> np.ndarray:
    """Velocity estimate aligned with the simulator's integration convention.

    The pose advances as ``x[t+1] = x[t] + v[t+1] * dt``, so a *backward*
    difference at ``t`` measures ``v[t]`` exactly.  Averaging ``2*half + 1``
    consecutive backward differences telescopes to

        v[t] ~= (x[t+half] - x[t-half-1]) / ((2*half + 1) * dt)

    which keeps that alignment while cutting the camera noise by the full span
    rather than by ``sqrt`` of it: only the two endpoints carry noise.  The
    price is smoothing across a window of ``2*half + 1`` steps, so the dataset
    trims the margins where the window would straddle a command step.
    """
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    half = int(max(0, min(half, (n - 2) // 2)))
    span = (2 * half + 1) * dt

    idx = np.arange(n)
    hi = np.clip(idx + half, 0, n - 1)
    lo = np.clip(idx - half - 1, 0, n - 1)
    diff = wrap_angle(a[hi] - a[lo]) if angular else (a[hi] - a[lo])
    scale = np.maximum(hi - lo, 1) * dt
    out = diff / scale
    # Ends use a shortened window; the dataset margin discards them anyway.
    out[: half + 1] = out[half + 1] if n > half + 1 else 0.0
    out[n - half :] = out[max(n - half - 1, 0)]
    del span
    return out


def _complementary(cam: np.ndarray, enc: np.ndarray, alpha: float) -> np.ndarray:
    """Camera at low frequency, encoder at high frequency.

    Kept as a documented alternative, not the default: a wheel-slip bias is a
    *fast* transient, so the high-frequency encoder path passes it through
    almost unattenuated -- exactly in the braking regime the safety case turns
    on.  :func:`estimator_comparison` quantifies that.
    """
    lp_cam = np.empty_like(cam)
    lp_enc = np.empty_like(enc)
    lp_cam[0], lp_enc[0] = cam[0], enc[0]
    for t in range(1, cam.shape[0]):
        lp_cam[t] = alpha * lp_cam[t - 1] + (1.0 - alpha) * cam[t]
        lp_enc[t] = alpha * lp_enc[t - 1] + (1.0 - alpha) * enc[t]
    return lp_cam + (enc - lp_enc)


COMPLEMENTARY_ALPHA = 0.75


def estimate_velocities(
    log: DrivingLog, cfg: Config, mode: str = "camera"
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward-speed and yaw-rate estimates from the logged sensor streams.

    ``mode`` selects the estimator: ``"camera"`` (default) differentiates the
    ArUco track, ``"encoder"`` takes wheel odometry at face value, and
    ``"complementary"`` fuses the two.  Only the camera estimator is unbiased
    while the wheels are slipping, which is why the twin is fitted to it.
    """
    dt = cfg.scenario.dt
    h = int(max(1, cfg.data.smooth_halfwidth))
    ax = _hold_invalid(log.aruco_x, log.aruco_valid)
    ay = _hold_invalid(log.aruco_y, log.aruco_valid)
    ath = _hold_invalid(log.aruco_theta, log.aruco_valid)

    vx = _aligned_derivative(ax, h, dt)
    vy = _aligned_derivative(ay, h, dt)
    cam_v = vx * np.cos(ath) + vy * np.sin(ath)  # ground speed along the heading
    cam_w = _aligned_derivative(ath, h, dt, angular=True)

    if mode == "camera":
        return cam_v, cam_w
    if mode == "encoder":
        return np.array(log.enc_v, dtype=float), np.array(log.enc_w, dtype=float)
    if mode == "complementary":
        return (
            _complementary(cam_v, log.enc_v, COMPLEMENTARY_ALPHA),
            _complementary(cam_w, log.enc_w, COMPLEMENTARY_ALPHA),
        )
    raise ValueError("unknown velocity estimator %r" % mode)


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    feature_names: Sequence[str] = FEATURE_NAMES
    target_names: Sequence[str] = TARGET_NAMES
    measurement_var: Optional[np.ndarray] = None  # noise variance of the targets
    info: Optional[Dict[str, object]] = None

    @property
    def n_train(self) -> int:
        return int(self.x_train.shape[0])

    @property
    def n_val(self) -> int:
        return int(self.x_val.shape[0])


def _clean_window(cmd: np.ndarray, sel: np.ndarray, reach: int, threshold: float) -> np.ndarray:
    """Mask out rows whose estimation window straddles a command step change.

    The velocity estimator averages over ``2 * reach + 1`` frames.  Where the
    command jumps -- the instant an emergency stop is requested -- that window
    mixes two dynamic regimes and the estimated acceleration is a blend of both,
    which would teach the twin that the robot brakes more weakly than it does.
    Gentle ramps change the command by a fraction of the threshold each step and
    are kept; only genuine discontinuities are cut out.
    """
    jump = np.zeros(cmd.shape[0], dtype=bool)
    jump[1:] = np.abs(np.diff(cmd)) > float(threshold)
    contaminated = np.zeros_like(jump)
    for shift in range(-int(reach), int(reach) + 1):
        contaminated |= np.roll(jump, shift)
    return ~contaminated[sel]


def rows_from_log(
    log: DrivingLog, cfg: Config, mode: str = "camera"
) -> Tuple[np.ndarray, np.ndarray]:
    """Feature/target rows from one mission, using only measurable quantities."""
    v_hat, w_hat = estimate_velocities(log, cfg, mode=mode)
    h = int(max(1, cfg.data.smooth_halfwidth))
    n = len(log)
    sel = np.arange(h + 1, n - h - 2)  # the derivative window needs a margin at both ends
    sel = sel[_clean_window(log.cmd_v, sel, h + 1, cfg.data.command_jump_threshold)]
    x = build_features(
        v_hat[sel],
        w_hat[sel],
        log.cmd_v[sel],
        log.cmd_w[sel],
        np.full(sel.shape[0], log.surface_mu),
    )
    y = np.stack((v_hat[sel + 1] - v_hat[sel], w_hat[sel + 1] - w_hat[sel]), axis=1)
    return x, y


def build_dataset(
    logs: Sequence[DrivingLog],
    cfg: Config,
    rng: np.random.Generator,
    measurement_var: Optional[np.ndarray] = None,
) -> Dataset:
    """Assemble the training set, splitting *by mission* to avoid leakage."""
    xs, ys, mission_id = [], [], []
    for log in logs:
        x, y = rows_from_log(log, cfg)
        xs.append(x)
        ys.append(y)
        mission_id.append(np.full(x.shape[0], log.mission))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    mid = np.concatenate(mission_id, axis=0)

    missions = np.unique(mid)
    n_val = int(round(cfg.data.val_fraction * len(missions)))
    val_missions = set(rng.permutation(missions)[:n_val].tolist())
    is_val = np.array([m in val_missions for m in mid])

    if measurement_var is None:
        measurement_var = estimate_target_noise(logs, cfg, rng)

    return Dataset(
        x_train=x[~is_val],
        y_train=y[~is_val],
        x_val=x[is_val],
        y_val=y[is_val],
        measurement_var=measurement_var,
        info={
            "n_missions": len(logs),
            "n_val_missions": int(n_val),
            "mu_range": [float(min(l.surface_mu for l in logs)), float(max(l.surface_mu for l in logs))],
            "speed_range": [float(x[:, 0].min()), float(x[:, 0].max())],
            "v_err_range": [float(x[:, 5].min()), float(x[:, 5].max())],
        },
    )


def estimate_target_noise(
    logs: Sequence[DrivingLog], cfg: Config, rng: np.random.Generator, n_probe: int = 12
) -> np.ndarray:
    """Variance the measurement chain contributes to the regression targets.

    Re-injects an independent draw of the *calibrated* sensor noise into the raw
    streams, re-runs the same estimator, and takes the variance of the
    difference.  Because the added draw has the same law as the noise already
    present, that variance equals the estimator's own noise variance -- obtained
    without ever touching the ground-truth state.
    """
    dcfg = cfg.data
    diffs: List[np.ndarray] = []
    for log in list(logs)[: int(n_probe)]:
        _, y0 = rows_from_log(log, cfg)
        n = len(log)
        noisy = DrivingLog(
            mission=log.mission,
            time_s=log.time_s,
            cmd_v=log.cmd_v,
            cmd_w=log.cmd_w,
            enc_v=log.enc_v + rng.normal(0.0, dcfg.encoder_noise_sd, size=n),
            enc_w=log.enc_w + rng.normal(0.0, dcfg.encoder_noise_sd, size=n),
            aruco_x=log.aruco_x + rng.normal(0.0, dcfg.aruco_xy_noise_sd, size=n),
            aruco_y=log.aruco_y + rng.normal(0.0, dcfg.aruco_xy_noise_sd, size=n),
            aruco_theta=log.aruco_theta + rng.normal(0.0, dcfg.aruco_theta_noise_sd, size=n),
            aruco_valid=log.aruco_valid,
            surface_mu=log.surface_mu,
        )
        _, y1 = rows_from_log(noisy, cfg)
        diffs.append(y1 - y0)
    d = np.concatenate(diffs, axis=0)
    return np.var(d, axis=0)


# --------------------------------------------------------------------------- #
# CSV persistence -- the on-disk contract with the hardware recorder
# --------------------------------------------------------------------------- #


def write_logs_csv(logs: Sequence[DrivingLog], directory: str) -> List[str]:
    """Write each mission to its own CSV in the schema of ``hardware/``."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for log in logs:
        path = out_dir / ("mission_%04d.csv" % log.mission)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(CSV_COLUMNS)
            for t in range(len(log)):
                writer.writerow(
                    [
                        log.mission,
                        t,
                        "%.4f" % log.time_s[t],
                        "%.6f" % log.cmd_v[t],
                        "%.6f" % log.cmd_w[t],
                        "%.6f" % log.enc_v[t],
                        "%.6f" % log.enc_w[t],
                        "%.6f" % log.aruco_x[t],
                        "%.6f" % log.aruco_y[t],
                        "%.6f" % log.aruco_theta[t],
                        int(bool(log.aruco_valid[t])),
                        "%.6f" % log.surface_mu,
                    ]
                )
        paths.append(str(path))
    return paths


def read_logs_csv(directory: str) -> List[DrivingLog]:
    """Read logs written by :func:`write_logs_csv` or by the ROS 2 recorder."""
    logs: List[DrivingLog] = []
    for path in sorted(Path(directory).glob("mission_*.csv")):
        cols: Dict[str, List[float]] = {c: [] for c in CSV_COLUMNS}
        with open(path, "r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                for c in CSV_COLUMNS:
                    cols[c].append(float(row[c]))
        logs.append(
            DrivingLog(
                mission=int(cols["mission"][0]),
                time_s=np.array(cols["time_s"]),
                cmd_v=np.array(cols["cmd_v"]),
                cmd_w=np.array(cols["cmd_w"]),
                enc_v=np.array(cols["enc_v"]),
                enc_w=np.array(cols["enc_w"]),
                aruco_x=np.array(cols["aruco_x"]),
                aruco_y=np.array(cols["aruco_y"]),
                aruco_theta=np.array(cols["aruco_theta"]),
                aruco_valid=np.array(cols["aruco_valid"]) > 0.5,
                surface_mu=float(cols["surface_mu"][0]),
            )
        )
    return logs


def estimator_comparison(logs: Sequence[DrivingLog], cfg: Config) -> Dict[str, object]:
    """Accuracy of each velocity estimator against the recorded truth.

    Diagnostic only -- it quantifies the measurement chain for the write-up and
    is never consulted while fitting.  The interesting column is the bias during
    slip: wheel odometry is precise but wrong exactly when the wheels lose grip.
    """
    h = int(max(1, cfg.data.smooth_halfwidth))
    out: Dict[str, object] = {}
    for mode in ("camera", "encoder", "complementary"):
        errs, slip_errs = [], []
        for log in logs:
            if log.true_v is None:
                continue
            v_hat, _ = estimate_velocities(log, cfg, mode=mode)
            n = len(log)
            sel = np.arange(h + 1, n - h - 2)
            err = v_hat[sel] - log.true_v[sel]
            errs.append(err)
            # "slipping" = the command is asking for more than the floor can give
            a_des = np.abs(log.cmd_v[sel] - log.true_v[sel]) / cfg.plant.tau_brake
            slipping = a_des > cfg.plant.traction_gain * log.surface_mu
            if slipping.any():
                slip_errs.append(err[slipping])
        if not errs:
            continue
        e = np.concatenate(errs)
        se = np.concatenate(slip_errs) if slip_errs else np.zeros(1)
        out[mode] = {
            "velocity_rmse": float(np.sqrt(np.mean(e ** 2))),
            "velocity_bias": float(np.mean(e)),
            "velocity_bias_while_slipping": float(np.mean(se)),
            "velocity_rmse_while_slipping": float(np.sqrt(np.mean(se ** 2))),
        }
    return out
