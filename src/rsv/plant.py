"""Ground-truth robot: the stand-in for the physical hardware.

This module is the *only* place that knows the true physics.  The twin never
sees these equations or parameters; it only ever sees logged data produced by
them.  Two stages are allowed to call it:

* :mod:`rsv.data` -- to generate the driving logs used to fit the twin, through
  a realistic measurement chain (wheel encoders + overhead ArUco camera);
* :mod:`rsv.sim2real` -- to replay scenarios "on hardware".

The important structural feature is the traction limit
``|a| <= traction_gain * mu``.  Ordinary logged driving uses gentle command
ramps that stay well inside this limit, so the limit is only weakly identified
from nominal data -- and it is exactly what decides whether an emergency stop on
a slippery floor succeeds.  That mismatch is the sim-to-real gap the closing
loop has to discover and repair; it is a property of the physics here, not a
contrivance bolted onto the experiment.

In a real deployment this file is replaced by the robot itself: the same
interfaces are implemented by the ROS 2 nodes under ``hardware/``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import PlantCfg
from .dynamics import Dynamics


class Plant(Dynamics):
    """True velocity dynamics of the differential-drive robot."""

    name = "plant"
    probabilistic = True

    def __init__(self, cfg: PlantCfg, dt: float) -> None:
        self.cfg = cfg
        self.dt = float(dt)

    # ------------------------------------------------------------------ #
    def accel_limit(self, mu: np.ndarray) -> np.ndarray:
        """Traction-limited longitudinal acceleration magnitude (m/s^2)."""
        return self.cfg.traction_gain * np.asarray(mu, dtype=float)

    def nominal_accel(
        self, v: np.ndarray, v_cmd: np.ndarray, mu: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(achieved_accel, slip_fraction)`` before noise.

        ``slip_fraction`` is how far the *requested* acceleration exceeds what
        the floor can deliver; it is zero in ordinary driving and large during
        an emergency stop on a low-traction surface.
        """
        c = self.cfg
        v = np.asarray(v, dtype=float)
        v_cmd = np.asarray(v_cmd, dtype=float)
        mu = np.asarray(mu, dtype=float)

        tau = np.where(v_cmd < v, c.tau_brake, c.tau_accel)
        a_des = (v_cmd - v) / tau
        a_max = self.accel_limit(mu)
        a = np.clip(a_des, -a_max, a_max) + c.drive_bias
        slip = np.clip(np.abs(a_des) / np.maximum(a_max, 1e-6) - 1.0, 0.0, 4.0)
        return a, slip

    def noise_scale(
        self, a: np.ndarray, slip: np.ndarray, mu: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Heteroscedastic process-noise standard deviations per control step."""
        c = self.cfg
        a_ref = c.traction_gain * c.mu_ref
        accel_term = np.abs(a) / a_ref
        slip_term = np.minimum(slip, 1.5)
        sd_v = c.noise_v_base + c.noise_v_accel * accel_term + c.noise_v_slip * slip_term
        sd_w = c.noise_w_base + c.noise_w_accel * accel_term
        return sd_v, sd_w

    # ------------------------------------------------------------------ #
    def step(
        self,
        v: np.ndarray,
        omega: np.ndarray,
        v_cmd: np.ndarray,
        omega_cmd: np.ndarray,
        mu: np.ndarray,
        eps: np.ndarray,
        model_index=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        del model_index  # the hardware has exactly one dynamics
        c = self.cfg
        dt = self.dt
        v = np.asarray(v, dtype=float)
        omega = np.asarray(omega, dtype=float)
        v_cmd = np.asarray(v_cmd, dtype=float)
        omega_cmd = np.asarray(omega_cmd, dtype=float)
        mu = np.asarray(mu, dtype=float)
        eps = np.atleast_2d(np.asarray(eps, dtype=float))

        a, slip = self.nominal_accel(v, v_cmd, mu)

        # Yaw axis: first-order lag, traction limited as well.
        a_w = (omega_cmd - omega) / c.tau_yaw
        a_w_max = c.yaw_traction_gain * mu
        a_w = np.clip(a_w, -a_w_max, a_w_max)

        sd_v, sd_w = self.noise_scale(a, slip, mu)
        dv = a * dt + sd_v * eps[:, 0]
        dw = a_w * dt + sd_w * eps[:, 1]

        # Command deadband / stiction: a near-zero command with a near-stopped
        # robot latches to rest instead of creeping.
        at_rest = (np.abs(v_cmd) < c.deadband_v) & (np.abs(v) < c.stiction_v)
        dv = np.where(at_rest, -v, dv)
        dw = np.where(at_rest & (np.abs(omega_cmd) < c.deadband_v), -omega, dw)

        # Saturation of the resulting velocity.
        v_next = np.clip(v + dv, -0.15, c.v_max)
        return v_next - v, dw

    # ------------------------------------------------------------------ #
    def predict(self, features: np.ndarray):
        """Ground-truth mean and aleatoric sd, for benchmarking the twin."""
        f = np.atleast_2d(np.asarray(features, dtype=float))
        v, omega, v_cmd, omega_cmd, mu = f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4]
        zero = np.zeros((f.shape[0], 2))
        mean_dv, mean_dw = self.step(v, omega, v_cmd, omega_cmd, mu, zero)
        a, slip = self.nominal_accel(v, v_cmd, mu)
        sd_v, sd_w = self.noise_scale(a, slip, mu)
        mean = np.stack((mean_dv, mean_dw), axis=1)
        sd = np.stack((sd_v, sd_w), axis=1)
        return mean, sd, np.zeros_like(sd)


def stopping_distance(cfg: PlantCfg, v0: float, mu: float) -> float:
    """Closed-form-ish stopping distance from ``v0`` on traction ``mu``.

    Only used for reporting and for sanity checks on the configuration: the
    emergency stop commands ``v_cmd = 0``, so the achieved deceleration is
    traction limited at ``traction_gain * mu`` for essentially the whole stop.
    """
    a = max(cfg.traction_gain * float(mu), 1e-6)
    return float(v0) ** 2 / (2.0 * a)
