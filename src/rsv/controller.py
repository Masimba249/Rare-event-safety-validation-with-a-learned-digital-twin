"""The safety function under test: cruise to a goal, stop for obstacles.

The robot runs a simple waypoint controller and an obstacle-stop monitor fed by
a forward-facing range finder modelled as a narrow fan of beams.  When the
measured clearance drops below ``stop_distance`` the monitor latches an
emergency stop (commanded velocity zero) and never releases it.

Three properties of this monitor create the rare-failure structure the rest of
the project studies:

* it acts on a *measured* clearance, so noise, a calibration bias and stale
  readings all delay the trigger;
* the stop command is only a request -- whether the robot actually stops in time
  depends on floor traction, which the monitor cannot observe;
* the beam fan is narrow, so an obstacle offset far enough to the side is
  invisible at close range yet still wide enough for the body to clip it.

The first two give a "stops too late" failure mode, the third a distinct
"never saw it" mode.  A unimodal importance-sampling proposal would miss one of
them, which is why the proposal in :mod:`rsv.estimate.importance` is a mixture.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import ControllerCfg


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angles to ``(-pi, pi]``."""
    return (np.asarray(a, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


class RangeFinder:
    """Forward-facing fan of range beams against a circular obstacle."""

    def __init__(self, cfg: ControllerCfg, robot_radius: float) -> None:
        self.cfg = cfg
        self.robot_radius = float(robot_radius)
        half = np.deg2rad(cfg.beam_half_angle_deg)
        n = int(max(1, cfg.n_beams))
        self.offsets = np.zeros(1) if n == 1 else np.linspace(-half, half, n)

    def clearance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
        obs_x: np.ndarray,
        obs_y: np.ndarray,
        obs_r: float,
    ) -> np.ndarray:
        """Minimum true clearance over the beam fan, in metres.

        Returns ``max_range`` when no beam intersects the obstacle -- the blind
        spot that makes the lateral failure mode possible.
        """
        x, y, theta = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (x, y, theta))
        dx = np.asarray(obs_x, dtype=float) - x
        dy = np.asarray(obs_y, dtype=float) - y

        ang = theta[:, None] + self.offsets[None, :]  # (B, n_beams)
        ux, uy = np.cos(ang), np.sin(ang)

        t_ca = dx[:, None] * ux + dy[:, None] * uy  # projection onto each beam
        d2 = (dx * dx + dy * dy)[:, None]
        perp2 = np.maximum(d2 - t_ca * t_ca, 0.0)

        disc = obs_r * obs_r - perp2
        hit = (disc > 0.0) & (t_ca > 0.0)
        t_hit = t_ca - np.sqrt(np.maximum(disc, 0.0))

        rng = np.where(hit, t_hit - self.robot_radius, self.cfg.max_range)
        return np.clip(np.min(rng, axis=1), 0.0, self.cfg.max_range)


class SafetyController:
    """Waypoint tracking plus the latching obstacle-stop monitor."""

    def __init__(self, cfg: ControllerCfg, dt: float) -> None:
        self.cfg = cfg
        self.dt = float(dt)

    def speed_command(
        self, v_cmd_prev: np.ndarray, latched: np.ndarray
    ) -> np.ndarray:
        """Ramp toward cruise speed, or command an immediate stop once latched.

        The ramp keeps ordinary driving inside the traction envelope; the
        emergency stop deliberately does not, which is what makes the stop
        traction limited on the real robot.
        """
        target = np.where(latched, 0.0, self.cfg.v_nominal)
        step = self.cfg.accel_limit * self.dt
        delta = np.clip(target - v_cmd_prev, -np.inf, step)
        v_cmd = np.where(latched, 0.0, v_cmd_prev + delta)
        return np.clip(v_cmd, 0.0, self.cfg.v_nominal)

    def yaw_command(
        self, x: np.ndarray, y: np.ndarray, theta: np.ndarray, goal_x: float
    ) -> np.ndarray:
        """Proportional heading control toward the goal point ``(goal_x, 0)``."""
        heading = np.arctan2(-y, np.maximum(goal_x - x, 1e-3))
        err = wrap_angle(heading - theta)
        return np.clip(self.cfg.heading_gain * err, -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)

    def monitor(
        self, measured_clearance: np.ndarray, latched: np.ndarray
    ) -> np.ndarray:
        """Latching obstacle-stop rule."""
        return latched | (measured_clearance <= self.cfg.stop_distance)


def expected_stop_margin(
    ctrl: ControllerCfg, traction_gain: float, mu: float, latency_steps: int, dt: float
) -> Tuple[float, float]:
    """Nominal margin bookkeeping for the configuration sanity report.

    Returns ``(margin, stopping_distance)`` in metres: how much clearance is
    left over after reaction delay and a traction-limited stop from cruise.
    """
    v = ctrl.v_nominal
    react = v * latency_steps * dt
    brake = v * v / (2.0 * max(traction_gain * mu, 1e-6))
    return ctrl.stop_distance - react - brake, brake
