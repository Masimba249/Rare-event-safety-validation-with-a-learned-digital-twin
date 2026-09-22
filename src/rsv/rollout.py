"""Closed-loop episodes: latent vector in, safety verdict out.

Two access patterns share one implementation of the step body:

* :func:`simulate` runs a whole batch of scenarios in lock-step, which is what
  makes a 200k-episode naive Monte Carlo tractable in pure NumPy;
* :class:`EpisodeSimulator` exposes the same step incrementally over a copyable
  :class:`EpisodeState`, which is what lets the MCTS stress tester branch a tree
  without replaying every episode from time zero.

The same code drives the learned twin and the hardware surrogate, so replaying a
scenario found in simulation on the plant is a one-argument change.

Robustness
----------
The safety property is "the body never touches the obstacle".  Its quantitative
margin is

    rho = min_t  ( ||p_t - p_obstacle|| - (r_robot + r_obstacle) )

so ``rho < 0`` is a collision.  The minimum stops being tracked at first
contact, since nothing in the model describes what happens after the robot
hits something.  Keeping the continuous margin rather than only
the boolean lets the stress tester descend toward failures it has not reached
yet -- the standard device from the Adaptive Stress Testing literature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .config import Config
from .controller import RangeFinder, SafetyController
from .dynamics import Dynamics
from .scenario import Disturbances, ScenarioSpace
from .utils import batched

# Fields of the decoded disturbance record, in the order the state slices them.
_PER_SAMPLE = ("mu", "obstacle_x", "obstacle_y", "range_bias", "latency", "model_index")
_PER_STEP = ("range_noise", "dropout", "residual")


@dataclass
class Traces:
    """Per-step histories, recorded only for the handful of plotted episodes."""

    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    v: np.ndarray
    omega: np.ndarray
    v_cmd: np.ndarray
    omega_cmd: np.ndarray
    clearance_true: np.ndarray
    clearance_measured: np.ndarray
    margin: np.ndarray
    latched: np.ndarray


@dataclass
class RolloutResult:
    """Outcome of a batch of episodes."""

    robustness: np.ndarray  # (B,) signed margin at closest approach, or at first contact
    failure: np.ndarray  # (B,) bool, robustness < 0
    impact_speed: np.ndarray  # (B,) speed at the closest approach (m/s)
    detect_step: np.ndarray  # (B,) step the stop latched, -1 if never
    closest_step: np.ndarray  # (B,) step of closest approach
    final_speed: np.ndarray  # (B,) speed at the end of the episode
    traces: Optional[Traces] = None

    @property
    def batch(self) -> int:
        return int(self.robustness.shape[0])


@dataclass
class EpisodeState:
    """Everything needed to resume an episode mid-flight."""

    t: int
    idx: np.ndarray  # row in the caller's output arrays
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    v: np.ndarray
    omega: np.ndarray
    v_cmd_prev: np.ndarray
    held: np.ndarray  # last accepted range reading
    latched: np.ndarray
    ring: np.ndarray  # (B, lat_max + 1, 2) command delay line
    robustness: np.ndarray
    impact_speed: np.ndarray
    detect_step: np.ndarray
    closest_step: np.ndarray
    dist: Dict[str, np.ndarray]

    @property
    def batch(self) -> int:
        return int(self.x.shape[0])

    def select(self, keep: np.ndarray) -> "EpisodeState":
        """Restrict to a subset of rows (used to retire finished episodes)."""
        return EpisodeState(
            t=self.t,
            idx=self.idx[keep],
            x=self.x[keep],
            y=self.y[keep],
            theta=self.theta[keep],
            v=self.v[keep],
            omega=self.omega[keep],
            v_cmd_prev=self.v_cmd_prev[keep],
            held=self.held[keep],
            latched=self.latched[keep],
            ring=self.ring[keep],
            robustness=self.robustness[keep],
            impact_speed=self.impact_speed[keep],
            detect_step=self.detect_step[keep],
            closest_step=self.closest_step[keep],
            dist={k: v[keep] for k, v in self.dist.items()},
        )

    def copy(self) -> "EpisodeState":
        """Deep copy, so a tree node can branch without disturbing its parent."""
        return EpisodeState(
            t=self.t,
            idx=self.idx.copy(),
            x=self.x.copy(),
            y=self.y.copy(),
            theta=self.theta.copy(),
            v=self.v.copy(),
            omega=self.omega.copy(),
            v_cmd_prev=self.v_cmd_prev.copy(),
            held=self.held.copy(),
            latched=self.latched.copy(),
            ring=self.ring.copy(),
            robustness=self.robustness.copy(),
            impact_speed=self.impact_speed.copy(),
            detect_step=self.detect_step.copy(),
            closest_step=self.closest_step.copy(),
            dist=dict(self.dist),  # arrays are swapped wholesale, never mutated
        )

    def repeat(self, n: int) -> "EpisodeState":
        """Tile this state into ``n`` identical rows.

        Used to evaluate one tree node with a batch of independent rollouts:
        a single-row NumPy step is dominated by per-call overhead, so running
        32 futures together costs about what one costs and returns a far less
        noisy value estimate.
        """
        n = int(n)
        tile1 = lambda a: np.repeat(a, n, axis=0)
        return EpisodeState(
            t=self.t,
            idx=np.arange(n),
            x=tile1(self.x),
            y=tile1(self.y),
            theta=tile1(self.theta),
            v=tile1(self.v),
            omega=tile1(self.omega),
            v_cmd_prev=tile1(self.v_cmd_prev),
            held=tile1(self.held),
            latched=tile1(self.latched),
            ring=np.repeat(self.ring, n, axis=0),
            robustness=tile1(self.robustness),
            impact_speed=tile1(self.impact_speed),
            detect_step=tile1(self.detect_step),
            closest_step=tile1(self.closest_step),
            dist={k: np.repeat(v, n, axis=0) for k, v in self.dist.items()},
        )

    def with_disturbances(self, dist: Disturbances) -> "EpisodeState":
        """Copy, swapping in a longer-prefix disturbance realisation."""
        out = self.copy()
        out.dist = dist_dict(dist)
        return out


class EpisodeSimulator:
    """One control step of the closed loop, shared by every access pattern."""

    def __init__(self, cfg: Config, dynamics: Dynamics) -> None:
        self.cfg = cfg
        self.dynamics = dynamics
        self.finder = RangeFinder(cfg.controller, cfg.scenario.robot_radius)
        self.ctrl = SafetyController(cfg.controller, cfg.scenario.dt)
        self.hit_radius = cfg.scenario.robot_radius + cfg.scenario.obstacle_radius
        self.lat_max = len(cfg.scenario.latency_probs)

    # ------------------------------------------------------------------ #
    def init(self, dist: Disturbances) -> EpisodeState:
        sc = self.cfg.scenario
        b = dist.batch
        return EpisodeState(
            t=0,
            idx=np.arange(b),
            x=np.full(b, sc.start_x, dtype=float),
            y=np.full(b, sc.start_y, dtype=float),
            theta=np.full(b, sc.start_theta, dtype=float),
            v=np.zeros(b),
            omega=np.zeros(b),
            v_cmd_prev=np.zeros(b),
            held=np.full(b, self.cfg.controller.max_range, dtype=float),
            latched=np.zeros(b, dtype=bool),
            ring=np.zeros((b, self.lat_max + 1, 2), dtype=float),
            robustness=np.full(b, np.inf),
            impact_speed=np.zeros(b),
            detect_step=np.full(b, -1, dtype=np.int64),
            closest_step=np.zeros(b, dtype=np.int64),
            dist=dist_dict(dist),
        )

    # ------------------------------------------------------------------ #
    def step(self, st: EpisodeState) -> Dict[str, np.ndarray]:
        """Advance one control step in place; return the step's diagnostics."""
        cc, sc = self.cfg.controller, self.cfg.scenario
        dt, t, d = sc.dt, st.t, st.dist

        # ---- perception -------------------------------------------------- #
        clear_true = self.finder.clearance(
            st.x, st.y, st.theta, d["obstacle_x"], d["obstacle_y"], sc.obstacle_radius
        )
        raw = np.clip(clear_true + d["range_bias"] + d["range_noise"][:, t], 0.0, cc.max_range)
        measured = np.where(d["dropout"][:, t], st.held, raw)
        st.held = measured

        # ---- safety monitor and control ---------------------------------- #
        newly = (~st.latched) & (measured <= cc.stop_distance)
        st.latched = st.latched | newly
        st.detect_step = np.where(newly, t, st.detect_step)

        v_cmd = self.ctrl.speed_command(st.v_cmd_prev, st.latched)
        omega_cmd = self.ctrl.yaw_command(st.x, st.y, st.theta, sc.goal_x)
        st.v_cmd_prev = v_cmd

        # ---- actuation latency ------------------------------------------- #
        slot = t % (self.lat_max + 1)
        st.ring[:, slot, 0] = v_cmd
        st.ring[:, slot, 1] = omega_cmd
        applied = st.ring[np.arange(st.batch), (t - d["latency"]) % (self.lat_max + 1)]
        ready = t >= d["latency"]
        v_app = np.where(ready, applied[:, 0], 0.0)
        w_app = np.where(ready, applied[:, 1], 0.0)

        # ---- dynamics ----------------------------------------------------- #
        dv, dw = self.dynamics.step(
            st.v,
            st.omega,
            v_app,
            w_app,
            d["mu"],
            d["residual"][:, t, :],
            model_index=d["model_index"],
        )
        # Numerical guards shared by both models, so a badly extrapolating twin
        # cannot produce non-physical states.
        st.v = np.clip(st.v + dv, -0.3, 1.6)
        st.omega = np.clip(st.omega + dw, -4.0, 4.0)

        # Kinematic integration of the pose (midpoint heading).
        theta_mid = st.theta + 0.5 * st.omega * dt
        st.x = st.x + st.v * np.cos(theta_mid) * dt
        st.y = st.y + st.v * np.sin(theta_mid) * dt
        st.theta = st.theta + st.omega * dt

        # ---- safety margin ------------------------------------------------ #
        gap = np.hypot(d["obstacle_x"] - st.x, d["obstacle_y"] - st.y) - self.hit_radius
        # Stop tracking once contact has happened: the dynamics model says nothing
        # about what the robot does after hitting something, so the recorded margin
        # is the margin *at first contact*.  This also makes the number independent
        # of whether the batch driver retired the episode early.
        improved = (gap < st.robustness) & (st.robustness >= 0.0)
        st.robustness = np.where(improved, gap, st.robustness)
        st.impact_speed = np.where(improved, st.v, st.impact_speed)
        st.closest_step = np.where(improved, t, st.closest_step)
        st.t = t + 1

        return {
            "gap": gap,
            "clearance_true": clear_true,
            "clearance_measured": measured,
            "v_cmd": v_cmd,
            "omega_cmd": omega_cmd,
        }

    # ------------------------------------------------------------------ #
    def finished(self, st: EpisodeState, gap: np.ndarray) -> np.ndarray:
        """Episodes whose verdict can no longer change.

        Safe to retire once the robot has collided (the model is meaningless
        after contact), has come to rest with the stop latched (it can never
        move again), or has driven past the obstacle.
        """
        collided = gap < 0.0
        at_rest = st.latched & (st.v < 0.015)
        passed = (st.x - st.dist["obstacle_x"] > self.hit_radius) & (st.v >= 0.0)
        return collided | at_rest | passed

    def run(self, st: EpisodeState, until: Optional[int] = None) -> EpisodeState:
        """Advance to step ``until`` (default: the end of the horizon).

        Stops early once every episode in the batch has a settled verdict,
        which is what keeps the MCTS rollouts cheap.
        """
        stop = int(self.cfg.scenario.horizon if until is None else until)
        while st.t < stop:
            gap = self.step(st)["gap"]
            if bool(self.finished(st, gap).all()):
                break
        return st


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #


def simulate(
    z: np.ndarray,
    dynamics: Dynamics,
    cfg: Config,
    space: ScenarioSpace,
    record: bool = False,
    compact: bool = True,
) -> RolloutResult:
    """Run one batch of closed-loop episodes defined by latent vectors ``z``."""
    z = np.atleast_2d(np.asarray(z, dtype=float))
    return simulate_disturbances(space.decode(z), dynamics, cfg, record=record, compact=compact)


def simulate_disturbances(
    dist: Disturbances,
    dynamics: Dynamics,
    cfg: Config,
    record: bool = False,
    compact: bool = True,
) -> RolloutResult:
    """Simulate from already-decoded disturbances (used by the replay stage)."""
    horizon = cfg.scenario.horizon
    b = dist.batch
    sim = EpisodeSimulator(cfg, dynamics)
    st = sim.init(dist)

    robustness = np.full(b, np.inf)
    impact_speed = np.zeros(b)
    detect_step = np.full(b, -1, dtype=np.int64)
    closest_step = np.zeros(b, dtype=np.int64)
    final_speed = np.zeros(b)

    rec: Optional[Dict[str, np.ndarray]] = None
    if record:
        compact = False
        keys = (
            "x",
            "y",
            "theta",
            "v",
            "omega",
            "v_cmd",
            "omega_cmd",
            "clearance_true",
            "clearance_measured",
            "margin",
        )
        rec = {k: np.zeros((b, horizon)) for k in keys}
        rec["latched"] = np.zeros((b, horizon), dtype=bool)

    def harvest(state: EpisodeState, rows: np.ndarray) -> None:
        out = state.idx[rows]
        robustness[out] = state.robustness[rows]
        impact_speed[out] = state.impact_speed[rows]
        detect_step[out] = state.detect_step[rows]
        closest_step[out] = state.closest_step[rows]
        final_speed[out] = state.v[rows]

    for t in range(horizon):
        info = sim.step(st)
        if rec is not None:
            rec["x"][st.idx, t] = st.x
            rec["y"][st.idx, t] = st.y
            rec["theta"][st.idx, t] = st.theta
            rec["v"][st.idx, t] = st.v
            rec["omega"][st.idx, t] = st.omega
            rec["v_cmd"][st.idx, t] = info["v_cmd"]
            rec["omega_cmd"][st.idx, t] = info["omega_cmd"]
            rec["clearance_true"][st.idx, t] = info["clearance_true"]
            rec["clearance_measured"][st.idx, t] = info["clearance_measured"]
            rec["margin"][st.idx, t] = info["gap"]
            rec["latched"][st.idx, t] = st.latched

        if compact and t % 8 == 7 and t < horizon - 1:
            done = sim.finished(st, info["gap"])
            if done.any():
                harvest(st, done)
                st = st.select(~done)
                if st.batch == 0:
                    break

    if st.batch:
        harvest(st, np.ones(st.batch, dtype=bool))

    failure = robustness < 0.0
    traces = Traces(**rec) if rec is not None else None
    return RolloutResult(
        robustness=robustness,
        failure=failure,
        impact_speed=np.where(failure, impact_speed, 0.0),
        detect_step=detect_step,
        closest_step=closest_step,
        final_speed=final_speed,
        traces=traces,
    )


def dist_dict(dist: Disturbances) -> Dict[str, np.ndarray]:
    """Expose a :class:`Disturbances` record as a dict of sliceable arrays."""
    return {name: getattr(dist, name) for name in _PER_SAMPLE + _PER_STEP}


# --------------------------------------------------------------------------- #
# Batched convenience drivers
# --------------------------------------------------------------------------- #


def robustness_of(
    z: np.ndarray,
    dynamics: Dynamics,
    cfg: Config,
    space: ScenarioSpace,
    batch: Optional[int] = None,
) -> np.ndarray:
    """Minimum safety margin for each latent vector, chunked over memory."""
    z = np.atleast_2d(np.asarray(z, dtype=float))
    n = z.shape[0]
    size = int(batch or cfg.estimate.batch)
    out = np.empty(n)
    for lo, hi in batched(n, size):
        out[lo:hi] = simulate(z[lo:hi], dynamics, cfg, space).robustness
    return out


def failures_of(
    z: np.ndarray,
    dynamics: Dynamics,
    cfg: Config,
    space: ScenarioSpace,
    batch: Optional[int] = None,
) -> np.ndarray:
    """Boolean failure indicator for each latent vector."""
    return robustness_of(z, dynamics, cfg, space, batch=batch) < 0.0
