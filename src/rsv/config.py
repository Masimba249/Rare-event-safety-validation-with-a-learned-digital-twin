"""Typed configuration for the rare-event safety-validation pipeline.

Every stage (data collection, twin fitting, adaptive stress testing, importance
sampling, hardware replay) reads its parameters from a single :class:`Config`,
so a run is fully described by one YAML file plus one master seed.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


@dataclass
class RunCfg:
    """Top-level bookkeeping."""

    seed: int = 20260922
    out_dir: str = "results"
    profile: str = "default"  # informational label carried into artifacts


@dataclass
class ScenarioCfg:
    """Episode geometry and the nominal (operational) disturbance distribution.

    The disturbance space is parameterised by a standard-normal latent vector
    ``z ~ N(0, I_d)``.  Every physical disturbance is a deterministic transform
    of ``z``: Gaussian channels are affine, discrete channels use an inverse
    CDF.  The nominal density therefore stays analytic, which is what makes the
    importance-sampling likelihood ratios exact.
    """

    dt: float = 0.05
    horizon: int = 140  # control steps -> 7.0 s

    # --- episode layout ---------------------------------------------------- #
    start_x: float = 0.0
    start_y: float = 0.0
    start_theta: float = 0.0
    goal_x: float = 4.0
    obstacle_x_mean: float = 2.60  # nominal obstacle placement (m)
    obstacle_x_sd: float = 0.06  # placement error of the test fixture (m)
    obstacle_y_sd: float = 0.07  # lateral placement error (m)
    obstacle_radius: float = 0.16
    robot_radius: float = 0.12

    # --- static latent channels -------------------------------------------- #
    mu_mean: float = 0.46  # floor traction coefficient
    mu_sd: float = 0.075
    mu_min: float = 0.06
    mu_max: float = 1.20
    sensor_bias_sd: float = 0.012  # range-finder calibration bias (m)
    latency_probs: List[float] = field(
        default_factory=lambda: [0.55, 0.33, 0.09, 0.03]
    )  # P(actuation latency = 1, 2, 3, 4 control steps)

    # --- per-step latent channels ------------------------------------------ #
    range_noise_sd: float = 0.016  # range-finder noise (m)
    dropout_prob: float = 0.035  # P(stale range reading at a step)

    # --- latent layout ----------------------------------------------------- #
    # static: [mu, obstacle_x, obstacle_y, sensor_bias, latency, model_index]
    n_static: int = 6
    # per step: [range noise, dropout, dynamics residual dv, residual dw]
    n_per_step: int = 4

    def __post_init__(self) -> None:
        total = float(sum(self.latency_probs))
        if total <= 0:
            raise ValueError("latency_probs must be positive")
        if abs(total - 1.0) > 1e-9:
            self.latency_probs = [p / total for p in self.latency_probs]

    @property
    def latent_dim(self) -> int:
        return self.n_static + self.n_per_step * self.horizon


@dataclass
class ControllerCfg:
    """The safety function under test: cruise to a goal with an obstacle stop."""

    v_nominal: float = 0.70  # cruise speed (m/s)
    stop_distance: float = 0.62  # clearance that triggers the emergency stop (m)
    beam_half_angle_deg: float = 8.0  # range-finder field of view
    n_beams: int = 3
    max_range: float = 4.0
    heading_gain: float = 1.4
    max_yaw_rate: float = 1.2
    accel_limit: float = 1.0  # ramp applied to the cruise command (m/s^2)


@dataclass
class PlantCfg:
    """Ground-truth robot -- "the hardware".

    Only :mod:`rsv.plant` and the hardware-replay stage may use these numbers;
    the twin never sees them.  The traction limit ``a_max = traction_gain * mu``
    is the nonlinearity that nominal logs barely excite, and it is therefore the
    source of the sim-to-real gap the closing loop has to repair.
    """

    tau_accel: float = 0.32  # first-order lag while accelerating (s)
    tau_brake: float = 0.11  # first-order lag while braking (s)
    tau_yaw: float = 0.18
    traction_gain: float = 3.2  # a_max = traction_gain * mu (m/s^2)
    yaw_traction_gain: float = 6.0
    deadband_v: float = 0.035  # command deadband (m/s)
    stiction_v: float = 0.02
    v_max: float = 1.10
    # heteroscedastic process noise: base + accel-driven + slip-driven
    noise_v_base: float = 0.0035
    noise_v_accel: float = 0.0090
    noise_v_slip: float = 0.0160
    noise_w_base: float = 0.0060
    noise_w_accel: float = 0.0110
    drive_bias: float = -0.004  # small persistent motor asymmetry (m/s^2)
    mu_ref: float = 0.50


@dataclass
class DataCfg:
    """The logging campaign used to fit the twin (the "real driving data")."""

    n_missions: int = 140
    mission_steps: int = 160
    # Operational surface distribution during logging: deliberately narrower and
    # higher-traction than the tail the stress tester will later explore.
    mu_mean: float = 0.52
    mu_sd: float = 0.060
    speed_lo: float = 0.25
    speed_hi: float = 0.85
    p_gentle_stop: float = 0.40  # fraction of missions containing a soft stop
    p_hard_stop: float = 0.30  # fraction that are emergency-brake tests
    # measurement chain
    encoder_noise_sd: float = 0.006  # wheel-odometry velocity noise (m/s)
    encoder_slip_gain: float = 0.85  # encoders over-read while wheels slip
    aruco_xy_noise_sd: float = 0.0008  # overhead camera position noise (m)
    aruco_theta_noise_sd: float = 0.0015  # rad
    aruco_dropout: float = 0.01
    smooth_halfwidth: int = 2  # half width of the camera velocity window
    command_jump_threshold: float = 0.10  # |d cmd_v| above this marks a discontinuity
    val_fraction: float = 0.2


@dataclass
class TwinCfg:
    """Probabilistic ensemble (deep ensemble with heteroscedastic Gaussian heads)."""

    backend: str = "ensemble"  # {"ensemble", "gp"}
    n_members: int = 5
    hidden: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "tanh"
    epochs: int = 220
    batch_size: int = 256
    lr: float = 3e-3
    weight_decay: float = 1e-5
    bootstrap: bool = True
    min_sigma: float = 5e-3  # bounds on the predictive sd, in standardised
    max_sigma: float = 3.0   # target units (see models/mlp.py)
    beta_nll: float = 0.5  # 0 = plain NLL, 1 = least-squares gradient on the mean
    # GP backend
    gp_inducing: int = 500
    gp_restarts: int = 2


@dataclass
class ASTCfg:
    """Adaptive stress testing: MCTS (Lee et al.) plus a cross-entropy searcher."""

    # --- MCTS -------------------------------------------------------------- #
    mcts_iterations: int = 2500
    mcts_depth_chunk: int = 5  # control steps committed per tree action
    mcts_c_uct: float = 1.1
    mcts_pw_k: float = 2.0  # progressive widening: |children| <= k * n^alpha
    mcts_pw_k_root: float = 9.0
    # The root deserves a far wider fan-out than the per-step levels, because
    # it chooses the episode's static disturbances and those decide whether a
    # failure is reachable at all.  At k = 2 the root acquires ~70 candidates
    # over a full search; the lateral failure mode occupies a band roughly 1%
    # of that sampling distribution wide, so the search found it zero times.
    # Widening the root is cheap -- each extra child is one rollout -- and it
    # is the difference between finding that mode and missing it entirely.
    mcts_pw_alpha: float = 0.45
    mcts_action_sd: float = 1.35  # spread of proposed per-step latent actions
    mcts_static_action_sd: float = 2.6  # spread of proposed static disturbances
    mcts_rollout_batch: int = 32  # independent futures evaluated per tree node
    mcts_miss_penalty: float = 120.0
    # Weight on the terminal miss distance.  It has to be large enough that a
    # collision always outranks a near miss: the penalty times a typical miss
    # (~0.2 m here) must exceed the log-likelihood cost of reaching the failure
    # region (~10 nats).  Too small and the search prefers staying nominal.
    mcts_top_k: int = 40  # failure paths kept for the proposal fit

    # --- cross-entropy ----------------------------------------------------- #
    cem_restarts: int = 13
    # 1 restart from nominal, then each of the 6 static disturbance channels
    # seeded in BOTH directions.  One-sided seeding is not enough: a mode can
    # live in either tail of a channel, and a symmetric pair (obstacle offset
    # to the left, obstacle offset to the right) is easy to half-miss -- which
    # costs half that mode's probability with no visible symptom.
    cem_iterations: int = 22
    cem_population: int = 1200
    cem_elite_frac: float = 0.15
    cem_smoothing: float = 0.65
    cem_sd_floor: float = 0.35
    cem_sd_init: float = 1.0
    cem_seed_shift: float = 2.0  # restart seed: tilt of one static channel

    # --- dimension screening ------------------------------------------------ #
    # A coordinate keeps its fitted shift only insofar as the shift exceeds
    # `screen_c` times its own standard error; otherwise it is returned to the
    # nominal law.  In 566 dimensions this is what keeps p/q from underflowing.
    # The search can afford a looser screen than the final proposal: leaking a
    # few hundredths of a standard deviation into irrelevant coordinates costs
    # well under a nat of Kullback-Leibler divergence, while screening hard
    # enough to zero them stalls the search before real shifts have grown.
    screen_c_search: float = 2.0
    screen_c_proposal: float = 2.5

    # --- proposal construction --------------------------------------------- #
    n_modes: int = 8  # mixture components fitted to the discovered failures
    cluster_dimensions: int = 24  # subspace used to separate the modes
    kmeans_iters: int = 40


@dataclass
class EstimateCfg:
    """Naive Monte Carlo and importance-sampling estimators."""

    n_mc: int = 200_000
    n_is: int = 20_000
    batch: int = 5_000
    defensive_alpha: float = 0.15  # weight on the nominal component of q
    hypothesis_weight: float = 0.12  # weight shared by the single-factor components
    hypothesis_shift: float = 3.5  # how far into the tail each one sits
    mode_sd_floor: float = 0.45
    mode_sd_ceiling: float = 1.30
    adapt_rounds: int = 3  # proposal-refinement rounds before the final estimate
    adapt_samples: int = 10_000  # episodes per refinement round
    min_component_mass: float = 0.02  # drop mixture components below this share
    min_fit_ess_fraction: float = 0.25  # tempering target for proposal-fitting weights
    bootstrap: int = 2_000
    confidence: float = 0.95
    target_rel_error: float = 0.10  # for the "simulations needed" comparison
    convergence_points: int = 24
    plant_reference_n: int = 3_000_000  # oracle MC on the hardware surrogate


@dataclass
class Sim2RealCfg:
    """Hardware replay and the DAgger-style model update."""

    n_replay: int = 60  # most-likely failure scenarios replayed
    n_repeats: int = 8  # fresh-noise repeats per scenario on hardware
    n_nominal_replay: int = 120
    # Nominal scenarios replayed alongside the predicted failures.  They serve
    # two purposes: as a control group they show the hardware does not simply
    # collide with everything, and as training data they keep the aggregated
    # set representative.  Replaying only failures makes every new row a
    # low-traction emergency stop, and refitting on that mixture buys accuracy
    # in the tail by losing it everywhere else -- measured here as braking error
    # above the logged traction range going from 0.37 to 0.44 m/s^2 with 40
    # nominal replays, against 0.15 with 120.
    refit_epochs: int = 260
    refit_weight: int = 1
    # Replication factor for the replayed rows.  DAgger aggregates datasets; it
    # does not reweight them, and there is a measured reason not to.  At 3x the
    # replayed rows became 58% of the training set and, being almost all
    # low-traction emergency stops, they pulled the fit so far that braking
    # authority at cruise speed became worse than before the loop (mean error
    # 0.29 m/s^2 against 0.23 before).  At 1x -- a 31% share -- the same data
    # halves that error instead, to 0.13.


@dataclass
class Config:
    run: RunCfg = field(default_factory=RunCfg)
    scenario: ScenarioCfg = field(default_factory=ScenarioCfg)
    controller: ControllerCfg = field(default_factory=ControllerCfg)
    plant: PlantCfg = field(default_factory=PlantCfg)
    data: DataCfg = field(default_factory=DataCfg)
    twin: TwinCfg = field(default_factory=TwinCfg)
    ast: ASTCfg = field(default_factory=ASTCfg)
    estimate: EstimateCfg = field(default_factory=EstimateCfg)
    sim2real: Sim2RealCfg = field(default_factory=Sim2RealCfg)

    # ------------------------------------------------------------------ #
    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "Config":
        return _build(Config, d or {})

    @staticmethod
    def load(path: Optional[str]) -> "Config":
        if path is None:
            return Config()
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return Config.from_dict(raw)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def override(self, dotted: Dict[str, Any]) -> "Config":
        """Return a copy with ``{"estimate.n_mc": 1000}``-style overrides applied."""
        d = self.to_dict()
        for key, value in dotted.items():
            parts = key.split(".")
            node = d
            for p in parts[:-1]:
                if p not in node:
                    raise KeyError("unknown config section '%s' in '%s'" % (p, key))
                node = node[p]
            if parts[-1] not in node:
                raise KeyError("unknown config key '%s'" % key)
            node[parts[-1]] = value
        return Config.from_dict(d)


_SECTIONS = {
    "RunCfg": RunCfg,
    "ScenarioCfg": ScenarioCfg,
    "ControllerCfg": ControllerCfg,
    "PlantCfg": PlantCfg,
    "DataCfg": DataCfg,
    "TwinCfg": TwinCfg,
    "ASTCfg": ASTCfg,
    "EstimateCfg": EstimateCfg,
    "Sim2RealCfg": Sim2RealCfg,
}


def _resolve(name: Any) -> Any:
    if isinstance(name, str):
        return _SECTIONS.get(name.strip("\"'"), name)
    return name


def _build(cls, data: Dict[str, Any]):
    """Recursively instantiate nested dataclasses, rejecting unknown keys."""
    kwargs: Dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in data.items():
        if key not in known:
            raise KeyError("unknown config key '%s' for %s" % (key, cls.__name__))
        ftype = _resolve(known[key].type)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _build(ftype, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def parse_overrides(items: Optional[List[str]]) -> Dict[str, Any]:
    """Parse ``section.key=value`` strings into a dict of YAML scalars."""
    out: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError("override '%s' must look like section.key=value" % item)
        key, value = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(value)
    return out
