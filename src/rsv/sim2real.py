"""Closing the loop: replay predicted failures on hardware, then repair the twin.

An estimate produced inside a learned model is a statement about the model.
This stage turns it into a statement about the robot, in three steps.

**Replay.**  The most likely failure scenarios found by stress testing are run
on the plant with the *same latent vector*, so the obstacle sits where the twin
said, the range finder drops the same readings, the actuation lag is the same,
and the standardised process-noise draws are the same.  Pairing this way
(common random numbers) means a disagreement is model error, not a different
roll of the dice.  Because a real robot cannot be made to slip on command, each
scenario is also repeated with fresh noise, which gives the per-scenario failure
*probability* on hardware rather than a single coin flip.

**Score.**  The headline number is the validation rate: of the failures the twin
predicted, what fraction actually happen.  It is reported next to the twin's own
epistemic uncertainty along each trajectory, because the interesting question is
not only how often the twin is wrong but whether it knew it might be.  A
nominal, non-failing control group is replayed alongside, so that a high
validation rate cannot be an artefact of a robot that collides with everything.

**Repair.**  The replays are logged through the same encoder-and-camera
measurement chain as the original driving campaign, appended to the training
set, and the twin is refitted -- a DAgger-style correction that adds data
exactly where the safety argument depends on it and where the original campaign
had none.  The whole estimate is then recomputed on the repaired twin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config
from .data import Dataset, DrivingLog, logs_from_trajectories, rows_from_log
from .dynamics import build_features
from .plant import Plant
from .rollout import RolloutResult, simulate
from .scenario import ScenarioSpace
from .twin import Twin
from .utils import kmeans


@dataclass
class ReplayOutcome:
    """Result of replaying a set of scenarios on the hardware surrogate."""

    z: np.ndarray  # (n, d) scenarios replayed
    twin_robustness: np.ndarray  # (n,) margin predicted by the twin
    plant_robustness: np.ndarray  # (n,) margin on hardware, same disturbance draw
    plant_robustness_repeats: np.ndarray  # (n, R) with fresh process noise
    twin_failure: np.ndarray  # (n,) bool
    plant_failure: np.ndarray  # (n,) bool, paired
    plant_failure_rate: np.ndarray  # (n,) fraction of repeats that collided
    epistemic: np.ndarray  # (n,) mean epistemic sd along the predicted trajectory
    logs: List[DrivingLog] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.z.shape[0])


@dataclass
class ValidationReport:
    """Sim-to-real scorecard."""

    n_replayed: int
    n_predicted_failures: int
    validation_rate_paired: float
    validation_rate_probabilistic: float
    validation_rate_ci: List[float]
    control_group: Dict[str, float]
    robustness_bias: float
    robustness_rmse: float
    robustness_correlation: float
    epistemic_of_confirmed: float
    epistemic_of_refuted: float
    missed_failures: Optional[Dict[str, float]] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_replayed": self.n_replayed,
            "n_predicted_failures": self.n_predicted_failures,
            "validation_rate_paired": self.validation_rate_paired,
            "validation_rate_probabilistic": self.validation_rate_probabilistic,
            "validation_rate_ci": self.validation_rate_ci,
            "control_group": self.control_group,
            "robustness_bias_m": self.robustness_bias,
            "robustness_rmse_m": self.robustness_rmse,
            "robustness_correlation": self.robustness_correlation,
            "mean_epistemic_confirmed": self.epistemic_of_confirmed,
            "mean_epistemic_refuted": self.epistemic_of_refuted,
            "missed_failures": self.missed_failures,
        }


# --------------------------------------------------------------------------- #
# Choosing what to put on the robot
# --------------------------------------------------------------------------- #


def select_replay_scenarios(
    failures: np.ndarray,
    space: ScenarioSpace,
    n: int,
    rng: np.random.Generator,
    n_groups: int = 6,
) -> np.ndarray:
    """Pick scenarios to run on hardware: the most likely one from each mode.

    Hardware time is the scarcest resource in the whole pipeline, so the choice
    matters.  Ranking purely by likelihood would spend every run on the single
    dominant mode; grouping first and then taking the most likely members of
    each group buys coverage of the different ways the function can fail, which
    is what a validation campaign is for.
    """
    failures = np.atleast_2d(np.asarray(failures, dtype=float))
    if failures.shape[0] == 0:
        return failures
    n = int(min(n, failures.shape[0]))

    from .estimate.screening import relevant_coordinates

    coords = relevant_coordinates(failures, np.ones(failures.shape[0]), n_keep=24)
    k = int(min(n_groups, failures.shape[0]))
    _, labels = kmeans(failures[:, coords], k, rng)

    log_p = space.log_p(failures)
    chosen: List[int] = []
    per_group = max(1, n // max(k, 1))
    for j in range(k):
        rows = np.flatnonzero(labels == j)
        if rows.size == 0:
            continue
        order = rows[np.argsort(-log_p[rows])]
        chosen.extend(order[:per_group].tolist())
    # Top up with the next most likely overall, whatever group they came from.
    if len(chosen) < n:
        rest = [i for i in np.argsort(-log_p) if i not in set(chosen)]
        chosen.extend(rest[: n - len(chosen)])
    return failures[np.asarray(chosen[:n], dtype=int)]


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #


def replay_on_hardware(
    z: np.ndarray,
    twin: Twin,
    plant: Plant,
    cfg: Config,
    space: ScenarioSpace,
    rng: np.random.Generator,
    n_repeats: Optional[int] = None,
    record_logs: bool = True,
    log_id_offset: int = 10_000,
) -> ReplayOutcome:
    """Run scenarios on the twin and on hardware, paired and then repeated.

    ``log_id_offset`` starts the mission numbering of the recorded replays.
    Each group replayed in a round needs its own range: the logs are written to
    one directory by mission id, so two groups sharing a range silently
    overwrite each other and the refit quietly trains on less data than it
    reports.
    """
    z = np.atleast_2d(np.asarray(z, dtype=float))
    if z.shape[0] == 0:
        empty = np.zeros(0)
        return ReplayOutcome(z, empty, empty, np.zeros((0, 0)), empty.astype(bool),
                             empty.astype(bool), empty, empty, [])
    repeats = int(cfg.sim2real.n_repeats if n_repeats is None else n_repeats)

    twin_res = simulate(z, twin, cfg, space, record=True, compact=False)
    plant_res = simulate(z, plant, cfg, space, record=True, compact=False)

    # Fresh process-noise draws: the residual channels are resampled while the
    # exogenous disturbances (obstacle placement, dropouts, latency) are held.
    resid = space.channel_indices(2), space.channel_indices(3)
    repeat_rob = np.empty((z.shape[0], repeats))
    for r in range(repeats):
        zr = z.copy()
        for idx in resid:
            zr[:, idx] = rng.standard_normal((z.shape[0], idx.size))
        repeat_rob[:, r] = simulate(zr, plant, cfg, space).robustness

    epistemic = _trajectory_epistemic(twin_res, space.decode(z).mu, twin)

    logs: List[DrivingLog] = []
    if record_logs and plant_res.traces is not None:
        logs = logs_from_trajectories(
            plant_res.traces, space.decode(z).mu, cfg, rng, first_mission=log_id_offset
        )

    return ReplayOutcome(
        z=z,
        twin_robustness=twin_res.robustness,
        plant_robustness=plant_res.robustness,
        plant_robustness_repeats=repeat_rob,
        twin_failure=twin_res.failure,
        plant_failure=plant_res.failure,
        plant_failure_rate=(repeat_rob < 0.0).mean(axis=1),
        epistemic=epistemic,
        logs=logs,
    )


def _trajectory_epistemic(res: RolloutResult, mu: np.ndarray, twin: Twin) -> np.ndarray:
    """Mean epistemic sd the twin reports along each predicted trajectory.

    Averaged over the braking phase only -- the steps where the stop is latched
    -- because that is where model error turns into a collision.
    """
    if res.traces is None:
        return np.zeros(res.batch)
    tr = res.traces
    n, horizon = tr.v.shape
    out = np.zeros(n)
    for i in range(n):
        mask = tr.latched[i]
        if not mask.any():
            mask = np.ones(horizon, dtype=bool)
        feats = build_features(
            tr.v[i][mask],
            tr.omega[i][mask],
            tr.v_cmd[i][mask],
            tr.omega_cmd[i][mask],
            np.full(int(mask.sum()), mu[i]),
        )
        out[i] = float(np.mean(twin.epistemic_at(feats)))
    return out


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def validation_report(
    failure_replay: ReplayOutcome,
    control_replay: Optional[ReplayOutcome],
    confidence: float = 0.95,
    missed: Optional[Dict[str, float]] = None,
) -> ValidationReport:
    """Score the twin's failure predictions against hardware."""
    from .estimate.monte_carlo import clopper_pearson

    predicted = failure_replay.twin_failure
    n_pred = int(predicted.sum())
    confirmed = int((predicted & failure_replay.plant_failure).sum())
    rate = confirmed / n_pred if n_pred else float("nan")
    prob_rate = (
        float(failure_replay.plant_failure_rate[predicted].mean()) if n_pred else float("nan")
    )
    ci = clopper_pearson(confirmed, n_pred, confidence) if n_pred else [float("nan")] * 2

    err = failure_replay.plant_robustness - failure_replay.twin_robustness
    corr = float("nan")
    if (
        failure_replay.n > 2
        and np.std(failure_replay.twin_robustness) > 1e-9
        and np.std(failure_replay.plant_robustness) > 1e-9
    ):
        corr = float(np.corrcoef(failure_replay.twin_robustness, failure_replay.plant_robustness)[0, 1])

    conf_mask = predicted & failure_replay.plant_failure
    ref_mask = predicted & ~failure_replay.plant_failure

    control: Dict[str, float] = {}
    if control_replay is not None and control_replay.n:
        control = {
            "n": float(control_replay.n),
            "twin_failure_rate": float(control_replay.twin_failure.mean()),
            "hardware_failure_rate": float(control_replay.plant_failure.mean()),
            "robustness_bias_m": float(
                np.mean(control_replay.plant_robustness - control_replay.twin_robustness)
            ),
        }

    return ValidationReport(
        n_replayed=failure_replay.n,
        n_predicted_failures=n_pred,
        validation_rate_paired=float(rate),
        validation_rate_probabilistic=prob_rate,
        validation_rate_ci=[float(ci[0]), float(ci[1])],
        control_group=control,
        robustness_bias=float(np.mean(err)) if err.size else float("nan"),
        robustness_rmse=float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan"),
        robustness_correlation=corr,
        epistemic_of_confirmed=float(failure_replay.epistemic[conf_mask].mean()) if conf_mask.any() else float("nan"),
        epistemic_of_refuted=float(failure_replay.epistemic[ref_mask].mean()) if ref_mask.any() else float("nan"),
        missed_failures=missed,
    )


def missed_failure_check(
    plant_failures: Optional[np.ndarray],
    twin: Twin,
    cfg: Config,
    space: ScenarioSpace,
    rng: np.random.Generator,
    n_repeats: int = 8,
) -> Optional[Dict[str, float]]:
    """The converse error: hardware failures the twin does not reproduce.

    A twin can score a perfect validation rate and still be useless if it misses
    most of the ways the robot actually fails, so real failures -- collected by
    the naive Monte Carlo run on the plant -- are replayed in the twin.  Each is
    run several times because the twin is stochastic: a scenario it fails one
    time in eight has not been missed, merely made less likely.
    """
    if plant_failures is None or np.size(plant_failures) == 0:
        return None
    z = np.atleast_2d(np.asarray(plant_failures, dtype=float))
    resid = space.channel_indices(2), space.channel_indices(3)
    hits = np.zeros(z.shape[0])
    for _ in range(int(n_repeats)):
        zr = z.copy()
        for idx in resid:
            zr[:, idx] = rng.standard_normal((z.shape[0], idx.size))
        hits += simulate(zr, twin, cfg, space).failure
    reproduced = hits / float(n_repeats)
    return {
        "n_hardware_failures": float(z.shape[0]),
        "reproduced_at_least_once": float(np.mean(hits > 0)),
        "mean_reproduction_rate": float(np.mean(reproduced)),
    }


# --------------------------------------------------------------------------- #
# Repair
# --------------------------------------------------------------------------- #


def augment_dataset(
    dataset: Dataset,
    logs: Sequence[DrivingLog],
    cfg: Config,
    rng: np.random.Generator,
) -> Tuple[Dataset, Dict[str, object]]:
    """Append replayed-hardware rows to the training set (DAgger-style).

    The replays are processed by exactly the same measurement chain and feature
    builder as the original campaign, so nothing downstream can tell where a row
    came from -- the twin is not handed privileged information about the failure
    region, only more driving data that happens to be from it.

    The new rows are replicated ``refit_weight`` times.  A few dozen replays
    against ~17k nominal rows would otherwise be statistically invisible, and the
    point is precisely that these rows describe a region the bulk of the data
    says nothing about.
    """
    if not logs:
        return dataset, {"n_new_rows": 0}

    xs, ys = [], []
    for log in logs:
        x, y = rows_from_log(log, cfg)
        if x.shape[0]:
            xs.append(x)
            ys.append(y)
    if not xs:
        return dataset, {"n_new_rows": 0}

    x_new = np.concatenate(xs, axis=0)
    y_new = np.concatenate(ys, axis=0)
    reps = int(max(1, cfg.sim2real.refit_weight))
    x_rep = np.tile(x_new, (reps, 1))
    y_rep = np.tile(y_new, (reps, 1))

    augmented = Dataset(
        x_train=np.concatenate((dataset.x_train, x_rep), axis=0),
        y_train=np.concatenate((dataset.y_train, y_rep), axis=0),
        x_val=dataset.x_val,
        y_val=dataset.y_val,
        measurement_var=dataset.measurement_var,
        info=dict(dataset.info or {}),
    )
    info = {
        "n_new_rows": int(x_new.shape[0]),
        "replication_factor": reps,
        "n_rows_after": int(augmented.x_train.shape[0]),
        "new_row_share": float(x_rep.shape[0] / augmented.x_train.shape[0]),
        "new_mu_range": [float(x_new[:, 4].min()), float(x_new[:, 4].max())],
        "new_v_err_range": [float(x_new[:, 5].min()), float(x_new[:, 5].max())],
    }
    return augmented, info


def refit_twin(
    dataset: Dataset,
    cfg: Config,
    rng: np.random.Generator,
) -> Twin:
    """Fit a fresh twin on the augmented dataset.

    Refitting from scratch rather than fine-tuning keeps the second twin an
    honest function of its data, so the two rounds can be compared without
    worrying about what the optimiser remembered from the first.
    """
    twin = Twin(cfg.twin, rng)
    twin.fit(dataset.x_train, dataset.y_train, rng, epochs=cfg.sim2real.refit_epochs)
    twin.set_measurement_noise(dataset.measurement_var)
    return twin
