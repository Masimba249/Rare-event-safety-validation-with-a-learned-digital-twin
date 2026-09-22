"""The end-to-end study, one function per stage.

Stages communicate through files in the output directory, so any of them can be
re-run on its own::

    collect   drive the logging campaign, build the training set
    fit       fit the probabilistic twin, check its calibration
    stress    adaptive stress testing on the twin (MCTS + cross-entropy)
    estimate  naive Monte Carlo and adaptive importance sampling, with bounds
    validate  replay predicted failures on hardware, score them
    refit     add the replays to the training set, refit, re-estimate
    reference the hardware oracle: a very large Monte Carlo on the plant
    report    assemble figures and the written summary

``run_all`` runs them in order.  Every stage returns a JSON-able dict which is
also written to disk, and every random draw comes from a named, independently
seeded stream, so re-running a stage reproduces it exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .ast import AdaptiveStressTest, multi_start_cem
from .config import Config
from .data import (
    Dataset,
    build_dataset,
    collect_logs,
    estimator_comparison,
    read_logs_csv,
    write_logs_csv,
)
from .estimate import (
    accuracy_report,
    adaptive_importance_sample,
    compare,
    coverage_diagnostic,
    naive_monte_carlo,
    per_member_estimates,
)
from .plant import Plant
from .controller import expected_stop_margin
from .rollout import simulate
from .scenario import ScenarioSpace
from .sim2real import (
    augment_dataset,
    missed_failure_check,
    refit_twin,
    replay_on_hardware,
    select_replay_scenarios,
    validation_report,
)
from .twin import Twin
from .utils import LOG, ensure_dir, load_json, load_npz, rng_for, save_json, save_npz, timed


class Artifacts:
    """Where each stage keeps its output."""

    def __init__(self, out_dir: str) -> None:
        self.root = ensure_dir(out_dir)
        ensure_dir(str(self.root / "figures"))

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def replay_logs_dir(self, rnd: int = 1) -> Path:
        """Replay logs are kept per round.

        Round 2 must not overwrite round 1: the round-1 replays are the training
        data the refit was built from, and a reader comparing the two rounds
        needs both on disk.
        """
        return self.root / ("logs_replay_round%d" % rnd)

    @property
    def dataset(self) -> Path:
        return self.root / "dataset.npz"

    def twin(self, rnd: int) -> Path:
        return self.root / ("twin_round%d.npz" % rnd)

    def stress(self, rnd: int) -> Path:
        return self.root / ("stress_round%d.npz" % rnd)

    def stage(self, name: str) -> Path:
        return self.root / ("%s.json" % name)

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def summary(self) -> Path:
        return self.root / "summary.md"


def _space(cfg: Config) -> ScenarioSpace:
    """The disturbance space, sized to the twin's ensemble."""
    return ScenarioSpace(cfg.scenario, n_models=cfg.twin.n_members)


def _load_dataset(art: Artifacts) -> Dataset:
    d = load_npz(str(art.dataset))
    return Dataset(
        x_train=d["x_train"],
        y_train=d["y_train"],
        x_val=d["x_val"],
        y_val=d["y_val"],
        measurement_var=d["measurement_var"],
        info=load_json(str(art.stage("collect"))).get("dataset"),
    )


def _load_twin(cfg: Config, art: Artifacts, rnd: int) -> Twin:
    twin = Twin.load(str(art.twin(rnd)), cfg.twin, rng_for(cfg.run.seed, "twin", rnd))
    return twin


# --------------------------------------------------------------------------- #
# 1. Data collection
# --------------------------------------------------------------------------- #


def stage_collect(cfg: Config, art: Artifacts, write_csv: bool = True) -> Dict[str, Any]:
    """Drive the logging campaign and build the twin's training set."""
    with timed("collect") as t:
        logs = collect_logs(cfg, rng_for(cfg.run.seed, "data"))
        dataset = build_dataset(logs, cfg, rng_for(cfg.run.seed, "data", 1))
        estimators = estimator_comparison(logs, cfg)
        if write_csv:
            write_logs_csv(logs, str(art.logs_dir))

    save_npz(
        str(art.dataset),
        x_train=dataset.x_train,
        y_train=dataset.y_train,
        x_val=dataset.x_val,
        y_val=dataset.y_val,
        measurement_var=np.asarray(dataset.measurement_var),
    )
    out = {
        "stage": "collect",
        "seconds": t["seconds"],
        "n_missions": len(logs),
        "mission_steps": len(logs[0]),
        "n_train_rows": dataset.n_train,
        "n_val_rows": dataset.n_val,
        "target_measurement_sd": np.sqrt(dataset.measurement_var).tolist(),
        "target_sd": dataset.y_train.std(axis=0).tolist(),
        "dataset": dataset.info,
        "velocity_estimators": estimators,
        "logs_csv": str(art.logs_dir) if write_csv else None,
    }
    save_json(str(art.stage("collect")), out)
    return out


# --------------------------------------------------------------------------- #
# 2. Twin
# --------------------------------------------------------------------------- #


def stage_fit(cfg: Config, art: Artifacts) -> Dict[str, Any]:
    """Fit the probabilistic twin and report how well calibrated it is."""
    dataset = _load_dataset(art)
    with timed("fit twin") as t:
        twin = Twin(cfg.twin, rng_for(cfg.run.seed, "twin"))
        twin.fit(dataset.x_train, dataset.y_train, rng_for(cfg.run.seed, "twin", 1))
        twin.set_measurement_noise(dataset.measurement_var)
    twin.save(str(art.twin(1)))

    out = _twin_report(cfg, twin, dataset, label="round 1")
    out.update({"stage": "fit", "seconds": t["seconds"]})
    save_json(str(art.stage("fit")), out)
    return out


def _twin_report(cfg: Config, twin: Twin, dataset: Dataset, label: str) -> Dict[str, Any]:
    """Calibration plus the decision-relevant slice: braking against traction."""
    plant = Plant(cfg.plant, cfg.scenario.dt)
    rep = twin.calibration_report(dataset.x_val, dataset.y_val, label=label)

    mu_grid = np.linspace(0.12, 0.80, 35)
    curve = twin.braking_curve(mu_grid, cfg.controller.v_nominal, cfg.scenario.dt)
    truth = -np.minimum(
        cfg.plant.traction_gain * mu_grid, cfg.controller.v_nominal / cfg.plant.tau_brake
    )
    logged_lo, logged_hi = float(dataset.x_train[:, 4].min()), float(dataset.x_train[:, 4].max())
    inside = (mu_grid >= logged_lo) & (mu_grid <= logged_hi)

    return {
        "backend": cfg.twin.backend,
        "n_members": twin.n_models,
        "n_train_rows": dataset.n_train,
        "calibration": rep,
        "logged_traction_range": [logged_lo, logged_hi],
        "braking_curve": {
            "mu": mu_grid.tolist(),
            "twin_accel": curve["accel_mean"].tolist(),
            "true_accel": truth.tolist(),
            "epistemic": curve["accel_epistemic"].tolist(),
            "aleatoric": curve["accel_aleatoric"].tolist(),
        },
        "braking_error": {
            "inside_logged_range_mae": float(np.mean(np.abs(curve["accel_mean"][inside] - truth[inside]))),
            "outside_logged_range_mae": float(
                np.mean(np.abs(curve["accel_mean"][~inside] - truth[~inside]))
            ) if (~inside).any() else float("nan"),
            "mean_epistemic_inside": float(np.mean(curve["accel_epistemic"][inside])),
            "mean_epistemic_outside": float(np.mean(curve["accel_epistemic"][~inside]))
            if (~inside).any()
            else float("nan"),
        },
    }


# --------------------------------------------------------------------------- #
# 3. Adaptive stress testing
# --------------------------------------------------------------------------- #


def stage_stress(cfg: Config, art: Artifacts, rnd: int = 1) -> Dict[str, Any]:
    """Search the twin for the most likely ways the obstacle stop fails."""
    space = _space(cfg)
    twin = _load_twin(cfg, art, rnd)

    with timed("MCTS stress test") as t_mcts:
        ast = AdaptiveStressTest(twin, cfg, space, rng_for(cfg.run.seed, "mcts", rnd))
        mcts = ast.search()

    with timed("cross-entropy stress test") as t_cem:
        cem_runs = multi_start_cem(twin, cfg, space, rng_for(cfg.run.seed, "cem", rnd))

    cem_failures = [r.failures for r in cem_runs if r.found_failure]
    cem_logq = [r.failure_logq for r in cem_runs if r.found_failure]
    if cem_failures:
        seed_z = np.concatenate(cem_failures, axis=0)
        seed_logw = space.log_p(seed_z) - np.concatenate(cem_logq)
    else:
        seed_z = np.zeros((0, space.dim))
        seed_logw = np.zeros(0)

    # MCTS failures come from a tree policy with no closed-form density, so they
    # are pooled in with a neutral weight; their value is the interpretable
    # most-likely path, and extra coverage for the proposal to start from.
    if mcts.n_failures:
        neutral = float(np.median(seed_logw)) if seed_logw.size else 0.0
        seed_z = np.concatenate((seed_z, mcts.failures), axis=0)
        seed_logw = np.concatenate((seed_logw, np.full(mcts.n_failures, neutral)))

    save_npz(
        str(art.stress(rnd)),
        failures=seed_z,
        log_weights=seed_logw,
        mcts_failures=mcts.failures,
        mcts_log_p=mcts.log_p,
        mcts_robustness=mcts.robustness,
    )

    top = [space.describe(mcts.failures[i]) for i in range(min(5, mcts.n_failures))]
    out = {
        "stage": "stress",
        "round": rnd,
        "mcts": {
            "seconds": t_mcts["seconds"],
            "iterations": mcts.iterations,
            "episodes": mcts.episodes,
            "tree_nodes": mcts.tree_nodes,
            "failures_returned": mcts.n_failures,
            "best_log_p": mcts.best_log_p,
            "history": mcts.history,
            "most_likely_failures": top,
        },
        "cross_entropy": {
            "seconds": t_cem["seconds"],
            "restarts": len(cem_runs),
            "restarts_with_failures": int(sum(r.found_failure for r in cem_runs)),
            "episodes": int(sum(r.evaluations for r in cem_runs)),
            "failures": int(sum(r.failures.shape[0] for r in cem_runs)),
            "final_hit_rate": [float(r.hit_rate) for r in cem_runs],
            "tilted_dimensions": [int(r.n_tilted) for r in cem_runs],
            "best_robustness": [float(r.best_robustness) for r in cem_runs],
        },
        "seed_failures": int(seed_z.shape[0]),
        "search_episodes": int(mcts.episodes + sum(r.evaluations for r in cem_runs)),
    }
    save_json(str(art.stage("stress_round%d" % rnd)), out)
    return out


# --------------------------------------------------------------------------- #
# 4. Estimation
# --------------------------------------------------------------------------- #


def stage_estimate(cfg: Config, art: Artifacts, rnd: int = 1) -> Dict[str, Any]:
    """Estimate the failure probability by brute force and by importance sampling."""
    space = _space(cfg)
    twin = _load_twin(cfg, art, rnd)
    stress = load_npz(str(art.stress(rnd)))

    with timed("naive Monte Carlo (twin)"):
        mc = naive_monte_carlo(
            twin, cfg, space, rng_for(cfg.run.seed, "mc", rnd), collect_failures=True
        )

    with timed("adaptive importance sampling (twin)"):
        ais = adaptive_importance_sample(
            twin,
            cfg,
            space,
            stress["failures"],
            rng_for(cfg.run.seed, "is", rnd),
            seed_log_weights=stress["log_weights"],
        )

    cov = coverage_diagnostic(mc.failures, ais.proposal, space)
    cmp = compare(mc, ais.estimate, cfg.estimate.target_rel_error)

    with timed("per-member estimates (model uncertainty)"):
        model_unc = per_member_estimates(
            twin,
            cfg,
            space,
            ais.proposal,
            rng_for(cfg.run.seed, "is", 10 + rnd),
            n_models=twin.n_models,
        )

    search_episodes = int(load_json(str(art.stage("stress_round%d" % rnd)))["search_episodes"])
    total_is = ais.estimate.n + ais.adaptation_episodes + search_episodes

    out = {
        "stage": "estimate",
        "round": rnd,
        "naive_monte_carlo": mc.as_dict(),
        "importance_sampling": ais.as_dict(),
        "proposal": ais.proposal.describe(),
        "coverage": cov.as_dict(),
        "model_uncertainty": model_unc.as_dict(),
        "comparison": cmp.as_dict(),
        "episode_budget": {
            "naive_monte_carlo": mc.n,
            "importance_sampling_final": ais.estimate.n,
            "importance_sampling_adaptation": ais.adaptation_episodes,
            "stress_search": search_episodes,
            "importance_sampling_total": total_is,
        },
        "convergence": {
            "naive_monte_carlo": mc.convergence,
            "importance_sampling": ais.estimate.convergence,
        },
    }
    save_json(str(art.stage("estimate_round%d" % rnd)), out)

    save_npz(
        str(art.root / ("estimate_round%d.npz" % rnd)),
        proposal_weights=ais.proposal.weights,
        proposal_means=ais.proposal.means,
        proposal_sds=ais.proposal.sds,
        mc_failures=mc.failures if mc.failures is not None else np.zeros((0, space.dim)),
        is_failures=ais.estimate.failures
        if ais.estimate.failures is not None
        else np.zeros((0, space.dim)),
    )
    return out


# --------------------------------------------------------------------------- #
# 5. Hardware reference
# --------------------------------------------------------------------------- #


def stage_reference(cfg: Config, art: Artifacts) -> Dict[str, Any]:
    """The oracle: a very large Monte Carlo on the hardware surrogate.

    Available here only because the "hardware" is itself simulated.  On a real
    robot this number does not exist -- which is the entire motivation for the
    project -- so nothing upstream of this stage is allowed to consult it.  It
    is computed purely to score the estimates afterwards.
    """
    space = _space(cfg)
    plant = Plant(cfg.plant, cfg.scenario.dt)
    with timed("hardware oracle Monte Carlo"):
        mc = naive_monte_carlo(
            plant,
            cfg,
            space,
            rng_for(cfg.run.seed, "plant_reference"),
            n=cfg.estimate.plant_reference_n,
            collect_failures=True,
        )

    modes = {}
    if mc.failures is not None and mc.failures.shape[0]:
        res = simulate(mc.failures, plant, cfg, space)
        never = res.detect_step < 0
        modes = {
            "never_detected_share": float(never.mean()),
            "stopped_too_late_share": float((~never).mean()),
            "mean_impact_speed": float(res.impact_speed.mean()),
            "median_traction": float(np.median(space.decode(mc.failures).mu)),
        }
        save_npz(str(art.root / "reference_failures.npz"), failures=mc.failures)

    out = {
        "stage": "reference",
        "p_reference": mc.p_hat,
        "n": mc.n,
        "n_failures": mc.n_failures,
        "ci": [mc.ci_low, mc.ci_high],
        "ci_method": mc.ci_method,
        "seconds": mc.seconds,
        "failure_modes": modes,
        "margin_bookkeeping": _margin_bookkeeping(cfg),
    }
    save_json(str(art.stage("reference")), out)
    return out


def _margin_bookkeeping(cfg: Config) -> Dict[str, Any]:
    """Where the safety margin goes, at nominal and at a tail traction."""
    out = {}
    for name, mu in (("nominal_traction", cfg.scenario.mu_mean), ("tail_traction", 0.25)):
        margin, brake = expected_stop_margin(
            cfg.controller, cfg.plant.traction_gain, mu, 2, cfg.scenario.dt
        )
        out[name] = {
            "mu": mu,
            "trigger_clearance_m": cfg.controller.stop_distance,
            "reaction_travel_m": cfg.controller.v_nominal * 2 * cfg.scenario.dt,
            "braking_distance_m": brake,
            "margin_m": margin,
        }
    return out


# --------------------------------------------------------------------------- #
# 6. Sim-to-real
# --------------------------------------------------------------------------- #


def stage_validate(cfg: Config, art: Artifacts, rnd: int = 1) -> Dict[str, Any]:
    """Replay predicted failures on hardware and score the twin."""
    space = _space(cfg)
    twin = _load_twin(cfg, art, rnd)
    plant = Plant(cfg.plant, cfg.scenario.dt)
    rng = rng_for(cfg.run.seed, "sim2real", rnd)

    est = load_npz(str(art.root / ("estimate_round%d.npz" % rnd)))
    pool = est["is_failures"]
    if pool.shape[0] == 0:
        pool = load_npz(str(art.stress(rnd)))["failures"]

    chosen = select_replay_scenarios(pool, space, cfg.sim2real.n_replay, rng)
    with timed("hardware replay of predicted failures"):
        failure_replay = replay_on_hardware(
            chosen, twin, plant, cfg, space, rng, log_id_offset=10_000
        )

    nominal = space.sample_nominal(rng, cfg.sim2real.n_nominal_replay)
    with timed("hardware replay of nominal control group"):
        control_replay = replay_on_hardware(
            nominal, twin, plant, cfg, space, rng, n_repeats=2, log_id_offset=20_000
        )

    ref_path = art.root / "reference_failures.npz"
    missed = None
    if ref_path.exists():
        ref = load_npz(str(ref_path))["failures"]
        missed = missed_failure_check(ref, twin, cfg, space, rng)

    report = validation_report(failure_replay, control_replay, cfg.estimate.confidence, missed)

    all_logs = list(failure_replay.logs) + list(control_replay.logs)
    write_logs_csv(all_logs, str(art.replay_logs_dir(rnd)))
    save_npz(
        str(art.root / ("replay_round%d.npz" % rnd)),
        z=failure_replay.z,
        twin_robustness=failure_replay.twin_robustness,
        plant_robustness=failure_replay.plant_robustness,
        plant_failure_rate=failure_replay.plant_failure_rate,
        twin_failure=failure_replay.twin_failure,
        plant_failure=failure_replay.plant_failure,
        epistemic=failure_replay.epistemic,
        control_twin_robustness=control_replay.twin_robustness,
        control_plant_robustness=control_replay.plant_robustness,
    )

    out = {"stage": "validate", "round": rnd, "n_replay_logs": len(all_logs)}
    out.update(report.as_dict())
    save_json(str(art.stage("validate_round%d" % rnd)), out)
    return out


# --------------------------------------------------------------------------- #
# 7. Repair and re-estimate
# --------------------------------------------------------------------------- #


def stage_refit(cfg: Config, art: Artifacts) -> Dict[str, Any]:
    """Add the hardware replays to the training set and refit the twin."""
    dataset = _load_dataset(art)
    # The refit learns from the round-1 replays: those are the episodes the
    # hardware actually ran in response to the round-1 twin's predictions.
    logs = read_logs_csv(str(art.replay_logs_dir(1)))
    rng = rng_for(cfg.run.seed, "twin_refit")

    augmented, info = augment_dataset(dataset, logs, cfg, rng)
    with timed("refit twin on augmented data") as t:
        twin2 = refit_twin(augmented, cfg, rng)
    twin2.save(str(art.twin(2)))

    save_npz(
        str(art.root / "dataset_round2.npz"),
        x_train=augmented.x_train,
        y_train=augmented.y_train,
        x_val=augmented.x_val,
        y_val=augmented.y_val,
        measurement_var=np.asarray(augmented.measurement_var),
    )

    out = _twin_report(cfg, twin2, augmented, label="round 2")
    out.update({"stage": "refit", "seconds": t["seconds"], "augmentation": info})
    save_json(str(art.stage("refit")), out)
    return out


def _load_round2_dataset(art: Artifacts) -> Dataset:
    d = load_npz(str(art.root / "dataset_round2.npz"))
    return Dataset(
        x_train=d["x_train"],
        y_train=d["y_train"],
        x_val=d["x_val"],
        y_val=d["y_val"],
        measurement_var=d["measurement_var"],
        info={},
    )


# --------------------------------------------------------------------------- #
# 8. Report
# --------------------------------------------------------------------------- #


def stage_report(cfg: Config, art: Artifacts, make_figures: bool = True) -> Dict[str, Any]:
    """Assemble the master summary, and the figures if matplotlib is present."""
    def maybe(name: str) -> Optional[Dict[str, Any]]:
        path = art.stage(name)
        return load_json(str(path)) if path.exists() else None

    collect = maybe("collect")
    fit = maybe("fit")
    reference = maybe("reference")
    est1 = maybe("estimate_round1")
    est2 = maybe("estimate_round2")
    val1 = maybe("validate_round1")
    val2 = maybe("validate_round2")
    refit = maybe("refit")

    accuracy: List[Dict[str, Any]] = []
    if reference and est1:
        p_ref = reference["p_reference"]
        accuracy.append(
            accuracy_report(
                "round 1 twin, importance sampling",
                est1["importance_sampling"]["p_hat"],
                est1["importance_sampling"]["ci"][0],
                est1["importance_sampling"]["ci"][1],
                p_ref,
            )
        )
        accuracy.append(
            accuracy_report(
                "round 1 twin, naive Monte Carlo",
                est1["naive_monte_carlo"]["p_hat"],
                est1["naive_monte_carlo"]["ci"][0],
                est1["naive_monte_carlo"]["ci"][1],
                p_ref,
            )
        )
        if est2:
            accuracy.append(
                accuracy_report(
                    "round 2 twin (after hardware loop), importance sampling",
                    est2["importance_sampling"]["p_hat"],
                    est2["importance_sampling"]["ci"][0],
                    est2["importance_sampling"]["ci"][1],
                    p_ref,
                )
            )

    summary = {
        "config": cfg.to_dict(),
        "collect": collect,
        "twin_round1": fit,
        "twin_round2": refit,
        "stress_round1": maybe("stress_round1"),
        "stress_round2": maybe("stress_round2"),
        "estimate_round1": est1,
        "estimate_round2": est2,
        "validate_round1": val1,
        "validate_round2": val2,
        "reference": reference,
        "accuracy_against_hardware": accuracy,
    }
    save_json(str(art.stage("results")), summary)

    if make_figures:
        try:
            from .plots import make_all_figures

            figs = make_all_figures(cfg, art, summary)
            summary["figures"] = figs
        except Exception as exc:  # pragma: no cover - plotting is optional
            LOG.warning("figures skipped: %s", exc)
            summary["figures"] = []

    from .report import write_summary

    write_summary(cfg, art, summary)
    save_json(str(art.stage("results")), summary)
    return summary


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run_all(cfg: Config, out_dir: Optional[str] = None, skip_reference: bool = False) -> Dict[str, Any]:
    """Run the whole study end to end."""
    art = Artifacts(out_dir or cfg.run.out_dir)
    cfg.save(str(art.config))

    stage_collect(cfg, art)
    stage_fit(cfg, art)
    if not skip_reference:
        stage_reference(cfg, art)
    stage_stress(cfg, art, rnd=1)
    stage_estimate(cfg, art, rnd=1)
    stage_validate(cfg, art, rnd=1)
    stage_refit(cfg, art)
    stage_stress(cfg, art, rnd=2)
    stage_estimate(cfg, art, rnd=2)
    stage_validate(cfg, art, rnd=2)
    return stage_report(cfg, art)
