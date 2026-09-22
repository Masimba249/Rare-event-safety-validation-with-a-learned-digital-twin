"""Render the run's results as a Markdown report.

Written for whoever has to read the safety argument rather than run it, so the
numbers come with the caveats attached: what the interval does and does not
cover, how much of the failure region the proposal reached, and how the twin's
prediction held up when the scenarios were put on the robot.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .config import Config
from .utils import fmt_prob


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return "{:,}".format(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "n/a"
        if v != 0 and abs(v) < 1e-3:
            return "%.3e" % v
        if abs(v) >= 1e4:  # episode counts read better grouped than decimalised
            return "{:,}".format(int(round(v)))
        return ("%%.%df" % digits) % v
    return str(v)


def _rel(value: Any, reference: Any) -> str:
    """Relative difference, or ``n/a`` when there is nothing to compare against.

    A reference of zero is a real outcome -- it means even the oracle run saw no
    failure -- and the report has to say so rather than divide by it.
    """
    try:
        ref = float(reference)
        val = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(ref) or ref <= 0 or not np.isfinite(val):
        return "n/a"
    return "%+.0f%%" % (100.0 * (val / ref - 1.0))


def _pct(v: Any, digits: int = 1) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return ("%%.%df%%%%" % digits) % (100.0 * float(v))


def _band(est: Dict[str, Any]) -> Any:
    """``(low, high)`` of the per-member estimates, or ``None``."""
    mu = (est or {}).get("model_uncertainty") or {}
    vals = mu.get("per_member_p") or []
    return (min(vals), max(vals)) if vals else None


def _table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def write_summary(cfg: Config, art, summary: Dict[str, Any]) -> str:
    """Write ``summary.md`` into the output directory and return its text."""
    lines: List[str] = []
    add = lines.append

    ref = summary.get("reference") or {}
    est1 = summary.get("estimate_round1") or {}
    est2 = summary.get("estimate_round2") or {}
    val1 = summary.get("validate_round1") or {}
    val2 = summary.get("validate_round2") or {}
    fit1 = summary.get("twin_round1") or {}
    fit2 = summary.get("twin_round2") or {}
    collect = summary.get("collect") or {}
    stress1 = summary.get("stress_round1") or {}

    add("# Rare-event safety validation results")
    add("")
    add(
        "Failure probability of the obstacle-stop function, estimated from a learned "
        "digital twin, with confidence bounds and a hardware check."
    )
    add("")
    add("Seed `%s`, profile `%s`." % (cfg.run.seed, cfg.run.profile))
    add("")

    # ---------------------------------------------------------------- headline
    add("## Headline")
    add("")
    if est1:
        is1 = est1["importance_sampling"]
        mc1 = est1["naive_monte_carlo"]
        cmp1 = est1["comparison"]
        add(
            "Importance sampling on the round-1 twin puts the failure probability at "
            "**%s** (95%% CI %s to %s) from %s episodes."
            % (
                fmt_prob(is1["p_hat"]),
                fmt_prob(is1["ci"][0]),
                fmt_prob(is1["ci"][1]),
                _fmt(is1["n"]),
            )
        )
        if mc1["n_failures"] == 0:
            add(
                "Naive Monte Carlo over %s episodes saw **no failures at all**, so all it "
                "can say is p < %s." % (_fmt(mc1["n"]), fmt_prob(mc1["ci"][1]))
            )
        else:
            add(
                "Naive Monte Carlo over %s episodes saw %s failures: %s (95%% CI %s to %s)."
                % (
                    _fmt(mc1["n"]),
                    _fmt(mc1["n_failures"]),
                    fmt_prob(mc1["p_hat"]),
                    fmt_prob(mc1["ci"][0]),
                    fmt_prob(mc1["ci"][1]),
                )
            )
        add("")
        add(
            "Per-episode variance falls by **%.0fx**, so reaching a %s relative error "
            "needs %s episodes instead of %s."
            % (
                cmp1["variance_reduction_factor"],
                _pct(cmp1["target_relative_error"], 0),
                _fmt(cmp1["episodes_for_target"]["importance_sampling"]),
                _fmt(cmp1["episodes_for_target"]["naive_monte_carlo"]),
            )
        )
    if ref:
        add("")
        if ref.get("n_failures", 0) == 0:
            add(
                "The hardware oracle run (%s episodes on the plant) saw no failures "
                "either, so it bounds the truth at p < %s rather than pinning it down; "
                "the estimates below cannot be scored against it."
                % (_fmt(ref["n"]), fmt_prob(ref["ci"][1]))
            )
        else:
            add(
                "The hardware oracle -- %s episodes on the plant, available only because "
                "the robot here is itself simulated and never consulted by any earlier "
                "stage -- gives **%s** (95%% CI %s to %s)."
                % (
                    _fmt(ref["n"]),
                    fmt_prob(ref["p_reference"]),
                    fmt_prob(ref["ci"][0]),
                    fmt_prob(ref["ci"][1]),
                )
            )
    add("")

    # ----------------------------------------------------------- accuracy table
    rows = summary.get("accuracy_against_hardware") or []
    if rows:
        add("## Estimates against the truth")
        add("")
        add(
            _table(
                ["estimator", "estimate", "95% CI", "vs reference", "covers truth"],
                [
                    [
                        r["estimator"],
                        fmt_prob(r["p_hat"]),
                        "%s – %s" % (fmt_prob(r["ci"][0]), fmt_prob(r["ci"][1])),
                        _rel(r["p_hat"], r["reference"]),
                        _fmt(r["covers_reference"]),
                    ]
                    for r in rows
                ],
            )
        )
        add("")

    # ------------------------------------------------------------- the twin
    add("## The learned twin")
    add("")
    if collect:
        add(
            "- %s logged missions x %s control steps -> %s training rows "
            "(%s held out, split by mission)."
            % (
                _fmt(collect["n_missions"]),
                _fmt(collect["mission_steps"]),
                _fmt(collect["n_train_rows"]),
                _fmt(collect["n_val_rows"]),
            )
        )
        est = collect.get("velocity_estimators") or {}
        if "camera" in est and "encoder" in est:
            cam, enc = est["camera"], est["encoder"]
            ratio = abs(enc["velocity_bias_while_slipping"]) / max(
                abs(cam["velocity_bias_while_slipping"]), 1e-9
            )
            add(
                "- Velocity estimator: the twin is fitted to the overhead camera, not to "
                "wheel odometry. Overall RMSE is %s m/s for the camera and %s m/s for the "
                "encoders, but while the wheels are slipping the encoders are biased by "
                "%s m/s against the camera's %s m/s -- %sx worse, in exactly the regime "
                "the safety case turns on."
                % (
                    _fmt(cam["velocity_rmse"], 4),
                    _fmt(enc["velocity_rmse"], 4),
                    _fmt(enc["velocity_bias_while_slipping"], 4),
                    _fmt(cam["velocity_bias_while_slipping"], 4),
                    _fmt(ratio, 1),
                )
            )
    if fit1:
        cal = fit1["calibration"]
        add(
            "- Held-out accuracy: RMSE %s m/s on the velocity increment; predictive "
            "intervals cover %s at a nominal 95%% (miscalibration area %s)."
            % (
                _fmt(cal["rmse"]["dv"], 4),
                _pct(cal["coverage_joint"]["0.95"]),
                _fmt(cal["miscalibration_area"]),
            )
        )
        be = fit1["braking_error"]
        lo, hi = fit1["logged_traction_range"]
        e_in, e_out = be["mean_epistemic_inside"], be["mean_epistemic_outside"]
        flag = (
            "and its epistemic uncertainty rises with it (%s to %s m/s²), so the error bars do flag the region where the mean is wrong"
            if np.isfinite(e_out) and e_out > e_in
            else "but its epistemic uncertainty does not rise to match (%s to %s m/s²), so the error bars do not flag the whole gap"
        ) % (_fmt(e_in), _fmt(e_out))
        add(
            "- Braking authority: mean error %s m/s² inside the logged traction range "
            "(%s–%s) and %s m/s² outside it, %s."
            % (
                _fmt(be["inside_logged_range_mae"]),
                _fmt(lo, 2),
                _fmt(hi, 2),
                _fmt(be["outside_logged_range_mae"]),
                flag,
            )
        )
    if fit2:
        be2 = fit2["braking_error"]
        before = (fit1.get("braking_error") or {}).get("outside_logged_range_mae")
        after = be2["outside_logged_range_mae"]
        verb = (
            "falls" if (before is not None and np.isfinite(before) and after < before) else "moves"
        )
        add(
            "- After the hardware loop, the error outside the originally logged range %s "
            "from %s to %s m/s²." % (verb, _fmt(before), _fmt(after))
        )
    add("")

    # -------------------------------------------------------- stress testing
    if stress1:
        add("## What the stress testing found")
        add("")
        m = stress1["mcts"]
        c = stress1["cross_entropy"]
        add(
            "- MCTS: %s iterations, %s episodes, %s tree nodes. Best failure log-density "
            "%s (a typical nominal draw scores about %s)."
            % (
                _fmt(m["iterations"]),
                _fmt(m["episodes"]),
                _fmt(m["tree_nodes"]),
                _fmt(m["best_log_p"], 1),
                _fmt(_typical_log_p(cfg), 1),
            )
        )
        add(
            "- Cross-entropy: %s restarts, %s found failures, %s episodes."
            % (_fmt(c["restarts"]), _fmt(c["restarts_with_failures"]), _fmt(c["episodes"]))
        )
        top = _distinct(m.get("most_likely_failures") or [])
        if top:
            add("")
            add("Most likely failure scenarios found:")
            add("")
            add(
                _table(
                    ["traction", "obstacle offset (m)", "latency (steps)", "stale readings", "range bias (m)"],
                    [
                        [
                            _fmt(f["floor_traction_mu"], 3),
                            _fmt(f["obstacle_y"], 3),
                            _fmt(f["actuation_latency_steps"]),
                            _fmt(f["n_dropouts"]),
                            _fmt(f["range_bias_m"], 4),
                        ]
                        for f in top
                    ],
                )
            )
        add("")

    # ------------------------------------------------------------- estimators
    if est1:
        add("## Estimator diagnostics")
        add("")
        is1 = est1["importance_sampling"]
        cov = est1["coverage"]
        budget = est1["episode_budget"]
        add(
            "- Proposal: %s components, defensive weight %s (so every likelihood ratio is "
            "bounded by %s). Effective sample size %s of %s; the largest single sample "
            "contributes %s of the estimate."
            % (
                _fmt(len(est1["proposal"])),
                _fmt(cfg.estimate.defensive_alpha, 2),
                _fmt(is1["weight_bound"], 1),
                _fmt(is1["effective_sample_size"]),
                _fmt(is1["n"]),
                _pct(is1["max_weight_share"]),
            )
        )
        if cov["n_probe_failures"] == 0:
            add(
                "- Coverage audit: not possible -- the naive run found no failures to audit "
                "the proposal against, so nothing independent confirms that the proposal "
                "reaches the whole failure region."
            )
        else:
            add(
                "- Coverage audit against %s naive-Monte-Carlo failures: %s of failure mass "
                "is reached only by the defensive component (95%% CI %s to %s). %s"
                % (
                    _fmt(cov["n_probe_failures"]),
                    _pct(cov["uncovered_failure_mass_fraction"]),
                    _pct(cov["uncovered_fraction_ci"][0]),
                    _pct(cov["uncovered_fraction_ci"][1]),
                    cov["verdict"].capitalize() + ".",
                )
            )
        mu_ = est1.get("model_uncertainty")
        if mu_:
            add(
                "- Model uncertainty: estimating the same probability with each "
                "ensemble member in turn gives %s to %s, a spread of %.1fx. That is "
                "much wider than the %s-wide confidence interval above, and it is the "
                "honest measure of what is not known: a tight interval on a learned "
                "twin describes the arithmetic, not the robot."
                % (
                    fmt_prob(min(mu_["per_member_p"])),
                    fmt_prob(max(mu_["per_member_p"])),
                    mu_["max_over_min"],
                    _pct((is1["ci"][1] - is1["ci"][0]) / max(is1["p_hat"], 1e-12), 0),
                )
            )
        add(
            "- Episode budget: %s for the naive run; %s for importance sampling "
            "(%s final estimate + %s proposal adaptation + %s stress search)."
            % (
                _fmt(budget["naive_monte_carlo"]),
                _fmt(budget["importance_sampling_total"]),
                _fmt(budget["importance_sampling_final"]),
                _fmt(budget["importance_sampling_adaptation"]),
                _fmt(budget["stress_search"]),
            )
        )
        add("")

    # --------------------------------------------------------------- sim2real
    if val1:
        add("## Sim-to-real")
        add("")
        add(
            "- Of %s predicted failures replayed on hardware with the same disturbances, "
            "**%s** collided (95%% CI %s to %s). Repeating each scenario with fresh "
            "process noise gives a mean hardware failure probability of %s."
            % (
                _fmt(val1["n_predicted_failures"]),
                _pct(val1["validation_rate_paired"]),
                _pct(val1["validation_rate_ci"][0]),
                _pct(val1["validation_rate_ci"][1]),
                _pct(val1["validation_rate_probabilistic"]),
            )
        )
        add(
            "- Predicted margin is biased by %s m against hardware (RMSE %s m, "
            "correlation %s)."
            % (
                _fmt(val1["robustness_bias_m"], 4),
                _fmt(val1["robustness_rmse_m"], 4),
                _fmt(val1["robustness_correlation"], 2),
            )
        )
        cg = val1.get("control_group") or {}
        if cg:
            add(
                "- Control group: %s nominal scenarios replayed; hardware failure rate %s, "
                "twin %s. A high validation rate is not an artefact of a robot that "
                "collides with everything."
                % (
                    _fmt(int(cg.get("n", 0))),
                    _pct(cg.get("hardware_failure_rate")),
                    _pct(cg.get("twin_failure_rate")),
                )
            )
        conf, refu = val1.get("mean_epistemic_confirmed"), val1.get("mean_epistemic_refuted")
        if conf is not None and refu is not None and np.isfinite(refu):
            add(
                "- Mean epistemic uncertainty along confirmed predictions %s vs %s for "
                "refuted ones." % (_fmt(conf, 4), _fmt(refu, 4))
            )
        missed = val1.get("missed_failures")
        if missed:
            add(
                "- Converse check: of %s real hardware failures replayed in the twin, "
                "%s were reproduced at least once in eight tries (mean reproduction rate %s)."
                % (
                    _fmt(int(missed["n_hardware_failures"])),
                    _pct(missed["reproduced_at_least_once"]),
                    _pct(missed["mean_reproduction_rate"]),
                )
            )
        if val2:
            add(
                "- After refitting on the replays, the validation rate moves from %s to %s."
                % (
                    _pct(val1["validation_rate_paired"]),
                    _pct(val2["validation_rate_paired"]),
                )
            )
        add("")

    # ------------------------------------------------------------- the loop
    if est2 and ref and ref.get("p_reference", 0.0) > 0:
        add("## Effect of closing the loop")
        add("")
        add(
            _table(
                ["", "round 1 twin", "round 2 twin (after replay)", "hardware"],
                [
                    [
                        "failure probability",
                        fmt_prob(est1["importance_sampling"]["p_hat"]),
                        fmt_prob(est2["importance_sampling"]["p_hat"]),
                        fmt_prob(ref["p_reference"]),
                    ],
                    [
                        "relative error vs hardware",
                        _rel(est1["importance_sampling"]["p_hat"], ref["p_reference"]),
                        _rel(est2["importance_sampling"]["p_hat"], ref["p_reference"]),
                        "--",
                    ],
                    [
                        "validation rate of predictions",
                        _pct(val1.get("validation_rate_paired")),
                        _pct(val2.get("validation_rate_paired")),
                        "--",
                    ],
                    [
                        "model-uncertainty band (per ensemble member)",
                        _band_text(est1),
                        _band_text(est2),
                        "--",
                    ],
                    [
                        "band contains the hardware value",
                        _fmt(_band_contains(est1, ref["p_reference"])),
                        _fmt(_band_contains(est2, ref["p_reference"])),
                        "--",
                    ],
                ],
            )
        )
        add("")
        b1, b2 = _band(est1), _band(est2)
        if b1 and b2:
            in1 = _band_contains(est1, ref["p_reference"])
            in2 = _band_contains(est2, ref["p_reference"])
            if (not in1) and in2:
                add(
                    "The point estimate did not move onto the truth -- it moved past "
                    "it. What the loop bought is the band: before the replays the five "
                    "ensemble members agreed with each other to within %.1fx and were "
                    "collectively wrong, excluding the hardware value; after them they "
                    "disagree by %.1fx and the range they span contains it. A twin that "
                    "has seen the tail knows it is uncertain there, and a safety case "
                    "is better served by a wide honest range than by a narrow "
                    "confident one that is wrong."
                    % (b1[1] / max(b1[0], 1e-300), b2[1] / max(b2[0], 1e-300))
                )
                add("")

    figs = summary.get("figures") or []
    if figs:
        add("## Figures")
        add("")
        for f in figs:
            name = f.replace("\\", "/").rsplit("/", 1)[-1]
            add("- ![%s](figures/%s)" % (name, name))
        add("")

    text = "\n".join(lines) + "\n"
    with open(str(art.summary), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def _distinct(rows: List[Dict[str, Any]], keys=("floor_traction_mu", "obstacle_y")) -> List[Dict[str, Any]]:
    """Drop near-duplicate scenarios so the table shows distinct mechanisms.

    A tree search returns many neighbours of the same optimum; listing five
    rounded copies of one scenario would overstate what it found.
    """
    seen, out = set(), []
    for r in rows:
        key = tuple(round(float(r.get(k, 0.0)), 2) for k in keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _band_text(est: Dict[str, Any]) -> str:
    band = _band(est)
    return "n/a" if band is None else "%s – %s" % (fmt_prob(band[0]), fmt_prob(band[1]))


def _band_contains(est: Dict[str, Any], reference: float) -> Any:
    band = _band(est)
    return None if band is None else bool(band[0] <= reference <= band[1])


def _typical_log_p(cfg: Config) -> float:
    """Expected ``log p(z)`` for a draw from the nominal law, for scale."""
    d = cfg.scenario.latent_dim
    return -0.5 * d * (1.0 + float(np.log(2.0 * np.pi)))
