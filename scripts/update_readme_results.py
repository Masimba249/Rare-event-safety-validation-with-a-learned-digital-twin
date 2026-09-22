#!/usr/bin/env python3
"""Splice a run's headline numbers into the README, and publish its figures.

The README quotes results.  Quoting them by hand is how a repository ends up
claiming numbers no run produces, so they are generated from
``results/results.json`` and written between the markers below.  Re-run this
after any run whose numbers the README should reflect.

    python scripts/update_readme_results.py [results_dir] [--no-figures]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"

# Figures the README embeds; the rest stay in the run directory.
PUBLISHED = (
    "02_braking_authority.png",
    "04_failure_modes.png",
    "08_estimates_vs_truth.png",
)


def fmt(p: float) -> str:
    return "0" if not p else ("%.4g" % p if p >= 1e-3 else "%.2e" % p)


def rel(value: float, reference: float) -> str:
    if not reference:
        return "n/a"
    return "%+.0f%%" % (100.0 * (value / reference - 1.0))


def build_block(res: dict) -> str:
    ref = (res.get("reference") or {}).get("p_reference", 0.0)
    e1 = res["estimate_round1"]
    e2 = res.get("estimate_round2") or {}
    v1 = res.get("validate_round1") or {}
    v2 = res.get("validate_round2") or {}
    f1 = res.get("twin_round1") or {}
    f2 = res.get("twin_round2") or {}

    is1, mc1, cmp1 = e1["importance_sampling"], e1["naive_monte_carlo"], e1["comparison"]
    cov = e1["coverage"]

    lines = [START, ""]
    lines.append(
        "From the committed run (`configs/default.yaml`, seed %s). Full detail in "
        "`results/summary.md`." % res["config"]["run"]["seed"]
    )
    lines.append("")
    lines.append("| | estimate | 95% interval | vs hardware |")
    lines.append("|---|---|---|---|")
    lines.append(
        "| **hardware oracle** (%s episodes on the plant) | **%s** | %s – %s | — |"
        % (
            "{:,}".format(res["reference"]["n"]),
            fmt(ref),
            fmt(res["reference"]["ci"][0]),
            fmt(res["reference"]["ci"][1]),
        )
    )
    lines.append(
        "| naive Monte Carlo on the twin (%s episodes) | %s | %s – %s | %s |"
        % (
            "{:,}".format(mc1["n"]),
            fmt(mc1["p_hat"]),
            fmt(mc1["ci"][0]),
            fmt(mc1["ci"][1]),
            rel(mc1["p_hat"], ref),
        )
    )
    lines.append(
        "| importance sampling, round-1 twin (%s episodes) | %s | %s – %s | %s |"
        % (
            "{:,}".format(is1["n"]),
            fmt(is1["p_hat"]),
            fmt(is1["ci"][0]),
            fmt(is1["ci"][1]),
            rel(is1["p_hat"], ref),
        )
    )
    if e2:
        is2 = e2["importance_sampling"]
        lines.append(
            "| importance sampling, round-2 twin (after the hardware loop) | %s | %s – %s | %s |"
            % (fmt(is2["p_hat"]), fmt(is2["ci"][0]), fmt(is2["ci"][1]), rel(is2["p_hat"], ref))
        )
    lines.append("")

    lines.append(
        "- **Variance reduction %.0f×.** Reaching a 10%% relative error takes %s "
        "episodes by naive Monte Carlo and %s by importance sampling. The same "
        "200,000-episode budget that gives the naive estimator a %.0f%%-wide "
        "interval gives importance sampling one of %.0f%%."
        % (
            cmp1["variance_reduction_factor"],
            "{:,.0f}".format(cmp1["episodes_for_target"]["naive_monte_carlo"]),
            "{:,.0f}".format(cmp1["episodes_for_target"]["importance_sampling"]),
            100 * (mc1["ci"][1] - mc1["ci"][0]) / max(mc1["p_hat"], 1e-12),
            100 * (is1["ci"][1] - is1["ci"][0]) / max(is1["p_hat"], 1e-12),
        )
    )
    lines.append(
        "- **Coverage audit: %.0f%% of the failure mass unreached** by the tilted "
        "components (%d naive-run failures probed). %s"
        % (
            100 * cov["uncovered_failure_mass_fraction"],
            cov["n_probe_failures"],
            cov["verdict"][0].upper() + cov["verdict"][1:] + ".",
        )
    )
    modes = res["reference"].get("failure_modes") or {}
    if modes:
        lines.append(
            "- **The hardware failures split %.0f%% \"never saw it\" / %.0f%% \"stopped "
            "too late\"**, at a median traction of %.2f."
            % (
                100 * modes["never_detected_share"],
                100 * modes["stopped_too_late_share"],
                modes["median_traction"],
            )
        )
    if f1 and f2:
        b1, b2 = f1["braking_error"], f2["braking_error"]
        lines.append(
            "- **The hardware loop repairs the tail.** Braking-authority error outside "
            "the logged traction range: %.2f m/s² before the loop, %.2f m/s² after "
            "(inside the range: %.2f → %.2f)."
            % (
                b1["outside_logged_range_mae"],
                b2["outside_logged_range_mae"],
                b1["inside_logged_range_mae"],
                b2["inside_logged_range_mae"],
            )
        )
    b1 = (e1.get("model_uncertainty") or {}).get("per_member_p") or []
    b2 = (e2.get("model_uncertainty") or {}).get("per_member_p") or []
    if b1:
        inside1 = min(b1) <= ref <= max(b1)
        lines.append(
            "- **Model uncertainty dwarfs sampling uncertainty.** Re-estimating with each ensemble member in turn spans %s – %s (%.1f×), against a confidence interval %.0f%% wide. That band %s the hardware value."
            % (
                fmt(min(b1)),
                fmt(max(b1)),
                max(b1) / max(min(b1), 1e-300),
                100 * (is1["ci"][1] - is1["ci"][0]) / max(is1["p_hat"], 1e-12),
                "contains" if inside1 else "excludes",
            )
        )
        if b2:
            inside2 = min(b2) <= ref <= max(b2)
            lines.append(
                "- **What the hardware loop buys is the band, not the point.** After the replays the members span %s – %s (%.1f×), which %s the hardware value: a twin that has seen the tail knows it is uncertain there."
                % (
                    fmt(min(b2)),
                    fmt(max(b2)),
                    max(b2) / max(min(b2), 1e-300),
                    "contains" if inside2 else "still excludes",
                )
            )
    if v1:
        line = (
            "- **Sim-to-real: %.0f%% of predicted failures reproduced on hardware** "
            "(%d replayed, 95%% CI %.0f–%.0f%%), against %.0f%% in the nominal control "
            "group."
            % (
                100 * v1["validation_rate_paired"],
                v1["n_predicted_failures"],
                100 * v1["validation_rate_ci"][0],
                100 * v1["validation_rate_ci"][1],
                100 * (v1.get("control_group") or {}).get("hardware_failure_rate", 0.0),
            )
        )
        if v2:
            line += " After the loop: %.0f%%." % (100 * v2["validation_rate_paired"])
        lines.append(line)
        conf, refu = v1.get("mean_epistemic_confirmed"), v1.get("mean_epistemic_refuted")
        if conf and refu and refu == refu:
            lines.append(
                "- **The twin's uncertainty is informative about its own mistakes:** "
                "predictions hardware refuted carried %.0f%% more epistemic uncertainty "
                "than ones it confirmed (%.4f vs %.4f m/s)."
                % (100 * (refu / conf - 1.0), refu, conf)
            )
    lines.append("")
    lines.append(END)
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    make_figures = "--no-figures" not in argv
    argv = [a for a in argv if not a.startswith("--")]
    results_dir = Path(argv[0] if argv else "results")

    root = Path(__file__).resolve().parents[1]
    with open(results_dir / "results.json", encoding="utf-8") as fh:
        res = json.load(fh)

    readme = root / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit("README is missing the %s / %s markers" % (START, END))
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(head + build_block(res) + tail, encoding="utf-8")
    print("README results block updated from %s" % (results_dir / "results.json"))

    if make_figures:
        out = root / "docs" / "figures"
        out.mkdir(parents=True, exist_ok=True)
        for name in PUBLISHED:
            src = results_dir / "figures" / name
            if src.exists():
                shutil.copyfile(src, out / name)
                print("published docs/figures/%s" % name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
