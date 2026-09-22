"""Figures for the report.

House style, applied everywhere:

* At most three categorical series per axes, taken in fixed order from slots
  1-3 of the reference palette (blue, orange, aqua) -- the subset documented as
  passing colour-vision separation on every pair.  A fourth thing to show is a
  fourth panel, not a fourth hue.
* Every series is directly labelled as well as legended, so identity never
  depends on colour alone.
* Reference lines (ground truth, the identity diagonal) are neutral grey, not a
  series colour: they are context, not data.
* One y-axis per panel, recessive grid and axes, thin marks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# --- palette ---------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8982"
GRID = "#e6e5e0"
AXIS = "#d7d6d1"

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "rsv_blue", ["#eef4fc", "#9cc3ee", "#4b90dd", "#2a78d6", "#12457f"]
)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": INK_2,
            "lines.linewidth": 2.0,
            "figure.dpi": 150,
            "font.size": 9.5,
        }
    )


def _clean(ax, top: bool = False, right: bool = False) -> None:
    ax.spines["top"].set_visible(top)
    ax.spines["right"].set_visible(right)


def _label_end(ax, x, y, text, color, dx=0.0, dy=0.0, ha="left", va="center") -> None:
    """Direct label at the end of a series, in ink rather than the series colour."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        color=INK_2,
        fontsize=8.5,
        ha=ha,
        va=va,
    )
    ax.plot([x], [y], marker="o", ms=5, color=color, zorder=5)


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
# Individual figures
# --------------------------------------------------------------------------- #


def figure_calibration(summary: Dict[str, Any], out: Path) -> Optional[str]:
    """Are the twin's error bars the right size?"""
    fit = summary.get("twin_round1")
    if not fit:
        return None
    cov = fit["calibration"]["coverage_joint"]
    levels = np.array([float(k) for k in cov])
    empirical = np.array([cov[k] for k in cov])
    order = np.argsort(levels)
    levels, empirical = levels[order], empirical[order]

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=1.4, ls=(0, (4, 3)), zorder=1)
    ax.annotate(
        "perfect calibration",
        xy=(0.62, 0.62),
        xytext=(6, -12),
        textcoords="offset points",
        color=INK_MUTED,
        fontsize=8,
        rotation=38,
    )
    ax.plot(levels, empirical, color=SERIES[0], marker="o", ms=6, zorder=3)
    _label_end(ax, levels[-1], empirical[-1], "  learned twin", SERIES[0], dx=8)

    ax.set_xlabel("nominal coverage of the predictive interval")
    ax.set_ylabel("fraction of held-out steps actually inside")
    ax.set_title("Twin uncertainty is close to calibrated")
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(0.3, 1.02)
    _clean(ax)
    area = fit["calibration"]["miscalibration_area"]
    ax.text(
        0.42,
        0.96,
        "miscalibration area %.3f\n(0 is perfect)" % area,
        color=INK_2,
        fontsize=8.5,
        va="top",
    )
    return _save(fig, out / "01_twin_calibration.png")


def figure_braking(summary: Dict[str, Any], out: Path) -> Optional[str]:
    """The decision-relevant slice: braking authority against floor traction."""
    panels = [(summary.get("twin_round1"), "Round 1: fitted to the driving campaign")]
    if summary.get("twin_round2"):
        panels.append((summary["twin_round2"], "Round 2: after the hardware loop"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (fit, title) in zip(axes, panels):
        bc = fit["braking_curve"]
        mu = np.array(bc["mu"])
        twin = np.array(bc["twin_accel"])
        truth = np.array(bc["true_accel"])
        epi = np.array(bc["epistemic"])
        lo, hi = fit["logged_traction_range"]

        ax.axvspan(lo, hi, color="#eef4fc", zorder=0)
        ax.annotate(
            "traction covered by\nthe logging campaign",
            xy=(0.5 * (lo + hi), 0.04),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            color=INK_MUTED,
            fontsize=8,
        )
        ax.fill_between(mu, twin - 2 * epi, twin + 2 * epi, color=SERIES[0], alpha=0.16, lw=0)
        ax.plot(mu, twin, color=SERIES[0], zorder=3)
        ax.plot(mu, truth, color=SERIES[1], zorder=3)
        _label_end(ax, mu[0], twin[0], "  twin (±2 epistemic sd)", SERIES[0], dx=8)
        _label_end(ax, mu[0], truth[0], "  hardware", SERIES[1], dx=8, dy=-12)
        ax.set_xlabel("floor traction coefficient")
        ax.set_title(title)
        ax.legend(
            handles=[
                Line2D([], [], color=SERIES[0], lw=2, label="learned twin"),
                Line2D([], [], color=SERIES[1], lw=2, label="hardware"),
            ],
            loc="upper right",
        )
        _clean(ax)
    axes[0].set_ylabel("achieved deceleration in an\nemergency stop (m/s²)")
    fig.suptitle(
        "Confident and wrong in the tail, then uncertain and closer"
        if len(panels) > 1
        else "Where the twin has no data it stays confident and optimistic",
        color=INK,
        fontsize=12,
        fontweight="semibold",
        y=1.02,
    )
    return _save(fig, out / "02_braking_authority.png")


def figure_uncertainty_map(cfg, twin, out: Path) -> Optional[str]:
    """Epistemic uncertainty across the states an emergency stop passes through."""
    v_grid = np.linspace(0.0, 0.95, 60)
    mu_grid = np.linspace(0.12, 0.80, 60)
    m = twin.uncertainty_map(v_grid, mu_grid, v_cmd=0.0)

    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    im = ax.pcolormesh(
        m["mu"], m["v"], m["epistemic"] / cfg.scenario.dt, cmap=SEQUENTIAL, shading="auto"
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("epistemic sd of predicted\ndeceleration (m/s²)", color=INK_2, fontsize=9)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    cb.outline.set_edgecolor(AXIS)

    ax.axvline(cfg.scenario.mu_mean, color=INK_2, lw=1.2, ls=(0, (4, 3)))
    ax.annotate(
        " nominal floor",
        xy=(cfg.scenario.mu_mean, 0.9),
        color=INK_2,
        fontsize=8,
    )
    ax.axhline(cfg.controller.v_nominal, color=INK_2, lw=1.2, ls=(0, (1, 2)))
    ax.annotate(" cruise speed", xy=(0.66, cfg.controller.v_nominal), color=INK_2, fontsize=8)
    ax.set_xlabel("floor traction coefficient")
    ax.set_ylabel("speed at the moment of braking (m/s)")
    ax.set_title("The twin knows where it is ignorant")
    ax.grid(False)
    _clean(ax)
    return _save(fig, out / "03_epistemic_map.png")


def figure_failure_modes(cfg, traces, labels, out: Path) -> Optional[str]:
    """Clearance over time for one representative episode per failure mode."""
    if not traces or not any(np.isfinite(np.asarray(m)).any() for m in traces):
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    t = np.arange(cfg.scenario.horizon) * cfg.scenario.dt

    ax.axhline(0.0, color=INK_2, lw=1.4)
    ax.annotate(" contact", xy=(t[-1], 0.0), color=INK_2, fontsize=8, ha="right", va="bottom")
    ax.axhline(
        cfg.controller.stop_distance, color=INK_MUTED, lw=1.2, ls=(0, (4, 3))
    )
    ax.annotate(
        " stop trigger",
        xy=(0.15, cfg.controller.stop_distance),
        color=INK_MUTED,
        fontsize=8,
        va="bottom",
    )

    handles = []
    for k, (margin, label) in enumerate(zip(traces, labels)):
        colour = SERIES[k % len(SERIES)]
        n = min(len(t), len(margin))
        ax.plot(t[:n], margin[:n], color=colour, zorder=3)
        j = int(np.argmin(margin[:n]))
        # Stagger the direct labels: the minima of the two failure modes sit
        # within a few centimetres of each other and would otherwise overlap.
        _label_end(ax, t[j], margin[j], "  " + label, colour, dx=8, dy=14 + 16 * k)
        handles.append(Line2D([], [], color=colour, lw=2, label=label))

    ax.set_xlabel("time (s)")
    ax.set_ylabel("clearance to the obstacle (m)")
    ax.set_title("Two ways the obstacle stop fails")
    ax.set_ylim(-0.15, 1.05)
    ax.set_xlim(1.4, t[-1])
    ax.legend(handles=handles, loc="upper right")
    _clean(ax)
    return _save(fig, out / "04_failure_modes.png")


def figure_convergence(summary: Dict[str, Any], out: Path) -> Optional[str]:
    """Estimate and interval against episodes spent, for both estimators."""
    est = summary.get("estimate_round1")
    if not est:
        return None
    conv = est["convergence"]
    ref = (summary.get("reference") or {}).get("p_reference")

    # Anchor the y-range on the answer.  A running naive estimate sits at exactly
    # zero until its first failure, which a log axis cannot show; letting it set
    # the scale would compress the part of the plot that carries the comparison
    # into a sliver. Those stretches are drawn as gaps instead, which is what
    # "no failure seen yet" actually looks like.
    anchor = ref if (ref and np.isfinite(ref) and ref > 0) else est["importance_sampling"]["p_hat"]
    anchor = anchor if anchor > 0 else 1e-4

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(anchor / 25.0, anchor * 25.0)

    for (key, colour, label) in (
        ("naive_monte_carlo", SERIES[1], "naive Monte Carlo"),
        ("importance_sampling", SERIES[0], "importance sampling"),
    ):
        c = conv.get(key) or {}
        n = np.asarray(c.get("n", []), dtype=float)
        p = np.asarray(c.get("p", []), dtype=float)
        lo = np.asarray(c.get("ci_low", []), dtype=float)
        hi = np.asarray(c.get("ci_high", []), dtype=float)
        if n.size == 0:
            continue
        seen = p > 0
        ax.fill_between(
            n,
            np.where(seen & (lo > 0), lo, np.nan),
            np.where(seen, hi, np.nan),
            color=colour,
            alpha=0.16,
            lw=0,
        )
        ax.plot(n, np.where(seen, p, np.nan), color=colour, zorder=3)
        if seen.any():
            j = int(np.flatnonzero(seen)[-1])
            _label_end(ax, n[j], p[j], "  " + label, colour, dx=6)

    if ref and np.isfinite(ref) and ref > 0:
        ax.axhline(ref, color=INK_2, lw=1.4, ls=(0, (4, 3)), zorder=2)
        ax.annotate(
            "hardware reference",
            xy=(0.01, ref),
            xycoords=("axes fraction", "data"),
            xytext=(0, 4),
            textcoords="offset points",
            color=INK_2,
            fontsize=8.5,
            va="bottom",
        )
    ax.set_xlabel("episodes simulated")
    ax.set_ylabel("estimated failure probability")
    ax.set_title("Same budget, very different precision")
    ax.legend(
        handles=[
            Line2D([], [], color=SERIES[0], lw=2, label="importance sampling"),
            Line2D([], [], color=SERIES[1], lw=2, label="naive Monte Carlo"),
        ],
        loc="lower left",
    )
    _clean(ax)
    return _save(fig, out / "05_estimator_convergence.png")


def figure_variance_reduction(summary: Dict[str, Any], out: Path) -> Optional[str]:
    """Episodes each estimator needs to reach the target relative error."""
    est = summary.get("estimate_round1")
    if not est:
        return None
    cmp = est["comparison"]
    target = cmp["target_relative_error"]
    values = [
        cmp["episodes_for_target"]["naive_monte_carlo"],
        cmp["episodes_for_target"]["importance_sampling"],
    ]
    names = ["naive Monte Carlo", "importance sampling"]
    colours = [SERIES[1], SERIES[0]]

    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ypos = np.arange(len(values))[::-1]
    ax.barh(ypos, values, height=0.5, color=colours, zorder=3)
    for y, v, name in zip(ypos, values, names):
        ax.annotate(
            "  %s episodes" % _si(v),
            xy=(v, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK_2,
            fontsize=9,
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, color=INK_2)
    ax.set_xscale("log")
    ax.set_xlim(right=max(values) * 6)
    ax.set_xlabel("episodes needed for a %.0f%% relative error" % (100 * target))
    ax.set_title("Variance reduction: %.0fx fewer episodes" % cmp["episode_saving_factor"])
    ax.grid(axis="y", visible=False)
    _clean(ax)
    return _save(fig, out / "06_variance_reduction.png")


def figure_sim2real(summary: Dict[str, Any], replay: Dict[str, np.ndarray], out: Path) -> Optional[str]:
    """Predicted margin against measured margin, for the replayed scenarios."""
    if replay is None or replay["twin_robustness"].size == 0:
        return None
    tw = replay["twin_robustness"]
    pl = replay["plant_robustness"]
    epi = replay["epistemic"]

    fig, ax = plt.subplots(figsize=(5.9, 4.6))
    lim = [min(tw.min(), pl.min()) - 0.03, max(tw.max(), pl.max()) + 0.03]
    ax.plot(lim, lim, color=INK_MUTED, lw=1.3, ls=(0, (4, 3)), zorder=1)
    ax.annotate("twin agrees with hardware", xy=(lim[1], lim[1]), xytext=(-6, -14),
                textcoords="offset points", ha="right", color=INK_MUTED, fontsize=8)
    ax.axhline(0, color=INK_2, lw=1.0)
    ax.axvline(0, color=INK_2, lw=1.0)
    # Shade only the quadrant where both agree it is a collision.
    ax.add_patch(
        Rectangle((lim[0], lim[0]), -lim[0], -lim[0], color="#f6efe9", zorder=0, lw=0)
    )

    sc = ax.scatter(tw, pl, c=epi, cmap=SEQUENTIAL, s=44,
                    edgecolor=SURFACE, linewidth=1.2, zorder=3)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("twin's epistemic sd on the\nvelocity increment (m/s)",
                 color=INK_2, fontsize=8.5)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    cb.outline.set_edgecolor(AXIS)

    ax.annotate("confirmed\ncollisions", xy=(lim[0] + 0.01, lim[0] + 0.01),
                color=INK_2, fontsize=8.5, va="bottom")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("margin predicted by the twin (m)")
    ax.set_ylabel("margin measured on hardware (m)")
    val = summary.get("validate_round1") or {}
    ax.set_title("Replay of predicted failures: %.0f%% confirmed"
                 % (100 * val.get("validation_rate_paired", float("nan"))))
    _clean(ax)
    return _save(fig, out / "07_sim_to_real.png")


def figure_estimates(summary: Dict[str, Any], out: Path) -> Optional[str]:
    """Every estimate, with its interval, against the hardware reference.

    An estimator that saw no failure at all has no point estimate to draw -- only
    the upper bound its zero count supports -- so it is drawn as a bound with an
    arrow rather than as a dot at zero, which a log axis cannot show and a reader
    would misread as "the probability is zero".
    """
    rows = summary.get("accuracy_against_hardware") or []
    if not rows:
        return None
    ref = rows[0]["reference"]

    # Fix the axis before drawing: a zero-failure row has no left end of its own,
    # so it needs somewhere defined to start from.  If nothing positive was
    # estimated at all there is no log axis to draw, and no figure to make.
    finite = [
        v
        for r in rows
        for v in (r["ci"][0], r["ci"][1], r["p_hat"])
        if np.isfinite(v) and v > 0
    ]
    if np.isfinite(ref) and ref > 0:
        finite.append(ref)
    if not finite:
        return None
    x_lo, x_hi = min(finite) / 3.0, max(finite) * 2.0

    fig, ax = plt.subplots(figsize=(7.6, 0.78 * len(rows) + 2.4))
    ypos = np.arange(len(rows))[::-1].astype(float)
    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    has_ref = bool(np.isfinite(ref) and ref > 0)
    if has_ref:
        ax.axvline(ref, color=INK_2, lw=1.4, ls=(0, (4, 3)), zorder=1)

    for y, row in zip(ypos, rows):
        colour = SERIES[0] if "importance" in row["estimator"] else SERIES[1]
        lo, hi = row["ci"]
        if row["p_hat"] <= 0.0:
            ax.annotate(
                "",
                xy=(x_lo * 1.25, y),
                xytext=(hi, y),
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.6, shrinkA=0, shrinkB=0),
            )
            ax.annotate(
                "  no failures seen: p < %s" % _sci(hi),
                xy=(hi, y),
                xytext=(10, 0),
                textcoords="offset points",
                va="center",
                color=INK_2,
                fontsize=8.5,
            )
            continue
        ax.plot([max(lo, x_lo * 1.05), hi], [y, y], color=colour, lw=2.8,
                solid_capstyle="round", zorder=3)
        ax.plot([row["p_hat"]], [y], marker="o", ms=8, color=colour,
                markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4)
        label = "  %s" % _sci(row["p_hat"])
        if has_ref and np.isfinite(row["relative_bias"]):
            label += "  (%+.0f%%)" % (100 * row["relative_bias"])
        ax.annotate(label, xy=(hi, y), xytext=(10, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=8.5)

    ax.set_yticks(ypos)
    ax.set_yticklabels([_wrap(r["estimator"], 30) for r in rows], color=INK_2, fontsize=8.5)
    ax.set_ylim(-0.75, len(rows) - 0.15)
    ax.set_xlabel("failure probability per episode")
    ax.set_title("Estimates and 95% intervals against the truth")
    if not has_ref:
        ax.set_title("Estimates and 95% intervals (no hardware reference available)")
        ax.grid(axis="y", visible=False)
        _clean(ax)
        return _save(fig, out / "08_estimates_vs_truth.png")
    ax.annotate(
        "hardware reference\n%s" % _sci(ref),
        xy=(ref, 0.97),
        xycoords=("data", "axes fraction"),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        color=INK_2,
        fontsize=8.5,
    )
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            Line2D([], [], color=SERIES[0], lw=3, label="importance sampling"),
            Line2D([], [], color=SERIES[1], lw=3, label="naive Monte Carlo"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
        ncol=2,
    )
    _clean(ax)
    return _save(fig, out / "08_estimates_vs_truth.png")


# --------------------------------------------------------------------------- #
# Helpers and driver
# --------------------------------------------------------------------------- #


def _si(v: float) -> str:
    if not np.isfinite(v):
        return "n/a"
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if v >= div:
            return "%.1f%s" % (v / div, suffix)
    return "%.0f" % v


def _sci(v: float) -> str:
    return "n/a" if not np.isfinite(v) else "%.2e" % v


def _wrap(text: str, width: int = 34) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return "\n".join(lines)


def make_all_figures(cfg, art, summary: Dict[str, Any]) -> List[str]:
    """Render every figure the available artifacts support."""
    from .rollout import simulate
    from .scenario import ScenarioSpace
    from .twin import Twin
    from .utils import load_npz, rng_for

    _style()
    out = Path(art.figures)
    made: List[str] = []

    for fn in (figure_calibration, figure_braking, figure_convergence,
               figure_variance_reduction, figure_estimates):
        path = fn(summary, out)
        if path:
            made.append(path)

    twin_path = Path(art.twin(1))
    if twin_path.exists():
        twin = Twin.load(str(twin_path), cfg.twin, rng_for(cfg.run.seed, "plots"))
        path = figure_uncertainty_map(cfg, twin, out)
        if path:
            made.append(path)

        space = ScenarioSpace(cfg.scenario, n_models=cfg.twin.n_members)
        traces, labels = _representative_episodes(cfg, art, space, twin)
        path = figure_failure_modes(cfg, traces, labels, out)
        if path:
            made.append(path)

    replay_path = Path(art.root) / "replay_round1.npz"
    if replay_path.exists():
        path = figure_sim2real(summary, load_npz(str(replay_path)), out)
        if path:
            made.append(path)

    return made


def _representative_episodes(cfg, art, space, twin):
    """One failing episode per mode, plus a nominal run, for the trajectory plot."""
    from .plant import Plant
    from .rollout import simulate
    from .utils import load_npz, rng_for

    path = Path(art.root) / "reference_failures.npz"
    plant = Plant(cfg.plant, cfg.scenario.dt)
    rng = rng_for(cfg.run.seed, "plots", 1)

    picks: List[np.ndarray] = []
    labels: List[str] = []
    if path.exists():
        z = load_npz(str(path))["failures"]
        if z.shape[0]:
            res = simulate(z, plant, cfg, space)
            never = res.detect_step < 0
            for mask, label in ((never, "never saw it"), (~never, "stopped too late")):
                idx = np.flatnonzero(mask)
                if idx.size:
                    best = idx[np.argmax(space.log_p(z[idx]))]
                    picks.append(z[best])
                    labels.append(label)
    picks.append(space.sample_nominal(rng, 1)[0])
    labels.append("nominal run")

    res = simulate(np.asarray(picks), plant, cfg, space, record=True, compact=False)
    margins = [res.traces.margin[i] for i in range(len(picks))]
    return margins, labels
