"""End-to-end: every stage runs, writes its artifacts, and they fit together."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsv.cli import main
from rsv.config import Config, parse_overrides
from rsv.pipeline import Artifacts, run_all
from rsv.sim2real import select_replay_scenarios
from rsv.scenario import ScenarioSpace
from rsv.utils import load_json, rng_for


def test_config_round_trip(tmp_path, default_cfg):
    path = tmp_path / "cfg.yaml"
    default_cfg.save(str(path))
    reloaded = Config.load(str(path))
    assert reloaded.to_dict() == default_cfg.to_dict()


def test_overrides_reject_unknown_keys(default_cfg):
    with pytest.raises(KeyError):
        default_cfg.override({"estimate.not_a_key": 1})
    with pytest.raises(KeyError):
        default_cfg.override({"nosuchsection.n_mc": 1})
    assert default_cfg.override(parse_overrides(["estimate.n_mc=7"])).estimate.n_mc == 7


def test_latency_probabilities_are_normalised():
    cfg = Config.from_dict({"scenario": {"latency_probs": [2.0, 2.0]}})
    assert sum(cfg.scenario.latency_probs) == pytest.approx(1.0)


def test_replay_selection_spreads_across_modes(default_cfg):
    """Hardware time is scarce, so the chosen scenarios must not all be one mode."""
    space = ScenarioSpace(default_cfg.scenario, n_models=1)
    rng = rng_for(91, "sim2real")
    failures = rng.standard_normal((300, space.dim))
    failures[:150, 0] = -3.5  # traction mode
    failures[150:, 2] = 3.8  # lateral mode
    chosen = select_replay_scenarios(failures, space, 12, rng, n_groups=4)
    assert chosen.shape[0] == 12
    assert (chosen[:, 0] < -2.0).any()
    assert (chosen[:, 2] > 2.0).any()


@pytest.mark.slow
def test_full_pipeline_produces_a_coherent_report(tmp_path, tiny_cfg):
    """The whole study at miniature scale: stages, artifacts and the summary."""
    art = Artifacts(str(tmp_path / "out"))
    summary = run_all(tiny_cfg, str(art.root))

    for name in ("collect", "fit", "reference", "estimate_round1", "validate_round1", "refit"):
        assert Path(art.stage(name)).exists(), name
    assert Path(art.twin(1)).exists() and Path(art.twin(2)).exists()
    assert Path(art.summary).exists()
    assert list(Path(art.logs_dir).glob("mission_*.csv"))
    assert list(Path(art.replay_logs_dir(1)).glob("mission_*.csv"))
    assert list(Path(art.replay_logs_dir(2)).glob("mission_*.csv"))

    text = Path(art.summary).read_text(encoding="utf-8")
    assert "Rare-event safety validation results" in text
    assert "Sim-to-real" in text

    est = summary["estimate_round1"]
    assert est["importance_sampling"]["p_hat"] >= 0.0
    assert est["naive_monte_carlo"]["ci"][0] <= est["naive_monte_carlo"]["ci"][1]
    assert est["comparison"]["variance_reduction_factor"] > 0.0
    # The episode budget must account for every simulated episode.
    budget = est["episode_budget"]
    assert budget["importance_sampling_total"] == (
        budget["importance_sampling_final"]
        + budget["importance_sampling_adaptation"]
        + budget["stress_search"]
    )

    # Results must be JSON-clean (no numpy scalars leaking into the artifact).
    json.dumps(load_json(str(art.stage("results"))))


@pytest.mark.slow
def test_stages_can_be_resumed_individually(tmp_path, tiny_cfg):
    """Re-running one stage against existing artifacts must work."""
    out = str(tmp_path / "resume")
    overrides = [
        "data.n_missions=6",
        "data.mission_steps=80",
        "twin.n_members=2",
        "twin.epochs=8",
        "twin.hidden=[12, 12]",
        "estimate.batch=400",
        "estimate.n_mc=800",
    ]
    args = []
    for o in overrides:
        args += ["--set", o]
    assert main(["collect", "--out", out] + args) == 0
    assert main(["fit", "--out", out] + args) == 0
    assert Path(out, "fit.json").exists()
    # A second fit must overwrite cleanly rather than fail.
    assert main(["fit", "--out", out] + args) == 0
    report = load_json(str(Path(out, "fit.json")))
    assert report["calibration"]["n"] > 0
