#!/usr/bin/env python3
"""Regenerate the YAML configs from the dataclass defaults.

The configs are checked in so a run is reproducible from the repository alone,
which means they can drift from the defaults in `rsv.config` whenever those
change -- and a stale value in the YAML silently wins, because the file is the
source of truth for a run.  Re-run this whenever a default changes.

    python scripts/write_configs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsv.config import Config  # noqa: E402

QUICK = {
    "run.profile": "quick",
    "data.n_missions": 60,
    "twin.epochs": 90,
    "twin.n_members": 4,
    "ast.mcts_iterations": 600,
    "ast.cem_iterations": 14,
    "ast.cem_population": 700,
    "estimate.n_mc": 40_000,
    "estimate.n_is": 8_000,
    "estimate.adapt_rounds": 2,
    "estimate.adapt_samples": 5_000,
    "estimate.plant_reference_n": 300_000,
    "sim2real.n_replay": 30,
    "sim2real.n_repeats": 4,
    "sim2real.n_nominal_replay": 20,
    "sim2real.refit_epochs": 90,
}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    Config().save(str(root / "configs" / "default.yaml"))
    Config().override(QUICK).save(str(root / "configs" / "quick.yaml"))
    print("wrote configs/default.yaml and configs/quick.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
