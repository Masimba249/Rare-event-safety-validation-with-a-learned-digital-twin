"""Command-line entry point.

    python -m rsv.cli all --config configs/default.yaml --out results

Each pipeline stage is also a subcommand, so a long run can be resumed or a
single stage re-run against the artifacts already on disk::

    python -m rsv.cli fit --out results
    python -m rsv.cli estimate --round 1 --out results

Any configuration value can be overridden from the command line, which is how
the fast smoke-test profile works::

    python -m rsv.cli all --set estimate.n_mc=20000 --set twin.epochs=40
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .config import Config, parse_overrides
from .pipeline import (
    Artifacts,
    run_all,
    stage_collect,
    stage_estimate,
    stage_fit,
    stage_refit,
    stage_reference,
    stage_report,
    stage_stress,
    stage_validate,
)
from .utils import LOG, setup_logging


def _common_options() -> argparse.ArgumentParser:
    """Options accepted either before or after the subcommand.

    ``SUPPRESS`` defaults matter: the same actions are attached to the top-level
    parser and to every subparser, and without it the subparser would overwrite
    a value given before the subcommand with its own default.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS, help="YAML configuration file")
    common.add_argument("--out", default=argparse.SUPPRESS, help="output directory")
    common.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="override the master seed")
    common.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=argparse.SUPPRESS,
        metavar="section.key=value",
        help="override a configuration value; repeatable",
    )
    common.add_argument(
        "--quiet", action="store_true", default=argparse.SUPPRESS, help="only warnings and errors"
    )
    return common


def build_parser() -> argparse.ArgumentParser:
    common = _common_options()
    parser = argparse.ArgumentParser(
        prog="rsv",
        parents=[common],
        description=(
            "Rare-event safety validation of a robot obstacle-stop function "
            "with a learned probabilistic digital twin."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_text, parents=[common])

    add("collect", "drive the logging campaign and build the training set")
    add("fit", "fit the probabilistic twin")
    add("stress", "adaptive stress testing on the twin").add_argument("--round", type=int, default=1)
    add("estimate", "naive Monte Carlo and importance sampling").add_argument("--round", type=int, default=1)
    add("validate", "replay predicted failures on hardware").add_argument("--round", type=int, default=1)
    add("refit", "add the replays to the training set and refit")
    add("reference", "hardware oracle Monte Carlo (ground truth)")
    add("report", "assemble figures and the written summary").add_argument(
        "--no-figures", action="store_true"
    )
    add("all", "run the whole study").add_argument(
        "--skip-reference",
        action="store_true",
        help="skip the hardware oracle run (much faster, no ground truth to score against)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(not getattr(args, "quiet", False))

    cfg = Config.load(getattr(args, "config", None))
    overrides = getattr(args, "overrides", None)
    if overrides:
        cfg = cfg.override(parse_overrides(overrides))
    seed = getattr(args, "seed", None)
    if seed is not None:
        cfg.run.seed = int(seed)
    out_dir = getattr(args, "out", None) or cfg.run.out_dir

    art = Artifacts(out_dir)
    cfg.save(str(art.config))
    LOG.info("output directory: %s", art.root)

    cmd = args.command
    if cmd == "all":
        run_all(cfg, out_dir, skip_reference=getattr(args, "skip_reference", False))
    elif cmd == "collect":
        stage_collect(cfg, art)
    elif cmd == "fit":
        stage_fit(cfg, art)
    elif cmd == "stress":
        stage_stress(cfg, art, rnd=args.round)
    elif cmd == "estimate":
        stage_estimate(cfg, art, rnd=args.round)
    elif cmd == "validate":
        stage_validate(cfg, art, rnd=args.round)
    elif cmd == "refit":
        stage_refit(cfg, art)
    elif cmd == "reference":
        stage_reference(cfg, art)
    elif cmd == "report":
        stage_report(cfg, art, make_figures=not getattr(args, "no_figures", False))
    else:  # pragma: no cover - argparse rejects anything else
        raise SystemExit("unknown command %r" % cmd)

    LOG.info("done; artifacts in %s", art.root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
