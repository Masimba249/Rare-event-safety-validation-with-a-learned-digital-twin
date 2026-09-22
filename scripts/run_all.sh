#!/usr/bin/env bash
# Run the whole study end to end.  Usage: scripts/run_all.sh [config] [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/default.yaml}"
OUT="${2:-results}"

export PYTHONPATH="${PYTHONPATH:-}:src"
python -m rsv.cli all --config "$CONFIG" --out "$OUT"

echo
echo "Done.  Summary: $OUT/summary.md   Figures: $OUT/figures/"
