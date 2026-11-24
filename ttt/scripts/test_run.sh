#!/usr/bin/env bash
# The test run of PROTOCOL.md: the six arms on the 26 test doubles, with the settings validation
# chose (scripts/select_ttt.py on VAL, default ../results/ttt/val_v2). The shared adapter is
# retrained on all 105 training doubles. Every arm runs at T = 1; icl, shared, ttt and
# shared_ttt also at T = 5; 2 draws each. scripts/analyze_ttt.py then writes analysis.md.
#
#   dibs submit --cpus 2 --mem 10G --vram 5G --time 3h30m --retry -- bash scripts/test_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."  # ttt/
val="${VAL:-../results/ttt/val_v2}"
out="${OUT:-../results/ttt/test_v2}"

test -f "$val/stage2_selection.json" || { echo "no stage 2 selection in $val" >&2; exit 1; }
choice=$(uv run python scripts/select_ttt.py shared "$val")
read -r scope shared_lr <<< "$choice"
uv run python scripts/norman_ttt.py --holdout test --scope "$scope" --shared-lr "$shared_lr" \
  --selection "$val/stage2_selection.json" --steps 1 5 --random-steps 1 --out "$out"
uv run python scripts/analyze_ttt.py "$out"
