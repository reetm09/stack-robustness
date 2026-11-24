#!/usr/bin/env bash
# Validation stage 2 of PROTOCOL.md, in one queue slot: tune test-time training on the
# validation doubles, with shared_ttt starting from the shared adapter stage 1 selected.
#   a. TTT learning rate {3e-5, 1e-4, 3e-4} at 50 steps, for ttt and shared_ttt;
#   b. 250 steps at each arm's best learning rate;
#   then each arm keeps the best of its four settings (scripts/select_ttt.py).
#
#   [OUT_ROOT=../results/ttt/val_v2] dibs submit --cpus 2 --mem 10G --vram 5G --time 2h --retry -- bash scripts/val_stage2.sh
set -euo pipefail
cd "$(dirname "$0")/.."  # ttt/
val="${OUT_ROOT:-../results/ttt/val_v2}"

choice=$(uv run python scripts/select_ttt.py shared "$val")
read -r scope shared_lr <<< "$choice"
shared=(--scope "$scope" --shared-lr "$shared_lr" --shared-adapter "$val/shared_${scope}_lr${shared_lr}/shared_adapter.pt")

run() {  # run LR STEPS ARM...
  local lr=$1 steps=$2
  shift 2
  uv run python scripts/norman_ttt.py --holdout val --arms "$@" "${shared[@]}" \
    --ttt-lr "$lr" --ttt-steps "$steps" --out "$val/ttt_lr${lr}_steps${steps}"
}

for lr in 3e-5 1e-4 3e-4; do  # select_ttt.RATES
  run "$lr" 50 ttt shared_ttt
done

choice=$(uv run python scripts/select_ttt.py lr "$val")
read -r lr_ttt lr_shared_ttt <<< "$choice"
if [ "$lr_ttt" = "$lr_shared_ttt" ]; then
  run "$lr_ttt" 250 ttt shared_ttt
else
  run "$lr_ttt" 250 ttt
  run "$lr_shared_ttt" 250 shared_ttt
fi
uv run python scripts/select_ttt.py final "$val" "$lr_ttt" "$lr_shared_ttt"
