#!/usr/bin/env bash
# Validation stage 2 of PROTOCOL.md, in one queue slot: tune test-time training on the
# validation doubles, with shared_ttt starting from stage 1's selected shared adapter.
#   a. TTT learning rate {3e-5, 1e-4} at 50 steps, for ttt and shared_ttt;
#   b. 250 steps at each arm's better learning rate;
#   then each arm keeps the best of its three settings (scripts/select_ttt.py).
#
#   dibs submit --cpus 2 --mem 10G --vram 4G --time 1h30m --retry -- bash scripts/val_stage2.sh
set -euo pipefail
cd "$(dirname "$0")/.."  # ttt/
val=../results/ttt/val
# Stage 1 selected the final block + decoder at learning rate 3e-4 (PROTOCOL.md, Decisions).
shared=(--scope final_block --shared-lr 3e-4 --shared-adapter "$val/shared_final_block_lr3e-4/shared_adapter.pt")

run() {  # run LR STEPS ARM...
  local lr=$1 steps=$2
  shift 2
  uv run python scripts/norman_ttt.py --holdout val --arms "$@" "${shared[@]}" \
    --ttt-lr "$lr" --ttt-steps "$steps" --out "$val/ttt_lr${lr}_steps${steps}"
}

for lr in 3e-5 1e-4; do
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
