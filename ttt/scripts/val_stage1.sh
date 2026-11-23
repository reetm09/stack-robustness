#!/usr/bin/env bash
# Validation stage 1 of PROTOCOL.md, in one queue slot: build the validation split, smoke-test
# norman_ttt.py on one validation double, then train the shared adapter at each grid setting
# (scope x learning rate, 2,000 steps) and score it on the validation doubles at T = 1.
#
#   SMOKE_DIR=/some/scratch/dir dibs submit --cpus 4 --mem 10G --vram 5G --time 1h45m -- bash scripts/val_stage1.sh
set -euo pipefail
cd "$(dirname "$0")/.."  # ttt/
smoke="${SMOKE_DIR:?set SMOKE_DIR to a scratch directory}"

if [ ! -f ../data/processed/norman2019_val_eval.npz ]; then
  (cd .. && uv run python scripts/norman_baselines.py --validation)
fi

rm -rf "$smoke"
uv run python scripts/norman_ttt.py --holdout val --doubles 1 --shared-steps 6 --ttt-steps 4 --draws 1 \
  --steps 1 2 --out "$smoke"
uv run python - "$smoke" <<'EOF'
import json
import sys
from pathlib import Path

import numpy as np

pair = next(Path(sys.argv[1], "pairs").glob("*.npz"))
saved = np.load(pair)
profiles = dict(zip(saved["names"], saved["values"]))
tasks = json.loads(pair.with_suffix(".json").read_text())["tasks"]
expected = {f"{arm}_T{t}_{kind}" for arm in ("icl", "shared", "ttt", "shared_ttt") for t in (1, 2) for kind in ("raw", "diff")}
expected |= {f"{arm}_T1_{kind}" for arm in ("ttt_random", "shared_ttt_random") for kind in ("raw", "diff")}
assert expected <= set(profiles), sorted(expected - set(profiles))
assert all(np.isfinite(v).all() for v in profiles.values()), "non-finite profile"
assert not np.allclose(profiles["icl_T1_raw"], profiles["shared_T1_raw"]), "the shared adapter changed nothing"
if tasks["ttt"] != tasks["ttt_random"]:  # equal only when the pair has no neighbours
    assert not np.allclose(profiles["ttt_T1_raw"], profiles["ttt_random_T1_raw"]), "neighbour and random TTT agree exactly"
print("smoke test passed:", len(profiles), "profiles")
EOF

for scope in final_block all_cross_cell; do
  for lr in 1e-4 3e-4; do
    arms="shared"
    if [ "$scope" = final_block ] && [ "$lr" = 1e-4 ]; then arms="icl shared"; fi
    uv run python scripts/norman_ttt.py --holdout val --arms $arms --scope "$scope" --shared-lr "$lr" \
      --out "../results/ttt/val/shared_${scope}_lr${lr}"
  done
done
