#!/usr/bin/env bash
# Validation stage 1 of PROTOCOL.md, in one queue slot: build the validation split, smoke-test
# norman_ttt.py on one validation double, then train the shared adapter at each grid setting
# (scope:learning rate, 2,000 steps) and score it on the validation doubles at T = 1.
#
#   SMOKE_DIR=/some/scratch/dir dibs submit --cpus 2 --mem 10G --vram 6G --time 1h45m --retry -- bash scripts/val_stage1.sh
#
# OUT_ROOT (default ../results/ttt/val_v2) holds one folder per setting. SETTINGS (default the
# whole grid) picks settings, e.g. "final_block:3e-4". Frozen STACK (icl) is scored once, beside
# the setting named by ICL_WITH (default final_block:3e-4; empty for none).
set -euo pipefail
cd "$(dirname "$0")/.."  # ttt/
smoke="${SMOKE_DIR:?set SMOKE_DIR to a scratch directory}"
out_root="${OUT_ROOT:-../results/ttt/val_v2}"
settings="${SETTINGS:-final_block:1e-4 final_block:3e-4 all_cross_cell:1e-4 all_cross_cell:3e-4}"
icl_with="${ICL_WITH-final_block:3e-4}"

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

assert json.loads(Path(sys.argv[1], "args.json").read_text())["reveal"] == "prompt", "not the version 2 layout"
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
curve = json.loads(Path(sys.argv[1], "shared_history.json").read_text())["monitor"]
assert [m["step"] for m in curve] == [0, 6], [m["step"] for m in curve]
assert all(np.isfinite(m[k]) for m in curve for k in ("train_loss", "val_loss")), curve
print("smoke test passed:", len(profiles), "profiles; loss on fixed windows", curve)
EOF

for setting in $settings; do
  scope="${setting%%:*}"
  lr="${setting##*:}"
  arms="shared"
  if [ "$setting" = "$icl_with" ]; then arms="icl shared"; fi
  uv run python scripts/norman_ttt.py --holdout val --arms $arms --scope "$scope" --shared-lr "$lr" \
    --out "$out_root/shared_${scope}_lr${lr}"
done
