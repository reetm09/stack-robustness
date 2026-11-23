"""The choices of validation stage 2 (PROTOCOL.md): each TTT arm's learning rate, then its setting.

    uv run python scripts/select_ttt.py lr ../results/ttt/val              # prints "<ttt lr> <shared_ttt lr>"
    uv run python scripts/select_ttt.py final ../results/ttt/val 1e-4 3e-5  # writes stage2_selection.json

Each run directory is ttt_lr<lr>_steps<steps>, written by scripts/val_stage2.sh.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ARMS = ("ttt", "shared_ttt")
RATES = ("3e-5", "1e-4")


def median(val: Path, lr: str, steps: int, arm: str) -> float:
    """Median validation wMSE of the arm's diff-scored T = 1 prediction in one run."""
    s = pd.read_csv(val / f"ttt_lr{lr}_steps{steps}" / "scores.csv")
    s = s[(s["metric"] == "wmse") & (s["panel"] == "stack_genes") & (s["model"] == f"{arm}_T1_diff")]
    assert not s.empty, f"no {arm} scores in ttt_lr{lr}_steps{steps}"
    return float(s["value"].median())


def main() -> None:
    stage, val = sys.argv[1], Path(sys.argv[2])
    if stage == "lr":  # step 2: the better learning rate at 50 steps
        print(*(min(RATES, key=lambda lr: median(val, lr, 50, arm)) for arm in ARMS))
        return
    better = dict(zip(ARMS, sys.argv[3:5]))
    selection = {}
    for arm in ARMS:  # step 3: the best of the arm's three settings; within 1% of it, fewer steps
        medians = {(steps, lr): median(val, lr, steps, arm) for steps, lr in [(50, r) for r in RATES] + [(250, better[arm])]}
        best = min(medians.values())
        steps, lr = min((k for k, v in medians.items() if v <= 1.01 * best), key=lambda k: (k[0], medians[k]))
        selection[arm] = {"lr": lr, "steps": steps,
                          "median_wmse": {f"lr{r}_steps{s}": v for (s, r), v in medians.items()}}
    for arm in ARMS:  # the random-pair arms copy their neighbour arm's settings
        selection[f"{arm}_random"] = {k: selection[arm][k] for k in ("lr", "steps")}
    (val / "stage2_selection.json").write_text(json.dumps(selection, indent=1) + "\n")
    print(json.dumps(selection, indent=1))


if __name__ == "__main__":
    main()
