"""The choices of validation (PROTOCOL.md): stage 1's shared adapter, then each TTT arm's setting.

    uv run python scripts/select_ttt.py shared ../results/ttt/val_v2             # prints "<scope> <lr>"
    uv run python scripts/select_ttt.py lr ../results/ttt/val_v2                 # prints "<ttt lr> <shared_ttt lr>"
    uv run python scripts/select_ttt.py final ../results/ttt/val_v2 1e-4 3e-4    # writes stage2_selection.json

Stage 1's runs are shared_<scope>_lr<lr>, stage 2's ttt_lr<lr>_steps<steps> (val_stage1.sh,
val_stage2.sh). Each choice takes the lowest median validation wMSE of the diff-scored T = 1
prediction; a setting within 1% of it goes to the cheaper option: fewer steps, then the
smaller scope.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ARMS = ("ttt", "shared_ttt")
RATES = ("3e-5", "1e-4", "3e-4")  # TTT learning rates at 50 steps
SCOPES = ("final_block", "all_cross_cell")  # smaller first
SHARED_RATES = ("1e-4", "3e-4")


def median(val: Path, run: str, model: str) -> float:
    """Median validation wMSE of one model's predictions in one run."""
    s = pd.read_csv(val / run / "scores.csv")
    s = s[(s["metric"] == "wmse") & (s["panel"] == "stack_genes") & (s["model"] == model)]
    assert not s.empty, f"no {model} scores in {run}"
    return float(s["value"].median())


def cheapest_within_one_percent(medians: dict, cost) -> tuple:
    best = min(medians.values())
    return min((k for k, v in medians.items() if v <= 1.01 * best), key=lambda k: (cost(k), medians[k]))


def main() -> None:
    stage, val = sys.argv[1], Path(sys.argv[2])
    if stage == "shared":  # step 1; every setting runs 2,000 steps, so ties go to the smaller scope
        medians = {(scope, lr): median(val, f"shared_{scope}_lr{lr}", "shared_T1_diff")
                   for scope in SCOPES for lr in SHARED_RATES}
        scope, lr = cheapest_within_one_percent(medians, lambda k: SCOPES.index(k[0]))
        (val / "stage1_selection.json").write_text(json.dumps(
            {"scope": scope, "lr": lr, "median_wmse": {f"{s}_lr{r}": v for (s, r), v in medians.items()}}, indent=1) + "\n")
        print(scope, lr)
        return
    if stage == "lr":  # step 2: the better learning rate at 50 steps
        print(*(min(RATES, key=lambda lr: median(val, f"ttt_lr{lr}_steps50", f"{arm}_T1_diff")) for arm in ARMS))
        return
    better = dict(zip(ARMS, sys.argv[3:5]))
    selection = {}
    for arm in ARMS:  # step 3: the arm's best setting; within 1% of it, fewer steps
        settings = [(50, r) for r in RATES] + [(250, better[arm])]
        medians = {(steps, lr): median(val, f"ttt_lr{lr}_steps{steps}", f"{arm}_T1_diff") for steps, lr in settings}
        steps, lr = cheapest_within_one_percent(medians, lambda k: k[0])
        selection[arm] = {"lr": lr, "steps": steps,
                          "median_wmse": {f"lr{r}_steps{s}": v for (s, r), v in medians.items()}}
    for arm in ARMS:  # the random-pair arms copy their neighbour arm's settings
        selection[f"{arm}_random"] = {k: selection[arm][k] for k in ("lr", "steps")}
    (val / "stage2_selection.json").write_text(json.dumps(selection, indent=1) + "\n")
    print(json.dumps(selection, indent=1))


if __name__ == "__main__":
    main()
