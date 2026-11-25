"""Compare the exploratory Experiment 4 variants with the registered validation runs
(PROTOCOL.md, "Two exploratory variants"). The criteria were fixed before any variant result
existed: median wMSE on `stack_genes`, diff-scored, T = 1; the per-pair difference B minus A
(positive: A is better) with a bootstrap 95% interval for its median (10,000 resamples of
pairs, as scripts/analyze_ttt.py); and wins out of the pairs, with differences within 1% of
B's value counted as ties. Run-to-run noise is about 1.1% of a median (PROTOCOL.md).

    uv run python scripts/compare_variants.py
"""

import json
from pathlib import Path

import pandas as pd

from stack_robustness.stats import paired_summary

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "results/ttt/val_v2"
OUT = ROOT / "results/ttt/val_v2_variants"
METRIC, PANEL, T, TIE = "wmse", "stack_genes", 1, 0.01

RUNS = {  # label -> (run folder, arm)
    "icl": (VAL / "shared_final_block_lr3e-4", "icl"),
    "shared": (VAL / "shared_all_cross_cell_lr3e-4", "shared"),
    "shared_ttt": (VAL / "ttt_lr3e-4_steps250", "shared_ttt"),
    "shared_ttt, ED only": (OUT / "ed_shared_ttt", "shared_ttt"),
    "shared_ttt, kNN": (OUT / "knn_shared_ttt", "shared_ttt"),
    "shared_ttt_random": (OUT / "registered_shared_ttt_random", "shared_ttt_random"),
    "ttt": (VAL / "ttt_lr1e-4_steps250", "ttt"),
    "ttt, ED only": (OUT / "ed_ttt", "ttt"),
    "ttt, kNN": (OUT / "knn_ttt", "ttt"),
    "ttt_random": (OUT / "registered_ttt_random", "ttt_random"),
}
COMPARISONS = [  # (A, B)
    ("shared_ttt, ED only", "shared_ttt"),
    ("shared_ttt, kNN", "shared_ttt"),
    ("shared_ttt, kNN", "shared_ttt_random"),
    ("shared_ttt", "shared_ttt_random"),
    ("shared_ttt, ED only", "shared"),
    ("shared_ttt", "shared"),
    ("ttt, ED only", "ttt"),
    ("ttt, kNN", "ttt"),
    ("ttt, kNN", "ttt_random"),
    ("ttt", "ttt_random"),
    ("ttt, ED only", "icl"),
    ("ttt", "icl"),
]


def scores(folder: Path, arm: str):
    path = folder / "scores.csv"
    if not path.exists():
        return None
    s = pd.read_csv(path)
    s = s[(s["metric"] == METRIC) & (s["panel"] == PANEL) & (s["model"] == f"{arm}_T{T}_diff")]
    return s.set_index("perturbation")["value"]


def main() -> None:
    found = {label: scores(*where) for label, where in RUNS.items()}
    lines = ["# Exploratory Experiment 4 variants on the validation doubles\n",
             f"Median {METRIC} on `{PANEL}` (lower is better), diff-scored, T = {T}. PROTOCOL.md, "
             "\"Two exploratory variants\", fixes how these are read.\n",
             "| run | pairs | median |", "|---|---|---|"]
    for label, s in found.items():
        lines.append(f"| {label} | {'pending' if s is None else len(s)} | "
                     f"{'' if s is None else f'{s.median():.5f}'} |")

    lines += ["\n## Paired differences\n",
              "B minus A per pair, so positive means A is better; bootstrap 95% interval for the median "
              f"(10,000 resamples); ties are differences within {TIE:.0%} of B's value.\n",
              "| A vs B | pairs | median | 95% interval | median change in % of B's median | A better / tie / B better |",
              "|---|---|---|---|---|---|"]
    for a, b in COMPARISONS:
        if found[a] is None or found[b] is None:
            lines.append(f"| {a} vs {b} | pending | | | | |")
            continue
        pairs = found[a].index.intersection(found[b].index)
        d = found[b][pairs] - found[a][pairs]
        rel = d / found[b][pairs]
        s = paired_summary(d, n_boot=10000, seed=0)
        better, worse = int((rel > TIE).sum()), int((rel < -TIE).sum())
        change = (found[a][pairs].median() - found[b][pairs].median()) / found[b][pairs].median()
        lines.append(f"| {a} vs {b} | {s['n']} | {s['median']:+.5f} | [{s['ci_low']:+.5f}, {s['ci_high']:+.5f}] | "
                     f"{change:+.1%} | {better} / {len(pairs) - better - worse} / {worse} |")

    for name in ("knn_shared_ttt", "knn_ttt"):
        path = OUT / name / "knn.json"
        if path.exists():
            chosen = json.loads(path.read_text())
            k = sum(len(c["genes"]) for c in chosen.values())
            overlap = sum(c["overlap"] for c in chosen.values())
            lines += [f"\n`{name}`: the nearest doubles include {overlap} of the {k} gene-sharing neighbours "
                      f"({overlap / max(k, 1):.0%}), over {len(chosen)} pairs."]

    text = "\n".join(lines) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
