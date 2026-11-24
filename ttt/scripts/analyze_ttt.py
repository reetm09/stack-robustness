"""Report a scripts/norman_ttt.py run as PROTOCOL.md fixes it, or compare validation runs.

    uv run python scripts/analyze_ttt.py ../results/ttt/test
    uv run python scripts/analyze_ttt.py ../results/ttt/val/* --select shared_T1_diff
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stack_robustness.evaluation import EvalContext
from stack_robustness.stats import paired_summary
from stack_ttt.diagnostics import prompt_use

ROOT = Path(__file__).resolve().parents[2]

PILOT = ["AHR+FEV", "AHR+KLF1", "BPGM+SAMD1"]
NON_ADDITIVE = ["DUSP9+SNAI1", "DUSP9+PRTG", "JUN+CEBPA", "AHR+FEV", "KLF1+COL2A1", "CBL+PTPN12",
                "BPGM+SAMD1", "MAP2K6+SPI1", "AHR+KLF1"]  # largest additive wMSE; fixed in PROTOCOL.md
ARMS = ["icl", "shared", "ttt", "ttt_random", "shared_ttt", "shared_ttt_random"]
REFERENCES = ["technical_duplicate", "interpolated_duplicate", "additive", "query", "query_one",
              "mean_baseline", "control_baseline"]
COMPARISONS = [("shared_ttt", "shared"), ("shared_ttt", "shared_ttt_random"), ("ttt", "icl"),
               ("ttt", "ttt_random"), ("shared", "icl")]
METRIC, PANEL = "wmse", "stack_genes"


def wide(scores: pd.DataFrame, metric: str = METRIC, panel: str = PANEL) -> pd.DataFrame:
    """Pairs x models table of one metric on one panel."""
    s = scores[(scores["metric"] == metric) & (scores["panel"] == panel)]
    return s.pivot_table(index="perturbation", columns="model", values="value")


def prompt_use_table(run: Path, arms, t: int) -> pd.DataFrame:
    """stack_ttt.diagnostics.prompt_use for every saved pair, prompt order and arm."""
    holdout = json.loads((run / "args.json").read_text())["holdout"]
    ctx = EvalContext.load(ROOT / f"data/processed/norman2019{'_val' if holdout == 'val' else ''}_eval.npz")
    genes = list(ctx.truth.columns)
    ctrl = ctx.ctrl[genes].to_numpy()
    rows = []
    for f in sorted((run / "pairs").glob("*.npz")):
        z = np.load(f)
        assert list(z["genes"]) == genes, f"{f} is not on the scoring genes"
        profiles = dict(zip(z["names"], z["values"]))
        rows += [{"arm": arm, "pair": f.stem, **o} for arm in arms if f"{arm}_T{t}_diff_one" in profiles
                 for o in prompt_use(profiles, ctrl, arm, steps=t)]
    return pd.DataFrame(rows)


def fmt(x: float) -> str:
    return f"{x:.4f}" if abs(x) < 0.1 else f"{x:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--select", nargs="*", help="print median validation wMSE of these models per run")
    parser.add_argument("--t", type=int, default=1, help="generation steps T of the main comparison")
    args = parser.parse_args()

    if args.select is not None:
        for run in args.runs:
            if not (run / "scores.csv").exists():
                continue
            w = wide(pd.read_csv(run / "scores.csv"))
            cells = [f"{m} {fmt(w[m].median())}" for m in args.select if m in w]
            print(f"{run.name:45s} {len(w)} pairs  " + "  ".join(cells))
        return

    run = args.runs[0]
    scores = pd.read_csv(run / "scores.csv")
    w = wide(scores)
    pairs = list(w.index)
    subsets = {
        "all": pairs,
        "without pilot": [p for p in pairs if p not in PILOT],
        "non-additive": [p for p in pairs if p in NON_ADDITIVE],
    }
    arms = [a for a in ARMS if f"{a}_T{args.t}_diff" in w]
    key = {a: f"{a}_T{args.t}_diff" for a in arms} | {r: r for r in REFERENCES if r in w}

    lines = [f"# Study 4 analysis: {run.name}\n",
             f"Primary metric: {METRIC} on `{PANEL}` (lower is better), diff-scored STACK, T = {args.t}. "
             f"Pairs: {len(pairs)}.\n",
             "## Median wMSE\n",
             "| prediction | " + " | ".join(f"{k} ({len(v)})" for k, v in subsets.items()) + " |",
             "|---|" + "---|" * len(subsets)]
    for name, column in [(f"{a} (diff)", key[a]) for a in arms] + [(f"{a} (raw)", f"{a}_T{args.t}_raw") for a in arms] \
            + [(r, r) for r in REFERENCES if r in w]:
        lines.append(f"| {name} | " + " | ".join(fmt(w.loc[v, column].median()) if v else "" for v in subsets.values()) + " |")

    lines += ["\n## Paired differences in wMSE\n",
              "Each row is B minus A per pair, so a positive value means A is better. The interval is a "
              "bootstrap 95% interval for the median (10,000 resamples of pairs).\n",
              "| A vs B | pairs | median | 95% interval | pairs where A is better |", "|---|---|---|---|---|"]
    comparisons = [c for c in COMPARISONS if c[0] in key and c[1] in key]
    comparisons += [(a, ref) for a in arms for ref in ("query", "additive") if ref in key]
    for a, b in comparisons:
        for subset, members in subsets.items():
            if not members:
                continue
            s = paired_summary(w.loc[members, key[b]] - w.loc[members, key[a]], n_boot=10000, seed=0)
            lines.append(f"| {a} vs {b} ({subset}) | {s['n']} | {fmt(s['median'])} | "
                         f"[{fmt(s['ci_low'])}, {fmt(s['ci_high'])}] | {s['n_positive']} of {s['n']} |")

    lines += ["\n## TTT gain against the additive baseline's error (Spearman, all pairs)\n"]
    for start, ttt in (("shared", "shared_ttt"), ("icl", "ttt")):
        if start in key and ttt in key and "additive" in key:
            gain = w[key[start]] - w[key[ttt]]
            rho, p = spearmanr(gain, w["additive"])
            lines.append(f"- {ttt} gain over {start}: rho = {rho:.2f} (p = {p:.2f}, {len(gain)} pairs)")

    t5 = [a for a in ARMS if f"{a}_T5_diff" in w and a in key]
    if t5 and args.t != 5:
        lines += ["\n## T = 5 against T = 1 (median wMSE, all pairs)\n", "| arm | T = 1 | T = 5 |", "|---|---|---|"]
        lines += [f"| {a} | {fmt(w[key[a]].median())} | {fmt(w[f'{a}_T5_diff'].median())} |" for a in t5]

    use = prompt_use_table(run, arms, args.t)
    if not use.empty:
        lines += [f"\n## Prompt use (exploratory, T = {args.t})\n",
                  "Replacing the control prompt with A's cells shifts the prediction. For each pair and prompt "
                  "order: the correlation of that shift with A's own effect on A's top 200 genes (r), and the "
                  "shift's size relative to that effect.\n",
                  "| arm | pair-orders | median r | r above +0.3 | r below -0.3 | median relative size |",
                  "|---|---|---|---|---|---|"]
        for arm in arms:
            u = use[use["arm"] == arm]
            if len(u):
                lines.append(f"| {arm} | {len(u)} | {u.r.median():+.2f} | {(u.r > 0.3).sum()} | {(u.r < -0.3).sum()} | "
                             f"{u.relative_size.median():.2f} |")

    lines += ["\n## Secondary metrics (median over all pairs, diff-scored, T = 1)\n"]
    for panel in sorted(scores["panel"].unique()):
        rows = []
        for metric in ("wmse", "pearson_deltactrl", "pearson_deltapert", "mse"):
            m = wide(scores, metric, panel)
            rows.append([metric] + [fmt(m[key[a]].median()) if key.get(a) in m else "" for a in arms + ["query", "additive"]])
        lines += [f"Panel `{panel}`:\n", "| metric | " + " | ".join(arms + ["query", "additive"]) + " |",
                  "|---|" + "---|" * (len(arms) + 2)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]

    (run / "analysis.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
