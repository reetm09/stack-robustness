"""Figures for Experiment 4 (test-time training) in the team report.

    uv run python scripts/paper_figures.py OUT_DIR

Writes OUT_DIR/ttt.pdf (paired differences on the test pairs, and TTT's gain against the
additive baseline's error) and OUT_DIR/ttt_curves.pdf (training and validation loss of the
shared adapter). Reads results/ttt/test_v2 and results/ttt/val_v2; the intervals use the
same bootstrap as scripts/analyze_ttt.py, so they match results/ttt/test_v2/analysis.md.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stack_robustness.stats import paired_summary

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "results/ttt/test_v2"
VAL = ROOT / "results/ttt/val_v2"
SCALE = 1e3  # wMSE is plotted in units of 10^-3

# (A, B, label): each row is B minus A per pair, so positive means A is better.
ROWS = [
    ("shared_ttt", "shared", "Shared adapter + TTT\nvs shared adapter"),
    ("shared_ttt", "shared_ttt_random", "Shared adapter + TTT:\nneighbours vs random"),
    ("ttt", "icl", "TTT vs frozen STACK"),
    ("ttt", "ttt_random", "TTT: neighbours\nvs random"),
    ("shared", "icl", "Shared adapter\nvs frozen STACK"),
]

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def wmse_table(run: Path) -> pd.DataFrame:
    """Pairs x predictions table of diff-scored wMSE on the STACK gene panel at T = 1."""
    s = pd.read_csv(run / "scores.csv")
    s = s[(s["metric"] == "wmse") & (s["panel"] == "stack_genes")]
    w = s.pivot_table(index="perturbation", columns="model", values="value")
    return w.rename(columns=lambda c: c.removesuffix("_T1_diff"))


def main_figure(out: Path) -> None:
    w = wmse_table(TEST)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.2), gridspec_kw={"width_ratios": [1.1, 1]})

    for i, (a, b, label) in enumerate(ROWS):
        s = paired_summary(w[b] - w[a], n_boot=10000, seed=0)
        key = i < 2
        color = "#1f77b4" if key else "0.45"
        y = len(ROWS) - 1 - i
        ax.plot([s["ci_low"] * SCALE, s["ci_high"] * SCALE], [y, y], color=color, lw=1.6 if key else 1.1)
        ax.plot(s["median"] * SCALE, y, "o", color=color, ms=4)
        ax.text(8.25, y, f"{s['n_positive']}/{s['n']}", va="center", ha="left", fontsize=6.5, color=color,
                clip_on=False)
    ax.axvline(0, color="0.2", lw=0.6, ls=":")
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[2] for r in ROWS][::-1])
    ax.set_xlim(-1, 8)
    ax.set_xlabel(r"Reduction in wMSE ($\times 10^{-3}$)")
    ax.set_title("(a) Paired differences (26 test pairs)", loc="left")

    gain = (w["shared"] - w["shared_ttt"]) * SCALE
    rho, p = spearmanr(gain, w["additive"])
    bx.axhline(0, color="0.2", lw=0.6, ls=":")
    bx.scatter(w["additive"], gain, s=11, color="#1f77b4", edgecolor="white", linewidth=0.4, zorder=3)
    bx.set_xscale("log")
    bx.set_xlabel("Additive baseline wMSE (log scale)")
    bx.set_ylabel(r"TTT gain ($\times 10^{-3}$)")
    bx.set_title("(b) TTT gain vs additive error", loc="left")
    bx.text(0.03, 0.95, f"Spearman $\\rho$ = {rho:.2f} (p = {p:.2f})", transform=bx.transAxes, va="top", fontsize=6.5)

    fig.subplots_adjust(left=0.235, right=0.985, bottom=0.19, top=0.88, wspace=0.55)
    fig.savefig(out / "ttt.pdf")
    plt.close(fig)


def curves_figure(out: Path) -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.0))

    loss = np.array(json.loads((TEST / "shared_history.json").read_text())["loss"], dtype=float)
    steps = np.arange(1, len(loss) + 1)
    ax.plot(steps, loss, color="0.75", lw=0.4, label="per step (one window)")
    k = 50
    smooth = np.convolve(loss, np.ones(k) / k, mode="valid")
    ax.plot(steps[k - 1:], smooth, color="#1f77b4", lw=1.2, label=f"mean of last {k} steps")
    ax.set_xlabel("Training step")
    ax.set_yscale("log")
    ax.set_ylabel("Training loss (log scale)")
    ax.set_title("(a) Final shared adapter (105 training pairs)", loc="left")
    ax.legend(frameon=False, loc="upper right")

    settings = [("final_block", "1e-4"), ("final_block", "3e-4"), ("all_cross_cell", "1e-4"),
                ("all_cross_cell", "3e-4")]
    for scope, lr in settings:
        h = json.loads((VAL / f"shared_{scope}_lr{lr}" / "shared_history.json").read_text())
        m = pd.DataFrame(h["monitor"])
        chosen = (scope, lr) == ("all_cross_cell", "3e-4")
        name = f"{'all layers' if scope == 'all_cross_cell' else 'final block'}, lr {lr}"
        bx.plot(m["step"], m["val_loss"], marker="o", ms=2.2, lw=1.4 if chosen else 0.8,
                color="#1f77b4" if chosen else "0.6", ls="-" if chosen or scope == "final_block" else "--",
                label=name + (" (chosen)" if chosen else ""), zorder=3 if chosen else 2)
        if chosen:
            bx.plot(m["step"], m["train_loss"], lw=0.9, color="#1f77b4", ls=":", label="same, training pairs")
    bx.set_xlabel("Training step")
    bx.set_ylabel("Loss on fixed windows")
    bx.set_title("(b) Validation pairs, model selection", loc="left")
    bx.legend(frameon=False, fontsize=5.6, loc="upper right")

    fig.tight_layout(w_pad=1.2)
    fig.savefig(out / "ttt_curves.pdf")
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    main_figure(out)
    curves_figure(out)
    print(f"wrote {out / 'ttt.pdf'} and {out / 'ttt_curves.pdf'}")
