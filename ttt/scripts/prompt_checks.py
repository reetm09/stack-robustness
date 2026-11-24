"""Two validation diagnostics for Study 4 (exploratory), run on 2026-11-23 before version 2,
with version 1's selected shared adapter (results/ttt/val/shared_final_block_lr3e-4) and
without gene renaming. Output: results/ttt/diagnostics/prompt_checks (job 4584).

1. The paper's layout: frozen STACK with each validation single's prompt (its infer half)
   onto control query cells, against a control prompt onto the same cells. Does the shift
   follow the prompt's own effect?
2. No context slots: icl and the selected shared adapter (final block, 3e-4) on the 21
   validation doubles at T = 1 with context_ratio 0 instead of 0.4. Training reveals true
   A+B cells where inference puts more prompt cells; with no such slots the two agree.
   Two pairs are first regenerated at 0.4 and must equal the saved profiles.

    cd ttt && uv run python <this file> --out ../results/ttt/diagnostics/prompt_checks
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stack_robustness.evaluation import EvalContext
from stack_robustness.genes import load_stack_genes
from stack_ttt.lora import adapters_disabled, inject_lora, load_adapter_state, stack_targets
from stack_ttt.model import load_stack
from stack_ttt.norman import load_norman
from stack_ttt.predict import generate, mean_log_profile, median_of_medians
from stack_ttt.tasks import CONTROL, cell_pools

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "results/ttt/val"
START = time.time()


def log(message):
    print(f"[{time.time() - START:6.0f}s] {message}", flush=True)


def pull(shift, effect, top=200):
    """Correlation of a shift with an effect, and its relative size, on the effect's top genes."""
    idx = np.argsort(-np.abs(effect))[:top]
    return float(np.corrcoef(shift[idx], effect[idx])[0, 1]), float(np.abs(shift[idx]).mean() / np.abs(effect[idx]).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_path = ROOT / "data/raw/stack_vocab/basecount_1000per_15000max.pkl"

    ctx = EvalContext.load(ROOT / "data/processed/norman2019_val_eval.npz")
    panel = list(ctx.truth.columns)
    split = pd.read_csv(ROOT / "splits/norman2019_val_conditions.csv")
    doubles = sorted(split.loc[split["split"].eq("val") & split["kind"].eq("combination"), "condition"])
    singles = sorted(split.loc[split["kind"].eq("single"), "condition"])
    vocab = load_stack_genes(vocab_path)
    cells = load_norman(ROOT / "data/raw/norman/NormanWeissman2019_filtered.h5ad", vocab, conditions=set(singles) | {CONTROL})
    pools = cell_pools(cells.condition, seed=0)  # the experiment's pools: doubles never move a cell (tested)
    reference = pools[(CONTROL, "infer")]
    log(f"loaded {cells.counts.shape[0]} cells; {len(doubles)} validation doubles")

    model, _ = load_stack(ROOT / "data/raw/stack/bc_large_aligned.ckpt", device=device)
    inject_lora(model, stack_targets(model, "final_block"), rank=8, alpha=16.0)
    cls_frozen = {k: v.clone() for k, v in model.cls.state_dict().items()}
    saved = torch.load(VAL / "shared_final_block_lr3e-4/shared_adapter.pt", map_location=device, weights_only=True)

    def profile(rows):
        return mean_log_profile(cells.counts[rows], cells.library[rows], cells.genes, panel)

    def gen(prompt_rows, query_rows, seed, cr):
        out = generate(model, cells.counts[prompt_rows], cells.counts[query_rows], cells.genes, vocab_path, num_steps=1,
                       seed=seed, batch_size=args.batch_size, context_ratio=cr, context_ratio_min=min(cr, 0.2))
        return mean_log_profile(out, cells.library[query_rows], cells.genes, panel)

    def predict(double, cr, draws=2):
        """predict_double with a context ratio: raw, diff and query, per order and combined."""
        a, b = double.split("+")
        found = {"raw": {}, "diff": {}, "query": {}}
        for p, q in [(a, b), (b, a)]:
            q_rows = pools[(q, "infer")]
            observed = profile(q_rows)
            found["query"][(p, q)] = [observed]
            found["raw"][(p, q)], found["diff"][(p, q)] = [], []
            for draw in range(draws):
                seed = 1000 * draw + 1
                g = gen(pools[(p, "infer")], q_rows, seed, cr)
                found["raw"][(p, q)].append(g)
                found["diff"][(p, q)].append(observed + g - gen(reference, q_rows, seed, cr))
        out = {}
        for name, by_order in found.items():
            out[f"{name}_one"] = pd.concat(by_order[(a, b)], axis=1).median(axis=1)
            out[name] = median_of_medians(by_order)
        return out

    def use(arm):
        if arm == "icl":
            model.cls.load_state_dict(cls_frozen)
            return adapters_disabled(model)
        load_adapter_state(model, saved["adapters"])
        model.cls.load_state_dict(saved["cls"])
        return torch.no_grad()  # a no-op context; the adapters stay on

    saved_runs = {"icl": VAL / "shared_final_block_lr1e-4", "shared": VAL / "shared_final_block_lr3e-4"}

    # Reproduce two saved pairs at 0.4 before trusting anything at 0.
    for arm in ("icl", "shared"):
        with use(arm):
            for double in doubles[:2]:
                got = predict(double, 0.4)
                z = np.load(saved_runs[arm] / "pairs" / f"{double}.npz")
                want = dict(zip(z["names"], z["values"]))
                for kind in ("raw", "diff"):
                    diff = float(np.abs(got[kind].to_numpy() - want[f"{arm}_T1_{kind}"]).max())
                    assert diff < 1e-5, f"{arm} {double} {kind}: regenerated at 0.4 differs from the saved run by {diff}"
    log("regenerated 2 pairs per arm at context 0.4: identical to the saved runs")

    # 1. The paper's layout: prompts onto control query cells.
    rng = np.random.default_rng(0)
    ctrl_query = np.sort(rng.choice(pools[(CONTROL, "query")], size=537, replace=False))  # three full windows at T = 1
    ctrl_obs = profile(ctrl_query)
    prompts = sorted({g for d in doubles for g in d.split("+")})
    rows = []
    with use("icl"):
        base = [gen(reference, ctrl_query, 1000 * d + 1, 0.4) for d in range(2)]
        for g in prompts:
            shifts = [gen(pools[(g, "infer")], ctrl_query, 1000 * d + 1, 0.4) - base[d] for d in range(2)]
            shift = pd.concat(shifts, axis=1).median(axis=1).to_numpy()
            effect = (profile(pools[(g, "infer")]) - ctrl_obs).to_numpy()
            r, size = pull(shift, effect)
            rows.append({"prompt": g, "cells": len(pools[(g, "infer")]), "r": r, "relative_size": size})
    layout = pd.DataFrame(rows)
    layout.to_csv(args.out / "control_query_layout.csv", index=False)
    log(f"1. frozen STACK, prompts onto control cells (T = 1, context 0.4), {len(layout)} prompts: "
        f"r(shift, prompt's effect) median {layout.r.median():+.2f} [{layout.r.quantile(.25):+.2f}, {layout.r.quantile(.75):+.2f}], "
        f"{(layout.r > 0.3).sum()} above +0.3, {(layout.r < -0.3).sum()} below -0.3; relative size median {layout.relative_size.median():.2f}")

    # 2. No context slots.
    frames, pulls = {}, []
    for arm in ("icl", "shared"):
        with use(arm):
            for double in doubles:
                got = predict(double, 0.0)
                for kind in ("raw", "diff"):
                    frames.setdefault(f"{arm}_T1_{kind}_ctx0", {})[double] = got[kind]
                a, b = double.split("+")
                obs_b = got["query_one"].to_numpy(); obs_a = 2 * got["query"].to_numpy() - obs_b
                dif_ab = got["diff_one"].to_numpy(); dif_ba = 2 * got["diff"].to_numpy() - dif_ab
                ctrl = ctx.ctrl[panel].to_numpy()
                for p, sh, eff in ((a, dif_ab - obs_b, obs_a - ctrl), (b, dif_ba - obs_a, obs_b - ctrl)):
                    r, size = pull(sh, eff)
                    pulls.append({"arm": arm, "double": double, "prompt": p, "r": r, "relative_size": size})
            log(f"2. {arm} at context 0 done")
    frames = {k: pd.DataFrame(v).T.loc[doubles, panel] for k, v in frames.items()}
    scores = ctx.score(frames, perturbations=doubles)
    scores.to_csv(args.out / "scores_ctx0.csv", index=False)
    pulls = pd.DataFrame(pulls)
    pulls.to_csv(args.out / "prompt_pull_ctx0.csv", index=False)
    w = scores[(scores.metric == "wmse") & (scores.panel == "stack_genes")].pivot_table(index="perturbation", columns="model", values="value")
    summary = {"control_query_layout": layout.describe().to_dict(), "median_wmse_ctx0": w.median().to_dict(),
               "pull_ctx0": pulls.groupby("arm")[["r", "relative_size"]].median().to_dict()}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    for arm in ("icl", "shared"):
        p = pulls[pulls.arm == arm]
        log(f"2. {arm} context 0: median wMSE diff {w[f'{arm}_T1_diff_ctx0'].median():.4f}, raw {w[f'{arm}_T1_raw_ctx0'].median():.4f}; "
            f"r(shift, prompt's effect) median {p.r.median():+.2f}, {(p.r > 0.3).sum()} of {len(p)} above +0.3; "
            f"relative size {p.relative_size.median():.2f}")


if __name__ == "__main__":
    main()
