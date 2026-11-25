"""Run the robustness benchmark on OpenProblems PBMC donor 1 and store every result.

    python -m stack_quality.run                        # everything (resumes finished conditions)
    python -m stack_quality.run --only clean --limit-drugs 2 --tag pilot   # timing pilot
    python -m stack_quality.run --skip-stack           # re-run tabular models and scoring only

Run from the repo root. One process, one model on the GPU at a time: STACK for every condition,
then (after STACK is freed) TabPFN, then scoring on the CPU.

Outputs
  data/processed/robustness/pbmc_donor1/            prepared contexts (cached), git-ignored
      preds/<condition>__d<draw>.npz                STACK and prompt-effect predictions per draw
      tabular.npz                                   tabular-model predictions
  results/robustness/pbmc_donor1[_<tag>]/
      scores.csv.gz      condition, arm, severity, target, draw, group, panel, model,
                         perturbation (target id), metric, value, gap_closed
      baselines.csv.gz   the fixed baselines and pca_ceiling, scored once
      timing.csv         seconds per STACK call
      summary.md         clean scores and per-arm robustness summaries
      config.json        settings and package versions
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .conditions import default_conditions, drugs_on, plates, sample_inputs
from .corrupt import background_profile
from .pbmc import RAW, VOCAB, Prepared, load_cells, prepare
from .score import Scorer, curves, plate_gap, summarise_arm
from .stack_run import run_plate
from .tabular import lodo_predict, pca_ceiling

START = time.time()
STACK_MODELS = ("stack", "stack_raw", "prompt_effect")


def log(message: str) -> None:
    print(f"[{time.time() - START:7.0f}s] {message}", flush=True)


def pred_path(root: Path, cond, draw: int) -> Path:
    return root / "preds" / f"{cond.name}__d{draw}.npz"


def save_preds(path: Path, out: dict, panel) -> None:
    ids = sorted(out["stack"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.array(ids, dtype=str),
                        **{m: np.stack([out[m][i] for i in ids]).astype(np.float32) for m in STACK_MODELS})


def load_preds(path: Path, panel) -> dict:
    with np.load(path) as z:
        ids = [str(i) for i in z["ids"]]
        return {m: pd.DataFrame(z[m], index=ids, columns=panel) for m in STACK_MODELS}


def run_stack(args, cells, prep: Prepared, conditions, root: Path, out_dir: Path) -> None:
    import torch

    from stack_ttt.model import load_stack

    todo = [(c, d) for c in conditions for d in range(c.draws) if not pred_path(root, c, d).exists()]
    if not todo:
        log("STACK: every condition already done")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"STACK: loading on {device}; {len(todo)} condition-draws to run")
    model, _ = load_stack(args.checkpoint, device=device)
    background = background_profile(cells.counts)
    cache, timing = {}, []
    for cond, draw in todo:
        t0 = time.time()
        out = {m: {} for m in STACK_MODELS}
        for plate in plates(cells):
            drugs = drugs_on(cells, plate, cond, prep.targets)[: args.limit_drugs or None]
            if not drugs:
                continue
            inputs = sample_inputs(cells, cond, draw, plate, drugs, seed=args.seed)
            with torch.no_grad():
                found, calls = run_plate(
                    model, cells, inputs, cond, draw, plate, prep.targets, prep.ctrl, prep.panel, args.vocab,
                    background, num_steps=args.num_steps, batch_size=args.batch_size, seed=args.seed,
                    control_cache=cache)
            for m in STACK_MODELS:
                out[m].update(found[m])
            timing += [{"condition": cond.name, "draw": draw, "plate": plate, "seconds": s} for s in calls]
        save_preds(pred_path(root, cond, draw), out, prep.panel)
        log(f"  {cond.name} draw {draw}: {len(out['stack'])} targets, {len(timing)} calls, {time.time() - t0:.0f}s")
        timing_csv = out_dir / "timing.csv"
        pd.DataFrame(timing).to_csv(timing_csv, index=False, mode="a", header=not timing_csv.exists())
        timing = []
    del model
    torch.cuda.empty_cache()


def run_tabular(args, prep: Prepared, stack_preds: dict, root: Path) -> dict:
    """{(condition, draw): {model: DataFrame target ids x hvg}} for each tabular model."""
    path = root / f"tabular{'_' + args.tag if args.tag else ''}.npz"
    keys = sorted(stack_preds)
    hvg = prep.hvg
    test_rows = []
    for key in keys:
        pe = stack_preds[key]["prompt_effect"]
        for tid in pe.index:
            drug, kind, _ = tid.split("|")
            test_rows.append({"key": key, "id": tid, "drug": drug, "kind": kind, "x": pe.loc[tid, hvg].to_numpy()})
    test = pd.DataFrame(test_rows)
    row_keys = np.array([f"{k[0]}__d{k[1]}__{i}" for k, i in zip(test["key"], test["id"])], dtype=str)
    if path.exists():
        with np.load(path) as z:
            stored = {m: z[m] for m in args.tabular if m in z.files}
            same_rows = "rows" in z.files and np.array_equal(z["rows"], row_keys)
        if same_rows and len(stored) == len(args.tabular):
            log(f"tabular: loaded {path}")
            return _split_tabular(test, stored, hvg)
    train = pd.DataFrame([{"drug": t.drug, "kind": t.kind, "x": prep.prompt_effects.loc[t.id, hvg].to_numpy(),
                           "y": prep.effects.loc[t.id, hvg].to_numpy()} for t in prep.targets])
    stored = {}
    for name in args.tabular:
        t0 = time.time()
        log(f"tabular: {name} on {len(test)} test rows")
        stored[name] = lodo_predict(name, train, test, prep.components, device=args.device,
                                    n_estimators=args.tabpfn_estimators, seed=args.seed, log=log).astype(np.float32)
        log(f"tabular: {name} done in {time.time() - t0:.0f}s")
    np.savez_compressed(path, rows=row_keys, **stored)
    return _split_tabular(test, stored, hvg)


def _split_tabular(test: pd.DataFrame, stored: dict, hvg) -> dict:
    out = {}
    for key, rows in test.groupby("key", sort=False):
        idx = rows.index.to_numpy()
        out[key] = {m: pd.DataFrame(z[idx], index=rows["id"].to_numpy(), columns=hvg) for m, z in stored.items()}
    return out


def _score_one(scorer, prep: Prepared, cond, draw, preds: dict, tab: dict, hvg_panel: str) -> pd.DataFrame:
    ctrl_of = {t.id: prep.ctrl.loc[t.group] for t in prep.targets}
    ids = list(preds["stack"].index)
    ctrl = pd.DataFrame([ctrl_of[i] for i in ids], index=ids)
    models = {"stack": preds["stack"], "stack_raw": preds["stack_raw"],
              "source_effect_copy": ctrl + preds["prompt_effect"]}
    for name, effect in tab.items():
        full = ctrl.copy()
        full.loc[effect.index, effect.columns] += effect
        models[name] = full
    panels_of = {name: [hvg_panel] for name in tab}
    s = scorer.score(models, panels_of)
    return s.assign(condition=cond.name, arm=cond.arm, severity=cond.severity, target=cond.target, draw=draw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--vocab", type=Path, default=VOCAB)
    parser.add_argument("--checkpoint", type=Path, default=Path("data/raw/stack/bc_large_aligned.ckpt"))
    parser.add_argument("--work", type=Path, default=Path("data/processed/robustness/pbmc_donor1"))
    parser.add_argument("--out", type=Path, default=Path("results/robustness/pbmc_donor1"))
    parser.add_argument("--tag", default="", help="suffix for a separate run (e.g. pilot); kept apart from the main one")
    parser.add_argument("--only", nargs="*", default=None, help="condition names to run (default: all)")
    parser.add_argument("--limit-drugs", type=int, default=0, help="drugs per plate (0: all), for pilots")
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--clean-draws", type=int, default=3)
    parser.add_argument("--plate-draws", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=1, help="STACK generation steps T")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tabular", nargs="*", default=["tabpfn", "ridge", "knn"])
    parser.add_argument("--tabpfn-estimators", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-stack", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(f"{args.out}_{args.tag}") if args.tag else args.out
    preds_root = Path(f"{args.work}_{args.tag}") if args.tag else args.work
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = load_cells(args.raw, args.vocab)
    log(f"loaded {len(cells.drug)} cells x {len(cells.data_genes)} genes")
    if (args.work / "prepared.json").exists():
        prep = Prepared.load(args.work)
        log(f"loaded prepared contexts from {args.work}")
    else:
        prep = prepare(cells, seed=args.seed, log=log)
        prep.save(args.work)
        log(f"saved prepared contexts to {args.work}")
    hvg_panel = next(p for p in next(iter(prep.contexts.values())).panels if p.startswith("hvg"))

    conditions = default_conditions(args.draws, args.clean_draws, args.plate_draws)
    if args.only:
        conditions = [c for c in conditions if c.name in args.only]
    if not args.skip_stack:
        run_stack(args, cells, prep, conditions, preds_root, out_dir)

    stack_preds = {(c.name, d): load_preds(pred_path(preds_root, c, d), prep.panel)
                   for c in conditions for d in range(c.draws) if pred_path(preds_root, c, d).exists()}
    log(f"{len(stack_preds)} condition-draws with STACK predictions")
    tab = run_tabular(args, prep, stack_preds, preds_root) if args.tabular else {}

    log("scoring")
    scorer = Scorer(prep.contexts)
    by_name = {c.name: c for c in conditions}
    frames = [_score_one(scorer, prep, by_name[name], draw, preds, tab.get((name, draw), {}), hvg_panel)
              for (name, draw), preds in stack_preds.items()]
    scores = pd.concat(frames, ignore_index=True)
    scores.to_csv(out_dir / "scores.csv.gz", index=False)

    truth_effect = pd.concat([ctx.truth.sub(ctx.ctrl, axis=1) for ctx in prep.contexts.values()])
    ceiling = pd.DataFrame(pca_ceiling(truth_effect[prep.hvg].to_numpy(), prep.components),
                           index=truth_effect.index, columns=prep.hvg)
    ctrl_rows = pd.DataFrame([prep.ctrl.loc[t.group] for t in prep.targets], index=[t.id for t in prep.targets])
    ceiling_full = ctrl_rows.copy()
    ceiling_full.loc[ceiling.index, prep.hvg] += ceiling
    baselines = pd.concat([scorer.baselines(), scorer.score({"pca_ceiling": ceiling_full}, {"pca_ceiling": [hvg_panel]})])
    baselines.to_csv(out_dir / "baselines.csv.gz", index=False)
    log(f"wrote {len(scores)} score rows and {len(baselines)} baseline rows to {out_dir}")

    write_summary(out_dir, scores, baselines, hvg_panel)
    import tabpfn
    import torch

    versions = {"python": platform.python_version(), "torch": torch.__version__, "tabpfn": tabpfn.__version__}
    config = {k: str(v) for k, v in vars(args).items()} | {"versions": versions,
                                                         "conditions": [c.__dict__ | {"drugs": list(c.drugs)}
                                                                        for c in conditions]}
    (out_dir / "config.json").write_text(json.dumps(config, indent=1))
    log("done")


def write_summary(out_dir: Path, scores: pd.DataFrame, baselines: pd.DataFrame, hvg_panel: str) -> None:
    lines = ["# Robustness on OpenProblems PBMC donor 1\n",
             "Generated by `python -m stack_quality.run`. Means over targets (drug, query cell type, plate); "
             "intervals are 95% bootstrap intervals over drugs.\n"]
    clean = scores[scores["arm"] == "clean"]
    baselines = baselines[baselines["perturbation"].isin(set(clean["perturbation"]))]  # same targets
    for panel in (hvg_panel, "stack_genes"):
        both = pd.concat([clean, baselines]).query("panel == @panel")
        both = both.groupby(["model", "perturbation", "metric"])["value"].mean().reset_index()
        for stat in ("mean", "median"):  # medians too: one outlying drug can dominate a mean error
            table = both.groupby(["metric", "model"])["value"].agg(stat).unstack("model")
            lines += [f"## Clean condition, panel `{panel}` ({stat} over targets)\n", table.round(3).to_markdown(), ""]
    for metric in ("pearson_deltactrl", "wmse"):
        curve = curves(scores[scores["arm"] != "plate"], metric, hvg_panel)
        lines += [f"## Robustness summaries: `{metric}`, panel `{hvg_panel}`\n",
                  summarise_arm(curve, metric).to_markdown(index=False), ""]
        if (scores["arm"] == "plate").any():
            lines += [f"### Covariate reliance Gamma_plate (`{metric}`; Belinostat and Dabrafenib only)\n",
                      plate_gap(scores, metric, hvg_panel).round(4).to_markdown(index=False), ""]
    lines += ["## Caveats\n",
              "- One donor: no donor or study holdout; plate stands in for a covariate, on two drugs only.",
              "- Tabular models (TabPFN, ridge, kNN) see 13 training drugs per held-out drug, one pseudobulk "
              "per target; repeated subsampled pseudobulks (replicates and extra context rows) are left for "
              "the full report. They never read the query cells, so query arms change only STACK.",
              "- STACK: frozen Stack-Large-Aligned, T = 1. Prompts below 333 cells are repeated within STACK's "
              "window, so the prompt-size arm measures a small prompt repeated, not fewer prompt slots."]
    (out_dir / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
