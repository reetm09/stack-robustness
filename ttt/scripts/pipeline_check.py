"""Is STACK's shift of Norman's cells caused by our pipeline, or by STACK itself?

Compares our code with Arc's own code, on Arc's tutorial data (OpenProblems PBMC donor 1,
huggingface.co/datasets/arcinstitute/Stack-DrugPBMC-Example) and on Norman et al.:

1. loading     our load_stack and Arc's load_model_from_checkpoint give the same weights
2. alignment   our Norman loader and Arc's CLI gene alignment give the same count matrix
3. generation  our generate() and Arc's CLI call give the same cells for the same inputs
4. artifact    with unperturbed cells as both prompt and query, how far STACK's generated
               cells sit from the measured query cells: on PBMC (Arc's data) and on Norman
5. positive control  Arc's tutorial task on PBMC: predict each drug's effect on held-out
               myeloid and B cells from T cells. STACK with Arc's synthetic control
               (generated drug minus generated DMSO, "diff") and without it ("raw"), beside
               copying the T-cell effect over (no model)

    dibs submit --cpus 4 --mem 16G --vram 6G --time 1h -- uv run python scripts/pipeline_check.py
"""

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from stack.cli.generation import _align_genes_to_target_list
from stack.model_loading import load_model_from_checkpoint

from stack_robustness.genes import load_stack_genes
from stack_ttt.model import load_stack
from stack_ttt.norman import load_norman
from stack_ttt.predict import generate, mean_log_profile

ROOT = Path(__file__).resolve().parents[2]
DMSO = "Dimethyl Sulfoxide"
START = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - START:6.0f}s] {message}", flush=True)


def check_loading(checkpoint: Path) -> tuple:
    ours, _ = load_stack(checkpoint, device="cpu")
    theirs = load_model_from_checkpoint(str(checkpoint), model_class="ICL_FinetunedModel", device=torch.device("cpu"))
    a, b = ours.state_dict(), theirs.state_dict()
    shared = set(a) & set(b)
    result = {
        "same_parameter_names": set(a) == set(b),
        "max_abs_difference": max((a[k].float() - b[k].float()).abs().max().item() for k in shared),
    }
    del theirs
    return ours, result


def arc_align(X, var_names, vocab) -> sparse.csr_matrix:
    """Counts on STACK's vocabulary, exactly as Arc's stack-generation CLI aligns them."""
    adata = ad.AnnData(X=sparse.csr_matrix(X), var=pd.DataFrame(index=pd.Index(list(var_names), dtype=object)))
    return sparse.csr_matrix(_align_genes_to_target_list(adata, list(vocab), None).X, dtype=np.float32)


def arc_generate(model, prompt, query, vocab, genelist, num_steps: int, seed: int, batch_size: int):
    """Arc's CLI call (stack/cli/generation.py, _run_incontext_generation) with its defaults."""
    var = pd.DataFrame(index=pd.Index(list(vocab), dtype=object))
    base = ad.AnnData(X=sparse.csr_matrix(prompt, dtype=np.float32), var=var)
    test = ad.AnnData(X=sparse.csr_matrix(query, dtype=np.float32), var=var.copy())
    out = model.get_incontext_generation(
        base_adata_or_path=base, test_adata_or_path=test, genelist_path=str(genelist),
        prompt_ratio=0.25, context_ratio=0.4, context_ratio_min=0.2, mask_rate=1.0, mode="mdm",
        num_steps=num_steps, gene_name_col=None, batch_size=batch_size, show_progress=False,
        num_workers=0, random_seed=seed,
    )
    return sparse.csr_matrix(out[0] if isinstance(out, tuple) else out)


def artifact(generated: pd.Series, observed: pd.Series) -> dict:
    """How far the generated profile sits from the measured one, over genes expressed in either."""
    expressed = (generated > 0.01) | (observed > 0.01)
    shift = (generated - observed)[expressed]
    top = shift.abs().sort_values(ascending=False).index[:12]
    return {
        "genes": int(expressed.sum()),
        "pearson_profiles": float(np.corrcoef(generated[expressed], observed[expressed])[0, 1]),
        "rms_shift": float(np.sqrt((shift**2).mean())),
        "genes_shifted_over_0.5": int((shift.abs() > 0.5).sum()),
        "top_shifts": {g: [round(float(observed[g]), 2), round(float(generated[g]), 2)] for g in top},
    }


def pearson(a: pd.Series, b: pd.Series) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/raw/stack/bc_large_aligned.ckpt")
    parser.add_argument("--vocab", type=Path, default=ROOT / "data/raw/stack_vocab/basecount_1000per_15000max.pkl")
    parser.add_argument("--pbmc", type=Path, default=ROOT / "data/raw/arc_example/openproblems_donor1.h5ad")
    parser.add_argument("--norman", type=Path, default=ROOT / "data/raw/norman/NormanWeissman2019_filtered.h5ad")
    parser.add_argument("--n-control", type=int, default=2000, help="Norman control cells per prompt and query")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", type=Path, default=ROOT / "results/ttt/pipeline_check")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = load_stack_genes(args.vocab)
    report = {}

    model, report["loading"] = check_loading(args.checkpoint)
    model = model.to(device).eval()
    log(f"1. loading: {report['loading']}")

    # 2. alignment on Norman: Arc's CLI alignment of the raw cells against our loader.
    conditions = ["control", "KLF1"]
    ours = load_norman(args.norman, vocab, conditions=set(conditions))
    backed = ad.read_h5ad(args.norman, backed="r")
    condition = backed.obs["perturbation"].astype(str).str.replace("_", "+", regex=False).to_numpy()
    rows = np.flatnonzero(np.isin(condition, conditions))
    raw = sparse.csc_matrix(backed.X[:]).tocsr()[rows]
    norman_genes = list(backed.var_names)
    backed.file.close()
    theirs = arc_align(raw, norman_genes, vocab)
    differ = np.flatnonzero(np.asarray(abs(theirs - ours.counts).sum(axis=0)).ravel())
    report["alignment_norman"] = {
        "cells": int(len(rows)),
        "same_shape": theirs.shape == ours.counts.shape,
        "columns_that_differ": int(len(differ)),
        "nonzero_vocabulary_columns": int((np.diff(theirs.tocsc().indptr) > 0).sum()),
    }
    log(f"2. alignment on Norman: {report['alignment_norman']}")

    # 3 and 4 on Norman: control prompt, a disjoint set of control query cells.
    rng = np.random.default_rng(0)
    control = rng.permutation(ours.rows("control"))
    prompt, query = control[: args.n_control], control[args.n_control : 2 * args.n_control]
    q_lib = ours.library[query]
    observed = mean_log_profile(ours.counts[query], q_lib, vocab, vocab)
    report["artifact_norman"] = {}
    for steps in (1, 5):
        mine = generate(model, ours.counts[prompt], ours.counts[query], vocab, args.vocab, num_steps=steps,
                        seed=0, batch_size=args.batch_size)
        if steps == 5:
            arcs = arc_generate(model, ours.counts[prompt], ours.counts[query], vocab, args.vocab, steps, 0,
                                args.batch_size)
            report["generation_norman_identical"] = bool(abs(mine - arcs).max() == 0)
            log(f"3. generation on Norman, ours vs Arc's call: identical = {report['generation_norman_identical']}")
        report["artifact_norman"][f"T{steps}"] = artifact(mean_log_profile(mine, q_lib, vocab, vocab), observed)
        log(f"4. Norman artifact T={steps}: { {k: v for k, v in report['artifact_norman'][f'T{steps}'].items() if k != 'top_shifts'} }")

    # PBMC (Arc's tutorial data): T cells prompt, myeloid and B cells query, as in the tutorial.
    pbmc = ad.read_h5ad(args.pbmc)
    counts = arc_align(pbmc.X, pbmc.var_names, vocab)
    library = np.asarray(pbmc.X.sum(axis=1), dtype=float).ravel()
    is_t = pbmc.obs["broad_cell_class"].eq("t cell").to_numpy()
    drug = pbmc.obs["sm_name"].astype(str).to_numpy()
    query = np.flatnonzero(~is_t & (drug == DMSO))
    t_dmso = np.flatnonzero(is_t & (drug == DMSO))
    q_lib = library[query]
    panel = [g for g, n in zip(vocab, np.diff(counts.tocsc().indptr)) if n > 0]
    observed = mean_log_profile(counts[query], q_lib, vocab, panel)

    def gen(prompt_rows, query_rows, steps, seed=0):
        return generate(model, counts[prompt_rows], counts[query_rows], vocab, args.vocab, num_steps=steps,
                        seed=seed, batch_size=args.batch_size)

    generated_dmso = {}
    report["artifact_pbmc"] = {}
    for steps in (1, 5):
        generated_dmso[steps] = gen(t_dmso, query, steps)
        if steps == 5:
            arcs = arc_generate(model, counts[t_dmso], counts[query], vocab, args.vocab, steps, 0, args.batch_size)
            report["generation_pbmc_identical"] = bool(abs(generated_dmso[steps] - arcs).max() == 0)
            log(f"3. generation on PBMC, ours vs Arc's call: identical = {report['generation_pbmc_identical']}")
        profile = mean_log_profile(generated_dmso[steps], q_lib, vocab, panel)
        report["artifact_pbmc"][f"T{steps}"] = artifact(profile, observed)
        log(f"4. PBMC artifact T={steps}: { {k: v for k, v in report['artifact_pbmc'][f'T{steps}'].items() if k != 'top_shifts'} }")

    # The same check within one cell type: DMSO T cells as prompt, other DMSO T cells as query.
    halves = rng.permutation(t_dmso)
    t_prompt, t_query = halves[: len(halves) // 2], halves[len(halves) // 2 :]
    same_type = gen(t_prompt, t_query, 5)
    report["artifact_pbmc"]["T5_t_cells_only"] = artifact(
        mean_log_profile(same_type, library[t_query], vocab, panel),
        mean_log_profile(counts[t_query], library[t_query], vocab, panel),
    )
    log(f"4. PBMC artifact, T cells only: { {k: v for k, v in report['artifact_pbmc']['T5_t_cells_only'].items() if k != 'top_shifts'} }")

    # 5. positive control: every drug's effect on the held-out myeloid and B cells.
    gen_dmso_profile = mean_log_profile(generated_dmso[5], q_lib, vocab, panel)
    t_dmso_profile = mean_log_profile(counts[t_dmso], library[t_dmso], vocab, panel)
    rows = []
    for name in sorted(set(drug) - {DMSO}):
        target = np.flatnonzero(~is_t & (drug == name))
        prompt_rows = np.flatnonzero(is_t & (drug == name))
        truth = mean_log_profile(counts[target], library[target], vocab, panel) - observed
        generated = mean_log_profile(gen(prompt_rows, query, 5), q_lib, vocab, panel)
        t_effect = mean_log_profile(counts[prompt_rows], library[prompt_rows], vocab, panel) - t_dmso_profile
        predictions = {
            "stack_diff": generated - gen_dmso_profile,
            "stack_raw": generated - observed,
            "t_cell_effect_copied": t_effect,
        }
        top = truth.abs().sort_values(ascending=False).index[:200]
        row = {"drug": name, "target_cells": len(target), "prompt_cells": len(prompt_rows),
               "effect_size": float(np.sqrt((truth[top] ** 2).mean()))}
        for key, pred in predictions.items():
            row[f"{key}_pearson"] = pearson(pred, truth)
            row[f"{key}_pearson_top200"] = pearson(pred[top], truth[top])
            row[f"{key}_mse"] = float(((pred - truth) ** 2).mean())
        rows.append(row)
        log(f"5. {name}: " + ", ".join(f"{k} {v:.3f}" for k, v in row.items() if k.endswith("pearson")))
    table = pd.DataFrame(rows).sort_values("effect_size", ascending=False)

    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "positive_control.csv", index=False)
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    log(f"wrote {args.out}")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
