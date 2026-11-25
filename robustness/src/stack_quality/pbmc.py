"""OpenProblems PBMC donor 1 (Arc's Stack-DrugPBMC-Example): cells, targets and scoring contexts.

Task (proposal Eq. 1): drug-treated T/NK cells are the prompt, DMSO T/NK cells the control
prompt, DMSO B or myeloid cells the query, and drug-treated B or myeloid cells the truth.

- A target is one (drug, query cell type, plate). Twelve drugs sit on one plate each; DMSO,
  Belinostat and Dabrafenib sit on both, so prompts, queries and controls always come from the
  target's plate. B and myeloid cells are scored separately because each drug's B/myeloid mix
  differs from DMSO's, which would otherwise read as a drug effect.
- Scoring is grouped by (cell type, plate), so each group has one control reference, as
  stack_robustness.EvalContext expects. Every target's cells are split into a truth and a
  duplicate half (Miller et al.).
- Genes are named in STACK's vocabulary spelling (matched case-insensitively, as arc-stack does).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from stack_robustness.baselines import assign_halves, interpolated_duplicate, pseudobulk
from stack_robustness.data import log_normalize
from stack_robustness.de import rank_genes
from stack_robustness.evaluation import EvalContext
from stack_robustness.genes import load_stack_genes

RAW = Path("data/raw/arc_example/openproblems_donor1.h5ad")
VOCAB = Path("data/raw/stack_vocab/basecount_1000per_15000max.pkl")
DMSO = "Dimethyl Sulfoxide"
KINDS = {"t cell": "T", "myeloid leukocyte": "My", "lymphocyte of b lineage": "B"}
QUERY_KINDS = ("B", "My")


@dataclass
class Cells:
    counts: sparse.csr_matrix  # cells x data genes, raw counts (every measured gene)
    data_genes: list
    drug: np.ndarray
    kind: np.ndarray  # "T" (T and NK cells, the prompt class), "B", "My"
    plate: np.ndarray
    vocab: list  # STACK's genes, in its order
    to_vocab: sparse.csr_matrix  # data genes x vocab, 0/1: counts @ to_vocab aligns to STACK

    def rows(self, drug=None, kind=None, plate=None) -> np.ndarray:
        mask = np.ones(len(self.drug), dtype=bool)
        for values, wanted in ((self.drug, drug), (self.kind, kind), (self.plate, plate)):
            if wanted is not None:
                mask &= values == wanted
        return np.flatnonzero(mask)

    def data_columns(self, genes) -> np.ndarray:
        """Data-matrix columns of vocabulary-spelled genes."""
        upper = {g.upper(): i for i, g in enumerate(self.data_genes)}
        return np.array([upper[g.upper()] for g in genes])


def vocab_projection(data_genes, vocab) -> sparse.csr_matrix:
    """0/1 matrix mapping data genes onto STACK's vocabulary, matched case-insensitively
    (arc-stack's _align_genes_to_target_list); vocabulary genes absent from the data stay zero."""
    upper = {g.upper(): i for i, g in enumerate(data_genes)}
    pairs = [(upper[v.upper()], j) for j, v in enumerate(vocab) if v.upper() in upper]
    r, c = zip(*pairs)
    return sparse.csr_matrix((np.ones(len(pairs), dtype=np.float32), (r, c)), shape=(len(data_genes), len(vocab)))


def load_cells(raw=RAW, vocab_path=VOCAB) -> Cells:
    adata = ad.read_h5ad(raw)
    vocab = load_stack_genes(vocab_path)
    data_genes = [str(g) for g in adata.var_names]
    return Cells(
        counts=sparse.csr_matrix(adata.X, dtype=np.float32),
        data_genes=data_genes,
        drug=adata.obs["sm_name"].astype(str).to_numpy(),
        kind=adata.obs["broad_cell_class"].astype(str).map(KINDS).to_numpy(),
        plate=adata.obs["plate_name"].astype(str).to_numpy(),
        vocab=vocab,
        to_vocab=vocab_projection(data_genes, vocab),
    )


def library(counts) -> np.ndarray:
    return np.asarray(sparse.csr_matrix(counts).sum(axis=1), dtype=np.float64).ravel()


def log_profile(counts, columns) -> np.ndarray:
    """Mean log1p(counts / library * 1e4) over cells, on the given data columns; the library is
    each cell's total over every gene (the ground-truth convention)."""
    counts = sparse.csr_matrix(counts)
    X = log_normalize(counts[:, columns], library(counts))
    return np.asarray(X.mean(axis=0)).ravel()


@dataclass(frozen=True)
class Target:
    drug: str
    kind: str
    plate: str

    @property
    def id(self) -> str:
        return f"{self.drug}|{self.kind}|{self.plate}"

    @property
    def group(self) -> str:
        return f"{self.kind}|{self.plate}"


def eligible_targets(cells: Cells, min_cells: int = 30) -> list:
    """(drug, B or myeloid, plate) with at least `min_cells` cells and DMSO controls on that plate."""
    out = []
    for drug in sorted(set(cells.drug) - {DMSO}):
        for kind in QUERY_KINDS:
            for plate in sorted(set(cells.plate)):
                if len(cells.rows(drug, kind, plate)) >= min_cells and len(cells.rows(DMSO, kind, plate)):
                    out.append(Target(drug, kind, plate))
    return out


@dataclass
class Prepared:
    """Everything fixed before any model runs, cached on disk."""

    contexts: dict  # group -> EvalContext (perturbations are target ids)
    targets: list  # Target
    panel: list  # stack_genes panel (vocabulary spelling)
    hvg: list  # hvg panel, a subset of `panel`
    effects: pd.DataFrame  # target id x panel: drug minus DMSO profile, all of the target's cells
    prompt_effects: pd.DataFrame  # target id x panel: T/NK effect, all prompt cells of the plate
    components: np.ndarray  # PCA of DMSO cells on the hvg panel (k x hvg)
    ctrl: pd.DataFrame  # group x panel: DMSO profile (Delta-ctrl reference)

    def save(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        for group, ctx in self.contexts.items():
            ctx.save(root / f"eval_{group.replace('|', '_')}.npz")
        np.savez_compressed(
            root / "prepared.npz",
            effects=self.effects.to_numpy(np.float32),
            prompt_effects=self.prompt_effects.loc[self.effects.index].to_numpy(np.float32),
            components=self.components.astype(np.float32),
            ctrl=self.ctrl.to_numpy(np.float32),
        )
        meta = {
            "targets": [[t.drug, t.kind, t.plate] for t in self.targets],
            "panel": self.panel,
            "hvg": self.hvg,
            "groups": list(self.ctrl.index),
        }
        (root / "prepared.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, root: Path) -> "Prepared":
        meta = json.loads((root / "prepared.json").read_text())
        targets = [Target(*t) for t in meta["targets"]]
        ids = [t.id for t in targets]
        with np.load(root / "prepared.npz") as z:
            # float32 on disk; float64 in memory so predictions can be added in place
            effects = pd.DataFrame(z["effects"].astype(float), index=ids, columns=meta["panel"])
            prompt_effects = pd.DataFrame(z["prompt_effects"].astype(float), index=ids, columns=meta["panel"])
            components = z["components"].astype(float)
            ctrl = pd.DataFrame(z["ctrl"].astype(float), index=meta["groups"], columns=meta["panel"])
        contexts = {g: EvalContext.load(root / f"eval_{g.replace('|', '_')}.npz") for g in meta["groups"]}
        return cls(contexts, targets, meta["panel"], meta["hvg"], effects, prompt_effects, components, ctrl)


def prepare(cells: Cells, *, seed: int = 0, min_cells: int = 30, min_detected: int = 3, n_hvg: int = 2000,
            n_components: int = 50, log=print) -> Prepared:
    """Truth/duplicate halves, DE tables, baselines, gene panels and the TabPFN PCA basis."""
    from sklearn.decomposition import PCA

    targets = eligible_targets(cells, min_cells)
    detected = np.diff(sparse.csc_matrix(cells.counts).indptr) >= min_detected
    vocab_upper = {v.upper(): v for v in cells.vocab}
    panel = [vocab_upper[g.upper()] for g, keep in zip(cells.data_genes, detected) if keep and g.upper() in vocab_upper]
    cols = cells.data_columns(panel)
    log(f"{len(targets)} targets; stack_genes panel {len(panel)} genes")

    def logx(rows):
        counts = cells.counts[rows]
        return log_normalize(counts[:, cols], library(counts))

    # HVGs and PCA from DMSO cells only, so no drug response leaks into either.
    dmso = cells.rows(DMSO)
    dmso_ad = ad.AnnData(X=logx(dmso), var=pd.DataFrame(index=panel))
    hv = sc.pp.highly_variable_genes(dmso_ad, n_top_genes=n_hvg, inplace=False)["highly_variable"].to_numpy()
    hvg = [g for g, h in zip(panel, hv) if h]
    pca = PCA(n_components=n_components, random_state=seed).fit(dmso_ad[:, hvg].X.toarray())
    log(f"hvg panel {len(hvg)} genes; PCA on {len(dmso)} DMSO cells explains {pca.explained_variance_ratio_.sum():.2f}")

    profile = {}  # (drug, kind, plate) -> all-cell profile
    for key in {(t.drug, t.kind, t.plate) for t in targets} | {(DMSO, k, p) for k in (*QUERY_KINDS, "T")
                                                              for p in set(cells.plate)}:
        profile[key] = np.asarray(logx(cells.rows(*key)).mean(axis=0)).ravel()
    effects = pd.DataFrame({t.id: profile[(t.drug, t.kind, t.plate)] - profile[(DMSO, t.kind, t.plate)]
                            for t in targets}, index=panel).T
    prompt_effects = pd.DataFrame(
        {t.id: np.asarray(logx(cells.rows(t.drug, "T", t.plate)).mean(axis=0)).ravel() - profile[(DMSO, "T", t.plate)]
         for t in targets}, index=panel).T

    groups = sorted({t.group for t in targets})
    ctrl = pd.DataFrame({g: profile[(DMSO, *g.split("|"))] for g in groups}, index=panel).T
    contexts = {}
    for group in groups:
        kind, plate = group.split("|")
        members = [t for t in targets if t.group == group]
        ids = [t.id for t in members]
        rows = np.concatenate([cells.rows(t.drug, kind, plate) for t in members] + [cells.rows(DMSO, kind, plate)])
        label = pd.Series(cells.drug[rows], index=np.arange(len(rows)))
        label = label.where(label.eq(DMSO), label + f"|{kind}|{plate}")
        halves = assign_halves(label, seed=seed)
        first, second = (halves == "first").to_numpy(), (halves == "second").to_numpy()
        is_ctrl = label.eq(DMSO).to_numpy()
        adata = ad.AnnData(X=logx(rows), obs=pd.DataFrame({"label": label.to_numpy()}), var=pd.DataFrame(index=panel))
        truth = pseudobulk(adata.X[first & ~is_ctrl], label[first & ~is_ctrl], panel).loc[ids]
        dup = pseudobulk(adata.X[second & ~is_ctrl], label[second & ~is_ctrl], panel).loc[ids]
        c = ctrl.loc[group]
        # Mean baseline, leave-one-drug-out: control plus the mean effect of the other drugs in this
        # cell type (either plate). pert_ref (the Delta-pert reference) uses every drug.
        same_kind = [t for t in targets if t.kind == kind]
        mean = pd.DataFrame({t.id: c + effects.loc[[s.id for s in same_kind if s.drug != t.drug]].mean() for t in members}).T
        pert_ref = c + effects.loc[[s.id for s in same_kind]].mean()
        log(f"{group}: {len(ids)} targets, {int(is_ctrl.sum())} DMSO cells; DE tests")
        de_rest_truth = rank_genes(adata[first & ~is_ctrl].copy(), "label", reference="rest")
        de_rest_dup = rank_genes(adata[second].copy(), "label", reference="rest")
        de_ctrl = rank_genes(adata[(first & ~is_ctrl) | is_ctrl].copy(), "label", reference=DMSO, method="wilcoxon")
        contexts[group] = EvalContext(
            truth=truth,
            ctrl=c,
            pert_ref=pert_ref,
            de_vs_ctrl={i: de_ctrl[i] for i in ids},
            de_vs_rest={i: de_rest_truth[i] for i in ids},
            baselines={
                "control_baseline": pd.DataFrame([c] * len(ids), index=ids),
                "mean_baseline": mean,
                "technical_duplicate": dup,
                "interpolated_duplicate": pd.DataFrame(
                    {i: interpolated_duplicate(dup.loc[i], mean.loc[i], de_rest_dup[i]["pval_adj"]) for i in ids}).T,
            },
            panels={"stack_genes": panel, f"hvg{n_hvg}": hvg},
        )
    return Prepared(contexts, targets, panel, hvg, effects, prompt_effects, pca.components_, ctrl)
