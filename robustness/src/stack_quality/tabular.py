"""Tabular models of the query effect from the prompt effect, leave-one-drug-out, in PCA space.

One row per target (drug, query cell type, plate), at the pseudobulk level: no cell in the
prompt is paired with a cell in the query, so pseudobulks are the unit that can be paired.

  features  the prompt (T/NK) effect projected on the PCA basis (fit on DMSO cells, hvg panel),
            plus an indicator for myeloid (vs B) query cells
  target    the query effect projected on the same basis; one regressor per component for
            TabPFN (as in the proposal), one multi-output fit for ridge and kNN
  context   every target of the other drugs, clean (all cells); the held-out drug's test rows
            carry the condition's subsampled or corrupted prompt effect

Predictions are mapped back to the hvg genes and added to the control reference, so these
models are scored on the hvg panel only. `pca_ceiling` (the truth projected on the basis and
back) bounds what any model working in this space can reach.

Caveat (to address in the full report): each training row is one pseudobulk draw. Several
subsampled pseudobulks per target, as extra context rows and as replicates, would show how much
the comparison depends on which cells formed each pseudobulk.
"""

import os

import numpy as np
import pandas as pd

# Keep the built TabPFN model in memory across the per-component fits (read when tabpfn loads).
os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "2")


def features(prompt_effects_hvg: np.ndarray, kinds, components: np.ndarray) -> np.ndarray:
    z = prompt_effects_hvg @ components.T
    return np.column_stack([z, (np.asarray(kinds) == "My").astype(float)])


def _tabpfn_factory(device: str, n_estimators: int, seed: int):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    def make():
        return TabPFNRegressor.create_default_for_version(
            ModelVersion.V3_5, device=device, n_estimators=n_estimators, random_state=seed)
    return make


def _fit_predict(name: str, X, Y, X_test, *, device: str, n_estimators: int, seed: int) -> np.ndarray:
    if name == "tabpfn":
        make = _tabpfn_factory(device, n_estimators, seed)
        return np.column_stack([make().fit(X, Y[:, k]).predict(X_test) for k in range(Y.shape[1])])
    from sklearn.linear_model import RidgeCV
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "ridge":
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
    elif name == "knn":
        model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="distance"))
    else:
        raise ValueError(name)
    return model.fit(X, Y).predict(X_test)


def lodo_predict(name: str, train: pd.DataFrame, test: pd.DataFrame, components: np.ndarray, *,
                 device: str = "cuda", n_estimators: int = 8, seed: int = 0, log=print) -> np.ndarray:
    """Predicted query effects on the hvg genes for every test row.

    `train`: one row per target with columns drug, kind, and `x` / `y` (prompt and query effects
    on the hvg genes, as arrays). `test`: columns drug, kind, x. Rows of a held-out drug are
    predicted from the other drugs' training rows only.
    """
    X_all = features(np.stack(train["x"]), train["kind"], components)
    Y_all = np.stack(train["y"]) @ components.T
    X_test = features(np.stack(test["x"]), test["kind"], components)
    Z = np.zeros((len(test), components.shape[0]))
    for drug in sorted(test["drug"].unique()):
        keep = (train["drug"] != drug).to_numpy()
        rows = (test["drug"] == drug).to_numpy()
        Z[rows] = _fit_predict(name, X_all[keep], Y_all[keep], X_test[rows], device=device,
                               n_estimators=n_estimators, seed=seed)
        log(f"  {name}: {drug} ({rows.sum()} test rows, {keep.sum()} context rows)")
    return Z @ components


def pca_ceiling(effects_hvg: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (effects_hvg @ components.T) @ components
