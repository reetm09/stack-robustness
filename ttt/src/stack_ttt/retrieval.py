"""Effect embeddings of single perturbations in frozen STACK's cell-embedding space, for choosing
test-time training examples by similarity (tasks.knn_tasks; an exploratory variant, PROTOCOL.md)."""

import zlib

import numpy as np
import torch

from .objective import teacher_embeddings
from .tasks import CONTROL


def mean_embeddings(model, counts, rows_by_condition: dict, device="cpu", window: int = 512, seed: int = 0) -> dict:
    """Mean frozen-STACK cell embedding of each condition's rows (counts: cells x genes, sparse).
    A condition's rows are shuffled and split into windows of at most `window` cells holding that
    condition alone, every cell a fully observed prompt cell: the way teacher_embeddings embeds
    target cells in training, with the adapters switched off."""
    means = {}
    for condition, rows in rows_by_condition.items():
        rows = np.random.default_rng([seed, zlib.crc32(condition.encode())]).permutation(np.asarray(rows))
        total = 0.0
        for chunk in np.array_split(rows, int(np.ceil(len(rows) / window))):
            x = torch.from_numpy(counts[chunk].toarray().astype(np.float32))[None].to(device)
            total = total + teacher_embeddings(model, x)[0].double().sum(dim=0).cpu().numpy()
        means[condition] = total / len(rows)
    return means


def effect_embeddings(model, cells, pools, singles, device="cpu", window: int = 512, seed: int = 0) -> dict:
    """e(g): the mean embedding of single g's "infer" cells minus that of the control "infer"
    cells. These are the cells predictions start from, so no held-out double's cells are used."""
    rows = {g: pools[(g, "infer")] for g in singles} | {CONTROL: pools[(CONTROL, "infer")]}
    means = mean_embeddings(model, cells.counts, rows, device=device, window=window, seed=seed)
    return {g: means[g] - means[CONTROL] for g in singles}
