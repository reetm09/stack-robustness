"""Test-time training tasks for a held-out Norman double (the proposal's H(g*)).

A task names the conditions whose cells fill a window: prompt, query and target. For a
held-out double A+B the tasks are:
- each training double sharing a gene with A+B, as (first single, second single, double)
  in both prompt orders; swapping prompt and query leaves the joint target unchanged;
- optionally the null task (control, control, control), whose target is zero effect;
- A and B alone: prompt-only windows on the double's own inference cells, which train
  masked reconstruction only, since their joint target is what we must predict.
"""

import zlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .norman import single_pools

CONTROL = "control"


@dataclass(frozen=True)
class Task:
    prompt: str
    query: Optional[str] = None
    target: Optional[str] = None


def neighbours(double: str, train_doubles) -> list:
    genes = set(double.split("+"))
    return [d for d in train_doubles if genes & set(d.split("+"))]


def shared_tasks(train_doubles, null: bool = False) -> list:
    """Every training double in both prompt orders: the tasks of one shared adapter, the
    spec's context-level adapter and Study 2's fine-tuned STACK."""
    tasks = [task for double in train_doubles for task in _both_orders(double)]
    return tasks + ([Task(CONTROL, CONTROL, CONTROL)] if null else [])


def norman_tasks(double: str, train_doubles, null: bool = False) -> list:
    return _instance_tasks(double, neighbours(double, train_doubles), null)


def random_tasks(double: str, train_doubles, seed: int, null: bool = False) -> list:
    """norman_tasks with the neighbours swapped for as many random training doubles that share
    no gene with `double`: the control showing whether choosing neighbours matters."""
    genes = set(double.split("+"))
    unrelated = [d for d in train_doubles if not genes & set(d.split("+"))]
    k = min(len(neighbours(double, train_doubles)), len(unrelated))
    rng = np.random.default_rng([seed, zlib.crc32(double.encode())])
    chosen = sorted(rng.choice(unrelated, size=k, replace=False).tolist()) if k else []
    return _instance_tasks(double, chosen, null)


def knn_neighbours(double: str, train_doubles, embedding: dict, k: int) -> list:
    """The k training doubles nearest `double`, most similar first, by the cosine similarity of
    summed single-gene effect embeddings, e(a+b) = e(a) + e(b) (retrieval.effect_embeddings).
    Only the singles' embeddings are used, so the held-out double's own cells never enter."""
    def vector(d: str) -> np.ndarray:
        a, b = d.split("+")
        return np.asarray(embedding[a], dtype=float) + np.asarray(embedding[b], dtype=float)

    target = vector(double)

    def cosine(d: str) -> float:
        v = vector(d)
        return float(v @ target / max(np.linalg.norm(v) * np.linalg.norm(target), 1e-12))

    return sorted(train_doubles, key=lambda d: (-cosine(d), d))[:k]


def knn_tasks(double: str, train_doubles, embedding: dict, null: bool = False) -> list:
    """norman_tasks with the neighbours swapped for as many training doubles nearest `double` in
    frozen STACK's embedding space (knn_neighbours): the selection the team's paper draft
    describes, an exploratory variant. Task counts match the neighbour arm's."""
    k = len(neighbours(double, train_doubles))
    return _instance_tasks(double, sorted(knn_neighbours(double, train_doubles, embedding, k)), null)


def _both_orders(double: str) -> list:
    a, b = double.split("+")
    return [Task(a, b, double), Task(b, a, double)]


def _instance_tasks(double: str, doubles, null: bool) -> list:
    tasks = [task for d in doubles for task in _both_orders(d)]
    if null:
        tasks.append(Task(CONTROL, CONTROL, CONTROL))
    return tasks + [Task(gene) for gene in double.split("+")]


def control_pools(condition: np.ndarray, seed: int) -> dict:
    """Control cells in four disjoint pools: prompt, query and target for the null task, so it
    never shows the model the same cell twice, and "infer" for the control prompt used as a
    reference at prediction time."""
    rows = np.random.default_rng(seed).permutation(np.flatnonzero(condition == CONTROL))
    roles = ("prompt", "query", "target", "infer")
    return {(CONTROL, role): part for role, part in zip(roles, np.array_split(rows, len(roles)))}


def cell_pools(condition: np.ndarray, seed: int) -> dict:
    """single_pools for every loaded single, plus control_pools. Doubles ("A+B") get no pool:
    their cells are only ever targets, so loading more doubles moves no cell between pools."""
    singles = sorted({c for c in set(condition.tolist()) if c != CONTROL and "+" not in c})
    return single_pools(condition, singles, seed=seed) | control_pools(condition, seed=seed)


def task_rows(task: Task, pools: dict, condition: np.ndarray):
    """(prompt, query, target) row indices for a task; query and target are None for a
    prompt-only task, which draws on the held-out double's own "infer" cells."""
    if task.query is None:
        return pools[(task.prompt, "infer")], None, None
    if task.target == CONTROL:
        return pools[(CONTROL, "prompt")], pools[(CONTROL, "query")], pools[(CONTROL, "target")]
    return pools[(task.prompt, "adapt")], pools[(task.query, "adapt")], np.flatnonzero(condition == task.target)
