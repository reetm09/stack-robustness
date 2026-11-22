"""Test-time training tasks for a held-out Norman double (the proposal's H(g*)).

A task names the conditions whose cells fill a window: prompt, query and target. For a
held-out double A+B the tasks are:
- each training double sharing a gene with A+B, as (first single, second single, double)
  in both prompt orders; swapping prompt and query leaves the joint target unchanged;
- optionally the null task (control, control, control), whose target is zero effect;
- A and B alone: prompt-only windows on the double's own inference cells, which train
  masked reconstruction only, since their joint target is what we must predict.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

CONTROL = "control"


@dataclass(frozen=True)
class Task:
    prompt: str
    query: Optional[str] = None
    target: Optional[str] = None


def neighbours(double: str, train_doubles) -> list:
    genes = set(double.split("+"))
    return [d for d in train_doubles if genes & set(d.split("+"))]


def norman_tasks(double: str, train_doubles, null: bool = False) -> list:
    tasks = []
    for neighbour in neighbours(double, train_doubles):
        a, b = neighbour.split("+")
        tasks += [Task(a, b, neighbour), Task(b, a, neighbour)]
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


def task_rows(task: Task, pools: dict, condition: np.ndarray):
    """(prompt, query, target) row indices for a task; query and target are None for a
    prompt-only task, which draws on the held-out double's own "infer" cells."""
    if task.query is None:
        return pools[(task.prompt, "infer")], None, None
    if task.target == CONTROL:
        return pools[(CONTROL, "prompt")], pools[(CONTROL, "query")], pools[(CONTROL, "target")]
    return pools[(task.prompt, "adapt")], pools[(task.query, "adapt")], np.flatnonzero(condition == task.target)
