import numpy as np
import pytest
from scipy import sparse

from stack_ttt.tasks import Task, control_pools, neighbours, norman_tasks, task_rows
from stack_ttt.windows import pick, task_window

TRAIN = ["A+C", "D+B", "E+F", "C+D"]


def test_neighbours_are_training_doubles_sharing_a_gene():
    assert neighbours("A+B", TRAIN) == ["A+C", "D+B"]


def test_norman_tasks_use_both_orders_of_each_neighbour_and_the_test_doubles_own_prompts():
    tasks = norman_tasks("A+B", TRAIN, null=True)

    assert tasks == [
        Task("A", "C", "A+C"),
        Task("C", "A", "A+C"),
        Task("D", "B", "D+B"),
        Task("B", "D", "D+B"),
        Task("control", "control", "control"),
        Task("A"),
        Task("B"),
    ]


def test_task_rows_keep_adaptation_and_inference_cells_apart():
    condition = np.array(["A"] * 4 + ["C"] * 4 + ["A+C"] * 3 + ["control"] * 12)
    pools = {("A", "adapt"): np.array([0, 1]), ("A", "infer"): np.array([2, 3]), ("C", "adapt"): np.array([4, 5])}
    pools.update(control_pools(condition, seed=0))

    prompt, query, target = task_rows(Task("A", "C", "A+C"), pools, condition)
    assert prompt.tolist() == [0, 1] and query.tolist() == [4, 5] and target.tolist() == [8, 9, 10]

    own_prompt, no_query, no_target = task_rows(Task("A"), pools, condition)
    assert own_prompt.tolist() == [2, 3] and no_query is None and no_target is None

    p, q, t = task_rows(Task("control", "control", "control"), pools, condition)
    reference = pools[("control", "infer")]  # the control prompt used at inference
    assert len(p) == len(q) == len(t) == len(reference) == 3
    parts = [set(p), set(q), set(t), set(reference)]
    assert all(not (x & y) for i, x in enumerate(parts) for y in parts[i + 1:])


def test_pick_draws_without_replacement_when_the_pool_is_big_enough():
    rng = np.random.default_rng(0)
    out = pick(np.arange(10), 3, rng)
    assert len(set(out)) == 3 and set(out) <= set(range(10))


def test_pick_uses_every_cell_before_repeating_any():
    rng = np.random.default_rng(0)
    out = pick(np.array([5, 6, 7]), 7, rng)
    assert len(out) == 7 and set(out) == {5, 6, 7}


def test_task_window_puts_prompt_cells_first_and_pairs_queries_with_targets():
    counts = sparse.csr_matrix(np.repeat(np.arange(30, dtype=np.float32)[:, None], 2, axis=1))  # row i holds i
    rng = np.random.default_rng(0)

    observed, truth = task_window(counts, np.arange(0, 10), np.arange(10, 20), np.arange(20, 30), 8, 2, rng)

    assert observed.shape == truth.shape == (8, 2)
    assert np.array_equal(observed[:2], truth[:2]) and set(observed[:2, 0]) <= set(range(10))
    assert set(observed[2:, 0]) <= set(range(10, 20)) and set(truth[2:, 0]) <= set(range(20, 30))


def test_a_prompt_only_window_has_no_query_cells():
    counts = sparse.csr_matrix(np.repeat(np.arange(6, dtype=np.float32)[:, None], 2, axis=1))

    observed, truth = task_window(counts, np.arange(6), None, None, 4, 4, np.random.default_rng(0))

    assert np.array_equal(observed, truth) and set(observed[:, 0]) <= set(range(6))
