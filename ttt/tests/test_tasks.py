import numpy as np
import pytest
from scipy import sparse

from stack_ttt.tasks import Task, cell_pools, control_pools, neighbours, norman_tasks, task_rows
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


def test_cell_pools_give_doubles_none_so_loading_more_doubles_moves_no_cell():
    # The validation run also loads its held-out doubles' cells, to measure a validation
    # loss. They must get no pool, and every single's adapt/infer split and the control
    # pools must keep the very same cells. load_norman keeps file order, so new rows land
    # between the old ones without reordering them.
    rng = np.random.default_rng(0)
    base = rng.permutation(np.array(["A"] * 9 + ["C"] * 7 + ["A+C"] * 5 + ["control"] * 12))
    where = np.sort(rng.choice(len(base) + 1, size=10))
    more = np.insert(base, where, "A+B")
    cell = np.insert(np.arange(len(base)), where, -1)  # each row's cell in `base`; -1 for new ones

    before, after = cell_pools(base, seed=0), cell_pools(more, seed=0)

    assert sorted(after) == sorted(before)
    assert all(cell[after[key]].tolist() == before[key].tolist() for key in before)
    assert not {key[0] for key in after} & {"A+C", "A+B"}


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

    observed, truth, revealed = task_window(counts, np.arange(0, 10), np.arange(10, 20), np.arange(20, 30), 8, 2, rng)

    assert revealed is None  # by default a revealed position shows its target, as in STACK's post-training
    assert observed.shape == truth.shape == (8, 2)
    assert np.array_equal(observed[:2], truth[:2]) and set(observed[:2, 0]) <= set(range(10))
    assert set(observed[2:, 0]) <= set(range(10, 20)) and set(truth[2:, 0]) <= set(range(20, 30))


def test_a_prompt_only_window_has_no_query_cells():
    counts = sparse.csr_matrix(np.repeat(np.arange(6, dtype=np.float32)[:, None], 2, axis=1))

    for reveal in ("target", "prompt"):
        observed, truth, revealed = task_window(counts, np.arange(6), None, None, 4, 4, np.random.default_rng(0),
                                                reveal=reveal)

        assert np.array_equal(observed, truth) and set(observed[:, 0]) <= set(range(6))
        assert revealed is None  # nothing follows the prompt, so nothing is revealed


def test_revealing_prompt_cells_draws_them_with_the_prompt_as_one_sample():
    # STACK's inference fills the slots between the prompt and the query with more prompt
    # cells, so adaptation windows can reveal those instead of target cells. Prompt and
    # revealed cells are one draw from the prompt pool: no cell repeats while the pool lasts.
    counts = sparse.csr_matrix(np.repeat(np.arange(40, dtype=np.float32)[:, None], 2, axis=1))  # row i holds i
    rng = np.random.default_rng(0)

    observed, truth, revealed = task_window(counts, np.arange(0, 20), np.arange(20, 30), np.arange(30, 40), 8, 2, rng,
                                            reveal="prompt")

    assert revealed.shape == observed.shape == truth.shape == (8, 2)
    assert np.array_equal(revealed[:2], observed[:2]) and np.array_equal(observed[:2], truth[:2])
    drawn = revealed[:, 0].tolist()
    assert len(set(drawn)) == 8 and set(drawn) <= set(range(20))
    assert set(observed[2:, 0]) <= set(range(20, 30)) and set(truth[2:, 0]) <= set(range(30, 40))


def test_an_unknown_reveal_is_refused():
    counts = sparse.csr_matrix(np.ones((6, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        task_window(counts, np.arange(3), np.arange(3, 6), np.arange(3, 6), 4, 2, np.random.default_rng(0), reveal="query")


def test_shared_tasks_use_every_training_double_in_both_orders():
    from stack_ttt.tasks import shared_tasks

    assert shared_tasks(["A+C", "D+B"], null=True) == [
        Task("A", "C", "A+C"), Task("C", "A", "A+C"), Task("D", "B", "D+B"), Task("B", "D", "D+B"),
        Task("control", "control", "control"),
    ]


def test_random_tasks_match_the_neighbour_tasks_except_for_which_doubles_they_use():
    from stack_ttt.tasks import random_tasks

    train = ["A+C", "D+B", "E+F", "G+H", "I+J", "C+D"]
    tasks = random_tasks("A+B", train, seed=0, null=True)
    doubles = sorted({t.target for t in tasks if t.query is not None and t.target != "control"})

    assert len(doubles) == len(neighbours("A+B", train)) == 2
    assert not set(doubles) & {"A+C", "D+B"}  # none shares a gene with A+B
    assert tasks[-3:] == [Task("control", "control", "control"), Task("A"), Task("B")]
    assert len(tasks) == len(norman_tasks("A+B", train, null=True))
    assert random_tasks("A+B", train, seed=0, null=True) == tasks  # reproducible


# Effect embeddings of singles, chosen so that the cosine ranking against e(A+B) = (1, 1, 0) is
# E+F (0.997), A+C (0.54), D+B (0.50), C+D (0.07), G+H (-1), with no ties.
EMBED = {g: np.array(v, dtype=float) for g, v in {
    "A": [1, 0, 0], "B": [0, 1, 0], "C": [0.2, 0, 1], "D": [0, 0, 1],
    "E": [1, 0.1, 0], "F": [0, 1, 0.1], "G": [-1, 0, 0], "H": [0, -1, 0]}.items()}
KNN_TRAIN = ["A+C", "D+B", "E+F", "G+H", "C+D"]


def test_knn_neighbours_rank_training_doubles_by_the_cosine_of_summed_single_embeddings():
    from stack_ttt.tasks import knn_neighbours

    assert knn_neighbours("A+B", KNN_TRAIN, EMBED, k=5) == ["E+F", "A+C", "D+B", "C+D", "G+H"]


def test_knn_tasks_take_as_many_doubles_as_share_a_gene_and_keep_the_other_tasks():
    from stack_ttt.tasks import knn_tasks

    tasks = knn_tasks("A+B", KNN_TRAIN, EMBED, null=True)
    doubles = sorted({t.target for t in tasks if t.query is not None and t.target != "control"})

    assert len(neighbours("A+B", KNN_TRAIN)) == 2
    assert doubles == ["A+C", "E+F"]  # the two nearest, one of them not sharing a gene
    assert tasks[-3:] == [Task("control", "control", "control"), Task("A"), Task("B")]
    assert len(tasks) == len(norman_tasks("A+B", KNN_TRAIN, null=True))


def test_knn_tasks_take_no_doubles_when_none_shares_a_gene():
    from stack_ttt.tasks import knn_tasks

    tasks = knn_tasks("A+B", ["E+F", "G+H"], EMBED, null=True)

    assert tasks == [Task("control", "control", "control"), Task("A"), Task("B")]


def test_effect_embeddings_use_only_infer_cells_and_subtract_control(tiny_model):
    from stack_ttt.retrieval import effect_embeddings, mean_embeddings

    rng = np.random.default_rng(0)
    counts = sparse.csr_matrix(rng.poisson(3.0, size=(60, 40)).astype(np.float32))
    pools = {("X", "infer"): np.arange(0, 20), ("X", "adapt"): np.arange(20, 40),
             ("control", "infer"): np.arange(40, 60)}
    cells = type("Cells", (), {"counts": counts})()

    effect = effect_embeddings(tiny_model, cells, pools, ["X"], device="cpu", window=8)
    means = mean_embeddings(tiny_model, counts, {"X": np.arange(0, 20), "control": np.arange(40, 60)},
                            device="cpu", window=8)
    np.testing.assert_allclose(effect["X"], means["X"] - means["control"], rtol=1e-6)

    changed = counts.tolil()
    changed[20:40] = changed[20:40] * 5  # the adapt half must not matter
    again = effect_embeddings(tiny_model, type("Cells", (), {"counts": changed.tocsr()})(), pools, ["X"],
                              device="cpu", window=8)
    np.testing.assert_allclose(again["X"], effect["X"], rtol=1e-6)
