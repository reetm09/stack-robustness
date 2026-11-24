import numpy as np
import pytest

from stack_ttt.diagnostics import prompt_use


def pair_profiles(arm, shift_ab, shift_ba, obs_a, obs_b):
    """One pair's saved profiles, as norman_ttt.py writes them, for two prompt orders whose
    prompt-induced shifts are given: A prompts B's cells (ab), and B prompts A's (ba)."""
    diff_ab, diff_ba = obs_b + shift_ab, obs_a + shift_ba
    return {"query_one": obs_b, "query": (obs_a + obs_b) / 2,
            f"{arm}_T1_diff_one": diff_ab, f"{arm}_T1_diff": (diff_ab + diff_ba) / 2}


def test_a_shift_along_the_prompts_effect_correlates_fully_and_has_its_size():
    rng = np.random.default_rng(0)
    ctrl = rng.normal(size=300)
    eff_a, eff_b = rng.normal(size=300), rng.normal(size=300)
    obs_a, obs_b = ctrl + eff_a, ctrl + eff_b

    out = prompt_use(pair_profiles("shared", 0.5 * eff_a, 0.5 * eff_b, obs_a, obs_b), ctrl, "shared")

    assert [o["prompt_order"] for o in out] == ["A->B", "B->A"]
    assert all(o["r"] == pytest.approx(1.0) and o["relative_size"] == pytest.approx(0.5) for o in out)


def test_a_shift_against_the_prompts_effect_correlates_negatively():
    rng = np.random.default_rng(1)
    ctrl = rng.normal(size=300)
    eff_a, eff_b = rng.normal(size=300), rng.normal(size=300)

    out = prompt_use(pair_profiles("icl", -0.1 * eff_a, -0.1 * eff_b, ctrl + eff_a, ctrl + eff_b), ctrl, "icl")

    assert all(o["r"] == pytest.approx(-1.0) and o["relative_size"] == pytest.approx(0.1) for o in out)


def test_the_statistic_uses_each_prompts_own_top_genes():
    # A shift that matches A's effect only off A's top genes earns no credit.
    ctrl = np.zeros(400)
    eff_a = np.r_[np.linspace(5, 3, 200), np.zeros(200)]  # A's top 200 genes are the first 200
    shift = np.r_[np.zeros(200), np.ones(200)]
    eff_b = np.r_[np.zeros(200), np.linspace(5, 3, 200)]

    out = prompt_use(pair_profiles("ttt", shift, np.zeros(400), ctrl + eff_a, ctrl + eff_b), ctrl, "ttt", top=200)

    assert out[0]["relative_size"] == 0.0
