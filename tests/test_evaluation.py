import numpy as np
import pandas as pd
import pytest

from stack_robustness.evaluation import EvalContext, median_tables

GENES = ["g0", "g1", "g2", "g3"]
PERTS = ["A+B", "C+D"]


def _de(scores):
    return pd.DataFrame({"score": scores, "pval": [0.01] * 4, "pval_adj": [0.02, 0.03, 0.5, 0.9]}, index=GENES)


@pytest.fixture
def ctx():
    truth = pd.DataFrame([[1.0, 2.0, 3.0, 4.0], [4.0, 1.0, 0.0, 2.0]], index=PERTS, columns=GENES)
    ctrl = pd.Series([1.0, 1.0, 1.0, 1.0], index=GENES)
    mean = pd.Series([2.0, 1.5, 1.0, 2.0], index=GENES)
    return EvalContext(
        truth=truth,
        ctrl=ctrl,
        pert_ref=mean,
        de_vs_ctrl={"A+B": _de([1.0, 2.0, 3.0, 4.0]), "C+D": _de([4.0, 0.0, -1.0, 1.0])},
        de_vs_rest={"A+B": _de([0.5, 2.0, -3.0, 1.0]), "C+D": _de([1.0, 1.0, 2.0, 0.0])},
        baselines={
            "control_baseline": pd.DataFrame([ctrl, ctrl], index=PERTS),
            "mean_baseline": pd.DataFrame([mean, mean], index=PERTS),
        },
        panels={"all": GENES, "two": ["g3", "g0"]},
    )


def test_a_saved_context_loads_back_unchanged(ctx, tmp_path):
    path = tmp_path / "ctx.npz"
    ctx.save(path)

    back = EvalContext.load(path)

    pd.testing.assert_frame_equal(back.truth, ctx.truth)
    pd.testing.assert_series_equal(back.ctrl, ctx.ctrl, check_names=False)
    pd.testing.assert_series_equal(back.pert_ref, ctx.pert_ref, check_names=False)
    for table in ("de_vs_ctrl", "de_vs_rest"):
        for p in PERTS:
            pd.testing.assert_frame_equal(getattr(back, table)[p], getattr(ctx, table)[p])
    pd.testing.assert_frame_equal(back.baselines["mean_baseline"], ctx.baselines["mean_baseline"], check_names=False)
    assert back.panels == ctx.panels


def test_score_rates_a_model_beside_the_baselines_on_every_panel(ctx):
    long = ctx.score({"oracle": ctx.truth.copy()})

    assert set(long["model"]) == {"control_baseline", "mean_baseline", "oracle"}
    assert set(long["panel"]) == {"all", "two"}
    oracle_mse = long[(long.model == "oracle") & (long.metric == "mse")]
    assert oracle_mse["value"].tolist() == [0.0] * 4  # 2 perturbations x 2 panels
    assert oracle_mse["gap_closed"].tolist() == pytest.approx([1.0] * 4, abs=1e-5)


def test_median_tables_give_metric_by_model_medians_per_panel(ctx):
    long = ctx.score({"oracle": ctx.truth.copy()})
    long.loc[(long.model == "oracle") & (long.metric == "mae") & (long.perturbation == "A+B"), "value"] = 3.0

    tables = median_tables(long, "value")

    assert list(tables) == ["all", "two"]
    assert list(tables["all"].columns) == ["control_baseline", "mean_baseline", "oracle"]
    assert tables["all"].index[0] == "pearson_deltactrl"
    assert tables["all"].loc["mae", "oracle"] == pytest.approx(1.5)  # median of 3.0 and 0.0


def test_score_uses_only_each_panels_genes(ctx):
    wrong_outside_panel = ctx.truth.copy()
    wrong_outside_panel[["g1", "g2"]] += 10.0

    long = ctx.score({"model": wrong_outside_panel})

    mse = long[(long.model == "model") & (long.metric == "mse")].set_index(["panel", "perturbation"])["value"]
    assert mse.loc[("two", "A+B")] == 0.0
    assert mse.loc[("all", "A+B")] == pytest.approx(50.0)  # (100 + 100) / 4 genes
