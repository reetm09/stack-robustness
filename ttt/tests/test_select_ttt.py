import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "select_ttt.py"


@pytest.fixture
def select():
    spec = importlib.util.spec_from_file_location("select_ttt", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run(root: Path, run: str, medians: dict) -> None:
    """A run folder whose scores.csv gives each model the median wMSE in `medians`."""
    rows = [dict(metric=metric, panel=panel, model=model, perturbation=f"P{i}", value=v * f)
            for model, v in medians.items() for i, f in enumerate([0.5, 1.0, 3.0])
            for metric, panel in [("wmse", "stack_genes"), ("wmse", "hvg"), ("mse", "stack_genes")]]
    (root / run).mkdir(parents=True)
    pd.DataFrame(rows).to_csv(root / run / "scores.csv", index=False)


def run_main(select, monkeypatch, capsys, *args):
    monkeypatch.setattr("sys.argv", ["select_ttt.py", *map(str, args)])
    select.main()
    return capsys.readouterr().out.strip()


def test_stage_one_prefers_the_smaller_scope_within_one_percent_of_the_best(select, tmp_path, monkeypatch, capsys):
    # Version 1's real medians: cross-cell at 3e-4 was best, the final block at 3e-4 0.91% behind.
    for run, m in {"shared_final_block_lr1e-4": 0.01959, "shared_final_block_lr3e-4": 0.01629,
                   "shared_all_cross_cell_lr1e-4": 0.01696, "shared_all_cross_cell_lr3e-4": 0.01615}.items():
        write_run(tmp_path, run, {"shared_T1_diff": m})

    assert run_main(select, monkeypatch, capsys, "shared", tmp_path) == "final_block 3e-4"
    assert json.loads((tmp_path / "stage1_selection.json").read_text())["scope"] == "final_block"


def test_stage_one_takes_the_larger_scope_when_it_is_clearly_better(select, tmp_path, monkeypatch, capsys):
    for run, m in {"shared_final_block_lr1e-4": 0.020, "shared_final_block_lr3e-4": 0.018,
                   "shared_all_cross_cell_lr1e-4": 0.019, "shared_all_cross_cell_lr3e-4": 0.015}.items():
        write_run(tmp_path, run, {"shared_T1_diff": m})

    assert run_main(select, monkeypatch, capsys, "shared", tmp_path) == "all_cross_cell 3e-4"


def test_each_ttt_arm_takes_its_best_rate_then_its_best_setting(select, tmp_path, monkeypatch, capsys):
    write_run(tmp_path, "ttt_lr3e-5_steps50", {"ttt_T1_diff": 0.020, "shared_ttt_T1_diff": 0.015})
    write_run(tmp_path, "ttt_lr1e-4_steps50", {"ttt_T1_diff": 0.018, "shared_ttt_T1_diff": 0.016})
    write_run(tmp_path, "ttt_lr3e-4_steps50", {"ttt_T1_diff": 0.017, "shared_ttt_T1_diff": 0.017})
    assert run_main(select, monkeypatch, capsys, "lr", tmp_path) == "3e-4 3e-5"  # the grid's new top rate can win

    write_run(tmp_path, "ttt_lr3e-4_steps250", {"ttt_T1_diff": 0.0169})        # best, but 50 steps is within 1%
    write_run(tmp_path, "ttt_lr3e-5_steps250", {"shared_ttt_T1_diff": 0.012})  # clearly best
    run_main(select, monkeypatch, capsys, "final", tmp_path, "3e-4", "3e-5")
    chosen = json.loads((tmp_path / "stage2_selection.json").read_text())

    assert (chosen["ttt"]["lr"], chosen["ttt"]["steps"]) == ("3e-4", 50)
    assert (chosen["shared_ttt"]["lr"], chosen["shared_ttt"]["steps"]) == ("3e-5", 250)
    assert chosen["ttt_random"] == {"lr": "3e-4", "steps": 50}
    assert chosen["shared_ttt_random"] == {"lr": "3e-5", "steps": 250}
