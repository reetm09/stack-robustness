"""The X-Atlas/Orion loader and command line, on synthetic files with the release's schema."""

import json
import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from stack_cellline.artifact import load_finetuned
from stack_cellline.run import main as run_main
from stack_cellline.split import main as split_main
from stack_cellline.xatlas import Subset, load_xatlas, plan_cells, vocabulary_columns
from stack_cellline.split import read_cells

from conftest import TINY_CONFIG

LINES = ("HEK293T", "HCT116")
CTRL = "Non-Targeting"
GENES = [f"G{i}" for i in range(10)]
N_VOCAB = TINY_CONFIG["n_genes"]  # 40
VOCAB = [f"V{i:02d}" for i in range(N_VOCAB)]


def gene_metadata() -> pd.DataFrame:
    """Tokens 0..39 are the vocabulary in lower case; 40..43 are outside it; 44 repeats V05's
    symbol; 45 has no symbol, so its name is its Ensembl ID."""
    names = [v.lower() for v in VOCAB] + ["OUT1", "OUT2", "OUT3", "OUT4", "V05", "ENSG00000000045"]
    return pd.DataFrame({"ensembl_id": [f"ENSG{i:011d}" for i in range(len(names))], "gene_name": names,
                         "gene_token_id": np.arange(len(names), dtype=np.int64)})


@pytest.fixture
def release(tmp_path):
    """Batch files in the release's layout, the per-line metadata tables, the split, the
    vocabulary file, and the true counts on the vocabulary for every cell."""
    rng = np.random.default_rng(0)
    columns = vocabulary_columns(gene_metadata(), VOCAB)
    data, obs_dir = tmp_path / "data", tmp_path / "obs"
    data.mkdir(), obs_dir.mkdir()
    truth, obs_frames = {}, {line: [] for line in LINES}
    for line in LINES:
        for b in (1, 2):
            sample = f"{line}_Batch{b}"
            cells = [(g, 8) for g in GENES] + [(CTRL, 12)]
            rows = []
            for gene, n in cells:
                for i in range(n):
                    barcode = f"AC{b}{i:03d}{gene}-{sample}"
                    tokens = np.sort(rng.choice(len(columns), size=6, replace=False))
                    values = rng.integers(1, 6, size=6).astype(float)
                    dense = np.zeros(N_VOCAB, np.float32)
                    np.add.at(dense, columns[tokens][columns[tokens] >= 0], values[columns[tokens] >= 0])
                    truth[barcode] = dense
                    rows.append({"gene_token_id": tokens.tolist(), "gene_expression": values.tolist(),
                                 "cell_barcode": barcode, "sample": sample, "num_features": 2,
                                 "guide_target": f"{gene}_P1P2-1|{gene}_P1P2-2", "gene_target": gene,
                                 "n_genes_by_counts": 6, "total_counts": float(values.sum() + 100),
                                 "total_counts_mt": 1.0, "pct_counts_mt": 1.0, "pass_guide_filter": 1})
            table = pa.Table.from_pylist(rows, schema=pa.schema([
                ("gene_token_id", pa.list_(pa.int64())), ("gene_expression", pa.list_(pa.float64())),
                ("cell_barcode", pa.string()), ("sample", pa.string()), ("num_features", pa.int64()),
                ("guide_target", pa.string()), ("gene_target", pa.string()), ("n_genes_by_counts", pa.int64()),
                ("total_counts", pa.float64()), ("total_counts_mt", pa.float64()),
                ("pct_counts_mt", pa.float64()), ("pass_guide_filter", pa.int64())]))
            pq.write_table(table, data / f"{sample}.parquet")
            obs_frames[line].append(table.drop_columns(["gene_token_id", "gene_expression"]).to_pandas())
    obs = [f"{line}={obs_dir / f'{line}_obs.parquet'}" for line in LINES]
    for line in LINES:
        pd.concat(obs_frames[line]).to_parquet(obs_dir / f"{line}_obs.parquet", index=False)
    split_main(["--obs", *obs, "--cell-id-col", "cell_barcode", "--gene-col", "gene_target", "--batch-col", "sample",
                "--control-label", CTRL, "--lines", *LINES, "--min-cells", "10", "--out", str(tmp_path / "split")])
    (tmp_path / "vocab.pkl").write_bytes(pickle.dumps([np.str_(v) for v in VOCAB], protocol=4))
    gene_metadata().to_parquet(tmp_path / "gene_metadata.parquet", index=False)
    return {"root": tmp_path, "data": str(data), "obs": obs, "split": tmp_path / "split", "truth": truth}


def _split_tables(release):
    genes = pd.read_csv(release["split"] / "genes.csv", index_col="gene", keep_default_na=False)
    controls = pd.read_csv(release["split"] / "controls.csv.gz", dtype=str)
    obs = read_cells(release["obs"], "gene_target", "sample", cell_id_col="cell_barcode")
    return obs, genes, controls


def test_vocabulary_columns_match_symbols_case_insensitively():
    columns = vocabulary_columns(gene_metadata(), VOCAB)
    assert list(columns[:N_VOCAB]) == list(range(N_VOCAB))
    assert list(columns[40:44]) == [-1] * 4 and columns[45] == -1
    assert columns[44] == columns[5] == 5  # the same symbol shares a column
    assert vocabulary_columns(gene_metadata(), VOCAB, aliases={"OUT1": "V38"})[40] == 38


def test_plan_takes_train_and_val_genes_but_never_test(release):
    obs, genes, controls = _split_tables(release)
    plan, report = plan_cells(obs, genes, controls, *LINES, CTRL)

    train, val, test = (set(genes.index[genes["split"] == s]) for s in ("train", "val", "test"))
    assert (len(train), len(val), len(test)) == (6, 2, 2)
    assert set(plan["gene"]) == train | val | {CTRL} and not set(plan["gene"]) & test
    pooled = plan[plan["pool"] != ""]
    assert set(pooled["line"]) == {"HCT116"} and set(pooled["pool"]) == {"train", "val"}
    test_pool = set(controls.loc[controls["pool"] == "test", "cell_id"])
    assert not test_pool & set(plan["cell_id"])
    assert report["cells"] == {"perturbed": {"HEK293T": 128, "HCT116": 128}, "controls": {"train": 16, "val": 4}}


def test_subset_caps_are_seeded_and_independent(release):
    obs, genes, controls = _split_tables(release)
    small = Subset(n_train_genes=3, n_val_genes=1, max_cells_per_gene=5, max_controls=6, batches_per_line=1)

    plan, report = plan_cells(obs, genes, controls, *LINES, CTRL, small)
    again, _ = plan_cells(obs.sample(frac=1.0, random_state=3), genes, controls, *LINES, CTRL, small)
    _, wide = plan_cells(obs, genes, controls, *LINES, CTRL, Subset(n_train_genes=3, n_val_genes=1))

    assert report["genes"] == {"train": 3, "val": 1} and report["batches"] == {"HEK293T": 1, "HCT116": 1}
    assert report["cells"]["perturbed"] == {"HEK293T": 20, "HCT116": 20}
    assert report["cells"]["controls"] == {"train": 6, "val": 2}
    pd.testing.assert_frame_equal(plan, again)
    assert wide["train_genes"] == report["train_genes"]  # reading fewer batches picks the same genes


def _load(release, **kwargs):
    return load_xatlas(release["obs"], release["split"], release["data"], gene_metadata(), VOCAB, *LINES,
                       log=lambda _: None, **kwargs)


def test_each_loaded_row_is_its_cell_on_the_vocabulary(release):
    loaded = _load(release, subset=Subset(n_train_genes=2, n_val_genes=1, max_controls=5))
    data, split, plan = loaded["data"], loaded["split"], loaded["plan"]

    dense = data.counts.toarray()
    expected = np.stack([release["truth"][c] for c in plan["cell_id"]])
    np.testing.assert_array_equal(dense, expected)  # duplicated symbols summed, outside genes dropped
    assert data.counts.dtype == np.float32 and dense.sum() > 0
    np.testing.assert_array_equal(data.line, plan["line"].to_numpy(dtype=str))
    np.testing.assert_array_equal(data.target, plan["gene"].to_numpy(dtype=str))
    np.testing.assert_array_equal(loaded["library"], dense.sum(axis=1) + 100 + _outside(release, plan))
    assert (split.train_genes, split.val_genes) == (loaded["report"]["train_genes"], loaded["report"]["val_genes"])
    assert set(data.target[split.train_controls]) == {CTRL} and set(data.line[split.val_controls]) == {"HCT116"}


def _outside(release, plan):
    """Counts of each planned cell's tokens outside the vocabulary (part of total_counts)."""
    out = []
    for cell in plan["cell_id"]:
        batch = cell.rsplit("-", 1)[1]
        table = pq.read_table(f"{release['data']}/{batch}.parquet").to_pandas().set_index("cell_barcode")
        row = table.loc[cell]
        inside = release["truth"][cell].sum()
        out.append(float(np.sum(row["gene_expression"])) - inside)
    return np.array(out, dtype=np.float32)


def test_refuses_a_batch_file_missing_planned_cells(release):
    controls = pd.read_csv(release["split"] / "controls.csv.gz", dtype=str)
    planned = controls.loc[(controls["pool"] == "train") & (controls["batch"] == "HCT116_Batch1"), "cell_id"].iloc[0]
    path = f"{release['data']}/HCT116_Batch1.parquet"
    table = pq.read_table(path)
    pq.write_table(table.filter(pc.not_equal(table["cell_barcode"], planned)), path)
    with pytest.raises(ValueError, match="planned cells"):
        _load(release)


def test_plan_only_reads_no_expression(release, monkeypatch):
    import stack_cellline.xatlas as xatlas

    monkeypatch.setattr(xatlas, "load_counts", lambda *a, **k: pytest.fail("read a batch file"))
    loaded = _load(release, plan_only=True)
    assert set(loaded) == {"plan", "report"} and loaded["report"]["batch_files_to_read"] == 4


def test_command_line_plans_then_fine_tunes_a_short_run(release, tiny_checkpoint):
    root = release["root"]
    common = ["--checkpoint", str(tiny_checkpoint), "--vocab", str(root / "vocab.pkl"),
              "--gene-metadata", str(root / "gene_metadata.parquet"), "--obs", *release["obs"],
              "--split", str(release["split"]), "--data", release["data"],
              "--train-genes", "3", "--val-genes", "1", "--max-controls", "8"]

    plan = run_main([*common, "--out", str(root / "plan"), "--plan-only"])
    assert plan["genes"] == {"train": 3, "val": 1} and not (root / "plan" / "adapter.pt").exists()

    out = run_main([*common, "--out", str(root / "run"), "--steps", "2", "--n-cells", "16", "--n-kept", "4",
                    "--val-windows", "1", "--device", "cpu"])
    assert np.isfinite(out["val_loss_before"]) and np.isfinite(out["val_loss_after"])
    for name in ("adapter.pt", "history.jsonl", "val.json", "run.json", "plan.json"):
        assert (root / "run" / name).exists(), name
    settings = json.loads((root / "run" / "run.json").read_text())["settings"]
    assert settings["train_genes"] == plan["train_genes"] and settings["target_line"] == "HCT116"
    assert not load_finetuned(tiny_checkpoint, root / "run" / "adapter.pt").training
