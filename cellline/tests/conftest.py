import os

# Tests run on CPU only; hide any GPU before torch loads (as ttt/tests/conftest.py does).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from scipy import sparse  # noqa: E402
from stack.model_finetune import ICL_FinetunedModel  # noqa: E402

from stack_cellline.tasks import CellLineData, Split  # noqa: E402

# A real STACK fine-tuning model, shrunk so tests run on CPU in seconds.
TINY_CONFIG = dict(n_genes=40, n_hidden=4, token_dim=8, n_cells=16, n_layers=2, n_heads=2, dropout=0.0, n_proj=16)

SOURCE, TARGET, CONTROL = "SRC", "TGT", "non-targeting"
TRAIN_GENES, VAL_GENES, TEST_GENES = ["G0", "G1", "G2"], ["G3", "G4"], ["G5"]


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return ICL_FinetunedModel(**TINY_CONFIG).eval()


@pytest.fixture
def tiny_checkpoint(tiny_model, tmp_path):
    """The tiny model saved in the released checkpoint's layout, which load_stack reads."""
    path = tmp_path / "tiny.ckpt"
    torch.save(
        {
            "hyper_parameters": {"model_config": TINY_CONFIG},
            "state_dict": {f"model.{k}": v for k, v in tiny_model.state_dict().items()},
        },
        path,
    )
    return path


def make_data(seed: int = 0):
    """Synthetic cells of two lines: 12 per (line, gene) for the training and validation genes,
    10 source-line controls and 30 target-line controls (20 training pool, 10 validation pool).
    No test-gene or test-pool cells, as a loader for the real data must guarantee."""
    rng = np.random.default_rng(seed)
    line, target = [], []
    for gene in TRAIN_GENES + VAL_GENES:
        for which in (SOURCE, TARGET):
            line += [which] * 12
            target += [gene] * 12
    line += [SOURCE] * 10 + [TARGET] * 30
    target += [CONTROL] * 40
    line, target = np.array(line), np.array(target)
    counts = sparse.csr_matrix(rng.poisson(3.0, size=(len(line), TINY_CONFIG["n_genes"])).astype(np.float32))
    data = CellLineData(counts=counts, genes=[f"V{i}" for i in range(TINY_CONFIG["n_genes"])], line=line, target=target)
    target_ctrl = np.flatnonzero((line == TARGET) & (target == CONTROL))
    split = Split(
        source_line=SOURCE,
        target_line=TARGET,
        control_label=CONTROL,
        train_genes=list(TRAIN_GENES),
        val_genes=list(VAL_GENES),
        test_genes=list(TEST_GENES),
        train_controls=target_ctrl[:20],
        val_controls=target_ctrl[20:],
    )
    return data, split


@pytest.fixture
def tiny_data():
    return make_data()
