import os

# Unit tests run on CPU. arc-stack's inference touches CUDA whenever a GPU is visible,
# and this machine's GPU is shared through the dibs queue, so hide it before torch loads.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest  # noqa: E402
import torch  # noqa: E402

from stack.model_finetune import ICL_FinetunedModel

# A real STACK fine-tuning model, shrunk so tests run on CPU in milliseconds.
TINY_CONFIG = dict(
    n_genes=40, n_hidden=4, token_dim=8, n_cells=16, n_layers=2, n_heads=2, dropout=0.0, n_proj=16
)


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return ICL_FinetunedModel(**TINY_CONFIG).eval()


@pytest.fixture
def tiny_counts():
    """Two windows of 16 cells x 40 genes of Poisson counts: observed and ground truth."""
    g = torch.Generator().manual_seed(1)
    observed = torch.poisson(torch.full((1, 16, 40), 3.0), generator=g)
    truth = torch.poisson(torch.full((1, 16, 40), 5.0), generator=g)
    truth[:, :6] = observed[:, :6]  # prompt positions: observed and truth are the same cells
    return observed, truth
