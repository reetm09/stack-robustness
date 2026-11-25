"""Download the model checkpoints and datasets used here into the Hugging Face cache ($HF_HOME/hub).

    python scripts/fetch_hf.py               # everything
    python scripts/fetch_hf.py --only pbmc   # entries whose name contains "pbmc"

Run from the repo root, inside the conda env (HF_HOME comes from .env if it is not set).
Re-running only checks that the files are present.

- STACK and the datasets are also symlinked to the paths the team's commands use
  (data/raw/stack/bc_large_aligned.ckpt, data/raw/arc_example/openproblems_donor1.h5ad, ...).
- TabPFN: the `tabpfn` package keeps its own model directory (TABPFN_MODEL_CACHE_DIR, default
  ~/.cache/tabpfn) and only downloads, behind a Prior Labs browser login, when a file is missing
  there. It also resolves symlinks and picks the loader by suffix, so the hub cache's
  snapshot symlinks (-> suffix-less blobs) cannot be passed directly. The checkpoints are therefore
  hard-linked (no extra disk) into $HF_HOME/tabpfn/, which setup_env.sh sets as
  TABPFN_MODEL_CACHE_DIR. Then, with no download:
      TabPFNRegressor.create_default_for_version(ModelVersion.V3_5)
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env", override=False)

from huggingface_hub import constants, hf_hub_download  # noqa: E402  (after HF_HOME is set)

TABPFN_REPO = "Prior-Labs/tabpfn_3_5"
STACK_REPO = "arcinstitute/Stack-Large-Aligned"
PBMC_REPO = "arcinstitute/Stack-DrugPBMC-Example"  # OpenProblems PBMC donor 1 (CC-BY-4.0)

# (name, repo, repo_type, filename, symlink under the repo root or None)
FILES = [
    ("tabpfn", TABPFN_REPO, "model", "tabpfn-v3.5-20260909.safetensors", None),  # ModelVersion.V3_5
    ("tabpfn-fast", TABPFN_REPO, "model", "tabpfn-v3.5-fast-20260909.safetensors", None),  # V3_5_FAST
    ("stack", STACK_REPO, "model", "bc_large_aligned.ckpt", "data/raw/stack/bc_large_aligned.ckpt"),
    ("stack-vocab", STACK_REPO, "model", "basecount_1000per_15000max.pkl",
     "data/raw/stack_vocab/basecount_1000per_15000max.pkl"),
    ("pbmc-donor1", PBMC_REPO, "dataset", "openproblems_donor1.h5ad",
     "data/raw/arc_example/openproblems_donor1.h5ad"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", default="", help="fetch only entries whose name contains this")
    args = parser.parse_args()
    print(f"HF_HOME={os.environ.get('HF_HOME', '(unset: ~/.cache/huggingface)')}")
    tabpfn_dir = Path(os.environ.get("TABPFN_MODEL_CACHE_DIR", Path(constants.HF_HOME) / "tabpfn"))
    for name, repo, repo_type, filename, link in FILES:
        if args.only not in name:
            continue
        path = Path(hf_hub_download(repo, filename, repo_type=repo_type))
        print(f"{repo}/{filename}\n    cached: {path}")
        if repo == TABPFN_REPO:
            target = tabpfn_dir / filename
            if not target.exists():
                tabpfn_dir.mkdir(parents=True, exist_ok=True)
                os.link(path.resolve(), target)
            print(f"    hard-linked for tabpfn: {target}")
        if link is None:
            continue
        link = REPO / link
        if link.is_symlink() or not link.exists():
            link.parent.mkdir(parents=True, exist_ok=True)
            link.unlink(missing_ok=True)
            link.symlink_to(path.resolve())
            print(f"    linked: {link.relative_to(REPO)}")
        else:
            print(f"    kept existing file: {link.relative_to(REPO)}")


if __name__ == "__main__":
    main()
