#!/usr/bin/env bash
# Create or update the `stack-robustness` conda env, keep Hugging Face downloads inside the repo
# (.cache/huggingface/, git-ignored), and fetch the model checkpoints and datasets (scripts/fetch_hf.py).
#
#   bash scripts/setup_env.sh            # env + models
#   bash scripts/setup_env.sh --no-models
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME=stack-robustness
HF_HOME_DIR="$REPO/.cache/huggingface"

cd "$REPO"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda env update -n "$ENV_NAME" -f environment.yml --prune
else
    conda env create -f environment.yml
fi

# Set on every `conda activate stack-robustness`, as absolute paths, so they hold from any directory.
# TABPFN_MODEL_CACHE_DIR holds hard links to the hub-cached TabPFN files (see fetch_hf.py).
VARS=(HF_HOME="$HF_HOME_DIR" TABPFN_MODEL_CACHE_DIR="$HF_HOME_DIR/tabpfn")
conda env config vars set -n "$ENV_NAME" "${VARS[@]}" >/dev/null
# The same for tools that read .env (VS Code's Python extension, python-dotenv, tabpfn's settings).
printf '%s\n' "${VARS[@]}" > .env
mkdir -p "$HF_HOME_DIR/tabpfn"
printf '  %s\n' "conda env vars and .env:" "${VARS[@]}"

if [[ "${1:-}" != "--no-models" ]]; then
    conda run --no-capture-output -n "$ENV_NAME" python scripts/fetch_hf.py
fi
