#!/usr/bin/env bash
# Version 2 after its first job (smoke test and final_block:3e-4 with icl): the rest of stage 1,
# stage 2 and the test run, in one queue slot. Each step resumes from saved pairs.
#
#   SMOKE_DIR=/some/scratch/dir dibs submit --cpus 2 --mem 10G --vram 6G --time 6h --retry -- bash scripts/v2_chain.sh
set -euo pipefail
cd "$(dirname "$0")"  # ttt/scripts
SETTINGS="final_block:1e-4 all_cross_cell:1e-4 all_cross_cell:3e-4" ICL_WITH="" bash val_stage1.sh
bash val_stage2.sh
bash test_run.sh
