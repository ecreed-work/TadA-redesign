#!/bin/bash
# RFdiffusion3 PARTIAL diffusion for the TadA redesign campaign: re-noise each
# relaxed parent in place by partial_t Angstroms while holding the arm's motif
# and the catalytic Zn rigid, with hotspots aimed at the nucleotides flanking
# the target base. One array task per cell (16 cells).
#   MODE=debug  -> 1 batch x 2 = 2 backbones/cell, 20 timesteps (the gate)
#   MODE=run    -> 4 batches x 8 = 32 backbones/cell
# Structural design only -- no base-editing function is implied.
#SBATCH --job-name=tada_rfd_partial
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/tada_rfd_partial_%A_%a.out
#SBATCH -e logs/tada_rfd_partial_%A_%a.err
set -euo pipefail

CONDA_BASE="/research_jude/rgs01_jude/groups/tsaigrp/projects/Genomics/common/miniforge3"
SUB="/research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign"
RUN="${RUN:-$SUB/outputs/20260805_tada_redesign}"
INPUTS="$RUN/rfd_in/rfd_inputs.yaml"
RFD3_CKPT="${RFD3_CKPT:-/research_jude/rgs01_jude/groups/tsaigrp/projects/Genomics/common/claude/foundry_ckpt/rfd3_latest.ckpt}"
MODE="${MODE:-debug}"

cd "$SUB"; mkdir -p logs
unset PYTHONPATH
source "${CONDA_BASE}/bin/activate" cas9-pam-design

[[ -s "$INPUTS" ]]    || { echo "[rfd_partial] ERROR: missing $INPUTS (run prep_rfd_inputs)" >&2; exit 1; }
[[ -s "$RFD3_CKPT" ]] || { echo "[rfd_partial] ERROR: missing ckpt $RFD3_CKPT" >&2; exit 1; }

# One cell per array task. Keys are sorted so the mapping is stable across runs.
CELL=$(python3 -c "
import sys, yaml
print(sorted(yaml.safe_load(open('$INPUTS')))[int(sys.argv[1]) - 1])
" "${SLURM_ARRAY_TASK_ID:?run as an array job, e.g. --array=1-16}")
CELL_YAML="$RUN/rfd_in/cell_${CELL}.yaml"
python3 -c "
import yaml
specs = yaml.safe_load(open('$INPUTS'))
yaml.safe_dump({'$CELL': specs['$CELL']}, open('$CELL_YAML', 'w'), sort_keys=False)
"

# RFD3 names outputs <yaml-stem>_<group>_<batch>_model_n.cif.gz, and cell ids
# embed a float (pt1.0), so filenames cannot be parsed back into a cell.
# Each cell gets its own output directory instead; filter_backbones reads the
# cell from the directory name, never from the filename (measured 2026-08-05).
OUT="$RUN/rfd/$CELL"
mkdir -p "$OUT"

echo "[rfd_partial] cell=$CELL mode=$MODE host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
if [ "$MODE" = "debug" ]; then
  rfd3 design out_dir="$OUT" inputs="$CELL_YAML" ckpt_path="$RFD3_CKPT" \
    prevalidate_inputs=true n_batches=1 diffusion_batch_size=2 \
    inference_sampler.num_timesteps=20 skip_existing=true
else
  rfd3 design out_dir="$OUT" inputs="$CELL_YAML" ckpt_path="$RFD3_CKPT" \
    n_batches=4 diffusion_batch_size=8 skip_existing=true
fi
echo "[rfd_partial] done cell=$CELL ; backbones now: $(ls "$OUT"/*.cif.gz 2>/dev/null | wc -l)"

