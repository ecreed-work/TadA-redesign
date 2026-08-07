#!/bin/bash
# ESMFold2 screen over designs.tsv, one array task per shard. Each task folds its
# shard in ONE process so the model load is amortised (see fold_many.py).
#   SHARDS=44 sbatch --array=1-44 fold_screen.slurm
# Set -t from the measured amortised cost in fold_probe/timing.json, with margin.
#SBATCH --job-name=tada_fold_screen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o logs/tada_fold_screen_%A_%a.out
#SBATCH -e logs/tada_fold_screen_%A_%a.err
set -euo pipefail

CONDA_BASE="/research_jude/rgs01_jude/groups/tsaigrp/projects/Genomics/common/miniforge3"
SUB="/research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign"
SHARDS="${SHARDS:-44}"

cd "$SUB"; mkdir -p logs
unset PYTHONPATH
source "${CONDA_BASE}/bin/activate" esmfold2

echo "[fold_screen.slurm] shard ${SLURM_ARRAY_TASK_ID:?run as an array job}/$SHARDS host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
PYTHONPATH="$SUB" python -m tada_redesign.fold_screen \
  --shard "$SLURM_ARRAY_TASK_ID" --n-shards "$SHARDS"
