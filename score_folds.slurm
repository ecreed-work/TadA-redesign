#!/bin/bash
# Score every folded design in fold_screen/ -> fold_screen.tsv.
#   sbatch score_folds.slurm
#
# CPU-only: NO --gres=gpu. Scoring is Kabsch/numpy over ~10.5k structures and
# never touches the GPU; reserving one would idle an H100 for hours. This
# cluster exposes only the `gpu` and `hpcf_test` SLURM partitions, so a
# CPU-only job still lands on `gpu` -- just without a device.
#
# Measured 2026-08-09 on the login node before it was moved here: ~26 designs
# per minute single-threaded, i.e. ~6.5 h for 10,542. Each design pays TWO
# iterative anchor refinements (motif_rmsd and cleft_clearance each call
# _anchor_arrays), which is where the time goes. -t is set with margin over that.
#SBATCH --job-name=tada_score_folds
#SBATCH --partition=gpu
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o logs/tada_score_folds_%j.out
#SBATCH -e logs/tada_score_folds_%j.err
set -euo pipefail

CONDA_BASE="/research_jude/rgs01_jude/groups/tsaigrp/projects/Genomics/common/miniforge3"
SUB="/research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign"

cd "$SUB"; mkdir -p logs
unset PYTHONPATH
source "${CONDA_BASE}/bin/activate" ligandmpnn_env

echo "[score_folds.slurm] host=$(hostname) start=$(date -Is)"
PYTHONPATH="$SUB" python -m tada_redesign.score_folds
echo "[score_folds.slurm] done=$(date -Is)"
