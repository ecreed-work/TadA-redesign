#!/bin/bash
# LigandMPNN sequence design for the TadA redesign campaign. One array task per
# (arm, temperature) plus the zero-bias control, using LigandMPNN's *_multi JSON
# flags so a single process designs every backbone of that arm.
#   --array=1-10   (2 arms x (4 temperatures + 1 control))
# The Zn and ssDNA context ride inside each backbone PDB; LigandMPNN reads
# non-protein atoms as ligand context.
# Structural design only -- no base-editing function is implied.
#SBATCH --job-name=tada_lmpnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o logs/tada_lmpnn_%A_%a.out
#SBATCH -e logs/tada_lmpnn_%A_%a.err
set -euo pipefail

CONDA_BASE="/research_jude/rgs01_jude/groups/tsaigrp/projects/Genomics/common/miniforge3"
R="/research/rgs01/home/clusterHome/ecreed/claude-proteindesign"
SUB="$R/tada-redesign"
RUN="${RUN:-$SUB/outputs/20260805_tada_redesign}"
CKPT="$R/design/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt"
NBATCH="${NBATCH:-5}"          # sequences per backbone at this temperature
SEED="${SEED:-111}"

cd "$SUB"; mkdir -p logs
unset PYTHONPATH
source "${CONDA_BASE}/bin/activate" ligandmpnn-sc
[[ -s "$CKPT" ]] || { echo "[lmpnn] ERROR: missing checkpoint $CKPT" >&2; exit 1; }

# Task -> (arm, temperature, biased?). Arms x 4 temps = 8, then 2 controls.
ARMS=(FULL MIN); TEMPS=(0.1 0.15 0.2 0.3); CONTROL_TEMP=0.15
IDX=$(( ${SLURM_ARRAY_TASK_ID:?run as an array job, e.g. --array=1-10} - 1 ))
if [ "$IDX" -lt 8 ]; then
  ARM="${ARMS[$(( IDX / 4 ))]}"; T="${TEMPS[$(( IDX % 4 ))]}"; BIASED=1
  TAG="T${T}"
else
  ARM="${ARMS[$(( IDX - 8 ))]}"; T="$CONTROL_TEMP"; BIASED=0
  TAG="control"
fi

IN="$RUN/mpnn_in/$ARM"
OUT="$RUN/lmpnn/$ARM/$TAG"
[[ -s "$IN/pdb_paths.json" ]] || { echo "[lmpnn] ERROR: missing $IN/pdb_paths.json (run prep_mpnn_inputs)" >&2; exit 1; }

BIAS_ARGS=()
if [ "$BIASED" = "1" ]; then
  BIAS_ARGS=(--bias_AA_per_residue_multi "$IN/bias.json")
fi

echo "[lmpnn] arm=$ARM tag=$TAG T=$T biased=$BIASED nbatch=$NBATCH host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
python3 "$SUB/tada_redesign/_run_ligandmpnn.py" \
  --model_type ligand_mpnn \
  --checkpoint_ligand_mpnn "$CKPT" \
  --pdb_path_multi "$IN/pdb_paths.json" \
  --fixed_residues_multi "$IN/fixed_residues.json" \
  "${BIAS_ARGS[@]}" \
  --out_folder "$OUT" \
  --temperature "$T" \
  --batch_size 1 \
  --number_of_batches "$NBATCH" \
  --seed "$SEED" \
  --save_stats 1 \
  --verbose 0
echo "[lmpnn] done -> $OUT/seqs ; fastas: $(ls "$OUT"/seqs/*.fa 2>/dev/null | wc -l)"
