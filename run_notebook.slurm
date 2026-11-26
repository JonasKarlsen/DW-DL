#!/bin/bash
#SBATCH --job-name=har_notebook
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00

set -e

NOTEBOOK_PATH="$1"

if [ -z "$NOTEBOOK_PATH" ]; then
    echo "Error: No notebook provided."
    echo "Usage: sbatch run_notebook.slurm path/to/notebook.ipynb"
    exit 1
fi

if [ ! -f "$NOTEBOOK_PATH" ]; then
    echo "Error: File '$NOTEBOOK_PATH' not found."
    exit 1
fi

mkdir -p logs executed_notebooks

BASENAME=$(basename "$NOTEBOOK_PATH" .ipynb)
OUT_NOTEBOOK="executed_notebooks/${BASENAME}_job${SLURM_JOB_ID}.ipynb"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Executing: $NOTEBOOK_PATH -> $OUT_NOTEBOOK"
nvidia-smi

# Activate your Miniconda environment on /cluster/datastore
source /cluster/datastore/jokar2607/miniconda/etc/profile.d/conda.sh
conda activate base

# Execute the notebook
jupyter nbconvert \
    --to notebook \
    --execute "$NOTEBOOK_PATH" \
    --output "$(realpath "$OUT_NOTEBOOK")" \
    --ExecutePreprocessor.timeout=-1

echo "Finished! Executed notebook saved to: $OUT_NOTEBOOK"
