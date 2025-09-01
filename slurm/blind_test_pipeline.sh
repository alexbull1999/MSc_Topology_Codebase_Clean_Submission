#!/bin/bash
#SBATCH --job-name=blind_test_pipeline
#SBATCH --partition=gpgpuB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm_blind_test_%j.out
#SBATCH --error=logs/slurm_blind_test_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahb24

echo "Starting order embeddings training job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Load CUDA first
echo "Loading CUDA..."
. /vol/cuda/12.0.0/setup.sh

# Activate your conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /vol/bitbucket/ahb24/tda_entailment_new

echo "Activated conda environment: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi
echo ""


echo ""
echo "Checking for required input data..."

# Change to your project directory
cd $SLURM_SUBMIT_DIR/..

# Run order embeddings training
python src/blind_test_pipeline.py

# Capture exit code
EXIT_CODE=$?

echo ""
echo "Pipline completed with exit code: $EXIT_CODE"
echo "Time: $(date)"

# Show final training results if successful
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== Pipeline SUCCESSFUL ==="
else
    echo ""
    echo "=== PIPELINE FAILED ==="
fi

echo ""
echo "Job finished."