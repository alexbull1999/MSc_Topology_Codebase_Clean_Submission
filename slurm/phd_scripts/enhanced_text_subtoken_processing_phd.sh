#!/bin/bash
#SBATCH --job-name=enhanced_text_subtoken_processing
#SBATCH --partition=gpgpuB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=../phd_logs/full_enhanced_text__subtoken_processing_roberta_PHD_%j.out
#SBATCH --error=../phd_logs/full_enhanced_text_subtoken_processing_roberta_PHD_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahb24

echo "Starting text processing test job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Load any required modules (adjust based on your setup)

# Load CUDA first
echo "Loading CUDA..."
. /vol/cuda/12.0.0/setup.sh

# Activate your conda environment - FIXED VERSION
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh


conda activate /vol/bitbucket/ahb24/tda_entailment_new

echo "Activated conda environment: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"
echo "Conda info:"
conda info --envs

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi
echo ""

echo "Running text_processing.py..."

# Change to your project directory
cd $SLURM_SUBMIT_DIR/../..

# Run text processing
python phd_method/src_phd/enhanced_subtoken_text_processing_phd.py

echo ""
echo "Text processing completed!"
echo "Time: $(date)"

# Show what files were created
echo "Files created in data/processed:"
ls -la data/processed/ 2>/dev/null || echo "No data/processed directory found"

echo "Job finished."