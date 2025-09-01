#!/bin/bash
#SBATCH --job-name=enhanced_subtoken_compute_phd
#SBATCH --partition=gpgpuC
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --output=../phd_logs/enhanced_subtoken_compute_phd_hdim0_%j.out
#SBATCH --error=../phd_logs/enhanced_subtoken_compute_phd_hdim0_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahb24

echo "Starting TDA integration job..."
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

# Test required packages
echo "Testing required packages..."
python -c "
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from pathlib import Path
import time
import tqdm
from gph.python import ripser_parallel
from loguru import logger"


echo ""
echo "Checking for required input files..."

# Change to your project directory
cd $SLURM_SUBMIT_DIR/../..

# Check if required files exist
missing_files=()

# Check for TDA-ready data from cone validation
if [ ! -f "phd_method/phd_data/processed/snli_10k_subset_balanced_roberta_subtokenized.pt" ]; then
    missing_files+=("snli_10k_subset_balanced_phd_roberta.pt")
fi

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "ERROR: Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    echo ""
    exit 1
fi

echo "All required files found!"

echo ""
echo "Starting PHD computation..."

# Run PHD computation
python phd_method/src_phd/subtoken_phd_computation.py

# Capture exit code
EXIT_CODE=$?

# Show analysis results if successful
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== PHD Comptutation SUCCESSFUL ==="
fi

echo ""
echo "Job finished."