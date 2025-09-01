#!/bin/bash
#SBATCH --job-name=cone_validation_tda
#SBATCH --partition=gpgpuB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm_cone_validation_tda_asymmetry_roberta_%j.out
#SBATCH --error=logs/slurm_cone_validation_tda_asymmetry_roberta_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahb24

echo "Starting hyperbolic cone validation and TDA data preparation job..."
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
import geoopt
import numpy as np
import matplotlib.pyplot as plt # Import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
print(f'PyTorch version: {torch.__version__}')
print(f'Geoopt version: {geoopt.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'Matplotlib version: {plt.__version__}') # Use plt.__version__
print(f'Seaborn version: {sns.__version__}')
print(f'Pandas version: {pd.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print('All packages loaded successfully!')
"

echo ""
echo "Checking for required input files and scripts..."

# Change to your project directory
cd $SLURM_SUBMIT_DIR/..

# Check if required files exist
missing_files=()

if [ ! -f "data/processed/snli_10k_subset_balanced.pt" ]; then
    missing_files+=("data/processed/snli_10k_subset_balanced.pt (from text_processing.py)")
fi

# The new script depends on hyperbolic_projection.py and entailment_cones.py
if [ ! -f "src/hyperbolic_projection.py" ]; then
    missing_files+=("src/hyperbolic_projection.py")
fi

if [ ! -f "src/entailment_cones.py" ]; then
    missing_files+=("src/entailment_cones.py")
fi

if [ ! -f "src/order_embeddings.py" ]; then
    missing_files+=("src/order_embeddings.py")
fi

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "ERROR: Missing required files/scripts:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Please ensure all necessary dependencies (e.g., text_processing.py, order_embeddings.py, hyperbolic_projection.py, entailment_cones.py) are in place and have been run successfully if they generate input files."
    exit 1
fi

echo "All required files and scripts found!"

echo ""
echo "Starting hyperbolic cone validation and TDA data preparation..."
echo "Analysis parameters:"
echo "  - Input: SNLI 10k subset"
echo "  - Cone validation and TDA data generation"
echo ""

# Run the new Python script
python src/tda_cone_validation_asymmetry.py

# Capture exit code
EXIT_CODE=$?

echo ""
echo "Hyperbolic cone validation and TDA data preparation completed with exit code: $EXIT_CODE"
echo "Time: $(date)"

# Show what files were created (adjust based on what the new script saves)
echo ""
echo "Files created in validation_results/:"
ls -la validation_results/*snli_10k* 2>/dev/null || echo "No validation results or TDA data found"

echo ""
echo "Current directory contents (relevant directories):"
echo "Validation Results:"
ls -la validation_results/ 2>/dev/null || echo "No validation_results directory"
echo "Data:"
ls -la data/processed/ 2>/dev/null || echo "No data/processed directory" # Corrected typo: /dev_null to /dev/null
echo "Src:"
ls -la src/ 2>/dev/null || echo "No src directory"


# Show analysis results if successful
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== HYPERBOLIC CONE VALIDATION AND TDA DATA PREPARATION SUCCESSFUL ==="
    echo "The cone validation and TDA data preparation script has completed successfully."
    echo ""
    echo "Key validation checks performed:"
    echo "  ✓ Cone Energy Hierarchy validated"
    echo "  ✓ Correlation with Order Energies validated"
    echo "  ✓ Geometric Properties validated (Aperture, Asymmetry, Transitivity Hint)"
    echo "  ✓ TDA-ready data (cone violations, labels, texts, metadata) collected"
    echo ""
    echo "Generated outputs:"
    echo "  - validation_results/cone_validation_results_snli_10k.pt (full validation report)" # Removed `ls -la` and just printed the path
    echo "  - validation_results/tda_ready_data_snli_10k.pt (data specifically for TDA analysis)" # Removed `ls -la` and just printed the path
    echo ""
    echo "Next steps:"
    echo "  1. Review the validation results in validation_results/cone_validation_results_snli_10k.pt." # Removed `ls -la` and just printed the path
    echo "  2. Utilize validation_results/tda_ready_data_snli_10k.pt for further Topological Data Analysis." # Removed `ls -la` and just printed the path
else
    echo ""
    echo "=== HYPERBOLIC CONE VALIDATION AND TDA DATA PREPARATION FAILED ==="
    echo "Please check the error output above for debugging information."
    echo ""
    echo "Common issues to check:"
    echo "  - Missing input data (snli_10k_subset_balanced.pt)"
    echo "  - Missing dependency scripts (hyperbolic_projection.py, entailment_cones.py, order_embeddings.py)"
    echo "  - Issues within tda_ready_cone_validation_texts_preserved.py itself"
    echo "  - GPU memory availability or other resource constraints"
fi

echo ""
echo "Job finished."