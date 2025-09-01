#!/bin/bash
#SBATCH --job-name=hyperbolic_projection
#SBATCH --partition=gpgpuB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm_hyperbolic_asymmetry%j.out
#SBATCH --error=logs/slurm_hyperbolic_asymmetry%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahb24

echo "Starting hyperbolic projection job..."
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
import matplotlib.pyplot as plt
print(f'PyTorch version: {torch.__version__}')
print(f'Geoopt version: {geoopt.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print('All packages loaded successfully!')
"

echo ""
echo "Checking for required input files..."

# Change to your project directory
cd $SLURM_SUBMIT_DIR/..

# Check if required files exist
missing_files=()

if [ ! -f "data/processed/snli_10k_subset_balanced.pt" ]; then
    missing_files+=("data/processed/snli_10k_subset_balanced.pt (from text_processing.py)")
fi

if [ ! -f "models/order_embeddings_snli_10k.pt" ]; then
    missing_files+=("models/order_embeddings_snli_10k.pt (from order_embeddings.py)")
fi

if [ ! -f "src/order_embeddings.py" ]; then
    missing_files+=("src/order_embeddings.py")
fi

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "ERROR: Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Please ensure the following have been run successfully:"
    echo "  1. text_processing.py (creates processed data)"
    echo "  2. order_embeddings.py (creates trained model)"
    exit 1
fi

echo "All required files found!"

echo ""
echo "Starting hyperbolic projection analysis..."
echo "Analysis parameters:"
echo "  - Input: SNLI 10k subset"
echo "  - Order model: Pre-trained order embeddings"
echo "  - Hyperbolic dimension: 30D"
echo "  - Target space: Poincaré ball"
echo ""

# Run hyperbolic projection
python src/hyperbolic_projection_asymmetry.py

# Capture exit code
EXIT_CODE=$?

echo ""
echo "Hyperbolic projection completed with exit code: $EXIT_CODE"
echo "Time: $(date)"

# Show what files were created
echo ""
echo "Files created in plots/:"
ls -la plots/*hyperbolic* 2>/dev/null || echo "No hyperbolic plots found"

echo ""
echo "Current directory contents:"
echo "Models:"
ls -la models/ 2>/dev/null || echo "No models directory"
echo "Plots:"
ls -la plots/ 2>/dev/null || echo "No plots directory"

# Show analysis results if successful
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== HYPERBOLIC PROJECTION SUCCESSFUL ==="
    echo "The hyperbolic projection has been completed successfully."
    echo ""
    echo "Key validation checks performed:"
    echo "  ✓ All points projected inside Poincaré ball unit sphere"
    echo "  ✓ Order energy hierarchy preservation verified"
    echo "  ✓ Hyperbolic distance statistics computed"
    echo "  ✓ Visualizations generated"
    echo ""
    echo "Generated outputs:"
    echo "  - Hyperbolic projection analysis plots"
    echo "  - Statistical validation of geometric properties"
    echo "  - Distance and energy distribution analysis"
    echo ""
    echo "Next steps:"
    echo "  1. Review the generated plots for geometric validation"
    echo "  2. Proceed to TDA cone analysis"
    echo "  3. Apply persistent homology to hyperbolic structures"
else
    echo ""
    echo "=== HYPERBOLIC PROJECTION FAILED ==="
    echo "Please check the error output above for debugging information."
    echo ""
    echo "Common issues to check:"
    echo "  - Geoopt library installation"
    echo "  - Order embeddings model compatibility"
    echo "  - GPU memory availability"
    echo "  - Input data format consistency"
fi

echo ""
echo "Job finished."