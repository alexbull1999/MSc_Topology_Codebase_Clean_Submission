#!/bin/bash
#SBATCH --job-name=cone_visualizations
#SBATCH --partition=gpgpuB
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --output=logs/slurm_cone_vis_asymmetry_%j.out
#SBATCH --error=logs/slurm_cone_vis_asymmetry_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahb24

echo "Starting real data cone visualizations job..."
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
import seaborn as sns
from sklearn.decomposition import PCA
import geoopt
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'Geoopt version: {geoopt.__version__}')
print('All required packages loaded successfully!')
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

if [ ! -f "src/entailment_cones.py" ]; then
    missing_files+=("src/entailment_cones.py (entailment cone implementation)")
fi

if [ ! -f "src/hyperbolic_projection.py" ]; then
    missing_files+=("src/hyperbolic_projection.py (hyperbolic projection)")
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
    echo "  3. hyperbolic_projection.py (hyperbolic embedding pipeline)"
    echo "  4. entailment_cones.py (cone analysis implementation)"
    exit 1
fi

echo "All required files found!"

# Run cone visualizations
python src/cone_visualisations_asymmetry.py

# Capture exit code
EXIT_CODE=$?

echo ""
echo "Cone visualizations completed with exit code: $EXIT_CODE"
echo "Time: $(date)"

# Show what files were created
echo ""
echo "Generated visualizations:"
ls -la plots/real_data_cone_visualizations/ 2>/dev/null || echo "No visualization directory found"

echo ""
echo "Directory structure:"
echo "Plots:"
ls -la plots/ 2>/dev/null || echo "No plots directory"

# Show analysis results if successful
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== CONE VISUALIZATIONS SUCCESSFUL ==="
    echo "Real data cone visualizations have been generated successfully."
    echo ""
    echo "Generated outputs:"
    echo "  ✓ PCA projection comparison plots"
    echo "  ✓ Energy distribution analysis"
    echo "  ✓ Representative example selections"
    echo "  ✓ Statistical validation of cone properties"
    echo ""
    echo "Key analyses performed:"
    echo "  - Cone violation energy ranking validation"
    echo "  - PCA-based 2D projections preserving cone structure"
    echo "  - Distribution analysis across entailment types"
    echo "  - Representative example identification"
    echo ""
    echo "Expected results:"
    echo "  - Entailment: Low cone violation energies (green)"
    echo "  - Neutral: Medium cone violation energies (blue)"
    echo "  - Contradiction: High cone violation energies (red)"
    echo ""
    echo "Next steps:"
    echo "  1. Review generated plots for cone structure validation"
    echo "  2. Proceed to TDA persistent homology analysis"
    echo "  3. Apply topological feature extraction"
else
    echo ""
    echo "=== CONE VISUALIZATIONS FAILED ==="
    echo "Please check the error output above for debugging information."
    echo ""
    echo "Common issues to check:"
    echo "  - Entailment cones implementation compatibility"
    echo "  - Hyperbolic pipeline initialization"
    echo "  - GPU memory availability for large dataset"
    echo "  - Missing dependencies (scikit-learn, seaborn)"
    echo ""
    echo "Troubleshooting steps:"
    echo "  1. Verify all prerequisite scripts have run successfully"
    echo "  2. Check GPU memory usage during processing"
    echo "  3. Ensure all required Python packages are installed"
fi

echo ""
echo "Job finished."