#!/bin/bash
#SBATCH --job-name=lattice_discovery
#SBATCH --partition=a16gpu
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=../logs/lattice_discovery_snlitestdata_3way_%j.out
#SBATCH --error=../logs/lattice_discovery_snlitestdata_3way_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahb24

echo "Starting Surface Distance Metric Analysis job..."
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

# Test PyTorch and CUDA
echo "Testing PyTorch and CUDA..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print('PyTorch setup verified!')
"

echo ""
echo "Checking for required input data and models..."

# Change to your project directory
cd ~/MSc_Topology_Codebase

# Check if processed BERT data exists
if [ ! -f "data/processed/snli_full_standard_BERT.pt" ]; then
    echo "ERROR: Required BERT data file not found: data/processed/snli_full_standard_BERT.pt"
    echo "Please ensure text_processing.py has been run successfully first."
    exit 1
fi

echo "Found BERT data: data/processed/snli_full_standard_BERT.pt"

# Check if trained order model exists
ORDER_MODEL=""
if [ -f "models/enhanced_order_embeddings_snli_full.pt" ]; then
    ORDER_MODEL="models/enhanced_order_embeddings_snli_full.pt"
    echo "✓ Found order model: $ORDER_MODEL"
else
    echo "ERROR: No trained order embedding model found in models/"
    echo "Please ensure order_embeddings_asymmetry.py has been run successfully first."
    exit 1
fi

echo ""
echo "Starting Lattice Metric Analysis..."
echo ""

python entailment_surfaces/lattice_metric_discovery.py 


# Capture exit code
EXIT_CODE=$?

echo ""
echo "Analysis completed with exit code: $EXIT_CODE"
echo "Time: $(date)"



if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== ANALYSIS SUCCESSFUL ==="
    echo ""
else
    echo ""
    echo "=== ANALYSIS FAILED ==="
    echo ""
fi

echo ""
echo "Job finished."