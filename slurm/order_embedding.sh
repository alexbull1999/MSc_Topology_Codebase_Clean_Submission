#!/bin/bash
#SBATCH --job-name=train_order_embeddings_3way
#SBATCH --partition=gpgpuB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/full_order_embeddings_SBERT_3way_%j.out
#SBATCH --error=logs/full_order_embeddings_SBERT_3way_%j.err
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
echo "Checking for required input data..."

# Change to your project directory
cd $SLURM_SUBMIT_DIR/..

# Check if processed data exists
if [ ! -f "data/processed/snli_full_standard_BERT.pt" ]; then
    echo "ERROR: Required processed data file not found: data/processed/snli_full_standard_BERT.pt"
    echo "Please ensure text_processing.py has been run successfully first."
    exit 1
fi

echo "Required data file found: data/processed/snli_full_standard_BERT.pt"


echo ""
echo "Starting order embeddings training..."
echo "Training parameters:"
echo "  - Epochs: 50"
echo "  - Batch size: 32" 
echo "  - Order dimension: 50"
echo "  - Random seed: 42"
echo ""

# Run order embeddings training
python src/order_embeddings_asymmetry_3classloss.py

# Capture exit code
EXIT_CODE=$?

echo ""
echo "Training completed with exit code: $EXIT_CODE"
echo "Time: $(date)"

# Show what files were created
echo ""
echo "Files created in models/:"
ls -la models/ 2>/dev/null || echo "No models directory found"

echo ""
echo "Files created in plots/:"
ls -la plots/ 2>/dev/null || echo "No plots directory found"

# Show final training results if successful
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== TRAINING SUCCESSFUL ==="
    echo "Order embeddings model has been trained and saved."
    echo "Check the output above for energy rankings validation."
    echo "Training plots should be available in plots/order_embedding_training.png"
else
    echo ""
    echo "=== TRAINING FAILED ==="
    echo "Please check the error output above for debugging information."
fi

echo ""
echo "Job finished."