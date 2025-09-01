#!/bin/bash
#SBATCH --job-name=persistence_diagram_prototypes
#SBATCH --partition=gpgpuC
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=../logs/COSINE_LATTICE_prototype_creator_persistence_diagrams_%j.out
#SBATCH --error=../logs/COSINE_LATTICE_prototype_creator_persistence_diagrams_%j.err
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


# Change to your project directory
cd ~/MSc_Topology_Codebase

echo ""
echo "Starting Persistence Diagram Prototype Creation..."
echo ""

export PYTHONUNBUFFERED=1

python entailment_surfaces/supervised_contrastive_autoencoder/src/persistence_diagram_prototype_creator.py


# Capture exit code
EXIT_CODE=$?

echo ""
echo "Analysis completed with exit code: $EXIT_CODE"
echo "Time: $(date)"


if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== ANALYSIS SUCCESSFUL ==="
    echo "Persistence Diagram Prototypes successful!"
    echo ""

else
    echo ""
    echo "=== ANALYSIS FAILED ==="
    echo "Please check the error output above for debugging information."
    echo ""
fi

echo ""
echo "Job finished."