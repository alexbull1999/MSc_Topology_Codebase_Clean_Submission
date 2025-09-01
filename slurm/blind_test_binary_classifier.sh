#!/bin/bash
#SBATCH --job-name=blind_test_binary_classifier
#SBATCH --output=logs/blind_test_binary_classifier_%j.out
#SBATCH --error=logs/blind_test_binary_classifier_%j.err
#SBATCH --time=6:00:00                    # 12 hours should be enough for 50 combinations
#SBATCH --partition=gpgpuC
#SBATCH --gres=gpu:1                       # Request 1 GPU
#SBATCH --cpus-per-task=4                  # 4 CPUs should be sufficient
#SBATCH --mem=4G                          # 32GB RAM
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahb24

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"

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

# Change to your project directory
cd $SLURM_SUBMIT_DIR/..


mkdir -p results/overnight_binary_hyperparam_search_normalizer/blind_evaluation


# Run the enhanced hyperparameter search with 50 combinations
echo "Starting enhanced hyperparameter search with 50 combinations..."
echo "Expected runtime: 8-12 hours"
echo "Data path: results/tda_integration/landmark_tda_features/neural_network_features_snli_10k.pt"

python classifiers/evaluate_blind_test.py

# Check if the job completed successfully
if [ $? -eq 0 ]; then
    echo "Blind test completed successfully at: $(date)"
    
else
    echo "Blind test failed or was interrupted at: $(date)"
    echo "Check the error log"
    exit 1
fi

# Print final job info
echo "Job finished at: $(date)"
echo "Total runtime: $SECONDS seconds"

