#!/bin/bash
#SBATCH --job-name=tda_normal_hyperparam_search
#SBATCH --output=logs/normal_hyperparam_search%j.out
#SBATCH --error=logs/normal_hyperparam_search%j.err
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


mkdir -p results/overnight_normal_hyperparam_search


# Run the enhanced hyperparameter search with 50 combinations
echo "Starting enhanced hyperparameter search with 50 combinations..."
echo "Expected runtime: 8-12 hours"
echo "Data path: results/tda_integration/landmark_tda_features/neural_network_features_snli_10k.pt"

python classifiers/train_classifier_landmark.py \
    --hyperparameter_search \
    --max_combinations 50 \
    --use_random_search \
    --final_training \
    --results_dir results/overnight_normal_hyperparam_search

# Check if the job completed successfully
if [ $? -eq 0 ]; then
    echo "Hyperparameter search completed successfully at: $(date)"
    
    # Quick summary of best results
    echo "=== QUICK RESULTS SUMMARY ==="
    if [ -f "results/overnight_enhanced_hyperparam_search/hyperparameter_search_results.json" ]; then
        echo "Best hyperparameters found:"
        python3 -c "
import json
with open('results/overnight_enhanced_hyperparam_search/hyperparameter_search_results.json') as f:
    data = json.load(f)
    if data['results']:
        best = data['results'][0]
        print(f\"Accuracy: {best['cv_mean_accuracy']:.2f}% ± {best['cv_std_accuracy']:.2f}%\")
        print(f\"F1-Score: {best['cv_mean_f1']:.4f} ± {best['cv_std_f1']:.4f}\")
        print(f\"Hyperparams: {best['hyperparameters']}\")
"
    fi
else
    echo "Hyperparameter search failed or was interrupted at: $(date)"
    echo "Check the error log: logs/enhanced_hyperparam_search_${SLURM_JOB_ID}.err"
    exit 1
fi

# Print final job info
echo "Job finished at: $(date)"
echo "Total runtime: $SECONDS seconds"

