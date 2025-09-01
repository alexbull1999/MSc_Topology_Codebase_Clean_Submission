#!/bin/bash
#SBATCH --job-name=feature_ablation
#SBATCH --output=logs/feature_ablation_%j.out
#SBATCH --error=logs/feature_ablation_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=gpgpuC
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahb24

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "=========================================="
echo "Feature Ablation Analysis Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "Submit directory: $SLURM_SUBMIT_DIR"

# Load CUDA first
echo "Loading CUDA..."
. /vol/cuda/12.0.0/setup.sh

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /vol/bitbucket/ahb24/tda_entailment_new

echo "Activated conda environment: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi
echo ""

# Change to your project directory
echo "Changing to project directory..."
cd $SLURM_SUBMIT_DIR/..


# Set the correct data path - update this based on your actual file location
DATA_PATH="results/tda_integration/landmark_tda_features/enhanced_neural_network_features_snli_10k.pt"

# Check if the data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found at: $DATA_PATH"
    echo "Available .pt files:"
    find . -name "*.pt" -type f 2>/dev/null | head -10
    exit 1
fi

echo "Using data file: $DATA_PATH"
echo "File size: $(ls -lh "$DATA_PATH" | awk '{print $5}')"

# Create results directory
mkdir -p results/feature_ablation

# Run the feature ablation analysis
echo "=========================================="
echo "Starting feature ablation analysis..."
echo "=========================================="

python classifiers/feature_ablation.py \
    --data_path "$DATA_PATH" \
    --output_dir "results/feature_ablation" \
    --random_seed 42 \
    --n_folds 5 \
    --save_plots \
    --verbose

# Check if the job completed successfully
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "Feature ablation completed successfully!"
    echo "=========================================="
    
    # Display key results if they exist
    if [ -f "results/feature_ablation/feature_ablation_results.csv" ]; then
        echo "Top 5 performing feature combinations:"
        head -6 results/feature_ablation/feature_ablation_results.csv
        echo ""
    fi
    
    if [ -f "results/feature_ablation/analysis_summary.json" ]; then
        echo "Analysis summary:"
        cat results/feature_ablation/analysis_summary.json
        echo ""
    fi
    
    # List output files
    echo "Generated files:"
    ls -la results/feature_ablation/
    
else
    echo "=========================================="
    echo "ERROR: Feature ablation analysis failed!"
    echo "Exit code: $EXIT_CODE"
    echo "=========================================="
    
    # Try to show Python error if available
    echo "Python traceback (if available):"
    tail -50 logs/feature_ablation_${SLURM_JOB_ID}.err
    
    exit $EXIT_CODE
fi

# Copy results to a timestamped directory for archival
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results/feature_ablation_${TIMESTAMP}"
cp -r results/feature_ablation "$RESULTS_DIR"

echo "Results archived to: $RESULTS_DIR"
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"

# Final summary
echo "=========================================="
echo "JOB SUMMARY"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Status: SUCCESS"
echo "Runtime: $SECONDS seconds"
echo "Results: $RESULTS_DIR"
echo "Main results: results/feature_ablation/"
echo "=========================================="