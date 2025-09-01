#!/bin/bash
#SBATCH --job-name=local_global_calibration
#SBATCH --partition=gpgpuC
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --output=../phd_logs/local_global_calibration_%j.out
#SBATCH --error=../phd_logs/local_global_calibration_%j.err
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


# Change to your project directory
cd $SLURM_SUBMIT_DIR/../..

# Check if required files exist
missing_files=()


echo ""
echo "Starting embedding tests..."
 
# Run PHD computation
python phd_method/src_phd/local_global_phd_calibration.py

# Capture exit code
EXIT_CODE=$?

# Show analysis results if successful
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== TESTS RAN SUCCESSFULLY ==="
fi

echo ""
echo "Job finished."