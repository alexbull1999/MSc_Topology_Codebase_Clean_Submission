#!/bin/bash
echo "Setting up GPU environment..."

# Create base environment
conda env create -f environment.yml

# Activate environment
conda activate entailment-tda

# Replace CPU PyTorch with GPU version
conda remove pytorch torchvision torchaudio cpuonly --force
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

echo "✅ GPU environment ready!"
echo "Activate with: conda activate entailment-tda"

# Test GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
