#!/bin/bash
echo "Setting up local development environment (CPU)..."

# Create environment
conda env create -f environment.yml

# Activate environment
conda activate entailment-tda

echo "✅ Local CPU environment ready!"
echo "Activate with: conda activate entailment-tda"