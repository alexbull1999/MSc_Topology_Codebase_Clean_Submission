"""
Dynamic Model Loader for Generative Applications
Loads models dynamically from config files, following the existing codebase patterns
"""

import torch
import torch.nn as nn
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.contrastive_autoencoder_model_global import ContrastiveAutoencoder


class DynamicModelLoader:
    """
    Dynamic model loader that works with config files
    Follows the pattern from full_pipeline_global.py
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.config = None
        
        print(f"DynamicModelLoader initialized on device: {self.device}")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        print(f"Loading config from: {config_path}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("Config loaded successfully!")
        return config
    
    def load_model_from_config(self, config: Dict[str, Any]) -> ContrastiveAutoencoder:
        """Create model from configuration"""
        model_config = config['model']
        
        print("Creating model from config:")
        print(f"  Input dim: {model_config['input_dim']}")
        print(f"  Latent dim: {model_config['latent_dim']}")
        print(f"  Hidden dims: {model_config['hidden_dims']}")
        print(f"  Dropout rate: {model_config['dropout_rate']}")
        
        model = ContrastiveAutoencoder(**model_config)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model created with {total_params:,} parameters")
        
        return model

    def load_checkpoint(self, model: ContrastiveAutoencoder, checkpoint_path: str):
        """Load model weights from checkpoint"""
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model_state_dict from checkpoint")

        model.to(self.device)
        model.eval()

        # Print additional checkpoint info if available
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                print(f"  Epoch: {checkpoint['epoch']}")
            if 'val_loss' in checkpoint:
                print(f"  Validation loss: {checkpoint['val_loss']:.6f}")
            if 'train_loss' in checkpoint:
                print(f"  Training loss: {checkpoint['train_loss']:.6f}")


    def load_from_experiment_dir(self, experiment_dir: str) -> ContrastiveAutoencoder:
        """
        Load model from experiment directory (like your best performing model)
        This follows the pattern from your existing pipeline
        """
        experiment_path = Path(experiment_dir)
        
        if not experiment_path.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
        
        # Load config
        config_path = experiment_path / "config.json"
        self.config = self.load_config(str(config_path))
        
        # Create model from config
        self.model = self.load_model_from_config(self.config)
        
        # Load best model checkpoint
        checkpoint_path = experiment_path / "checkpoints" / "best_model.pt"
        self.load_checkpoint(self.model, str(checkpoint_path))
        
        return self.model

    def test_model_functionality(self):
        """Test basic model functionality"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_from_experiment_dir() or load_from_paths() first.")
        
        print("\nTesting model functionality...")
        print("=" * 50)
        
        # Get input dimension from config
        input_dim = self.config['model']['input_dim']
        batch_size = 2
        
        # Create test input based on embedding type
        embedding_type = self.config['data'].get('embedding_type', 'unknown')
        print(f"Embedding type: {embedding_type}")
        print(f"Input dimension: {input_dim}")
        
        dummy_input = torch.randn(batch_size, input_dim).to(self.device)
        
        with torch.no_grad():
            # Test forward pass - matches original model interface
            latent, reconstructed = self.model(dummy_input)
            
            print(f"Input shape: {dummy_input.shape}")
            print(f"Latent shape: {latent.shape}")
            print(f"Reconstruction shape: {reconstructed.shape}")
            
            # Test individual components
            latent_only = self.model.encode(dummy_input)
            reconstruction_only = self.model.decode(latent)
            
            # Print sample outputs
            print("\nSample outputs:")
            print(f"Latent vector (first 10 dims): {latent[0, :10].cpu().numpy()}")
            
            # Calculate reconstruction quality
            reconstruction_mse = torch.mean((dummy_input - reconstructed) ** 2)
            print(f"Reconstruction MSE: {reconstruction_mse.item():.6f}")
            
            # Note about classification
            print("\nNote: Classification is done via KNN on latent representations")
            print("      (not through a trained classifier head)")
        
        print("Model functionality test completed successfully!")

    def print_model_info(self):
        """Print detailed model and config information"""
        if self.model is None or self.config is None:
            raise ValueError("Model and config not loaded.")
        
        print("\n" + "=" * 60)
        print("MODEL AND CONFIG INFORMATION")
        print("=" * 60)
        
        # Print experiment info
        if 'output' in self.config and 'experiment_name' in self.config['output']:
            print(f"Experiment: {self.config['output']['experiment_name']}")
        
        # Print data config
        print("\nData Configuration:")
        for key, value in self.config['data'].items():
            print(f"  {key}: {value}")
        
        # Print model config
        print("\nModel Configuration:")
        for key, value in self.config['model'].items():
            print(f"  {key}: {value}")
        
        # Print model architecture details
        print(f"\nModel Architecture:")
        print(f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        print("\nEncoder layers:")
        for i, layer in enumerate(self.model.encoder):
            if hasattr(layer, 'weight'):
                print(f"  {i}: {layer}")
        
        print("\nDecoder layers:")
        for i, layer in enumerate(self.model.decoder):
            if hasattr(layer, 'weight'):
                print(f"  {i}: {layer}")
        
        print(f"\nClassification Method: KNN on latent representations")
        print(f"Expected Classification Accuracy: ~83% (from your results)")


def main():
    """Test the dynamic model loader"""
    print("TESTING DYNAMIC MODEL LOADER")
    print("=" * 60)
    
    loader = DynamicModelLoader()
        
    # Load from experiment directory
    print("\nLoad from experiment directory")
    experiment_dir = 'entailment_surfaces/supervised_contrastive_autoencoder/experiments/coarse_embeddingcosine_concat_hiddendims[1024, 768, 512, 256, 128]_dropout0.2_optimAdam_lr0.0001_20250715_204239'
    
    model = loader.load_from_experiment_dir(experiment_dir)
        
    # Test functionality
    loader.test_model_functionality()
        
    # Print detailed info
    loader.print_model_info()
        
    print("\n" + "=" * 60)
    print("MODEL LOADED SUCCESSFULLY!")
    print("Ready to proceed to manifold_analyzer.py")
    print("=" * 60)
        
    return loader.model, loader.config, loader.device


if __name__ == "__main__":
    main()