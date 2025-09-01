"""
Simple script to run evaluator_global.py with identity autoencoder baseline
"""

import torch
import sys
import os

# Add your source directory to path
from identity_autoencoder import IdentityAutoencoder
from evaluator_global import GlobalContrastiveEvaluator
from data_loader_global import GlobalDataLoader


class IdentityAutoencoder(torch.nn.Module):
    """Identity autoencoder - does nothing to embeddings"""
    
    def __init__(self, input_dim=1536):
        super(IdentityAutoencoder, self).__init__()
        self.input_dim = input_dim
    
    def forward(self, x):
        return x, x  # latent, reconstructed


def run_identity_baseline_evaluation(train_loader, val_loader, test_loader, save_dir="identity_baseline_results"):
    """
    Run evaluator_global.py tests with identity autoencoder
    
    Args:
        data_loader: Your SBERT embeddings dataloader
        save_dir: Directory to save results
    """
    print("Running Identity Autoencoder Baseline Evaluation")
    print("=" * 50)
    
    # Create identity autoencoder
    model = IdentityAutoencoder(input_dim=1536)
    
    # Set up evaluator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = GlobalContrastiveEvaluator(model, device)
    
    print(f"Using device: {device}")
    print(f"Model: Identity Autoencoder (no processing)")
    print(f"Input/Output dim: 1536D")
    
    # Run comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    results = evaluator.comprehensive_evaluation(
        train_dataloader=train_loader,
        val_dataloader=val_loader,  # Use same data for baseline
        test_dataloader=test_loader
    )
    
    # Print results summary
    print("\n" + "=" * 50)
    print("IDENTITY BASELINE RESULTS SUMMARY")
    print("=" * 50)
    evaluator.print_summary()
    
    # Save results
    results_file = evaluator.save_evaluation_results(save_dir)
    
    print(f"\nBaseline results saved to: {results_file}")
    print("\nThis baseline shows performance with:")
    print("  ✅ No dimensionality reduction (1536D → 1536D)")
    print("  ✅ Perfect reconstruction (MSE = 0.0)")
    print("  ✅ Original SBERT semantic structure preserved")
    print("  ✅ No learned transformations")
    
    return results


def load_data(config):
    """
    Load and prepare data
    """
    print("Loading data...")
    print("=" * 40)
    
    # Create data loader
    data_loader = GlobalDataLoader(
        train_path=config['data']['train_path'],
        val_path=config['data']['val_path'], 
        test_path=config['data']['test_path'],
        embedding_type=config['data']['embedding_type'],
        sample_size=config['data']['sample_size'],
        random_state=config['data']['random_state']
    )
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = data_loader.load_data()
        
    # Create data loaders
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        batch_size=config['data']['batch_size'],
        balanced_sampling=config['data']['balanced_sampling']
    )
    
    print(f"Data loading completed!")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader



if __name__ == "__main__":
    # For testing with synthetic data
    print("Creating sample dataloader for testing...")
    config = {
        # Data configuration
        'data': {
            'train_path': 'data/processed/snli_full_standard_SBERT.pt',
            'val_path': 'data/processed/snli_full_standard_SBERT_validation.pt',
            'test_path': 'data/processed/snli_full_standard_SBERT_test.pt',
            'embedding_type': 'concat',  # 'lattice', 'concat', 'difference', 'cosine_concat'
            'batch_size': 1020,
            'sample_size': None,  # Use all data
            'balanced_sampling': True,
            'random_state': 42
        }
    }
    train_loader, val_loader, test_loader = load_data(config)
    
    # Run baseline evaluation
    results = run_identity_baseline_evaluation(train_loader,val_loader, test_loader)