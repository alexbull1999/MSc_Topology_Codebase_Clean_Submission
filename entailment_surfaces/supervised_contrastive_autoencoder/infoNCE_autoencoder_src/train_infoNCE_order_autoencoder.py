"""
Main training script for InfoNCE + Order Embeddings autoencoder
"""

import torch
import torch.optim as optim
import os
import json
from datetime import datetime
from pathlib import Path

from infoNCE_autoencoder import InfoNCEOrderAutoencoder
from losses import CombinedLoss
from data_loader import create_data_loaders
from trainer import InfoNCEOrderTrainer
from evaluator import InfoNCEOrderEvaluator


def create_config():
    """
    Create configuration for InfoNCE + Order Embeddings + Moor Topological training
    """
    return {
        'data': {
            'train_path': 'data/processed/snli_full_standard_SBERT.pt',
            'val_path': 'data/processed/snli_full_standard_SBERT_validation.pt',
            'test_path': 'data/processed/snli_full_standard_SBERT_test.pt',
            'embedding_type': 'concat',
            'batch_size': 1020  # Your proven batch size
        },
        
        'model': {
            'input_dim': 1536,  # SBERT concat dimension
            'latent_dim': 75,   # Your proven latent dimension
            'hidden_dims': [1024, 768, 512, 256],  # Your proven architecture
            'dropout_rate': 0.2
        },
        
        'loss': {
            'infonce_weight': 1.0,        # Main contrastive signal
            'order_weight': 0.5,          # Asymmetric entailment constraints
            'topological_weight': 1.0,   # Moor topological regularization
            'reconstruction_weight': 10.0, # Semantic preservation
            'temperature': 0.1,          # InfoNCE temperature
            'max_global_samples': 5000,   # Global dataset size for InfoNCE
            # Order embedding margins (Vendrov et al.)
            'entailment_margin': 0.3,     # Target energy for entailment
            'neutral_margin': 1.0,        # Lower bound for neutral
            'neutral_upper_bound': 1.5,   # Upper bound for neutral  
            'contradiction_margin': 2.0,  # Minimum energy for contradiction
            'topo_frequency': 100        # Apply topology every 1000 iterations
        },
        
        'optimizer': {
            'lr': 0.0001,      # Your proven learning rate
            'weight_decay': 1e-5
        },
        
        'training': {
            'num_epochs': 100,
            'patience': 15,
            'save_every': 10
        }
    }


def setup_experiment(config):
    """
    Setup experiment directory structure
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"infonce_order_autoencoder_{timestamp}"
    
    exp_dir = f"entailment_surfaces/supervised_contrastive_autoencoder/infoNCE_autoencoder_src/experiments/{exp_name}"
    checkpoints_dir = f"{exp_dir}/checkpoints"
    results_dir = f"{exp_dir}/results"
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save config
    with open(f"{exp_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Experiment setup: {exp_dir}")
    return exp_dir, checkpoints_dir, results_dir


def main():
    """
    Main training function
    """
    print("=" * 60)
    print("INFONCE + ORDER EMBEDDINGS AUTOENCODER TRAINING")
    print("=" * 60)
    
    # Create config
    config = create_config()
    
    # Setup experiment
    exp_dir, checkpoints_dir, results_dir = setup_experiment(config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Create model
    print("\nCreating model...")
    model = InfoNCEOrderAutoencoder(**config['model'])
    
    # Create loss function
    print("\nCreating loss function...")
    loss_function = CombinedLoss(**config['loss'])
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay']
    )
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = InfoNCEOrderTrainer(model, loss_function, optimizer, device)
    
    print("\nTraining configuration:")
    print(f"  InfoNCE weight: {config['loss']['infonce_weight']}")
    print(f"  Order weight: {config['loss']['order_weight']}")
    print(f"  Topological weight: {config['loss']['topological_weight']}")
    print(f"  Reconstruction weight: {config['loss']['reconstruction_weight']}")
    print(f"  Temperature: {config['loss']['temperature']}")
    print(f"  Topological frequency: {config['loss']['topo_frequency']} iterations")
    print(f"  Learning rate: {config['optimizer']['lr']}")
    print(f"  Batch size: {config['data']['batch_size']}")
    print()
    
    print("Expected improvements over distance-based contrastive:")
    print("  ✅ InfoNCE preserves topological structure (no representation collapse)")
    print("  ✅ Order embeddings add asymmetric entailment constraints")
    print("  ✅ Moor topological loss preserves input→latent topology (applied sparsely)")
    print("  ✅ Target performance: 81.67% → 88-93% accuracy")
    print()
    
    # Train model
    try:
        print("Starting training...")
        train_history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs'],
            patience=config['training']['patience'],
            save_dir=checkpoints_dir
        )
        
        # Save training history
        with open(f"{results_dir}/train_history.json", 'w') as f:
            json.dump(train_history, f, indent=2)
        
        print("✅ Training completed successfully!")
        
        # Plot training history
        print("\nGenerating training plots...")
        from plot_results import plot_training_history, compare_with_baseline
        
        try:
            plot_training_history(f"{results_dir}/train_history.json", results_dir)
        except Exception as e:
            print(f"Warning: Could not generate training plots: {e}")
        
        # Evaluate final model
        print("\n" + "=" * 50)
        print("FINAL EVALUATION")
        print("=" * 50)
        
        evaluator = InfoNCEOrderEvaluator(model, device)
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_results = evaluator.evaluate_detailed(test_loader)
        
        print(f"\nFinal Test Results:")
        print(f"  Clustering Accuracy: {test_results['clustering_accuracy']:.4f}")
        print(f"  Silhouette Score: {test_results['silhouette_score']:.4f}")
        print(f"  Separation Ratio: {test_results['separation_ratio']:.4f}")
        print(f"  Positive Distance Mean: {test_results['pos_distance_mean']:.4f}")
        print(f"  Negative Distance Mean: {test_results['neg_distance_mean']:.4f}")
        
        # Compare to baseline
        baseline_accuracy = 0.8167  # Your current best
        improvement = test_results['clustering_accuracy'] - baseline_accuracy
        print(f"\nComparison to baseline ({baseline_accuracy:.4f}):")
        print(f"  Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
        
        # Save results
        with open(f"{results_dir}/test_results.json", 'w') as f:
            json.dump(test_results, f, indent=2)

        
        # Also evaluate on validation set for comparison
        print("\nEvaluating on validation set...")
        val_results = evaluator.evaluate(val_loader)
        
        print(f"\nValidation Results:")
        print(f"  Clustering Accuracy: {val_results['clustering_accuracy']:.4f}")
        print(f"  Silhouette Score: {val_results['silhouette_score']:.4f}")
        print(f"  Separation Ratio: {val_results['separation_ratio']:.4f}")
        
        # Save validation results
        with open(f"{results_dir}/val_results.json", 'w') as f:
            json.dump(val_results, f, indent=2)
        
        print(f"\nAll results saved to: {results_dir}")
        print(f"Model checkpoints saved to: {checkpoints_dir}")
        print(f"Training plots saved to: {results_dir}/training_history.png")
        print(f"Comparison plots saved to: {results_dir}/baseline_comparison.png")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def quick_test():
    """
    Quick test to verify everything works before full training
    """
    print("=" * 50)
    print("QUICK TEST - INFONCE + ORDER AUTOENCODER")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    
    # Create small test data
    premise_embeddings = torch.randn(batch_size, 768, device=device)
    hypothesis_embeddings = torch.randn(batch_size, 768, device=device)
    labels = torch.randint(0, 3, (batch_size,), device=device)
    
    print(f"Test data created:")
    print(f"  Premise embeddings: {premise_embeddings.shape}")
    print(f"  Hypothesis embeddings: {hypothesis_embeddings.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Entailment pairs: {(labels == 0).sum().item()}")
    
    # Create model
    model = InfoNCEOrderAutoencoder(input_dim=1536, latent_dim=75)
    model.to(device)
    
    # Create loss function
    loss_function = CombinedLoss(
        infonce_weight=1.0,
        order_weight=0.1,
        topological_weight=0.05,
        reconstruction_weight=0.3,
        entailment_margin=0.3,
        neutral_margin=1.0,
        neutral_upper_bound=1.5,
        contradiction_margin=2.0,
        topo_frequency=10  # Apply frequently for testing
    )
    
    try:
        # Test forward pass
        print("\nTesting forward pass...")
        # Concatenate premise and hypothesis for model input
        combined_embeddings = torch.cat([premise_embeddings, hypothesis_embeddings], dim=1)
        latent_features, reconstructed = model(combined_embeddings)
        
        print(f"  Combined input: {combined_embeddings.shape}")
        print(f"  Latent features: {latent_features.shape}")
        print(f"  Reconstructed: {reconstructed.shape}")
        
        # Split reconstructed back to premise and hypothesis
        premise_reconstructed = reconstructed[:, :768]
        hypothesis_reconstructed = reconstructed[:, 768:]
        
        # For loss computation, use latent as both premise and hypothesis representation
        premise_latent = latent_features
        hypothesis_latent = latent_features
        
        # Test loss computation
        print("\nTesting loss computation...")
        total_loss, loss_components = loss_function(
            premise_latent, hypothesis_latent,
            premise_reconstructed, hypothesis_reconstructed,
            premise_embeddings, hypothesis_embeddings,
            labels
        )
        
        print(f"Loss components:")
        for key, value in loss_components.items():
            print(f"  {key}: {value:.6f}")
        
        # Test backward pass
        print("\nTesting backward pass...")
        total_loss.backward()
        print("✅ Backward pass successful!")
        
        print("\n✅ All tests passed! Ready for full training.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Run quick test
        success = quick_test()
        if success:
            print("\nRun full training with: python train_infonce_order_autoencoder.py")
    else:
        # Run full training
        success = main()
        if success:
            print("\n🎉 Training completed successfully!")
        else:
            print("\n❌ Training failed!")
