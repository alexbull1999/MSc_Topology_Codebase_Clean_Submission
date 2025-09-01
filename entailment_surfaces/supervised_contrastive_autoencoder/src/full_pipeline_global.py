"""
Complete Pipeline for Global Dataset Contrastive Training
Clean implementation from scratch
"""

import torch
import torch.optim as optim
import os
import json
from datetime import datetime
from pathlib import Path
import numpy as np

# Import our clean modules
from contrastive_autoencoder_model_global import ContrastiveAutoencoder
from losses_global import FullDatasetCombinedLoss
from trainer_global import GlobalDatasetTrainer
from data_loader_global import GlobalDataLoader
from evaluator_global import GlobalContrastiveEvaluator
from attention_autoencoder_model import AttentionAutoencoder
from loss_plot_utils import plot_training_losses


def np_encoder(object):
    """ Custom encoder for numpy data types """
    if isinstance(object, np.integer):
        return int(object)
    elif isinstance(object, np.floating):
        return float(object)
    elif isinstance(object, np.ndarray):
        return object.tolist()
    else:
        # Let the default encoder raise the TypeError for other types
        return object


def create_experiment_config():
    """
    Create configuration for global dataset training
    """
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
        },
        
        # Model configuration
        'model': {
            'input_dim': 1536,  # Will be updated based on embedding_type
            'latent_dim': 75,
            'hidden_dims': [1024, 768, 512, 256, 128],
            'dropout_rate': 0.2
        },
        
        # Loss configuration - GLOBAL DATASET
        'loss': {
            'contrastive_weight': 1.0,
            'reconstruction_weight': 100.0,  # Start with pure contrastive
            'margin': 2.0,
            'update_frequency': 3,  # Update global dataset every 3 epochs
            'max_global_samples': 5000,  # Subsample global dataset for efficiency
            # NEW: scheduling parameters
            'schedule_reconstruction': False,
            'warmup_epochs': 0,
            'max_reconstruction_weight': 100.0,
            'schedule_type': 'linear'  # or 'exponential'
        },
        
        # Optimizer configuration
        'optimizer': {
            'optimizer_type': 'Adam',
            'lr': 0.0001,
            'weight_decay': 1e-5
        },
        
        # Training configuration
        'training': {
            'num_epochs': 200,  # Fewer epochs since global updates are expensive
            'patience': 8,
            'save_every': 20,
            'debug_frequency': 25  # More frequent debug output
        },
        
        # Output configuration
        'output': {
            'save_results': True,
            'save_plots': True,
            'experiment_name': 'cosine_concat'
        }
    }
    
    return config


def setup_experiment(config):
    """
    Setup experiment directories and logging
    """
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config['output']['experiment_name']
    exp_dir = f"entailment_surfaces/supervised_contrastive_autoencoder/experiments/{experiment_name}_{timestamp}"
    
    # Create subdirectories
    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    results_dir = os.path.join(exp_dir, 'results')
    
    for directory in [exp_dir, checkpoints_dir, results_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Experiment setup completed:")
    print(f"  Experiment directory: {exp_dir}")
    print(f"  Config saved to: {config_path}")
    
    return exp_dir, checkpoints_dir, results_dir


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
    
    # Update model input dimension based on embedding type
    config['model']['input_dim'] = data_loader.embedder.get_output_dim()
    print(f"Updated model input_dim to: {config['model']['input_dim']}")
    
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


def create_model_and_trainer(config, device):
    """
    Create model, loss function, optimizer, and trainer
    """
    print("Creating model and trainer...")
    print("=" * 40)
    
    # Create model
    model = ContrastiveAutoencoder(**config['model'])
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create global dataset loss function
    loss_function = FullDatasetCombinedLoss(**config['loss'])

    # Create optimizer based on type
    optimizer_type = config['optimizer'].get('optimizer_type', 'Adam')
    
    # Extract only the parameters that the optimizer expects
    optimizer_params = {
        'lr': config['optimizer'].get('lr', 0.0001),
        'weight_decay': config['optimizer'].get('weight_decay', 1e-5)
    }
    
    # Create optimizer
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), **optimizer_params)
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), **optimizer_params)
    else:
        raise ValueError(f"Unknown optimizer type")


    print(f"Optimizer created: {optimizer_type} (lr={optimizer_params['lr']})")
    
    # Create global dataset trainer
    trainer = GlobalDatasetTrainer(
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device
    )
    
    return model, trainer


def train_model(trainer, train_loader, val_loader, config, checkpoints_dir):
    """
    Train the model
    """
    print("Starting training...")
    print("=" * 40)
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        patience=config['training']['patience'],
        save_dir=checkpoints_dir,
        save_every=config['training']['save_every'],
        debug_frequency=config['training']['debug_frequency']
    )
    
    print("Training completed!")
    return trainer.train_history


def evaluate_model(model, train_loader, val_loader, test_loader, config, results_dir, device):
    """
    Comprehensive model evaluation
    """
    print("Starting model evaluation...")
    print("=" * 40)
    
    # Create evaluator
    evaluator = GlobalContrastiveEvaluator(model, device)
    
    # Run comprehensive evaluation
    evaluation_results = evaluator.comprehensive_evaluation(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader
    )
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    if config['output']['save_results']:
        results_path = evaluator.save_evaluation_results(results_dir)
        print(f"Evaluation results saved to: {results_path}")
    
    return evaluation_results


def save_final_results(config, train_history, evaluation_results, exp_dir):
    """
    Save final experiment results
    """
    print("Saving final experiment results...")
    
    final_results = {
        'experiment_config': config,
        'training_history': train_history,
        'evaluation_results': evaluation_results,
        'experiment_metadata': {
            'completion_time': datetime.now().isoformat(),
            'experiment_directory': exp_dir,
            'pipeline_version': 'global_dataset_v1.0'
        }
    }
    
    # Save comprehensive results
    final_results_path = os.path.join(exp_dir, 'final_results.json')
    with open(final_results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=np_encoder)
    
    print(f"Final results saved to: {final_results_path}")
    return final_results_path

def create_and_save_plots(train_history, exp_dir, experiment_name):
    """
    Create and save loss plots after training completes
    """
    print("Creating loss plots...")
        
    # Create plots
    plot_path = plot_training_losses(
        train_history=train_history,
        save_dir=exp_dir,
        experiment_name=experiment_name
    )
    
    print(f"Loss plots saved to: {plot_path}")
    return plot_path

def main(config_override=None):
    """
    Main execution function
    """
    print("GLOBAL DATASET CONTRASTIVE AUTOENCODER PIPELINE")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create configuration
    config = create_experiment_config()
    if config_override:
        # Deep merge config override
        for key, value in config_override.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
    
    # Print configuration summary
    print(f"\nConfiguration:")
    print(f"  Embedding type: {config['data']['embedding_type']}")
    print(f"  Batch size: {config['data']['batch_size']}")
    print(f"  Global update frequency: {config['loss']['update_frequency']} epochs")
    print(f"  Training epochs: {config['training']['num_epochs']}")
    
    # Setup experiment
    exp_dir, checkpoints_dir, results_dir = setup_experiment(config)
    
    try:
        # Load data
        train_loader, val_loader, test_loader = load_data(config)
        
        # Create model and trainer
        model, trainer = create_model_and_trainer(config, device)
        
        # Train model
        train_history = train_model(trainer, train_loader, val_loader, config, checkpoints_dir)
        create_and_save_plots(train_history, exp_dir, config['output']['experiment_name'])
        
        # Load best model for evaluation
        best_model_path = os.path.join(checkpoints_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate model
        evaluation_results = evaluate_model(
            model, train_loader, val_loader, test_loader, config, results_dir, device
        )
        
        # Save final results
        final_results_path = save_final_results(config, train_history, evaluation_results, exp_dir)
        
        # Print completion summary
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Experiment directory: {exp_dir}")
        print(f"Best model: {best_model_path}")
        print(f"Final results: {final_results_path}")
        
        # Print key results
        if evaluation_results and 'separation' in evaluation_results:
            separation = evaluation_results['separation']
            if 'error' not in separation:
                print(f"\nKey Results:")
                print(f"  Separation ratio: {separation['separation_ratio']:.2f}x")
                print(f"  Perfect separation: {separation['perfect_separation']}")
                print(f"  Classification accuracy: {evaluation_results['classification']['accuracy']:.4f}")
        
        return exp_dir, train_history, evaluation_results
        
    except Exception as e:
        print(f"\nPIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def run_lattice_experiment():
    """
    Run experiment with lattice containment embeddings
    """
    config_override = {
        'data': {
            'embedding_type': 'lattice'
        },
        'output': {
            'experiment_name': 'global_lattice_test'
        }
    }
    
    return main(config_override)


def run_concat_experiment():
    """
    Run experiment with concatenation embeddings for comparison
    """
    config_override = {
        'data': {
            'embedding_type': 'concat'
        },
        'output': {
            'experiment_name': 'global_concat_test_attention'
        }
    }
    
    return main(config_override)

def run_cosine_experiment():
    """
    Run experiment with cosine_concat embeddings for comparison
    """
    config_override = {
        'data': {
            'embedding_type': 'cosine_concat'
        },
        'output': {
            'experiment_name': 'global_cosine_test_no_attention'
        }
    }
    
    return main(config_override)

def run_difference_experiment():
    """
    Run experiment with difference embeddings for comparison
    """
    config_override = {
        'data': {
            'embedding_type': 'difference'
        },
        'output': {
            'experiment_name': 'global_difference_test'
        }
    }
    
    return main(config_override)


def test_pipeline():
    """
    Test pipeline with small synthetic dataset
    """
    print("Testing pipeline with synthetic data...")
    
    # Create tiny synthetic dataset
    import tempfile
    
    n_samples = 300
    premise_embeddings = torch.randn(n_samples, 768)
    hypothesis_embeddings = torch.randn(n_samples, 768)
    labels = ['entailment' if i % 3 == 0 else 'neutral' if i % 3 == 1 else 'contradiction' 
              for i in range(n_samples)]
    
    synthetic_data = {
        'premise_embeddings': premise_embeddings,
        'hypothesis_embeddings': hypothesis_embeddings,
        'labels': labels
    }
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(synthetic_data, f.name)
        temp_path = f.name
    
    try:
        config_override = {
            'data': {
                'train_path': temp_path,
                'val_path': temp_path,
                'test_path': temp_path,
                'sample_size': 150,
                'batch_size': 32
            },
            'training': {
                'num_epochs': 2,
                'debug_frequency': 5
            },
            'output': {
                'experiment_name': 'pipeline_test'
            }
        }
        
        result = main(config_override)
        
        if result[0] is not None:
            print("Pipeline test completed successfully!")
        else:
            print("Pipeline test failed!")
            
    finally:
        # Clean up
        os.unlink(temp_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            test_pipeline()
        elif sys.argv[1] == 'lattice':
            run_lattice_experiment()
        elif sys.argv[1] == 'concat':
            run_concat_experiment()
        elif sys.argv[1] == 'cosine':
            run_cosine_experiment()
        elif sys.argv[1] == 'difference':
            run_difference_experiment()
        else:
            print("Usage: python full_pipeline.py [test|lattice|concat]")
    else:
        # Default: run lattice experiment
        print("Running default lattice containment experiment...")
        run_lattice_experiment()