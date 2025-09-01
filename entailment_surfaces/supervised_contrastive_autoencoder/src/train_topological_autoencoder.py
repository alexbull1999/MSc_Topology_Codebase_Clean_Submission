import torch
import torch.optim as optim
import os
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from full_pipeline_global import setup_experiment, load_data
from contrastive_autoencoder_model_global import ContrastiveAutoencoder
from attention_autoencoder_model import AttentionAutoencoder
from losses_global_topological import TopologicallyRegularizedCombinedLoss
from trainer_topological import TopologicalTrainer
from evaluator_global import GlobalContrastiveEvaluator


def create_topological_config():
    """
    Create configuration for topological autoencoder training.
    Based on your existing config structure but adapted for TorchPH.
    """
    config = {
        'data': {
            'train_path': 'data/processed/snli_full_standard_SBERT.pt',
            'val_path': 'data/processed/snli_full_standard_SBERT_validation.pt',
            'test_path': 'data/processed/snli_full_standard_SBERT_test.pt',
            'embedding_type': 'concat',  # Use your best performing type
            'batch_size': 3000,
            'sample_size': None,
            'balanced_sampling': True,
            'random_state': 42
        },
        
        'model': {
            'input_dim': 1536,  
            'latent_dim': 75,
            'hidden_dims': [1024, 768, 512, 256, 128],
            'dropout_rate': 0.2
        },
        
        'loss': {
            # Phase 1: Topological + Reconstruction (NO contrastive initially)
            'contrastive_weight': 0.0,  # Start with 0
            'reconstruction_weight': 100.0,  # INCREASED: Strong semantic preservation signal
            
            # Topological loss settings
            'topological_weight':1.0,  # Main learning signal
            'max_topological_weight': 1.0,
            'topological_warmup_epochs': 0,  # FIXED: Start immediately (no warmup)
            'prototypes_path': None,
            
            # Reconstruction scheduling (for compatibility with FullDatasetCombinedLoss)
            'schedule_reconstruction': False,  # Keep constant for Phase 1
            'warmup_epochs': 0,  # No warmup needed
            'max_reconstruction_weight': 100.0,
            'schedule_type': 'linear',
            
            # Global dataset settings (required for FullDatasetCombinedLoss compatibility)
            'positive_margin': 2.0,  # ADDED: Required even with contrastive_weight=0
            'negative_margin': 10.0,
            'update_frequency': 3,
            'max_global_samples': 5000
        },
        
        
        'optimizer': {
            'lr': 0.001,  # Conservative learning rate #WAS 0.0001
            'weight_decay': 1e-5
        },
        
        'training': {
            'num_epochs': 100,  # More epochs needed for topological learning
            'patience': 10,  # More patience for topology to emerge // was 10 -- removing patience essentially
            'save_every': 10,
            'debug_frequency': 25
        },
        
        'output': {
            'save_results': True,
            'save_plots': True,
            'experiment_name': 'signature_moor_lifted_autoencoder_attention'
        }
    }
    
    return config


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


def create_and_save_topological_plots(train_history, exp_dir, experiment_name):
    """
    Create and save loss plots after topological training completes
    """
    print("Creating topological loss plots...")
    
    # Import plotting functions
    from loss_plot_utils_topological import plot_topological_training_losses, plot_curriculum_learning_analysis
    
    # Create main training plots
    main_plot_path = plot_topological_training_losses(
        train_history=train_history,
        save_dir=exp_dir,
        experiment_name=experiment_name
    )
    
    # Create curriculum learning analysis if relevant data exists
    curriculum_plot_path = None
    if any(key in train_history for key in ['topological_weight', 'persistence_weight']):
        curriculum_plot_path = plot_curriculum_learning_analysis(
            train_history=train_history,
            save_dir=exp_dir,
            experiment_name=experiment_name
        )
    
    print(f"Main training plots saved to: {main_plot_path}")
    if curriculum_plot_path:
        print(f"Curriculum analysis saved to: {curriculum_plot_path}")
    
    return main_plot_path, curriculum_plot_path


def save_topological_training_summary(train_history, save_dir, experiment_name):
    """
    Save a text summary of the topological training progress
    """
    summary_path = os.path.join(save_dir, f'{experiment_name}_training_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write(f"TOPOLOGICAL AUTOENCODER TRAINING SUMMARY\n")
        f.write(f"{'='*50}\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Total Epochs: {len(train_history['epoch'])}\n\n")
        
        # Final losses
        if train_history['train_loss']:
            f.write(f"Final Losses:\n")
            f.write(f"  Total Loss (Train): {train_history['train_loss'][-1]:.6f}\n")
            f.write(f"  Total Loss (Val):   {train_history['val_loss'][-1]:.6f}\n")
            
            if 'train_contrastive_loss' in train_history:
                f.write(f"  Contrastive (Train): {train_history['train_contrastive_loss'][-1]:.6f}\n")
                f.write(f"  Contrastive (Val):   {train_history['val_contrastive_loss'][-1]:.6f}\n")
            
            if 'train_topological_loss' in train_history:
                f.write(f"  Topological (Train): {train_history['train_topological_loss'][-1]:.6f}\n")
                f.write(f"  Topological (Val):   {train_history['val_topological_loss'][-1]:.6f}\n")
            elif 'train_persistence_loss' in train_history:
                f.write(f"  Persistence (Train): {train_history['train_persistence_loss'][-1]:.6f}\n")
                f.write(f"  Persistence (Val):   {train_history['val_persistence_loss'][-1]:.6f}\n")
            
            if 'train_reconstruction_loss' in train_history:
                f.write(f"  Reconstruction (Train): {train_history['train_reconstruction_loss'][-1]:.6f}\n")
                f.write(f"  Reconstruction (Val):   {train_history['val_reconstruction_loss'][-1]:.6f}\n")
        
        # Separation metrics
        if 'train_separation_ratio' in train_history:
            f.write(f"\nSeparation Metrics:\n")
            f.write(f"  Final Train Separation: {train_history['train_separation_ratio'][-1]:.2f}x\n")
            f.write(f"  Final Val Separation:   {train_history['val_separation_ratio'][-1]:.2f}x\n")
        
        # Topological metrics
        if 'total_persistence' in train_history:
            f.write(f"\nTopological Metrics:\n")
            f.write(f"  Final Total Persistence: {train_history['total_persistence'][-1]:.6f}\n")
        
        # Weight schedule info
        if 'topological_weight' in train_history:
            f.write(f"\nCurriculum Learning:\n")
            f.write(f"  Initial Topological Weight: {train_history['topological_weight'][0]:.6f}\n")
            f.write(f"  Final Topological Weight:   {train_history['topological_weight'][-1]:.6f}\n")
        elif 'persistence_weight' in train_history:
            f.write(f"\nCurriculum Learning:\n")
            f.write(f"  Initial Persistence Weight: {train_history['persistence_weight'][0]:.6f}\n")
            f.write(f"  Final Persistence Weight:   {train_history['persistence_weight'][-1]:.6f}\n")
    
    print(f"Training summary saved to: {summary_path}")
    return summary_path


def main_topological_training():
    """
    Main function for topological autoencoder training using TorchPH.
    """
    print("="*60)
    print("TOPOLOGICAL AUTOENCODER TRAINING WITH TORCHPH")
    print("="*60)
    
    # Create config
    config = create_topological_config()
    
    # Setup experiment (use your existing setup function)
    
    exp_dir, checkpoints_dir, results_dir = setup_experiment(config)
    
    # Create data loaders (reuse your existing function)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = load_data(config)
    
    # Create model (reuse your existing model)
    model = ContrastiveAutoencoder(**config['model'])
    
    # Create topological loss function
    loss_function = TopologicallyRegularizedCombinedLoss(**config['loss'])
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), **config['optimizer'])
    
    # Create trainer (reuse your existing trainer)
    trainer = TopologicalTrainer(model, loss_function, optimizer, device)
    
    print("Starting Phase 1: Pure Topological Training")
    print(f"  Contrastive weight: {config['loss']['contrastive_weight']}")
    print(f"  Topological weight: {config['loss']['topological_weight']}")
    print(f"  Reconstruction weight: {config['loss']['reconstruction_weight']}")
    
    # Train model
    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs'],
            patience=config['training']['patience'],
            save_dir=checkpoints_dir,
            save_every=config['training']['save_every'],
            debug_frequency=config['training']['debug_frequency']
        )
        
        print("✅ Topological training completed successfully!")
        
        train_history = trainer.train_history
        create_and_save_topological_plots(train_history, exp_dir, config['output']['experiment_name'])
        save_topological_training_summary(train_history, exp_dir, config['output']['experiment_name'])

        # Save results
        print("Saving results...")
        # You can reuse your existing evaluation functions here

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

        # Print key results
        if evaluation_results and 'separation' in evaluation_results:
            separation = evaluation_results['separation']
            if 'error' not in separation:
                print(f"\nKey Results:")
                print(f"  Separation ratio: {separation['separation_ratio']:.2f}x")
                print(f"  Perfect separation: {separation['perfect_separation']}")
                print(f"  Classification accuracy: {evaluation_results['classification']['accuracy']:.4f}")

        # Analyze topological success
        print("\n" + "="*60)
        print("TOPOLOGICAL TRAINING ANALYSIS")
        print("="*60)
        
        # Get topological learning diagnosis
        diagnosis = trainer.diagnose_topological_progress()
        
        if diagnosis:
            topo_percentage = diagnosis['topology_percentage']
            if topo_percentage > 0.8:
                print("🚀 EXCELLENT: Consistent topological learning achieved!")
            elif topo_percentage > 0.5:
                print("✅ GOOD: Reasonable topological learning")
            elif topo_percentage > 0.2:
                print("⚠️  PARTIAL: Some topological learning but inconsistent")
            else:
                print("❌ POOR: Very limited topological learning")
            
            print(f"Final topological loss: {diagnosis['current_loss']:.4f}")
            print(f"Epochs with topology: {diagnosis['epochs_with_topology']}/{diagnosis['total_epochs']}")
        
        # Check if we avoided "three balls" problem
        clustering_results = evaluation_results.get('clustering', {})
        if 'clustering_accuracy' in clustering_results:
            clustering_acc = clustering_results['clustering_accuracy']
            if clustering_acc > 0.9:
                print(f"🎯 Excellent clustering accuracy: {clustering_acc:.3f}")
            elif clustering_acc > 0.7:
                print(f"👍 Good clustering accuracy: {clustering_acc:.3f}")
            else:
                print(f"⚠️  Poor clustering accuracy: {clustering_acc:.3f}")
        
        # Save final analysis
        final_analysis = {
            'experiment_config': config,
            'training_diagnosis': diagnosis,
            'evaluation_results': evaluation_results,
            'experiment_metadata': {
                'completion_time': datetime.now().isoformat(),
                'experiment_directory': exp_dir,
                'approach': 'contrastive_primary_topological_regularization',
                'insight': 'Pure topological loss fails because targets equal inputs'
            }
        }
        
        analysis_path = os.path.join(results_dir, 'final_analysis.json')
        with open(analysis_path, 'w') as f:
            # Convert any non-serializable objects
            import json
            serializable_analysis = {}
            for key, value in final_analysis.items():
                try:
                    json.dumps(value)  # Test if serializable
                    serializable_analysis[key] = value
                except:
                    serializable_analysis[key] = str(value)  # Convert to string if not
            
            json.dump(serializable_analysis, f, indent=2)
        
        print(f"\nFinal analysis saved to: {analysis_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Experiment saved to: {exp_dir}")


if __name__ == "__main__":
    main_topological_training()