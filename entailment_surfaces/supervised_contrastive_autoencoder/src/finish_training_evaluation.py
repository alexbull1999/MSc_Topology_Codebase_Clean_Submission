"""
Complete Topological Analysis Script
Loads trained topological models and completes the analysis that failed due to import error.
"""

import torch
import os
import json
from datetime import datetime
from pathlib import Path
import sys

# Add the src directory to path to import modules
sys.path.append('entailment_surfaces/supervised_contrastive_autoencoder/src')

# Import required modules
from loss_plot_utils_topological import plot_topological_training_losses, plot_curriculum_learning_analysis
from contrastive_autoencoder_model_global import ContrastiveAutoencoder
from losses_global_topological import TopologicallyRegularizedCombinedLoss
from trainer_topological import TopologicalTrainer
from data_loader_global import GlobalDataLoader
from evaluator_global import GlobalContrastiveEvaluator
from attention_autoencoder_model import AttentionAutoencoder


def create_and_save_topological_plots(train_history, exp_dir, experiment_name):
    """
    Create and save loss plots after topological training completes
    """
    print("Creating topological loss plots...")
    
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


def save_topological_training_summary(train_history, trainer, save_dir, experiment_name):
    """
    Save a text summary of the topological training progress
    """
    summary_path = os.path.join(save_dir, f'{experiment_name}_training_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write(f"TOPOLOGICAL AUTOENCODER TRAINING SUMMARY\n")
        f.write(f"{'='*50}\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Total Epochs: {len(train_history['epoch'])}\n")
        f.write(f"Best Epoch: {trainer.best_epoch if hasattr(trainer, 'best_epoch') else 'N/A'}\n")
        f.write(f"Best Val Loss: {trainer.best_val_loss if hasattr(trainer, 'best_val_loss') else 'N/A':.6f}\n\n")
        
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


def load_model_and_trainer_from_checkpoint(exp_dir, config):
    """
    Load the trained model and reconstruct trainer from checkpoint
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    best_model_path = os.path.join(checkpoints_dir, 'checkpoint_epoch_50.pt')
    
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")
    
    print(f"Loading model from {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device)
    
    # Recreate model
    model = ContrastiveAutoencoder(**config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Recreate loss function and optimizer
    loss_function = TopologicallyRegularizedCombinedLoss(**config['loss'])
    optimizer = torch.optim.Adam(model.parameters(), **config['optimizer'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Recreate trainer
    trainer = TopologicalTrainer(model, loss_function, optimizer, device)
    
    # Restore trainer state
    trainer.train_history = checkpoint['train_history']
    trainer.best_val_loss = checkpoint['best_val_loss']
    trainer.best_epoch = checkpoint['best_epoch']
    
    return model, trainer, device


def recreate_config_from_experiment(exp_dir):
    """
    Try to recreate config from experiment directory or use reasonable defaults
    """
    # Try to load existing config if available
    config_files = ['final_analysis.json', 'experiment_config.json', 'config.json']
    
    for config_file in config_files:
        config_path = os.path.join(exp_dir, config_file)
        if os.path.exists(config_path):
            print(f"Loading config from {config_path}")
            with open(config_path, 'r') as f:
                data = json.load(f)
                return data


def load_data(config):
    """
    Load data using existing data loader
    """
    data_loader = GlobalDataLoader(
        train_path=config['data']['train_path'],
        val_path=config['data']['val_path'],
        test_path=config['data']['test_path'],
        embedding_type=config['data']['embedding_type'],
        sample_size=config['data']['sample_size'],
        random_state=config['data']['random_state']
    )
    
    data_loader.load_data()

    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        batch_size=config['data']['batch_size'],
        balanced_sampling=config['data']['balanced_sampling']
    )

    return train_loader, val_loader, test_loader


def evaluate_model(model, train_loader, val_loader, test_loader, config, results_dir, device):
    """
    Comprehensive model evaluation
    """
    print("Starting model evaluation...")
    
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
    results_path = evaluator.save_evaluation_results(results_dir)
    print(f"Evaluation results saved to: {results_path}")
    
    return evaluation_results


def complete_analysis_for_experiment(exp_dir):
    """
    Complete the analysis for a single experiment directory
    """
    print(f"\n{'='*60}")
    print(f"COMPLETING ANALYSIS FOR: {exp_dir}")
    print(f"{'='*60}")
    
    # Get experiment name from directory
    exp_name = os.path.basename(exp_dir)
    
    # Recreate config
    config = recreate_config_from_experiment(exp_dir)
    
    # Load model and trainer
    try:
        model, trainer, device = load_model_and_trainer_from_checkpoint(exp_dir, config)
        print("✅ Model and trainer loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Create plots
    try:
        create_and_save_topological_plots(trainer.train_history, exp_dir, exp_name)
        save_topological_training_summary(trainer.train_history, trainer, exp_dir, exp_name)
        print("✅ Plots and summary created successfully")
    except Exception as e:
        print(f"❌ Failed to create plots: {e}")
        return False
    
    # Load data and evaluate
    try:
        train_loader, val_loader, test_loader = load_data(config)
        print("✅ Data loaded successfully")
        
        results_dir = os.path.join(exp_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        evaluation_results = evaluate_model(
            model, train_loader, val_loader, test_loader, config, results_dir, device
        )
        print("✅ Model evaluation completed successfully")
        
    except Exception as e:
        print(f"❌ Failed to evaluate model: {e}")
        evaluation_results = None
    
    # Print key results if evaluation succeeded
    if evaluation_results and 'separation' in evaluation_results:
        separation = evaluation_results['separation']
        if 'error' not in separation:
            print(f"\nKey Results:")
            print(f"  Separation ratio: {separation['separation_ratio']:.2f}x")
            print(f"  Perfect separation: {separation['perfect_separation']}")
            if 'classification' in evaluation_results:
                print(f"  Classification accuracy: {evaluation_results['classification']['accuracy']:.4f}")
    
    # Analyze topological success
    try:
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
        
        # Check clustering results
        if evaluation_results and 'clustering' in evaluation_results:
            clustering_results = evaluation_results['clustering']
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
            'training_diagnosis': diagnosis if 'diagnosis' in locals() else None,
            'evaluation_results': evaluation_results,
            'experiment_metadata': {
                'completion_time': datetime.now().isoformat(),
                'experiment_directory': exp_dir,
                'approach': 'pure_topological_autoencoder',
                'analysis_completed_by': 'complete_topological_analysis.py'
            }
        }
        
        results_dir = os.path.join(exp_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        analysis_path = os.path.join(results_dir, 'final_analysis.json')
        
        with open(analysis_path, 'w') as f:
            # Convert any non-serializable objects
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
        print(f"❌ Failed to complete topological analysis: {e}")
    
    print(f"✅ Analysis completed for: {exp_dir}")
    return True


def main():
    """
    Main function to complete analysis for specified experiment directories
    """
    # Your experiment directory
    exp_dir = "entailment_surfaces/supervised_contrastive_autoencoder/experiments/moor_topo-contrastive_autoencoder_noattention_20250725_170549"
    
    # Check if directory exists
    if not os.path.exists(exp_dir):
        print(f"❌ Experiment directory not found: {exp_dir}")
        
        # List available experiments
        exp_base_dir = "entailment_surfaces/supervised_contrastive_autoencoder/experiments"
        if os.path.exists(exp_base_dir):
            print("\nAvailable experiment directories:")
            for d in sorted(os.listdir(exp_base_dir)):
                if os.path.isdir(os.path.join(exp_base_dir, d)):
                    print(f"  {d}")
        return
    
    # Complete analysis
    success = complete_analysis_for_experiment(exp_dir)
    
    if success:
        print(f"\n{'='*60}")
        print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Results saved to: {exp_dir}")
    else:
        print(f"\n{'='*60}")
        print("❌ ANALYSIS FAILED")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()