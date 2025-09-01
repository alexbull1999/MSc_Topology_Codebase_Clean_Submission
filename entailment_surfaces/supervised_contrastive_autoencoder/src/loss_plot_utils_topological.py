"""
Simple plotting utilities for topological autoencoder loss visualization
"""

import matplotlib.pyplot as plt
import os
from pathlib import Path


def plot_topological_training_losses(train_history, save_dir, experiment_name="topological_experiment"):
    """
    Create and save loss plots from topological training history
    
    Args:
        train_history: Dictionary containing training metrics
        save_dir: Directory to save plots
        experiment_name: Name for the experiment (for file naming)
    """
    
    # Create plots directory
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set up the plots - 2x3 grid for topological version
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Topological Training Progress - {experiment_name}', fontsize=16)
    
    epochs = train_history['epoch']
    
    # Plot 1: Total Loss
    axes[0, 0].plot(epochs, train_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, train_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Contrastive Loss
    if 'train_contrastive_loss' in train_history and 'val_contrastive_loss' in train_history:
        axes[0, 1].plot(epochs, train_history['train_contrastive_loss'], 'b-', label='Train Contrastive', linewidth=2)
        axes[0, 1].plot(epochs, train_history['val_contrastive_loss'], 'r-', label='Val Contrastive', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Contrastive Loss')
        axes[0, 1].set_title('Contrastive Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Contrastive Loss\nNot Available', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Contrastive Loss')
    
    # Plot 3: Topological Loss (main difference from regular version)
    if 'train_topological_loss' in train_history and 'val_topological_loss' in train_history:
        axes[0, 2].plot(epochs, train_history['train_topological_loss'], 'g-', label='Train Topological', linewidth=2)
        axes[0, 2].plot(epochs, train_history['val_topological_loss'], 'orange', label='Val Topological', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Topological Loss')
        axes[0, 2].set_title('Topological Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    elif 'train_persistence_loss' in train_history and 'val_persistence_loss' in train_history:
        axes[0, 2].plot(epochs, train_history['train_persistence_loss'], 'g-', label='Train Persistence', linewidth=2)
        axes[0, 2].plot(epochs, train_history['val_persistence_loss'], 'orange', label='Val Persistence', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Persistence Loss')
        axes[0, 2].set_title('Persistence Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'Topological Loss\nNot Available', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Topological Loss')
    
    # Plot 4: Reconstruction Loss
    if 'train_reconstruction_loss' in train_history and 'val_reconstruction_loss' in train_history:
        axes[1, 0].plot(epochs, train_history['train_reconstruction_loss'], 'b-', label='Train Reconstruction', linewidth=2)
        axes[1, 0].plot(epochs, train_history['val_reconstruction_loss'], 'r-', label='Val Reconstruction', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Reconstruction Loss')
        axes[1, 0].set_title('Reconstruction Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Reconstruction Loss\nNot Available', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Reconstruction Loss')
    
    # Plot 5: Separation Ratio or Learning Rate
    if 'train_separation_ratio' in train_history and 'val_separation_ratio' in train_history:
        axes[1, 1].plot(epochs, train_history['train_separation_ratio'], 'b-', label='Train Separation', linewidth=2)
        axes[1, 1].plot(epochs, train_history['val_separation_ratio'], 'r-', label='Val Separation', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Separation Ratio')
        axes[1, 1].set_title('Distance Separation Ratio')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    elif 'learning_rate' in train_history:
        axes[1, 1].plot(epochs, train_history['learning_rate'], 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Separation Ratio\nNot Available', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Separation Ratio')
    
    # Plot 6: Topological Weight (if using curriculum learning)
    if 'topological_weight' in train_history:
        axes[1, 2].plot(epochs, train_history['topological_weight'], 'purple', linewidth=2, label='Topological Weight')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Weight')
        axes[1, 2].set_title('Topological Weight Schedule')
        axes[1, 2].grid(True, alpha=0.3)
    elif 'persistence_weight' in train_history:
        axes[1, 2].plot(epochs, train_history['persistence_weight'], 'purple', linewidth=2, label='Persistence Weight')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Weight')
        axes[1, 2].set_title('Persistence Weight Schedule')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        # Show topological metrics if available
        if 'total_persistence' in train_history:
            axes[1, 2].plot(epochs, train_history['total_persistence'], 'purple', linewidth=2)
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Total Persistence')
            axes[1, 2].set_title('Total Persistence')
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'Topological Metrics\nNot Available', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Topological Metrics')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, f'{experiment_name}_topological_training_losses.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Topological loss plots saved to: {plot_path}")
    return plot_path


def plot_topological_single_metric(train_history, metric_name, save_dir, experiment_name="topological_experiment"):
    """
    Plot a single metric from topological training history
    
    Args:
        train_history: Dictionary containing training metrics
        metric_name: Name of the metric to plot (without train_/val_ prefix)
        save_dir: Directory to save plots
        experiment_name: Name for the experiment
    """
    
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    epochs = train_history['epoch']
    train_key = f'train_{metric_name}'
    val_key = f'val_{metric_name}'
    
    if train_key in train_history:
        plt.plot(epochs, train_history[train_key], 'b-', label=f'Train {metric_name.title()}', linewidth=2)
    
    if val_key in train_history:
        plt.plot(epochs, train_history[val_key], 'r-', label=f'Val {metric_name.title()}', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f'{metric_name.replace("_", " ").title()} - {experiment_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(plots_dir, f'{experiment_name}_{metric_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"{metric_name} plot saved to: {plot_path}")
    return plot_path


def plot_curriculum_learning_analysis(train_history, save_dir, experiment_name="topological_experiment"):
    """
    Create specialized plots for curriculum learning analysis
    
    Args:
        train_history: Dictionary containing training metrics
        save_dir: Directory to save plots
        experiment_name: Name for the experiment
    """
    
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Curriculum Learning Analysis - {experiment_name}', fontsize=16)
    
    epochs = train_history['epoch']
    
    # Plot 1: Loss components evolution
    if all(key in train_history for key in ['train_contrastive_loss', 'train_topological_loss', 'train_reconstruction_loss']):
        axes[0].plot(epochs, train_history['train_contrastive_loss'], 'b-', label='Contrastive', linewidth=2)
        axes[0].plot(epochs, train_history['train_topological_loss'], 'g-', label='Topological', linewidth=2)
        axes[0].plot(epochs, train_history['train_reconstruction_loss'], 'orange', label='Reconstruction', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Components Evolution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')  # Log scale to see different magnitudes
    
    # Plot 2: Weight schedule
    if 'topological_weight' in train_history or 'persistence_weight' in train_history:
        weight_key = 'topological_weight' if 'topological_weight' in train_history else 'persistence_weight'
        axes[1].plot(epochs, train_history[weight_key], 'purple', linewidth=3)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Weight')
        axes[1].set_title('Topological Weight Schedule')
        axes[1].grid(True, alpha=0.3)
        axes[1].fill_between(epochs, train_history[weight_key], alpha=0.3, color='purple')
    
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, f'{experiment_name}_curriculum_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Curriculum learning analysis saved to: {plot_path}")
    return plot_path