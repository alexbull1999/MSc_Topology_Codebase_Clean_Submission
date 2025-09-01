"""
Simple plotting utilities for loss visualization
"""

import matplotlib.pyplot as plt
import os
from pathlib import Path


def plot_training_losses(train_history, save_dir, experiment_name="experiment"):
    """
    Create and save loss plots from training history
    
    Args:
        train_history: Dictionary containing training metrics
        save_dir: Directory to save plots
        experiment_name: Name for the experiment (for file naming)
    """
    
    # Create plots directory
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set up the plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Progress - {experiment_name}', fontsize=16)
    
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
    axes[0, 1].plot(epochs, train_history['train_contrastive_loss'], 'b-', label='Train Contrastive', linewidth=2)
    axes[0, 1].plot(epochs, train_history['val_contrastive_loss'], 'r-', label='Val Contrastive', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Contrastive Loss')
    axes[0, 1].set_title('Contrastive Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Reconstruction Loss
    axes[1, 0].plot(epochs, train_history['train_reconstruction_loss'], 'b-', label='Train Reconstruction', linewidth=2)
    axes[1, 0].plot(epochs, train_history['val_reconstruction_loss'], 'r-', label='Val Reconstruction', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Reconstruction Loss')
    axes[1, 0].set_title('Reconstruction Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Separation Ratio
    if 'train_separation_ratio' in train_history and 'val_separation_ratio' in train_history:
        axes[1, 1].plot(epochs, train_history['train_separation_ratio'], 'b-', label='Train Separation', linewidth=2)
        axes[1, 1].plot(epochs, train_history['val_separation_ratio'], 'r-', label='Val Separation', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Separation Ratio')
        axes[1, 1].set_title('Distance Separation Ratio')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # If no separation ratio data, plot learning rate
        axes[1, 1].plot(epochs, train_history['learning_rate'], 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, f'{experiment_name}_training_losses.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss plots saved to: {plot_path}")
    return plot_path


def plot_single_metric(train_history, metric_name, save_dir, experiment_name="experiment"):
    """
    Plot a single metric from training history
    
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