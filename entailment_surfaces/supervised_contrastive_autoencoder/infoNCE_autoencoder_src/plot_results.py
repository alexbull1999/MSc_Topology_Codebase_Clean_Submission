"""
Plotting utilities for InfoNCE + Order Embeddings results
"""

import matplotlib.pyplot as plt
import json
import numpy as np
import os


def plot_training_history(history_path, save_dir):
    """
    Plot training history from JSON file
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = history['epoch']
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('InfoNCE + Order Embeddings Training Progress', fontsize=16)
    
    # Total Loss
    axes[0, 0].plot(epochs, history['train_total_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_total_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # InfoNCE Loss
    axes[0, 1].plot(epochs, history['train_infonce_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_infonce_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('InfoNCE Loss')
    axes[0, 1].set_title('InfoNCE Contrastive Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Order Loss
    axes[1, 0].plot(epochs, history['train_order_loss'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_order_loss'], 'r-', label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Order Loss')
    axes[1, 0].set_title('Order Embedding Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reconstruction Loss
    axes[1, 1].plot(epochs, history['train_reconstruction_loss'], 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs, history['val_reconstruction_loss'], 'r-', label='Validation', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Reconstruction Loss')
    axes[1, 1].set_title('Reconstruction Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to: {plot_path}")
    return plot_path


def compare_with_baseline(results_path, baseline_accuracy=0.8167):
    """
    Create comparison chart with baseline
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    metrics = ['clustering_accuracy', 'silhouette_score', 'separation_ratio']
    current_values = [results[metric] for metric in metrics]
    
    # For comparison, assume baseline values (adjust based on your actual baseline)
    baseline_values = [baseline_accuracy, 0.53, 3.55]  # From your project docs
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline (Distance-based)', color='lightcoral')
    bars2 = ax.bar(x + width/2, current_values, width, label='InfoNCE + Order Embeddings', color='skyblue')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('InfoNCE + Order Embeddings vs Baseline Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['Clustering Accuracy', 'Silhouette Score', 'Separation Ratio'])
    ax.legend()
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    plt.tight_layout()
    
    # Save comparison plot
    save_dir = os.path.dirname(results_path)
    plot_path = os.path.join(save_dir, 'baseline_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Baseline comparison plot saved to: {plot_path}")
    return plot_path