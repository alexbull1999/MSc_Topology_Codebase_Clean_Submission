"""
Training Pipeline Module for Supervised Contrastive Autoencoder
Handles model training, validation, and checkpoint management
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from contrastive_autoencoder_model import ContrastiveAutoencoder
from losses import CombinedLoss
from data_loader import EntailmentDataLoader


class ContrastiveAutoencoderTrainer:
    """
    Trainer for the supervised contrastive autoencoder
    """
    
    def __init__(self, model, loss_function, optimizer, device='cuda'):
        """
        Initialize trainer
        
        Args:
            model: ContrastiveAutoencoder instance
            loss_function: CombinedLoss instance
            optimizer: PyTorch optimizer
            device: Device to train on
        """
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Training history
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_contrastive_loss': [],
            'train_reconstruction_loss': [],
            'val_loss': [],
            'val_contrastive_loss': [],
            'val_reconstruction_loss': [],
            'learning_rate': [],
            # Distance debugging stats
            'train_pos_distances': [],
            'train_neg_distances': [],
            'train_separation_ratio': [],
            'val_pos_distances': [],
            'val_neg_distances': [],
            'val_separation_ratio': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        print(f"Trainer initialized on device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, train_loader, current_epoch, beta_config=None):
        """
        SIMPLIFIED: Train for one epoch - NO BETA SCHEDULING
        """
        self.model.train()
    
        epoch_losses = {
            'total_loss': 0.0,
            'contrastive_loss': 0.0,
            'reconstruction_loss': 0.0,
            'contrastive_weight': self.loss_function.contrastive_weight,
            'num_batches': 0
        }

        # Distance tracking for debugging
        all_pos_distances = []
        all_neg_distances = []
    
        for batch_idx, batch in enumerate(train_loader):
            embeddings = batch['embeddings'].to(self.device)
            labels = batch['labels'].to(self.device)
        
            # Forward pass
            latent, reconstructed = self.model(embeddings)
        
            # Compute loss - contrastive_weight is always 1.0
            total_loss, contrastive_loss, reconstruction_loss = self.loss_function(
                latent, labels, reconstructed, embeddings
            )

            if torch.isnan(total_loss):
                print(f"!!! NaN loss detected at batch {batch_idx}. Stopping epoch. !!!")
                exit(1)
        
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
            # Accumulate losses
            epoch_losses['total_loss'] += total_loss.item()
            epoch_losses['contrastive_loss'] += contrastive_loss.item()
            epoch_losses['reconstruction_loss'] += reconstruction_loss.item()
            epoch_losses['num_batches'] += 1
        
            # Print progress
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}: "
                    f"Loss = {total_loss.item():.4f} "
                    f"(C: {contrastive_loss.item():.4f}, "
                    f"R: {reconstruction_loss.item():.4f})")

            # Debug output every debug_frequency batches
            if batch_idx % 2000 == 0:
                # Get distance statistics
                with torch.no_grad():
                    distance_stats = self.loss_function.get_distance_stats(latent, labels)
                
                print(f"Batch {batch_idx}/{len(train_loader)}: Loss = {total_loss.item():.4f} "
                      f"(C: {contrastive_loss.item():.4f}, R: {reconstruction_loss.item():.4f})")
                
                # Print distance debug info
                if distance_stats['pos_mean'] > 0:  # Only if we have valid distances
                    print(f"  Distance Debug: Pos={distance_stats['pos_mean']:.2f}±{distance_stats['pos_std']:.2f}, "
                          f"Neg={distance_stats['neg_mean']:.2f}±{distance_stats['neg_std']:.2f}, "
                          f"Ratio={distance_stats['separation_ratio']:.2f}x, Gap={distance_stats['gap']:.2f}")
                    
                    # Track for epoch averages
                    all_pos_distances.append(distance_stats['pos_mean'])
                    all_neg_distances.append(distance_stats['neg_mean'])
    
        # Calculate average losses
        avg_losses = {
            'total_loss': epoch_losses['total_loss'] / epoch_losses['num_batches'],
            'contrastive_loss': epoch_losses['contrastive_loss'] / epoch_losses['num_batches'],
            'reconstruction_loss': epoch_losses['reconstruction_loss'] / epoch_losses['num_batches'],
            'contrastive_weight': epoch_losses['contrastive_weight']
        }

        # Add distance stats to history
        if all_pos_distances:
            avg_losses['avg_pos_distance'] = np.mean(all_pos_distances)
            avg_losses['avg_neg_distance'] = np.mean(all_neg_distances)
            avg_losses['avg_separation_ratio'] = np.mean(all_neg_distances) / np.mean(all_pos_distances)
        else:
            avg_losses['avg_pos_distance'] = 0.0
            avg_losses['avg_neg_distance'] = 0.0
            avg_losses['avg_separation_ratio'] = 0.0
    
        return avg_losses
    
    def validate_epoch(self, val_loader, contrastive_weight=None):
        """
        SIMPLIFIED: Validate for one epoch - NO DYNAMIC WEIGHTS
        """
        self.model.eval()

    
        epoch_losses = {
            'total_loss': 0.0,
            'contrastive_loss': 0.0,
            'reconstruction_loss': 0.0,
            'num_batches': 0
        }

        all_pos_distances = []
        all_neg_distances = []
    
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['labels'].to(self.device)
            
                # Forward pass
                latent, reconstructed = self.model(embeddings)
            
                # Compute loss - contrastive_weight is always 1.0
                total_loss, contrastive_loss, reconstruction_loss = self.loss_function(
                    latent, labels, reconstructed, embeddings
                )

                # Get distance statistics
                distance_stats = self.loss_function.get_distance_stats(latent, labels)
                if distance_stats['pos_mean'] > 0:
                    all_pos_distances.append(distance_stats['pos_mean'])
                    all_neg_distances.append(distance_stats['neg_mean'])

                if batch_idx == 0:
                    print(f"DEBUG: First validation batch:")
                    print(f"  contrastive_loss raw: {contrastive_loss.item()}")
                    print(f"  reconstruction_loss raw: {reconstruction_loss.item()}")
                    print(f"  total_loss raw: {total_loss.item()}")

                # Check for NaN immediately after loss calculation
                if torch.isnan(total_loss) or torch.isnan(contrastive_loss):
                    print(f"!!! NaN detected in validation at batch {batch_idx}!")
                    print(f"  Total loss: {total_loss.item()}")
                    print(f"  Contrastive loss: {contrastive_loss.item()}")
                    print(f"  Latent stats: min={latent.min().item():.6f}, max={latent.max().item():.6f}")
                    print(f"  Labels unique: {torch.unique(labels)}")
                    exit(1)
            
                # Accumulate losses
                epoch_losses['total_loss'] += total_loss.item()
                epoch_losses['contrastive_loss'] += contrastive_loss.item()
                epoch_losses['reconstruction_loss'] += reconstruction_loss.item()
                epoch_losses['num_batches'] += 1
    
        # Calculate average losses
        avg_losses = {
            'total_loss': epoch_losses['total_loss'] / epoch_losses['num_batches'],
            'contrastive_loss': epoch_losses['contrastive_loss'] / epoch_losses['num_batches'],
            'reconstruction_loss': epoch_losses['reconstruction_loss'] / epoch_losses['num_batches']
        }

        # Add distance stats
        if all_pos_distances:
            avg_losses['avg_pos_distance'] = np.mean(all_pos_distances)
            avg_losses['avg_neg_distance'] = np.mean(all_neg_distances)
            avg_losses['avg_separation_ratio'] = np.mean(all_neg_distances) / np.mean(all_pos_distances)
        else:
            avg_losses['avg_pos_distance'] = 0.0
            avg_losses['avg_neg_distance'] = 0.0
            avg_losses['avg_separation_ratio'] = 0.0
    
        return avg_losses
    
    def train(self, train_loader, val_loader, num_epochs=50, patience=10, 
          save_dir='checkpoints', save_every=5, beta_config=None):
        """
        SIMPLIFIED: Main training loop - NO BETA SCHEDULING
        """
        print("Starting training...")
        print("=" * 50)
    
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
        # Training loop
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
        
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 30)
        
            # Training phase - NO BETA CONFIG
            train_losses = self.train_epoch(train_loader, epoch, beta_config=None)
        
            # Validation phase - NO DYNAMIC WEIGHTS
            val_losses = self.validate_epoch(val_loader, contrastive_weight=1.0)
        
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
        
            # Print epoch summary
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            print(f"Train Loss: {train_losses['total_loss']:.4f} "
                f"(C: {train_losses['contrastive_loss']:.4f})")
            print(f"Val Loss: {val_losses['total_loss']:.4f} "
                f"(C: {val_losses['contrastive_loss']:.4f})")
        
            # Print distance separation stats
            if train_losses['avg_separation_ratio'] > 0:
                print(f"Distance Stats:")
                print(f"  Train: Pos={train_losses['avg_pos_distance']:.2f}, "
                      f"Neg={train_losses['avg_neg_distance']:.2f}, "
                      f"Ratio={train_losses['avg_separation_ratio']:.2f}x")
                print(f"  Val:   Pos={val_losses['avg_pos_distance']:.2f}, "
                      f"Neg={val_losses['avg_neg_distance']:.2f}, "
                      f"Ratio={val_losses['avg_separation_ratio']:.2f}x")
            
            # Update training history
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_losses['total_loss'])
            self.train_history['train_contrastive_loss'].append(train_losses['contrastive_loss'])
            self.train_history['train_reconstruction_loss'].append(train_losses['reconstruction_loss'])
            self.train_history['val_loss'].append(val_losses['total_loss'])
            self.train_history['val_contrastive_loss'].append(val_losses['contrastive_loss'])
            self.train_history['val_reconstruction_loss'].append(val_losses['reconstruction_loss'])
            self.train_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Add distance stats to history
            self.train_history['train_pos_distances'].append(train_losses['avg_pos_distance'])
            self.train_history['train_neg_distances'].append(train_losses['avg_neg_distance'])
            self.train_history['train_separation_ratio'].append(train_losses['avg_separation_ratio'])
            self.train_history['val_pos_distances'].append(val_losses['avg_pos_distance'])
            self.train_history['val_neg_distances'].append(val_losses['avg_neg_distance'])
            self.train_history['val_separation_ratio'].append(val_losses['avg_separation_ratio'])
        
            # Check for improvement
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.best_epoch = epoch + 1
                self.patience_counter = 0
            
                # Save best model
                self.save_checkpoint(save_dir, f'best_model.pt', epoch + 1, is_best=True)
                print(f"New best model saved (Val Loss: {self.best_val_loss:.4f})")
            else:
                self.patience_counter += 1
                print(f"No improvement for {self.patience_counter} epochs")
        
            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(save_dir, f'checkpoint_epoch_{epoch + 1}.pt', epoch + 1)
        
            # Early stopping
            if self.patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
        print("\nSimplified training completed!")
    
        # Save final model
        self.save_checkpoint(save_dir, 'final_model.pt', epoch + 1, is_final=True)
    
        # Save training history
        self.save_training_history(save_dir)
    
    def save_checkpoint(self, save_dir, filename, epoch, is_best=False, is_final=False):
        """
        Save model checkpoint
        
        Args:
            save_dir: Directory to save checkpoint
            filename: Checkpoint filename
            epoch: Current epoch
            is_best: Whether this is the best model
            is_final: Whether this is the final model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'train_history': self.train_history,
            'model_config': {
                'input_dim': self.model.input_dim,
                'latent_dim': self.model.latent_dim,
                'hidden_dims': self.model.hidden_dims,
                'dropout_rate': self.model.dropout_rate
            }
        }
        
        if is_best:
            checkpoint['is_best'] = True
        if is_final:
            checkpoint['is_final'] = True
        
        filepath = os.path.join(save_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def save_training_history(self, save_dir):
        """
        Save training history as JSON
        
        Args:
            save_dir: Directory to save history
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = os.path.join(save_dir, f'training_history_{timestamp}.json')
        
        with open(history_file, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        print(f"Training history saved: {history_file}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        self.train_history = checkpoint['train_history']
        
        print(f"Checkpoint loaded: Epoch {checkpoint['epoch']}, "
              f"Best Val Loss: {self.best_val_loss:.4f}")
        
        return checkpoint['epoch']


class FullDatasetTrainer:
    """
    Trainer that periodically updates global dataset for contrastive learning
    """
    
    def __init__(self, model, loss_function, optimizer, device='cuda'):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Training history
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_contrastive_loss': [],
            'train_reconstruction_loss': [],
            'val_loss': [],
            'val_contrastive_loss': [],
            'val_reconstruction_loss': [],
            'learning_rate': [],
            # Distance debugging stats
            'train_pos_distances': [],
            'train_neg_distances': [],
            'train_separation_ratio': [],
            'val_pos_distances': [],
            'val_neg_distances': [],
            'val_separation_ratio': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        print(f"FullDatasetTrainer initialized on device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, train_loader, current_epoch=0, debug_frequency=500):
        """
        Train epoch with periodic global dataset updates
        """
        self.model.train()
        
        # Update global dataset every few epochs
        if hasattr(self.loss_function, 'contrastive_loss') and \
           hasattr(self.loss_function.contrastive_loss, 'update_global_dataset'):
            if current_epoch % getattr(self.loss_function.contrastive_loss, 'update_frequency', 5) == 0:
                print(f"🌍 Updating global dataset at epoch {current_epoch}")
                self.loss_function.contrastive_loss.update_global_dataset(
                    train_loader, self.model, self.device
                )
        
        epoch_losses = {
            'total_loss': 0.0,
            'contrastive_loss': 0.0,
            'reconstruction_loss': 0.0,
            'num_batches': 0,
            'contrastive_weight': self.loss_function.contrastive_weight
        }
        
        # Distance tracking for debugging
        all_pos_distances = []
        all_neg_distances = []
        
        print(f"\nEpoch {current_epoch + 1}")
        print("-" * 30)
        
        for batch_idx, batch in enumerate(train_loader):
            embeddings = batch['embeddings'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            latent, reconstructed = self.model(embeddings)
            
            # Compute loss with global dataset
            total_loss, contrastive_loss, reconstruction_loss = self.loss_function(
                latent, labels, reconstructed, embeddings
            )
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            epoch_losses['total_loss'] += total_loss.item()
            epoch_losses['contrastive_loss'] += contrastive_loss.item()
            epoch_losses['reconstruction_loss'] += reconstruction_loss.item()
            epoch_losses['num_batches'] += 1
            
            # Debug output every debug_frequency batches
            if batch_idx % debug_frequency == 0:
                # Get distance statistics
                with torch.no_grad():
                    if hasattr(self.loss_function, 'get_distance_stats'):
                        distance_stats = self.loss_function.get_distance_stats(latent, labels)
                    else:
                        distance_stats = self._get_distance_stats(latent, labels)
                
                print(f"Batch {batch_idx}/{len(train_loader)}: Loss = {total_loss.item():.4f} "
                      f"(C: {contrastive_loss.item():.4f}, R: {reconstruction_loss.item():.4f})")
                
                # Print distance debug info
                if distance_stats['pos_mean'] > 0:
                    print(f"  Distance Debug: Pos={distance_stats['pos_mean']:.2f}±{distance_stats['pos_std']:.2f}, "
                          f"Neg={distance_stats['neg_mean']:.2f}±{distance_stats['neg_std']:.2f}, "
                          f"Ratio={distance_stats['separation_ratio']:.2f}x, Gap={distance_stats['gap']:.2f}")
                    
                    all_pos_distances.append(distance_stats['pos_mean'])
                    all_neg_distances.append(distance_stats['neg_mean'])
        
        # Calculate average losses
        avg_losses = {
            'total_loss': epoch_losses['total_loss'] / epoch_losses['num_batches'],
            'contrastive_loss': epoch_losses['contrastive_loss'] / epoch_losses['num_batches'],
            'reconstruction_loss': epoch_losses['reconstruction_loss'] / epoch_losses['num_batches'],
            'contrastive_weight': epoch_losses['contrastive_weight']
        }
        
        # Add distance stats to history
        if all_pos_distances:
            avg_losses['avg_pos_distance'] = np.mean(all_pos_distances)
            avg_losses['avg_neg_distance'] = np.mean(all_neg_distances)
            avg_losses['avg_separation_ratio'] = np.mean(all_neg_distances) / np.mean(all_pos_distances)
        else:
            avg_losses['avg_pos_distance'] = 0.0
            avg_losses['avg_neg_distance'] = 0.0
            avg_losses['avg_separation_ratio'] = 0.0
        
        return avg_losses
    
    def validate_epoch(self, val_loader):
        """
        Validate for one epoch with distance debugging
        """
        self.model.eval()
        
        epoch_losses = {
            'total_loss': 0.0,
            'contrastive_loss': 0.0,
            'reconstruction_loss': 0.0,
            'num_batches': 0
        }
        
        all_pos_distances = []
        all_neg_distances = []
        
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                latent, reconstructed = self.model(embeddings)
                
                # Compute loss (validation typically doesn't use global dataset for efficiency)
                total_loss, contrastive_loss, reconstruction_loss = self.loss_function(
                    latent, labels, reconstructed, embeddings
                )
                
                # Get distance statistics
                if hasattr(self.loss_function, 'get_distance_stats'):
                    distance_stats = self.loss_function.get_distance_stats(latent, labels)
                else:
                    distance_stats = self._get_distance_stats(latent, labels)
                
                if distance_stats['pos_mean'] > 0:
                    all_pos_distances.append(distance_stats['pos_mean'])
                    all_neg_distances.append(distance_stats['neg_mean'])
                
                # Accumulate losses
                epoch_losses['total_loss'] += total_loss.item()
                epoch_losses['contrastive_loss'] += contrastive_loss.item()
                epoch_losses['reconstruction_loss'] += reconstruction_loss.item()
                epoch_losses['num_batches'] += 1
        
        # Calculate average losses
        avg_losses = {
            'total_loss': epoch_losses['total_loss'] / epoch_losses['num_batches'],
            'contrastive_loss': epoch_losses['contrastive_loss'] / epoch_losses['num_batches'],
            'reconstruction_loss': epoch_losses['reconstruction_loss'] / epoch_losses['num_batches']
        }
        
        # Add distance stats
        if all_pos_distances:
            avg_losses['avg_pos_distance'] = np.mean(all_pos_distances)
            avg_losses['avg_neg_distance'] = np.mean(all_neg_distances)
            avg_losses['avg_separation_ratio'] = np.mean(all_neg_distances) / np.mean(all_pos_distances)
        else:
            avg_losses['avg_pos_distance'] = 0.0
            avg_losses['avg_neg_distance'] = 0.0
            avg_losses['avg_separation_ratio'] = 0.0
        
        return avg_losses
    
    def _get_distance_stats(self, latent_features, labels):
        """
        Get distance statistics for debugging
        """
        batch_size = latent_features.shape[0]
        device = latent_features.device
        
        distances = torch.cdist(latent_features, latent_features, p=2)
        
        labels_expanded = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels_expanded, labels_expanded.T).float().to(device)
        mask_no_diagonal = 1 - torch.eye(batch_size, device=device)
        mask_positive = mask_positive * mask_no_diagonal
        mask_negative = (1 - torch.eq(labels_expanded, labels_expanded.T).float().to(device)) * mask_no_diagonal
        
        pos_distances = distances[mask_positive.bool()]
        neg_distances = distances[mask_negative.bool()]
        
        if len(pos_distances) == 0 or len(neg_distances) == 0:
            return {
                'pos_mean': 0.0, 'pos_std': 0.0, 'pos_min': 0.0, 'pos_max': 0.0,
                'neg_mean': 0.0, 'neg_std': 0.0, 'neg_min': 0.0, 'neg_max': 0.0,
                'separation_ratio': 0.0, 'gap': 0.0
            }
        
        stats = {
            'pos_mean': pos_distances.mean().item(),
            'pos_std': pos_distances.std().item(),
            'pos_min': pos_distances.min().item(),
            'pos_max': pos_distances.max().item(),
            'neg_mean': neg_distances.mean().item(),
            'neg_std': neg_distances.std().item(),
            'neg_min': neg_distances.min().item(),
            'neg_max': neg_distances.max().item(),
            'separation_ratio': (neg_distances.mean() / pos_distances.mean()).item(),
            'gap': (neg_distances.min() - pos_distances.max()).item()
        }
        
        return stats
    
    def train(self, train_loader, val_loader, num_epochs=50, patience=10, 
              save_dir='checkpoints', save_every=5, debug_frequency=500):
        """
        Main training loop with enhanced debugging and global dataset updates
        """
        print("Starting FULL DATASET training...")
        print("=" * 50)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_losses = self.train_epoch(train_loader, epoch, debug_frequency)
            
            # Validation phase
            val_losses = self.validate_epoch(val_loader)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary with distance stats
            print(f"\nEpoch {epoch + 1} completed in {epoch_time:.2f}s")
            print(f"Train Loss: {train_losses['total_loss']:.4f} "
                  f"(C: {train_losses['contrastive_loss']:.4f}, "
                  f"R: {train_losses['reconstruction_loss']:.4f})")
            print(f"Val Loss: {val_losses['total_loss']:.4f} "
                  f"(C: {val_losses['contrastive_loss']:.4f}, "
                  f"R: {val_losses['reconstruction_loss']:.4f})")
            
            # Print distance separation stats
            if train_losses['avg_separation_ratio'] > 0:
                print(f"Distance Stats:")
                print(f"  Train: Pos={train_losses['avg_pos_distance']:.2f}, "
                      f"Neg={train_losses['avg_neg_distance']:.2f}, "
                      f"Ratio={train_losses['avg_separation_ratio']:.2f}x")
                print(f"  Val:   Pos={val_losses['avg_pos_distance']:.2f}, "
                      f"Neg={val_losses['avg_neg_distance']:.2f}, "
                      f"Ratio={val_losses['avg_separation_ratio']:.2f}x")
            
            # Update training history
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_losses['total_loss'])
            self.train_history['train_contrastive_loss'].append(train_losses['contrastive_loss'])
            self.train_history['train_reconstruction_loss'].append(train_losses['reconstruction_loss'])
            self.train_history['val_loss'].append(val_losses['total_loss'])
            self.train_history['val_contrastive_loss'].append(val_losses['contrastive_loss'])
            self.train_history['val_reconstruction_loss'].append(val_losses['reconstruction_loss'])
            self.train_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Add distance stats to history
            self.train_history['train_pos_distances'].append(train_losses['avg_pos_distance'])
            self.train_history['train_neg_distances'].append(train_losses['avg_neg_distance'])
            self.train_history['train_separation_ratio'].append(train_losses['avg_separation_ratio'])
            self.train_history['val_pos_distances'].append(val_losses['avg_pos_distance'])
            self.train_history['val_neg_distances'].append(val_losses['avg_neg_distance'])
            self.train_history['val_separation_ratio'].append(val_losses['avg_separation_ratio'])
            
            # Check for improvement
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(save_dir, f'best_model.pt', epoch + 1, is_best=True)
                print(f"✅ New best model saved (Val Loss: {self.best_val_loss:.4f})")
            else:
                self.patience_counter += 1
                print(f"No improvement for {self.patience_counter} epochs")
            
            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(save_dir, f'checkpoint_epoch_{epoch + 1}.pt', epoch + 1)
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                print(f"Best model was at epoch {self.best_epoch} with Val Loss: {self.best_val_loss:.4f}")
                break
        
        print("\nFull dataset training completed!")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Print final distance stats
        if len(self.train_history['train_separation_ratio']) > 0:
            final_train_ratio = self.train_history['train_separation_ratio'][-1]
            final_val_ratio = self.train_history['val_separation_ratio'][-1]
            print(f"Final separation ratios: Train={final_train_ratio:.2f}x, Val={final_val_ratio:.2f}x")
    
    def save_checkpoint(self, save_dir, filename, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'trainer_type': 'FullDatasetTrainer'
        }
        
        checkpoint_path = os.path.join(save_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            print(f"Best model checkpoint saved: {checkpoint_path}")
        
        # Save training history as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = os.path.join(save_dir, f'training_history_{timestamp}.json')
        
        with open(history_file, 'w') as f:
            json.dump(self.train_history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        self.train_history = checkpoint['train_history']
        
        print(f"Checkpoint loaded: Epoch {checkpoint['epoch']}, "
              f"Best Val Loss: {self.best_val_loss:.4f}")
        
        return checkpoint['epoch']



def create_trainer(model_config, loss_config, optimizer_config, device='cuda'):
    """
    Factory function to create trainer with all components
    
    Args:
        model_config: Dictionary with model configuration
        loss_config: Dictionary with loss configuration
        optimizer_config: Dictionary with optimizer configuration
        device: Device to use for training
        
    Returns:
        Configured trainer instance
    """
    # Create model
    model = ContrastiveAutoencoder(**model_config)
    
    # Create loss function
    loss_fn = CombinedLoss(**loss_config)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), **optimizer_config)
    
    # Create trainer
    trainer = ContrastiveAutoencoderTrainer(model, loss_fn, optimizer, device)
    
    return trainer
    


def test_trainer():
    """Test trainer functionality with mock components"""
    print("Testing Trainer Module")
    print("=" * 40)
    
    # Create model
    print("Creating model...")
    model = ContrastiveAutoencoder(
        input_dim=768,
        latent_dim=75,
        hidden_dims=[512, 256],
        dropout_rate=0.2
    )
    
    # Create loss function
    print("Creating loss function...")
    loss_fn = CombinedLoss(
        contrastive_weight=1.0,
        reconstruction_weight=1.0,
        temperature=0.1
    )
    
    # Create optimizer
    print("Creating optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create trainer
    print("Creating trainer...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = ContrastiveAutoencoderTrainer(
        model=model,
        loss_function=loss_fn,
        optimizer=optimizer,
        device=device
    )

    print(f"Trainer created successfully on {device}")
    
    # Test with synthetic data
    print("\nTesting with synthetic data...")
    batch_size = 32
    num_batches = 5
    
    # Create synthetic dataset
    synthetic_embeddings = torch.randn(batch_size * num_batches, 768)
    synthetic_labels = torch.randint(0, 3, (batch_size * num_batches,))
    
    # Create simple dataset and dataloader
    from torch.utils.data import TensorDataset, DataLoader
    
    synthetic_dataset = TensorDataset(synthetic_embeddings, synthetic_labels)
    synthetic_loader = DataLoader(synthetic_dataset, batch_size=batch_size, shuffle=True)
    
    # Convert to expected format
    class SyntheticDataset:
        def __init__(self, embeddings, labels):
            self.embeddings = embeddings
            self.labels = labels
        
        def __len__(self):
            return len(self.embeddings)
        
        def __getitem__(self, idx):
            return {
                'embeddings': self.embeddings[idx],
                'labels': self.labels[idx]
            }
    
    synthetic_dataset = SyntheticDataset(synthetic_embeddings, synthetic_labels)
    synthetic_loader = DataLoader(synthetic_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Created synthetic dataset with {len(synthetic_dataset)} samples")
    
    # Test single training epoch
    print("\nTesting single training epoch...")
    train_losses = trainer.train_epoch(synthetic_loader)
    
    print(f"Training losses: {train_losses}")
    
    # Test validation epoch
    print("\nTesting validation epoch...")
    val_losses = trainer.validate_epoch(synthetic_loader)
    
    print(f"Validation losses: {val_losses}")
    
    # Test checkpoint saving
    print("\nTesting checkpoint saving...")
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer.save_checkpoint(temp_dir, 'test_checkpoint.pt', epoch=1)
        
        # Test checkpoint loading
        print("Testing checkpoint loading...")
        loaded_epoch = trainer.load_checkpoint(os.path.join(temp_dir, 'test_checkpoint.pt'))
        print(f"Loaded checkpoint from epoch: {loaded_epoch}")
    
    # Test short training run
    print("\nTesting short training run...")
    trainer.train(
        train_loader=synthetic_loader,
        val_loader=synthetic_loader,
        num_epochs=3,
        patience=2,
        save_dir='test_checkpoints',
        save_every=2
    )
    
    print("\nTrainer testing completed successfully!")
    print("\nTraining history summary:")
    print(f"Epochs trained: {len(trainer.train_history['epoch'])}")
    print(f"Final train loss: {trainer.train_history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {trainer.train_history['val_loss'][-1]:.4f}")
    print(f"Best epoch: {trainer.best_epoch}")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    
    # Clean up test files
    import shutil
    if os.path.exists('test_checkpoints'):
        shutil.rmtree('test_checkpoints')
        print("Cleaned up test checkpoint directory")


if __name__ == "__main__":
    test_trainer()