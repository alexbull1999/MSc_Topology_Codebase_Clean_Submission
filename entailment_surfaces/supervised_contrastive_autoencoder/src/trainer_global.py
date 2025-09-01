"""
Global Dataset Trainer - Complete Fixed Version
Clean implementation for full dataset contrastive training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class GlobalDatasetTrainer:
    """
    Trainer for global dataset contrastive learning
    """
    
    def __init__(self, model, loss_function, optimizer, device='cuda'):
        """
        Initialize trainer
        
        Args:
            model: ContrastiveAutoencoder instance
            loss_function: FullDatasetCombinedLoss instance
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
            'train_separation_ratio': [],
            'val_separation_ratio': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        print(f"GlobalDatasetTrainer initialized on device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, train_loader, current_epoch=0, debug_frequency=50):
        """
        Train for one epoch with global dataset updates
        """
        self.model.train()
        
        # Update global dataset at specified intervals
        if (hasattr(self.loss_function, 'contrastive_loss') and 
           hasattr(self.loss_function.contrastive_loss, 'update_global_dataset') and 
           hasattr(self.loss_function, 'contrastive_weight') and self.loss_function.contrastive_weight > 0):
            update_freq = getattr(self.loss_function.contrastive_loss, 'update_frequency', 3)
            if current_epoch % update_freq == 0:
                print(f"\n🌍 Updating global dataset at epoch {current_epoch + 1}")
                self.loss_function.contrastive_loss.update_global_dataset(
                    train_loader, self.model, self.device
                )
        
        epoch_losses = {
            'total_loss': 0.0,
            'contrastive_loss': 0.0,
            'reconstruction_loss': 0.0,
            'num_batches': 0
        }
        
        separation_ratios = []
        
        print(f"\nEpoch {current_epoch + 1} Training")
        print("-" * 40)
        
        for batch_idx, batch in enumerate(train_loader):
            embeddings = batch['embeddings'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            latent, reconstructed = self.model(embeddings)
            
            # Compute loss
            total_loss, contrastive_loss, reconstruction_loss = self.loss_function(
                latent, labels, reconstructed, embeddings, current_epoch=current_epoch
            )
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            epoch_losses['total_loss'] += total_loss.item()
            epoch_losses['contrastive_loss'] += contrastive_loss.item()
            epoch_losses['reconstruction_loss'] += reconstruction_loss.item()
            epoch_losses['num_batches'] += 1
            
            # Debug output
            if batch_idx % debug_frequency == 0:
                # Get distance statistics
                with torch.no_grad():
                    stats = self.loss_function.get_distance_stats(latent, labels)
                    if stats['separation_ratio'] > 0:
                        separation_ratios.append(stats['separation_ratio'])
                
                print(f"Batch {batch_idx:3d}/{len(train_loader)}: "
                      f"Loss={total_loss.item():.4f} "
                      f"(C:{contrastive_loss.item():.4f}, R:{reconstruction_loss.item():.4f})")
        
        # Calculate average losses
        avg_losses = {
            'total_loss': epoch_losses['total_loss'] / epoch_losses['num_batches'],
            'contrastive_loss': epoch_losses['contrastive_loss'] / epoch_losses['num_batches'],
            'reconstruction_loss': epoch_losses['reconstruction_loss'] / epoch_losses['num_batches'],
            'avg_separation_ratio': np.mean(separation_ratios) if separation_ratios else 0.0
        }
        
        return avg_losses
    
    def validate_epoch(self, val_loader, current_epoch=0):
        """
        Validate for one epoch
        """
        self.model.eval()
        
        epoch_losses = {
            'total_loss': 0.0,
            'contrastive_loss': 0.0,
            'reconstruction_loss': 0.0,
            'num_batches': 0
        }
        
        separation_ratios = []
        
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                latent, reconstructed = self.model(embeddings)
                
                # Compute loss (validation uses current batch only for efficiency)
                total_loss, contrastive_loss, reconstruction_loss = self.loss_function(
                    latent, labels, reconstructed, embeddings, current_epoch=current_epoch
                )
                
                # Get distance statistics
                stats = self.loss_function.get_distance_stats(latent, labels)
                if stats['separation_ratio'] > 0:
                    separation_ratios.append(stats['separation_ratio'])
                
                # Accumulate losses
                epoch_losses['total_loss'] += total_loss.item()
                epoch_losses['contrastive_loss'] += contrastive_loss.item()
                epoch_losses['reconstruction_loss'] += reconstruction_loss.item()
                epoch_losses['num_batches'] += 1
        
        # Calculate average losses
        avg_losses = {
            'total_loss': epoch_losses['total_loss'] / epoch_losses['num_batches'],
            'contrastive_loss': epoch_losses['contrastive_loss'] / epoch_losses['num_batches'],
            'reconstruction_loss': epoch_losses['reconstruction_loss'] / epoch_losses['num_batches'],
            'avg_separation_ratio': np.mean(separation_ratios) if separation_ratios else 0.0
        }
        
        return avg_losses
    
    def train(self, train_loader, val_loader, num_epochs=70, patience=8, 
              save_dir='checkpoints', save_every=5, debug_frequency=50):
        """
        Main training loop
        """
        print("Starting Global Dataset Training...")
        print("=" * 60)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_losses = self.train_epoch(train_loader, epoch, debug_frequency)
            
            # Validation phase
            val_losses = self.validate_epoch(val_loader, epoch)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            current_recon_weight = self.loss_function.get_reconstruction_weight(epoch + 1)
            
            # Print epoch summary
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.1f}s")
            print(f"Train: Loss={train_losses['total_loss']:.4f} "
                  f"(C:{train_losses['contrastive_loss']:.4f}, R:{train_losses['reconstruction_loss']:.4f}) "
                  f"Ratio={train_losses['avg_separation_ratio']:.2f}x")
            print(f"Val:   Loss={val_losses['total_loss']:.4f} "
                  f"(C:{val_losses['contrastive_loss']:.4f}, R:{val_losses['reconstruction_loss']:.4f}) "
                  f"Ratio={val_losses['avg_separation_ratio']:.2f}x")
            print(f"Reconstruction weight: {current_recon_weight:.3f}")

            
            # Update training history
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_losses['total_loss'])
            self.train_history['train_contrastive_loss'].append(train_losses['contrastive_loss'])
            self.train_history['train_reconstruction_loss'].append(train_losses['reconstruction_loss'])
            self.train_history['val_loss'].append(val_losses['total_loss'])
            self.train_history['val_contrastive_loss'].append(val_losses['contrastive_loss'])
            self.train_history['val_reconstruction_loss'].append(val_losses['reconstruction_loss'])
            self.train_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.train_history['train_separation_ratio'].append(train_losses['avg_separation_ratio'])
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
                print(f"Checkpoint saved at epoch {epoch + 1}")
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best model was at epoch {self.best_epoch} with Val Loss: {self.best_val_loss:.4f}")
                break
            
            print(f"{'='*60}")
        
        print(f"\nGlobal Dataset Training Completed!")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Print final separation stats
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
            'trainer_type': 'GlobalDatasetTrainer'
        }
        
        checkpoint_path = os.path.join(save_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        # Save training history as JSON
        if is_best:
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
        
        print(f"Checkpoint loaded: Epoch {checkpoint['epoch']}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        
        return checkpoint['epoch']
    
    def get_latent_representations(self, dataloader):
        """
        Extract latent representations for entire dataset
        """
        self.model.eval()
        all_latents = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['labels']
                
                latent, _ = self.model(embeddings)
                
                all_latents.append(latent.cpu())
                all_labels.append(labels)
        
        return torch.cat(all_latents, dim=0), torch.cat(all_labels, dim=0)


def test_trainer():
    """Test trainer with synthetic data"""
    print("Testing GlobalDatasetTrainer...")
    
    # Import dependencies
    from contrastive_autoencoder_model_global import ContrastiveAutoencoder
    from losses_global import FullDatasetCombinedLoss
    
    # Create model and loss
    model = ContrastiveAutoencoder(input_dim=768, latent_dim=75)
    loss_fn = FullDatasetCombinedLoss(
        contrastive_weight=1.0,
        reconstruction_weight=0.0,
        margin=2.0,
        update_frequency=2
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = GlobalDatasetTrainer(model, loss_fn, optimizer, device)
    
    # Create synthetic data
    from torch.utils.data import DataLoader
    
    n_samples = 300
    embeddings = torch.randn(n_samples, 768)
    labels = torch.randint(0, 3, (n_samples,))
    
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
    
    synthetic_dataset = SyntheticDataset(embeddings, labels)
    dataloader = DataLoader(synthetic_dataset, batch_size=32, shuffle=True)
    
    print(f"Created synthetic dataset: {len(synthetic_dataset)} samples")
    
    # Test training for a few epochs
    print("Testing training loop...")
    trainer.train(
        train_loader=dataloader,
        val_loader=dataloader,
        num_epochs=3,
        patience=5,
        save_dir='test_checkpoints',
        debug_frequency=5
    )
    
    print("✅ Trainer test completed!")
    
    # Clean up
    import shutil
    if os.path.exists('test_checkpoints'):
        shutil.rmtree('test_checkpoints')


if __name__ == "__main__":
    test_trainer()