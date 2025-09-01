"""
Trainer for InfoNCE + Order Embeddings autoencoder
"""

import torch
import os
import json
from datetime import datetime


class InfoNCEOrderTrainer:
    """
    Clean trainer for InfoNCE + Order Embeddings approach
    """
    
    def __init__(self, model, loss_function, optimizer, device='cuda'):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        
        self.model.to(device)
        
        # Training history
        self.train_history = {
            'epoch': [],
            'train_total_loss': [],
            'train_infonce_loss': [],
            'train_order_loss': [],
            'train_topological_loss': [],
            'train_reconstruction_loss': [],
            'val_total_loss': [],
            'val_infonce_loss': [],
            'val_order_loss': [],
            'val_topological_loss': [],
            'val_reconstruction_loss': [],
            'learning_rate': [],
            'topological_applications': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        print(f"InfoNCEOrderTrainer initialized on {device}")
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'infonce_loss': 0.0,
            'order_loss': 0.0,
            'topological_loss': 0.0,
            'reconstruction_loss': 0.0
        }
        
        num_batches = len(train_loader)
        topological_applications = 0
        
        for batch_idx, batch in enumerate(train_loader):
            premise_embeddings = batch['premise_embedding'].to(self.device)
            hypothesis_embeddings = batch['hypothesis_embedding'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Concatenate premise and hypothesis embeddings for the model
            # Model expects 1536-dim input (768 + 768)
            premise_hyp_concat = torch.cat([premise_embeddings, hypothesis_embeddings], dim=1)
            
            # Forward pass - model processes concatenated embeddings
            latent_features, reconstructed = self.model(premise_hyp_concat)
            
            # Split the latent features back into premise and hypothesis representations
            batch_size = premise_embeddings.shape[0]
            latent_dim = latent_features.shape[1]
            
            # For loss computation, we treat the latent as representing the pair
            # We can split it or use it as is - let's split it for order embeddings
            premise_latent = latent_features  # Use full latent as premise representation
            hypothesis_latent = latent_features  # Use same for hypothesis (they represent the pair)
            
            # For reconstruction, split the output back to premise and hypothesis
            premise_reconstructed = reconstructed[:, :768]  # First 768 dims
            hypothesis_reconstructed = reconstructed[:, 768:]  # Last 768 dims
            
            # Compute loss
            total_loss, loss_components = self.loss_function(
                premise_latent, hypothesis_latent,
                premise_reconstructed, hypothesis_reconstructed,
                premise_embeddings, hypothesis_embeddings,
                labels
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            for key in epoch_metrics.keys():
                epoch_metrics[key] += loss_components[key]
            
            # Track topological applications
            if loss_components.get('topological_applied', False):
                topological_applications += 1
            
            # Print progress
            if batch_idx % 50 == 0:
                topo_status = " [TOPO]" if loss_components.get('topological_applied', False) else ""
                print(f"  Batch {batch_idx}/{num_batches}: Loss = {total_loss.item():.6f}{topo_status}")
        
        # Average metrics
        for key in epoch_metrics.keys():
            epoch_metrics[key] /= num_batches
        
        # Add topological application count
        epoch_metrics['topological_applications'] = topological_applications
        
        return epoch_metrics
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'infonce_loss': 0.0,
            'order_loss': 0.0,
            'topological_loss': 0.0,
            'reconstruction_loss': 0.0
        }
        
        num_batches = len(val_loader)
        topological_applications = 0
        
        with torch.no_grad():
            for batch in val_loader:
                premise_embeddings = batch['premise_embedding'].to(self.device)
                hypothesis_embeddings = batch['hypothesis_embedding'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Concatenate premise and hypothesis embeddings for the model
                premise_hyp_concat = torch.cat([premise_embeddings, hypothesis_embeddings], dim=1)
                
                # Forward pass
                latent_features, reconstructed = self.model(premise_hyp_concat)
                
                # Split latent and reconstructed outputs
                premise_latent = latent_features
                hypothesis_latent = latent_features
                premise_reconstructed = reconstructed[:, :768]
                hypothesis_reconstructed = reconstructed[:, 768:]
                
                # Compute loss
                total_loss, loss_components = self.loss_function(
                    premise_latent, hypothesis_latent,
                    premise_reconstructed, hypothesis_reconstructed,
                    premise_embeddings, hypothesis_embeddings,
                    labels
                )
                
                # Update metrics
                for key in epoch_metrics.keys():
                    epoch_metrics[key] += loss_components[key]
                
                # Track topological applications
                if loss_components.get('topological_applied', False):
                    topological_applications += 1
        
        # Average metrics
        for key in epoch_metrics.keys():
            epoch_metrics[key] /= num_batches
        
        # Add topological application count
        epoch_metrics['topological_applications'] = topological_applications
        
        return epoch_metrics
    
    def train(self, train_loader, val_loader, num_epochs, patience=15, save_dir=None):
        """Full training loop with global dataset updates"""
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Update global dataset every 3 epochs
            if epoch % 3 == 0:
                print(f"🌍 Updating global dataset at epoch {epoch + 1}")
                self.loss_function.update_global_dataset(train_loader, self.model, self.device)
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Update history
            self.train_history['epoch'].append(epoch)
            
            # Update standard loss components
            standard_metrics = ['total_loss', 'infonce_loss', 'order_loss', 'topological_loss', 'reconstruction_loss']
            for key in standard_metrics:
                if key in train_metrics:
                    self.train_history[f'train_{key}'].append(train_metrics[key])
                if key in val_metrics:
                    self.train_history[f'val_{key}'].append(val_metrics[key])
            
            # Update special metrics separately
            self.train_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.train_history['topological_applications'].append(train_metrics.get('topological_applications', 0))
            
            # Print epoch summary
            print(f"Train - Total: {train_metrics['total_loss']:.6f}, "
                  f"InfoNCE: {train_metrics['infonce_loss']:.6f}, "
                  f"Order: {train_metrics['order_loss']:.6f}, "
                  f"Topo: {train_metrics['topological_loss']:.6f}, "
                  f"Recon: {train_metrics['reconstruction_loss']:.6f}")
            print(f"Val   - Total: {val_metrics['total_loss']:.6f}, "
                  f"InfoNCE: {val_metrics['infonce_loss']:.6f}, "
                  f"Order: {val_metrics['order_loss']:.6f}, "
                  f"Topo: {val_metrics['topological_loss']:.6f}, "
                  f"Recon: {val_metrics['reconstruction_loss']:.6f}")
            
            # Print topological application info
            if train_metrics.get('topological_applications', 0) > 0:
                print(f"Topological regularization applied {train_metrics['topological_applications']} times this epoch")
            
            # Early stopping
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best model
                if save_dir:
                    self.save_model(save_dir, 'best_model.pt')
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement")
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch+1}")
        
        return self.train_history
    
    def save_model(self, save_dir, filename):
        """Save model checkpoint"""
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }
        torch.save(checkpoint, os.path.join(save_dir, filename))
        print(f"Model saved to {os.path.join(save_dir, filename)}")