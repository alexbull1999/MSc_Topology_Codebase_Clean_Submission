# ============================================================================
# NEW FILE: trainer_topological.py
# ============================================================================

"""
TopologicalTrainer - Independent trainer specifically for topological autoencoder training
Clean implementation without inheritance, focused purely on topological learning
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


class TopologicalTrainer:
    """
    Independent trainer specifically for topological autoencoder training.
    Focused on topological learning with enhanced monitoring and debugging.
    """
    
    def __init__(self, model, loss_function, optimizer, device='cuda'):
        """
        Initialize topological trainer
        
        Args:
            model: ContrastiveAutoencoder instance
            loss_function: TopologicallyRegularizedCombinedLoss instance
            optimizer: PyTorch optimizer
            device: Device to train on
        """
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Training history focused on topological learning
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_contrastive_loss': [],
            'train_reconstruction_loss': [],
            'train_topological_loss': [],
            'val_loss': [],
            'val_contrastive_loss': [],
            'val_reconstruction_loss': [],
            'val_topological_loss': [],
            'learning_rate': [],
            'topological_weight': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Topological training specific tracking
        self.topological_milestones = {
            'first_nonzero_epoch': None,
            'first_nonzero_loss': None,
            'best_topological_loss': float('inf'),
            'epochs_with_topology': 0,
            'consecutive_topology_epochs': 0,
            'max_consecutive_topology': 0
        }
        
        print(f"TopologicalTrainer initialized on device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Enhanced with topological loss monitoring")
        
    def train_epoch(self, train_loader, current_epoch=0, debug_frequency=50):
        """
        Train for one epoch with enhanced topological debugging
        """
        self.model.train()
        
        # Set epoch for topological loss scheduling
        if hasattr(self.loss_function, 'set_epoch'):
            self.loss_function.set_epoch(current_epoch)

            # Update global dataset at specified intervals
        if (hasattr(self.loss_function, 'base_loss') and 
            hasattr(self.loss_function.base_loss, 'contrastive_loss') and
            hasattr(self.loss_function.base_loss.contrastive_loss, 'update_global_dataset') and
            hasattr(self.loss_function.base_loss, 'contrastive_weight') and self.loss_function.base_loss.contrastive_weight > 0):
            
            update_freq = getattr(self.loss_function.base_loss.contrastive_loss, 'update_frequency', 3)
            if current_epoch % update_freq == 0:
                print(f"\n🌍 Updating global dataset at epoch {current_epoch + 1}")
                self.loss_function.update_global_datasets(
                    train_loader, self.model, self.device
                )
        
        # Initialize epoch tracking
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_reconstruction_loss = 0.0
        total_topological_loss = 0.0
        topological_weight_used = 0.0
        
        num_batches = len(train_loader)
        batches_with_topology = 0
        
        # Print epoch header with topological status
        self._print_epoch_header(current_epoch, train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            embeddings = batch['embeddings'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            latent_features, reconstructed = self.model(embeddings)
            
            # Compute loss
            loss, loss_components = self.loss_function(
                latent_features, reconstructed, embeddings, labels
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            total_contrastive_loss += loss_components.get('contrastive_loss', 0.0)
            total_reconstruction_loss += loss_components.get('reconstruction_loss', 0.0)
            total_topological_loss += loss_components.get('topological_loss', 0.0)
            topological_weight_used = loss_components.get('topological_weight', 0.0)
            
            # Track topological learning
            if loss_components.get('topological_loss', 0.0) > 0:
                batches_with_topology += 1
            
            # Enhanced debug output
            if batch_idx % debug_frequency == 0:
                self._print_batch_debug(batch_idx, num_batches, loss, loss_components, 
                                      topological_weight_used)
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_reconstruction_loss = total_reconstruction_loss / num_batches
        avg_topological_loss = total_topological_loss / num_batches
        
        # Update topological milestones
        self._update_topological_milestones(current_epoch, avg_topological_loss)
        
        # Print epoch summary
        self._print_training_summary(current_epoch, avg_loss, avg_contrastive_loss, 
                                   avg_reconstruction_loss, avg_topological_loss, 
                                   topological_weight_used, batches_with_topology, num_batches)
        
        return {
            'total_loss': avg_loss,
            'contrastive_loss': avg_contrastive_loss,
            'reconstruction_loss': avg_reconstruction_loss,
            'topological_loss': avg_topological_loss,
            'topological_weight': topological_weight_used
        }
    
    def validate_epoch(self, val_loader, current_epoch=0):
        """
        Validate for one epoch with topological tracking
        """
        self.model.eval()
        
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_reconstruction_loss = 0.0
        total_topological_loss = 0.0
        topological_weight_used = 0.0
        
        num_batches = len(val_loader)
        batches_with_topology = 0
        
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                latent_features, reconstructed = self.model(embeddings)
                
                # Compute loss
                loss, loss_components = self.loss_function(
                    latent_features, reconstructed, embeddings, labels
                )
                
                # Track losses
                total_loss += loss.item()
                total_contrastive_loss += loss_components.get('contrastive_loss', 0.0)
                total_reconstruction_loss += loss_components.get('reconstruction_loss', 0.0)
                total_topological_loss += loss_components.get('topological_loss', 0.0)
                topological_weight_used = loss_components.get('topological_weight', 0.0)
                
                if loss_components.get('topological_loss', 0.0) > 0:
                    batches_with_topology += 1
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_reconstruction_loss = total_reconstruction_loss / num_batches
        avg_topological_loss = total_topological_loss / num_batches
        
        # Print validation summary
        self._print_validation_summary(avg_loss, avg_contrastive_loss, 
                                     avg_reconstruction_loss, avg_topological_loss, 
                                     topological_weight_used, batches_with_topology, num_batches)
        
        return {
            'total_loss': avg_loss,
            'contrastive_loss': avg_contrastive_loss,
            'reconstruction_loss': avg_reconstruction_loss,
            'topological_loss': avg_topological_loss,
            'topological_weight': topological_weight_used
        }
    
    def train(self, train_loader, val_loader, num_epochs, patience=10, save_dir='checkpoints', 
              save_every=5, debug_frequency=50):
        """
        Main training loop with enhanced topological monitoring
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("🧠 TOPOLOGICAL AUTOENCODER TRAINING STARTED")
        print("="*70)
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_losses = self.train_epoch(train_loader, epoch, debug_frequency)
            
            # Validation  
            val_losses = self.validate_epoch(val_loader, epoch)
            
            # Update training history
            self._update_training_history(epoch, train_losses, val_losses)
            
            # Check for improvement
            #RESET PATIENCE FOR TOPOLOGICAL WARMUP/CURRICULUM LEARNING
            if epoch == 11:
                self.best_val_loss = float('inf') #Reset patience
 
    
            is_best = val_losses['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total_loss']
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best model
                self._save_checkpoint(epoch, save_dir, 'best_model.pt', train_losses, val_losses)
                print("✅ New best model saved!")
            else:
                self.patience_counter += 1
            
            # Save periodic checkpoints
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(epoch, save_dir, f'checkpoint_epoch_{epoch+1}.pt', 
                                    train_losses, val_losses)
            
            # Print epoch completion summary
            epoch_time = time.time() - epoch_start_time
            self._print_epoch_completion(epoch, num_epochs, epoch_time, train_losses, 
                                       val_losses, is_best)
            
            # Early stopping check
            if self.patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
                print(f"Best model was at epoch {self.best_epoch+1} with Val Loss: {self.best_val_loss:.4f}")
                break
        
        # Print final topological analysis
        self._print_final_topological_analysis()
        
        print("\n" + "="*70)
        print("🎯 TOPOLOGICAL AUTOENCODER TRAINING COMPLETED")
        print("="*70)
    
    def _print_epoch_header(self, epoch, train_loader):
        """Print epoch header with topological status"""
        current_topo_weight = 0.0
        if hasattr(self.loss_function, 'get_current_topological_weight'):
            current_topo_weight = self.loss_function.get_current_topological_weight()
        
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1} | Batches: {len(train_loader)} | Topological Weight: {current_topo_weight:.4f}")
        
        if current_topo_weight == 0:
            print("🔄 Topological warmup phase")
        elif current_topo_weight < 0.05:
            print("🌱 Early topological learning")
        else:
            print("🧠 Full topological learning active")
        print("="*60)
    
    def _print_batch_debug(self, batch_idx, num_batches, loss, loss_components, topo_weight):
        """Print enhanced batch debug information"""
        # Topological status emoji
        topo_loss = loss_components.get('topological_loss', 0.0)
        if topo_loss == 0:
            topo_status = "❌"
        elif topo_loss < 1.0:
            topo_status = "🎉"
        elif topo_loss < 10.0:
            topo_status = "🚀"
        else:
            topo_status = "⚠️"
        
        print(f"Batch {batch_idx:3d}/{num_batches}: "
              f"Loss={loss.item():.4f} "
              f"(C:{loss_components.get('contrastive_loss', 0.0):.4f}, "
              f"R:{loss_components.get('reconstruction_loss', 0.0):.4f}, "
              f"T:{topo_loss:.4f}(w:{topo_weight:.3f}){topo_status})")
    
    def _print_training_summary(self, epoch, avg_loss, avg_c, avg_r, avg_t, topo_weight, 
                               batches_with_topo, total_batches):
        """Print training epoch summary"""
        print(f"\n📊 EPOCH {epoch+1} TRAINING SUMMARY:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Contrastive: {avg_c:.4f}")
        print(f"  Reconstruction: {avg_r:.4f}")
        print(f"  Topological: {avg_t:.4f} (weight: {topo_weight:.3f})")
        print(f"  Batches with topology: {batches_with_topo}/{total_batches} ({100*batches_with_topo/total_batches:.1f}%)")
    
    def _print_validation_summary(self, avg_loss, avg_c, avg_r, avg_t, topo_weight, 
                                 batches_with_topo, total_batches):
        """Print validation summary"""
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Contrastive: {avg_c:.4f}")
        print(f"  Reconstruction: {avg_r:.4f}")
        print(f"  Topological: {avg_t:.4f} (weight: {topo_weight:.3f})")
        print(f"  Batches with topology: {batches_with_topo}/{total_batches} ({100*batches_with_topo/total_batches:.1f}%)")
    
    def _print_epoch_completion(self, epoch, num_epochs, epoch_time, train_losses, val_losses, is_best):
        """Print epoch completion summary"""
        print(f"\n🎯 EPOCH {epoch+1}/{num_epochs} COMPLETE ({epoch_time:.1f}s)")
        print(f"Train Loss: {train_losses['total_loss']:.4f} "
              f"(C:{train_losses['contrastive_loss']:.4f}, "
              f"R:{train_losses['reconstruction_loss']:.4f}, "
              f"T:{train_losses['topological_loss']:.4f})")
        print(f"Val Loss:   {val_losses['total_loss']:.4f} "
              f"(C:{val_losses['contrastive_loss']:.4f}, "
              f"R:{val_losses['reconstruction_loss']:.4f}, "
              f"T:{val_losses['topological_loss']:.4f})")
        
        # Topological progress indicator
        if train_losses['topological_loss'] > 0:
            if train_losses['topological_loss'] < 1.0:
                print("🎉 Topological complexity detected!")
            elif train_losses['topological_loss'] < 10.0:
                print("🚀 Good topological learning progress")
            else:
                print("⚠️  High topological loss - may need adjustment")
        else:
            print("❌ No topological learning yet")
        
        if is_best:
            print("⭐ Best model so far!")
        
        print("-" * 60)
    
    def _update_topological_milestones(self, epoch, avg_topological_loss):
        """Update topological learning milestones"""
        if avg_topological_loss > 0:
            # First time detecting topology
            if self.topological_milestones['first_nonzero_epoch'] is None:
                self.topological_milestones['first_nonzero_epoch'] = epoch
                self.topological_milestones['first_nonzero_loss'] = avg_topological_loss
                print(f"🎉 MILESTONE: First topological learning detected at epoch {epoch+1}!")
                print(f"   Initial topological loss: {avg_topological_loss:.4f}")
            
            # Track consecutive epochs with topology
            self.topological_milestones['epochs_with_topology'] += 1
            self.topological_milestones['consecutive_topology_epochs'] += 1
            
            # Update max consecutive
            if self.topological_milestones['consecutive_topology_epochs'] > self.topological_milestones['max_consecutive_topology']:
                self.topological_milestones['max_consecutive_topology'] = self.topological_milestones['consecutive_topology_epochs']
            
            # Track best topological loss
            if avg_topological_loss < self.topological_milestones['best_topological_loss']:
                self.topological_milestones['best_topological_loss'] = avg_topological_loss
                print(f"📈 New best topological loss: {avg_topological_loss:.4f}")
        else:
            # Reset consecutive counter if no topology this epoch
            self.topological_milestones['consecutive_topology_epochs'] = 0
    
    def _update_training_history(self, epoch, train_losses, val_losses):
        """Update training history with topological tracking"""
        self.train_history['epoch'].append(epoch)
        self.train_history['train_loss'].append(train_losses['total_loss'])
        self.train_history['train_contrastive_loss'].append(train_losses['contrastive_loss'])
        self.train_history['train_reconstruction_loss'].append(train_losses['reconstruction_loss'])
        self.train_history['train_topological_loss'].append(train_losses['topological_loss'])
        
        self.train_history['val_loss'].append(val_losses['total_loss'])
        self.train_history['val_contrastive_loss'].append(val_losses['contrastive_loss'])
        self.train_history['val_reconstruction_loss'].append(val_losses['reconstruction_loss'])
        self.train_history['val_topological_loss'].append(val_losses['topological_loss'])
        
        self.train_history['topological_weight'].append(train_losses['topological_weight'])
        
        # Store current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.train_history['learning_rate'].append(current_lr)
    
    def _save_checkpoint(self, epoch, save_dir, filename, train_losses, val_losses):
        """Save checkpoint with topological information"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'train_history': self.train_history,
            'topological_milestones': self.topological_milestones,
            'current_train_losses': train_losses,
            'current_val_losses': val_losses
        }
        
        torch.save(checkpoint, os.path.join(save_dir, filename))
    
    def _print_final_topological_analysis(self):
        """Print final analysis of topological learning"""
        print("\n" + "="*70)
        print("📈 FINAL TOPOLOGICAL LEARNING ANALYSIS")
        print("="*70)
        
        milestones = self.topological_milestones
        topo_losses = self.train_history['train_topological_loss']
        total_epochs = len(topo_losses)
        
        print(f"First topological learning: Epoch {milestones['first_nonzero_epoch']+1 if milestones['first_nonzero_epoch'] is not None else 'Never'}")
        print(f"Epochs with topology: {milestones['epochs_with_topology']}/{total_epochs}")
        print(f"Max consecutive topology epochs: {milestones['max_consecutive_topology']}")
        print(f"Best topological loss: {milestones['best_topological_loss']:.4f}")
        print(f"Final topological loss: {topo_losses[-1]:.4f}")
        
        # Success assessment
        if milestones['first_nonzero_epoch'] is not None:
            print("✅ SUCCESS: Topological learning achieved!")
            
            # Detailed success analysis
            topology_percentage = milestones['epochs_with_topology'] / total_epochs
            if topology_percentage > 0.8:
                print("🚀 EXCELLENT: Very consistent topological learning (>80%)")
            elif topology_percentage > 0.5:
                print("👍 GOOD: Fairly consistent topological learning (>50%)")
            elif topology_percentage > 0.2:
                print("⚠️  PARTIAL: Some topological learning but inconsistent (>20%)")
            else:
                print("🔍 MINIMAL: Very limited topological learning (<20%)")
                
            # Check if learning is stable
            if milestones['max_consecutive_topology'] > 10:
                print("📈 Topological learning appears stable")
            else:
                print("📊 Topological learning appears unstable")
        else:
            print("❌ FAILURE: No topological learning detected")
            print("   Recommendations:")
            print("   - Increase topological weight")
            print("   - Check TorchPH implementation")
            print("   - Verify prototypes are loaded correctly")
            print("   - Try longer warmup period")
    
    def diagnose_topological_progress(self):
        """
        Analyze topological learning progress from training history.
        """
        if 'train_topological_loss' not in self.train_history:
            print("No topological loss history available")
            return None
        
        topo_losses = self.train_history['train_topological_loss']
        topo_weights = self.train_history['topological_weight']
        
        print("\n📈 TOPOLOGICAL LEARNING DIAGNOSIS:")
        print(f"  Total epochs: {len(topo_losses)}")
        print(f"  Epochs with topological learning: {sum(1 for x in topo_losses if x > 0)}")
        print(f"  Current topological loss: {topo_losses[-1]:.4f}")
        print(f"  Current topological weight: {topo_weights[-1]:.4f}")
        
        # Check for progress trend
        recent_losses = topo_losses[-5:] if len(topo_losses) >= 5 else topo_losses
        if all(x > 0 for x in recent_losses):
            if recent_losses[-1] < recent_losses[0]:
                print("  ✅ Topological loss is decreasing (good progress)")
            else:
                print("  ⚠️  Topological loss is increasing (may need tuning)")
        else:
            print("  ❌ Topological loss still zero (no structure learned yet)")
        
        return {
            'epochs_with_topology': sum(1 for x in topo_losses if x > 0),
            'total_epochs': len(topo_losses),
            'current_loss': topo_losses[-1],
            'current_weight': topo_weights[-1],
            'milestones': self.topological_milestones,
            'topology_percentage': sum(1 for x in topo_losses if x > 0) / len(topo_losses) if topo_losses else 0
        }
    
    def load_checkpoint(self, checkpoint_path):
        """Load a training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model and optimizer state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        self.train_history = checkpoint['train_history']
        
        # Load topological milestones if available
        if 'topological_milestones' in checkpoint:
            self.topological_milestones = checkpoint['topological_milestones']
        
        print(f"Checkpoint loaded: Epoch {checkpoint['epoch']}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        
        return checkpoint['epoch']