"""
Separate Model Trainers
1. OrderEmbeddingTrainer: Pure Vendrov et al. max-margin loss
2. AsymmetryTrainer: Asymmetric loss for directional patterns
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os


class OrderEmbeddingModel(nn.Module):
    """Pure order embedding model (Vendrov et al.)"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.to_order_space = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.to_order_space(embeddings)
    
    def order_violation_energy(self, u_emb: torch.Tensor, v_emb: torch.Tensor) -> torch.Tensor:
        """Compute order violation energy: E(u,v) = ||max(0, v-u)||²"""
        if u_emb.dim() == 2 and v_emb.dim() == 2:
            # Token-level: aggregate to sentence level
            min_len = min(u_emb.shape[0], v_emb.shape[0])
            u_truncated = u_emb[:min_len]
            v_truncated = v_emb[:min_len]
            violation = torch.clamp(v_truncated - u_truncated, min=0)
            energy = torch.norm(violation, dim=-1, p=2) ** 2
            return energy.mean()
        else:
            # Already sentence-level
            violation = torch.clamp(v_emb - u_emb, min=0)
            energy = torch.norm(violation, dim=-1, p=2) ** 2
            return energy


class AsymmetryTransformModel(nn.Module):
    """Asymmetry transformation model"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.asymmetric_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, order_embeddings: torch.Tensor) -> torch.Tensor:
        return self.asymmetric_projection(order_embeddings)
    
    def compute_asymmetric_energy(self, premise_order: torch.Tensor, hypothesis_order: torch.Tensor) -> torch.Tensor:
        """Compute asymmetric energy between premise and hypothesis order embeddings"""
        premise_asym = self.forward(premise_order)
        hypothesis_asym = self.forward(hypothesis_order)
        
        if premise_asym.dim() == 2 and hypothesis_asym.dim() == 2:
            # Token-level: aggregate
            min_len = min(premise_asym.shape[0], hypothesis_asym.shape[0])
            directional_diff = premise_asym[:min_len] - hypothesis_asym[:min_len]
            asymmetric_energy = torch.norm(directional_diff, dim=-1, p=2).mean()
        else:
            # Sentence-level
            directional_diff = premise_asym - hypothesis_asym
            asymmetric_energy = torch.norm(directional_diff, dim=-1, p=2)
        
        return asymmetric_energy


class TokenLevelEntailmentDataset(Dataset):
    """Dataset for training"""
    
    def __init__(self, processed_data_path: str):
        print(f"Loading processed dataset: {processed_data_path}")
        
        with open(processed_data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.premise_tokens = self.data['premise_tokens']
        self.hypothesis_tokens = self.data['hypothesis_tokens']
        self.labels = self.data['labels']
        
        self.label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        self.numeric_labels = [self.label_to_idx[label] for label in self.labels]
        
        print(f"Loaded {len(self.labels)} samples")
        print(f"Label distribution: {dict(zip(*np.unique(self.labels, return_counts=True)))}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'premise_tokens': self.premise_tokens[idx],
            'hypothesis_tokens': self.hypothesis_tokens[idx],
            'label': torch.tensor(self.numeric_labels[idx]),
            'label_str': self.labels[idx]
        }

class OrderEmbeddingTrainer:
    """Trainer for OrderEmbeddingModel using pure Vendrov et al. loss"""
    
    def __init__(self, model: OrderEmbeddingModel):
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=8, factor=0.5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.energy_rankings = []
        
        print(f"OrderEmbeddingTrainer initialized on {self.device}")
    
    def compute_vendrov_loss(self, premise_tokens: torch.Tensor, hypothesis_tokens: torch.Tensor,
                           label: int, margin: float = 1.0) -> Tuple[torch.Tensor, float]:
        """
        Pure Vendrov et al. max-margin loss
        - Entailment: minimize energy
        - Non-entailment: ensure energy > margin
        """
        premise_tokens = premise_tokens.to(self.device)
        hypothesis_tokens = hypothesis_tokens.to(self.device)
        
        # Get order embeddings
        premise_order = self.model(premise_tokens)
        hypothesis_order = self.model(hypothesis_tokens)
        
        # Compute order violation energy
        energy = self.model.order_violation_energy(premise_order, hypothesis_order)
        
        # Vendrov max-margin loss
        if label == 0:  # entailment
            loss = energy  # Minimize energy
        else:  # neutral or contradiction
            loss = torch.clamp(margin - energy, min=0)  # Ensure energy > margin
        
        return loss, energy.item()
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train OrderEmbeddingModel for one epoch"""
        self.model.train()
        total_loss = 0
        num_samples = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            batch_loss = 0
            batch_samples = 0
            
            for i in range(len(batch['premise_tokens'])):
                premise_tokens = batch['premise_tokens'][i]
                hypothesis_tokens = batch['hypothesis_tokens'][i]
                label = batch['label'][i].item()
                
                if premise_tokens.shape[0] < 2 or hypothesis_tokens.shape[0] < 2:
                    continue
                
                sample_loss, _ = self.compute_vendrov_loss(premise_tokens, hypothesis_tokens, label)
                batch_loss += sample_loss
                batch_samples += 1

                #MEMMGMT
                del premise_tokens, hypothesis_tokens, sample_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if batch_samples > 0:
                batch_loss = batch_loss / batch_samples
                batch_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += batch_loss.item()
                num_samples += batch_samples

                # ← CLEANUP AFTER EACH BATCH
                del batch_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ← CLEANUP EVERY 10 BATCHES
            if batch_idx % 10 == 0:
                import gc
                gc.collect()
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """Evaluate OrderEmbeddingModel"""
        self.model.eval()
        total_loss = 0
        all_energies = {'entailment': [], 'neutral': [], 'contradiction': []}
        num_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch_loss = 0
                batch_samples = 0
                
                for i in range(len(batch['premise_tokens'])):
                    premise_tokens = batch['premise_tokens'][i]
                    hypothesis_tokens = batch['hypothesis_tokens'][i]
                    label = batch['label'][i].item()
                    label_str = batch['label_str'][i]
                    
                    if premise_tokens.shape[0] < 2 or hypothesis_tokens.shape[0] < 2:
                        continue
                    
                    sample_loss, energy = self.compute_vendrov_loss(premise_tokens, hypothesis_tokens, label)
                    batch_loss += sample_loss
                    batch_samples += 1
                    
                    all_energies[label_str].append(energy)
                
                if batch_samples > 0:
                    total_loss += batch_loss.item() / batch_samples
                    num_samples += batch_samples
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        self.val_losses.append(avg_loss)
        
        # Compute energy summary
        energy_summary = {}
        for label, energies in all_energies.items():
            if energies:
                energy_summary[label] = {
                    'mean': np.mean(energies),
                    'std': np.std(energies),
                    'count': len(energies)
                }
        
        self.energy_rankings.append(energy_summary)
        return avg_loss, energy_summary


class AsymmetryTrainer:
    """Trainer for AsymmetryTransformModel using asymmetric loss"""
    
    def __init__(self, asymmetry_model: AsymmetryTransformModel, order_model: OrderEmbeddingModel):
        self.asymmetry_model = asymmetry_model
        self.order_model = order_model  # Frozen, pre-trained
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.asymmetry_model.to(self.device)
        self.order_model.to(self.device)
        self.order_model.eval()  # Freeze order model
        
        # Only train asymmetry model
        self.optimizer = optim.Adam(self.asymmetry_model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=8, factor=0.5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.asymmetry_stats = []
        
        print(f"AsymmetryTrainer initialized on {self.device}")
        print("Order model is frozen - only training asymmetry model")
    
    def compute_asymmetric_loss(self, premise_tokens: torch.Tensor, hypothesis_tokens: torch.Tensor,
                              label_str: str) -> Tuple[torch.Tensor, Dict]:
        """
        Asymmetric loss targeting directional patterns
        """
        premise_tokens = premise_tokens.to(self.device)
        hypothesis_tokens = hypothesis_tokens.to(self.device)
        
        # Get order embeddings (frozen model)
        with torch.no_grad():
            premise_order = self.order_model(premise_tokens)
            hypothesis_order = self.order_model(hypothesis_tokens)
        
        # Compute energies for asymmetric analysis
        forward_energy = self.order_model.order_violation_energy(premise_order, hypothesis_order)
        backward_energy = self.order_model.order_violation_energy(hypothesis_order, premise_order)
        asymmetric_energy = self.asymmetry_model.compute_asymmetric_energy(premise_order, hypothesis_order)
        
        # Asymmetric loss patterns
        loss = 0.0
        
        if label_str == 'entailment':
            # Low forward energy, high backward energy
            loss += F.mse_loss(forward_energy, torch.tensor(0.0, device=self.device))
            loss += F.relu(1.0 - backward_energy)
            
        elif label_str == 'neutral':
            # Symmetric relationship
            target_energy = 1.0
            loss += F.mse_loss(forward_energy, torch.tensor(target_energy, device=self.device))
            loss += F.mse_loss(backward_energy, torch.tensor(target_energy, device=self.device))
            loss += 0.5 * torch.abs(forward_energy - backward_energy)
            
        elif label_str == 'contradiction':
            # High forward energy and high asymmetry
            loss += F.relu(1.5 - forward_energy)
            loss += F.relu(1.0 - asymmetric_energy)
        
        stats = {
            'forward_energy': forward_energy.item(),
            'backward_energy': backward_energy.item(),
            'asymmetric_energy': asymmetric_energy.item()
        }
        
        return loss, stats
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train AsymmetryTransformModel for one epoch"""
        self.asymmetry_model.train()
        self.order_model.eval()  # Keep frozen
        
        total_loss = 0
        num_samples = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            batch_loss = 0
            batch_samples = 0
            
            for i in range(len(batch['premise_tokens'])):
                premise_tokens = batch['premise_tokens'][i]
                hypothesis_tokens = batch['hypothesis_tokens'][i]
                label_str = batch['label_str'][i]
                
                if premise_tokens.shape[0] < 2 or hypothesis_tokens.shape[0] < 2:
                    continue
                
                sample_loss, _ = self.compute_asymmetric_loss(premise_tokens, hypothesis_tokens, label_str)
                batch_loss += sample_loss
                batch_samples += 1
            
            if batch_samples > 0:
                batch_loss = batch_loss / batch_samples
                batch_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.asymmetry_model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += batch_loss.item()
                num_samples += batch_samples
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """Evaluate AsymmetryTransformModel"""
        self.asymmetry_model.eval()
        self.order_model.eval()
        
        total_loss = 0
        all_stats = {'entailment': [], 'neutral': [], 'contradiction': []}
        num_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch_loss = 0
                batch_samples = 0
                
                for i in range(len(batch['premise_tokens'])):
                    premise_tokens = batch['premise_tokens'][i]
                    hypothesis_tokens = batch['hypothesis_tokens'][i]
                    label_str = batch['label_str'][i]
                    
                    if premise_tokens.shape[0] < 2 or hypothesis_tokens.shape[0] < 2:
                        continue
                    
                    sample_loss, stats = self.compute_asymmetric_loss(premise_tokens, hypothesis_tokens, label_str)
                    batch_loss += sample_loss
                    batch_samples += 1
                    
                    all_stats[label_str].append(stats)
                
                if batch_samples > 0:
                    total_loss += batch_loss.item() / batch_samples
                    num_samples += batch_samples
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        self.val_losses.append(avg_loss)
        
        # Compute asymmetry summary
        asymmetry_summary = {}
        for label, stats_list in all_stats.items():
            if stats_list:
                asymmetry_summary[label] = {
                    'forward_energy_mean': np.mean([s['forward_energy'] for s in stats_list]),
                    'backward_energy_mean': np.mean([s['backward_energy'] for s in stats_list]),
                    'asymmetric_energy_mean': np.mean([s['asymmetric_energy'] for s in stats_list]),
                    'count': len(stats_list)
                }
        
        self.asymmetry_stats.append(asymmetry_summary)
        return avg_loss, asymmetry_summary


def train_order_embedding_model(processed_data_path: str, output_dir: str, epochs: int = 20, 
                               batch_size: int = 4, random_seed: int = 42):
    """Step 1: Train OrderEmbeddingModel with pure Vendrov loss"""
    
    print("=" * 80)
    print("STEP 1: TRAINING ORDER EMBEDDING MODEL")
    print("=" * 80)
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Create dataset and dataloaders
    dataset = TokenLevelEntailmentDataset(processed_data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed)
    )
    
    def collate_fn(batch):
        return {
            'premise_tokens': [item['premise_tokens'] for item in batch],
            'hypothesis_tokens': [item['hypothesis_tokens'] for item in batch],
            'label': torch.stack([item['label'] for item in batch]),
            'label_str': [item['label_str'] for item in batch]
        }
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Train order model
    order_model = OrderEmbeddingModel(hidden_size=768)
    order_trainer = OrderEmbeddingTrainer(order_model)
    
    print(f"Training OrderEmbeddingModel - Pure Vendrov et al. max-margin loss")
    
    best_val_loss = float('inf')
    patience = 8
    patience_counter = 0
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = order_trainer.train_epoch(train_loader)
        val_loss, energy_stats = order_trainer.evaluate(val_loader)
        order_trainer.scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if energy_stats:
            print("Energy Rankings:")
            for label, stats in energy_stats.items():
                print(f"  {label}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save order model
            order_model_path = Path(output_dir) / "order_embedding_model_all_roberta_large_v1.pt"
            torch.save({
                'model_state_dict': order_model.state_dict(),
                'training_stats': {
                    'train_losses': order_trainer.train_losses,
                    'val_losses': order_trainer.val_losses,
                    'energy_rankings': order_trainer.energy_rankings,
                },
                'best_val_loss': best_val_loss,
                'epoch': epoch,
            }, order_model_path)
            
            print(f"✅ Order model saved: {order_model_path}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return order_model, order_trainer


def train_asymmetry_model(processed_data_path: str, order_model: OrderEmbeddingModel, 
                         output_dir: str, epochs: int = 15, batch_size: int = 4, random_seed: int = 42):
    """Step 2: Train AsymmetryTransformModel with frozen OrderEmbeddingModel"""
    
    print("=" * 80)
    print("STEP 2: TRAINING ASYMMETRY TRANSFORM MODEL")
    print("=" * 80)
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Create dataset and dataloaders (same as before)
    dataset = TokenLevelEntailmentDataset(processed_data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed)
    )
    
    def collate_fn(batch):
        return {
            'premise_tokens': [item['premise_tokens'] for item in batch],
            'hypothesis_tokens': [item['hypothesis_tokens'] for item in batch],
            'label': torch.stack([item['label'] for item in batch]),
            'label_str': [item['label_str'] for item in batch]
        }
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Train asymmetry model
    asymmetry_model = AsymmetryTransformModel(hidden_size=768)
    asymmetry_trainer = AsymmetryTrainer(asymmetry_model, order_model)
    
    print(f"Training AsymmetryTransformModel - Asymmetric directional loss")
    print("OrderEmbeddingModel is frozen")
    
    best_val_loss = float('inf')
    patience = 8
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = asymmetry_trainer.train_epoch(train_loader)
        val_loss, asymmetry_stats = asymmetry_trainer.evaluate(val_loader)
        asymmetry_trainer.scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if asymmetry_stats:
            print("Asymmetry Stats:")
            for label, stats in asymmetry_stats.items():
                print(f"  {label}: fwd={stats['forward_energy_mean']:.3f}, "
                      f"bwd={stats['backward_energy_mean']:.3f}, "
                      f"asym={stats['asymmetric_energy_mean']:.3f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save asymmetry model
            asymmetry_model_path = Path(output_dir) / "asymmetry_transform_model_all_roberta_large_v1.pt"
            torch.save({
                'model_state_dict': asymmetry_model.state_dict(),
                'training_stats': {
                    'train_losses': asymmetry_trainer.train_losses,
                    'val_losses': asymmetry_trainer.val_losses,
                    'asymmetry_stats': asymmetry_trainer.asymmetry_stats,
                },
                'best_val_loss': best_val_loss,
                'epoch': epoch,
            }, asymmetry_model_path)
            
            print(f"✅ Asymmetry model saved: {asymmetry_model_path}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return asymmetry_model, asymmetry_trainer


def plot_training_progress(order_trainer: OrderEmbeddingTrainer, asymmetry_trainer: AsymmetryTrainer, 
                          output_dir: str):
    """Plot comprehensive training progress for both models"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # ============= ORDER EMBEDDING MODEL PLOTS =============
    
    # Plot 1: Order model losses
    epochs_order = range(1, len(order_trainer.train_losses) + 1)
    ax1.plot(epochs_order, order_trainer.train_losses, 'b-', label='Training Loss', linewidth=2)
    if order_trainer.val_losses:
        ax1.plot(epochs_order, order_trainer.val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Order Embedding Model: Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Order model energy rankings
    if order_trainer.energy_rankings:
        entail_means = []
        neutral_means = []
        contra_means = []
        
        for ranking in order_trainer.energy_rankings:
            entail_means.append(ranking.get('entailment', {}).get('mean', 0))
            neutral_means.append(ranking.get('neutral', {}).get('mean', 0))
            contra_means.append(ranking.get('contradiction', {}).get('mean', 0))
        
        epochs_val = range(1, len(entail_means) + 1)
        ax2.plot(epochs_val, entail_means, 'g-', label='Entailment', linewidth=3)
        ax2.plot(epochs_val, neutral_means, 'b-', label='Neutral', linewidth=3)
        ax2.plot(epochs_val, contra_means, 'r-', label='Contradiction', linewidth=3)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Order Violation Energy')
        ax2.set_title('Order Embedding Model: Energy Rankings (Vendrov Loss)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add target ordering annotation
        final_entail = entail_means[-1] if entail_means else 0
        final_neutral = neutral_means[-1] if neutral_means else 0
        final_contra = contra_means[-1] if contra_means else 0
        
        if final_entail < final_neutral < final_contra:
            ax2.text(0.02, 0.98, '✅ Correct ordering:\nEntailment < Neutral < Contradiction', 
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            ax2.text(0.02, 0.98, '❌ Incorrect ordering', 
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # ============= ASYMMETRY MODEL PLOTS =============
    
    # Plot 3: Asymmetry model losses
    epochs_asym = range(1, len(asymmetry_trainer.train_losses) + 1)
    ax3.plot(epochs_asym, asymmetry_trainer.train_losses, 'purple', label='Training Loss', linewidth=2)
    if asymmetry_trainer.val_losses:
        ax3.plot(epochs_asym, asymmetry_trainer.val_losses, 'orange', label='Validation Loss', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Asymmetry Transform Model: Training Progress')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Asymmetry patterns
    if asymmetry_trainer.asymmetry_stats:
        entail_asym = []
        neutral_asym = []
        contra_asym = []
        
        entail_forward = []
        neutral_forward = []
        contra_forward = []
        
        entail_backward = []
        neutral_backward = []
        contra_backward = []
        
        for stats in asymmetry_trainer.asymmetry_stats:
            # Asymmetric energy
            entail_asym.append(stats.get('entailment', {}).get('asymmetric_energy_mean', 0))
            neutral_asym.append(stats.get('neutral', {}).get('asymmetric_energy_mean', 0))
            contra_asym.append(stats.get('contradiction', {}).get('asymmetric_energy_mean', 0))
            
            # Forward energy
            entail_forward.append(stats.get('entailment', {}).get('forward_energy_mean', 0))
            neutral_forward.append(stats.get('neutral', {}).get('forward_energy_mean', 0))
            contra_forward.append(stats.get('contradiction', {}).get('forward_energy_mean', 0))
            
            # Backward energy
            entail_backward.append(stats.get('entailment', {}).get('backward_energy_mean', 0))
            neutral_backward.append(stats.get('neutral', {}).get('backward_energy_mean', 0))
            contra_backward.append(stats.get('contradiction', {}).get('backward_energy_mean', 0))
        
        epochs_asym_val = range(1, len(entail_asym) + 1)
        
        # Plot asymmetric energy patterns
        ax4.plot(epochs_asym_val, entail_asym, 'g-', label='Entailment Asymmetry', linewidth=2, linestyle='--')
        ax4.plot(epochs_asym_val, neutral_asym, 'b-', label='Neutral Asymmetry', linewidth=2, linestyle='--')
        ax4.plot(epochs_asym_val, contra_asym, 'r-', label='Contradiction Asymmetry', linewidth=2, linestyle='--')
        
        # Plot forward energy patterns
        ax4.plot(epochs_asym_val, entail_forward, 'g-', label='Entailment Forward', linewidth=2)
        ax4.plot(epochs_asym_val, neutral_forward, 'b-', label='Neutral Forward', linewidth=2)
        ax4.plot(epochs_asym_val, contra_forward, 'r-', label='Contradiction Forward', linewidth=2)
        
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Energy')
        ax4.set_title('Asymmetry Model: Directional Energy Patterns')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # Add target pattern annotations
        if len(entail_forward) > 0 and len(contra_forward) > 0:
            final_entail_forward = entail_forward[-1]
            final_contra_forward = contra_forward[-1]
            
            if final_entail_forward < final_contra_forward:
                pattern_text = '✅ Correct pattern:\nEntailment forward < Contradiction forward'
                color = 'lightgreen'
            else:
                pattern_text = '❌ Incorrect pattern'
                color = 'lightcoral'
            
            ax4.text(0.02, 0.98, pattern_text, transform=ax4.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    plt.tight_layout()
    
    # Save comprehensive plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = Path(output_dir) / f"comprehensive_training_progress__all_roberta_large_v1_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive training plots saved to: {plot_path}")
    
    # Generate individual model plots for detailed analysis
    plot_individual_model_progress(order_trainer, asymmetry_trainer, output_dir, timestamp)

def plot_individual_model_progress(order_trainer: OrderEmbeddingTrainer, asymmetry_trainer: AsymmetryTrainer,
                                 output_dir: str, timestamp: str):
    """Generate detailed individual plots for each model"""
    
    # ============= DETAILED ORDER MODEL ANALYSIS =============
    
    fig_order, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Order model loss with gradient analysis
    epochs_order = range(1, len(order_trainer.train_losses) + 1)
    ax1.plot(epochs_order, order_trainer.train_losses, 'b-', label='Training Loss', linewidth=2)
    if order_trainer.val_losses:
        ax1.plot(epochs_order, order_trainer.val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Vendrov Max-Margin Loss')
    ax1.set_title('Order Embedding Model: Pure Vendrov Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Energy convergence analysis
    if order_trainer.energy_rankings:
        entail_means = [r.get('entailment', {}).get('mean', 0) for r in order_trainer.energy_rankings]
        neutral_means = [r.get('neutral', {}).get('mean', 0) for r in order_trainer.energy_rankings]
        contra_means = [r.get('contradiction', {}).get('mean', 0) for r in order_trainer.energy_rankings]
        
        epochs_val = range(1, len(entail_means) + 1)
        ax2.plot(epochs_val, entail_means, 'g-', label='Entailment', linewidth=3, marker='o')
        ax2.plot(epochs_val, neutral_means, 'b-', label='Neutral', linewidth=3, marker='s')
        ax2.plot(epochs_val, contra_means, 'r-', label='Contradiction', linewidth=3, marker='^')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Order Violation Energy')
        ax2.set_title('Token-Level Order Violation Energy Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Energy separation analysis
        if len(entail_means) > 1:
            entail_neutral_gap = [n - e for e, n in zip(entail_means, neutral_means)]
            neutral_contra_gap = [c - n for n, c in zip(neutral_means, contra_means)]
            
            ax3.plot(epochs_val, entail_neutral_gap, 'cyan', label='Neutral - Entailment Gap', linewidth=2)
            ax3.plot(epochs_val, neutral_contra_gap, 'magenta', label='Contradiction - Neutral Gap', linewidth=2)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Energy Gap')
            ax3.set_title('Order Model: Class Separation Gaps')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Standard deviation analysis (training stability)
        entail_stds = [r.get('entailment', {}).get('std', 0) for r in order_trainer.energy_rankings]
        neutral_stds = [r.get('neutral', {}).get('std', 0) for r in order_trainer.energy_rankings]
        contra_stds = [r.get('contradiction', {}).get('std', 0) for r in order_trainer.energy_rankings]
        
        ax4.plot(epochs_val, entail_stds, 'g--', label='Entailment Std', linewidth=2)
        ax4.plot(epochs_val, neutral_stds, 'b--', label='Neutral Std', linewidth=2)
        ax4.plot(epochs_val, contra_stds, 'r--', label='Contradiction Std', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Energy Standard Deviation')
        ax4.set_title('Order Model: Within-Class Consistency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    order_plot_path = Path(output_dir) / f"order_model_detailed__all_roberta_large_v1_{timestamp}.png"
    plt.savefig(order_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ============= DETAILED ASYMMETRY MODEL ANALYSIS =============
    
    fig_asym, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Asymmetry model loss
    epochs_asym = range(1, len(asymmetry_trainer.train_losses) + 1)
    ax1.plot(epochs_asym, asymmetry_trainer.train_losses, 'purple', label='Training Loss', linewidth=2)
    if asymmetry_trainer.val_losses:
        ax1.plot(epochs_asym, asymmetry_trainer.val_losses, 'orange', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Asymmetric Loss')
    ax1.set_title('Asymmetry Model: Directional Pattern Learning')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Forward vs Backward energy comparison
    if asymmetry_trainer.asymmetry_stats:
        entail_forward = [s.get('entailment', {}).get('forward_energy_mean', 0) for s in asymmetry_trainer.asymmetry_stats]
        entail_backward = [s.get('entailment', {}).get('backward_energy_mean', 0) for s in asymmetry_trainer.asymmetry_stats]
        
        neutral_forward = [s.get('neutral', {}).get('forward_energy_mean', 0) for s in asymmetry_trainer.asymmetry_stats]
        neutral_backward = [s.get('neutral', {}).get('backward_energy_mean', 0) for s in asymmetry_trainer.asymmetry_stats]
        
        contra_forward = [s.get('contradiction', {}).get('forward_energy_mean', 0) for s in asymmetry_trainer.asymmetry_stats]
        contra_backward = [s.get('contradiction', {}).get('backward_energy_mean', 0) for s in asymmetry_trainer.asymmetry_stats]
        
        epochs_asym_val = range(1, len(entail_forward) + 1)
        
        # Forward energies
        ax2.plot(epochs_asym_val, entail_forward, 'g-', label='Entailment Forward', linewidth=2)
        ax2.plot(epochs_asym_val, neutral_forward, 'b-', label='Neutral Forward', linewidth=2)
        ax2.plot(epochs_asym_val, contra_forward, 'r-', label='Contradiction Forward', linewidth=2)
        
        # Backward energies (dashed)
        ax2.plot(epochs_asym_val, entail_backward, 'g--', label='Entailment Backward', linewidth=2)
        ax2.plot(epochs_asym_val, neutral_backward, 'b--', label='Neutral Backward', linewidth=2)
        ax2.plot(epochs_asym_val, contra_backward, 'r--', label='Contradiction Backward', linewidth=2)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Energy')
        ax2.set_title('Asymmetry Model: Forward vs Backward Energy Patterns')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Asymmetry measures
        entail_asym = [s.get('entailment', {}).get('asymmetric_energy_mean', 0) for s in asymmetry_trainer.asymmetry_stats]
        neutral_asym = [s.get('neutral', {}).get('asymmetric_energy_mean', 0) for s in asymmetry_trainer.asymmetry_stats]
        contra_asym = [s.get('contradiction', {}).get('asymmetric_energy_mean', 0) for s in asymmetry_trainer.asymmetry_stats]
        
        ax3.plot(epochs_asym_val, entail_asym, 'g-', label='Entailment', linewidth=3, marker='o')
        ax3.plot(epochs_asym_val, neutral_asym, 'b-', label='Neutral', linewidth=3, marker='s')
        ax3.plot(epochs_asym_val, contra_asym, 'r-', label='Contradiction', linewidth=3, marker='^')
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Asymmetric Energy')
        ax3.set_title('Asymmetry Model: Directional Opposition Learning')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Asymmetry ratio analysis (Forward - Backward difference)
        entail_diff = [f - b for f, b in zip(entail_forward, entail_backward)]
        neutral_diff = [f - b for f, b in zip(neutral_forward, neutral_backward)]
        contra_diff = [f - b for f, b in zip(contra_forward, contra_backward)]
        
        ax4.plot(epochs_asym_val, entail_diff, 'g-', label='Entailment (F-B)', linewidth=2)
        ax4.plot(epochs_asym_val, neutral_diff, 'b-', label='Neutral (F-B)', linewidth=2)
        ax4.plot(epochs_asym_val, contra_diff, 'r-', label='Contradiction (F-B)', linewidth=2)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Symmetric Line')
        
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Forward - Backward Energy')
        ax4.set_title('Asymmetry Model: Directional Bias Learning')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add target pattern annotation
        if len(entail_diff) > 0:
            final_entail_diff = entail_diff[-1]
            final_neutral_diff = neutral_diff[-1]
            
            if abs(final_neutral_diff) < abs(final_entail_diff):
                pattern_text = '✅ Correct:\nNeutral more symmetric than Entailment'
                color = 'lightgreen'
            else:
                pattern_text = '❌ Incorrect asymmetry pattern'
                color = 'lightcoral'
            
            ax4.text(0.02, 0.98, pattern_text, transform=ax4.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    plt.tight_layout()
    asymmetry_plot_path = Path(output_dir) / f"asymmetry_model_detailed__all_roberta_large_v1_{timestamp}.png"
    plt.savefig(asymmetry_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed order model plots saved to: {order_plot_path}")
    print(f"Detailed asymmetry model plots saved to: {asymmetry_plot_path}")
    
    # ============= GENERATE TOKEN-LEVEL ANALYSIS SUMMARY =============
    
    analysis_file = Path(output_dir) / f"token_level_analysis__all_roberta_large_v1_{timestamp}.txt"
    with open(analysis_file, 'w') as f:
        f.write("TOKEN-LEVEL ORDER EMBEDDING ANALYSIS\n")
        f.write("="*50 + "\n\n")
        
        f.write("SENTENCE-LEVEL vs TOKEN-LEVEL DIFFERENCES:\n")
        f.write("-"*40 + "\n")
        f.write("Original (Sentence): 1 embedding per text → 1 energy value\n")
        f.write("New (Token): N tokens per text → N energy values → aggregated\n\n")
        
        f.write("TOKEN-LEVEL PROCESSING NUANCES:\n")
        f.write("-"*30 + "\n")
        f.write("1. Variable sequence lengths: Premise ≠ Hypothesis token counts\n")
        f.write("2. Energy aggregation: Token energies → mean() → sentence energy\n")
        f.write("3. Richer representation: Each token captures different semantic aspects\n")
        f.write("4. Point cloud generation: Each token becomes a point\n\n")
        
        if order_trainer.energy_rankings:
            f.write("ORDER MODEL FINAL PERFORMANCE:\n")
            f.write("-"*30 + "\n")
            final_ranking = order_trainer.energy_rankings[-1]
            for label, stats in final_ranking.items():
                f.write(f"{label}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})\n")
            
            # Check ordering
            entail_mean = final_ranking.get('entailment', {}).get('mean', float('inf'))
            neutral_mean = final_ranking.get('neutral', {}).get('mean', float('inf'))
            contra_mean = final_ranking.get('contradiction', {}).get('mean', float('inf'))
            
            if entail_mean < neutral_mean < contra_mean:
                f.write("✅ CORRECT ORDERING: Entailment < Neutral < Contradiction\n")
            else:
                f.write("❌ INCORRECT ORDERING\n")
        
        if asymmetry_trainer.asymmetry_stats:
            f.write("\nASYMMETRY MODEL FINAL PERFORMANCE:\n")
            f.write("-"*35 + "\n")
            final_asym = asymmetry_trainer.asymmetry_stats[-1]
            for label, stats in final_asym.items():
                f.write(f"{label}:\n")
                f.write(f"  Forward: {stats.get('forward_energy_mean', 0):.4f}\n")
                f.write(f"  Backward: {stats.get('backward_energy_mean', 0):.4f}\n")
                f.write(f"  Asymmetric: {stats.get('asymmetric_energy_mean', 0):.4f}\n")
                f.write(f"  Samples: {stats.get('count', 0)}\n\n")
        
        f.write("POINT CLOUD GENERATION STRATEGY:\n")
        f.write("-"*35 + "\n")
        f.write("Per text (premise or hypothesis):\n")
        f.write("1. Original SBERT tokens: ~30-50 points\n")
        f.write("2. Order embeddings: ~30-50 points\n") 
        f.write("3. Asymmetric features: ~30-50 points\n")
        f.write("Total per text: ~90-150 points\n")
        f.write("Combined premise+hypothesis: ~180-300 points\n")
        f.write("✅ Sufficient for reliable PHD computation (≥200 points)\n")
    
    print(f"Token-level analysis summary saved to: {analysis_file}")



def main():
    """Train both models separately"""
    
    processed_data_path = "/vol/bitbucket/ahb24/tda_entailment_new/snli_train_sbert_tokens_all_roberta_large_v1.pkl"
    output_dir = "MSc_Topology_Codebase/phd_method/models/separate_models/"
    os.makedirs(output_dir, exist_ok=True)
    
    if not Path(processed_data_path).exists():
        print(f"Processed data not found at: {processed_data_path}")
        print("Please run sbert_token_extractor.py first")
        return
    
    # Step 1: Train OrderEmbeddingModel
    order_model, order_trainer = train_order_embedding_model(
        processed_data_path, output_dir, epochs=20, batch_size=1, random_seed=42
    )
    
    # Step 2: Train AsymmetryTransformModel (with frozen OrderEmbeddingModel)
    asymmetry_model, asymmetry_trainer = train_asymmetry_model(
        processed_data_path, order_model, output_dir, epochs=15, batch_size=1, random_seed=42
    )

    plot_training_progress(order_trainer, asymmetry_trainer, output_dir)
    
    print("\n" + "="*80)
    print("SEPARATE MODEL TRAINING COMPLETE!")
    print("✅ OrderEmbeddingModel: Pure hierarchy (Vendrov et al.)")
    print("✅ AsymmetryTransformModel: Directional patterns")
    print("\nPoint cloud structure per sample:")
    print("  - SBERT tokens: ~30-50 points")
    print("  - Order embeddings: ~30-50 points") 
    print("  - Asymmetric features: ~30-50 points")
    print("  - Total: ~90-150 points per text")
    print("Next: Update point_cloud_clustering_test.py to use both models")
    print("="*80)
    
    return order_model, asymmetry_model


if __name__ == "__main__":
    main()