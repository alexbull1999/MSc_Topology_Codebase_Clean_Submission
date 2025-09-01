"""
Longer training + Separate class margins in loss for neutral and contradiction, incl. upper bound limit for neutral 
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



class OrderEmbeddingTrainerSeparateMargins:
    """Trainer for OrderEmbeddingModel using separate margins for neutral vs contradiction"""
    
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
        
        print(f"OrderEmbeddingTrainerSeparateMargins initialized on {self.device}")
    
    def compute_separate_margin_loss(self, premise_tokens: torch.Tensor, hypothesis_tokens: torch.Tensor,
                                   label: int, neutral_margin: float = 1.0, contradiction_margin: float = 2.0,
                                   neutral_upper_bound: float = 1.5) -> Tuple[torch.Tensor, float]:
        """
        Separate margin loss with neutral sandwiching for better class separation
        - Entailment: minimize energy (target ~0.3)
        - Neutral: keep energy in [neutral_margin, neutral_upper_bound] (target ~1.3)
        - Contradiction: ensure energy > contradiction_margin (target ~1.9)
        """
        premise_tokens = premise_tokens.to(self.device)
        hypothesis_tokens = hypothesis_tokens.to(self.device)
        
        # Get order embeddings
        premise_order = self.model(premise_tokens)
        hypothesis_order = self.model(hypothesis_tokens)
        
        # Compute order violation energy
        energy = self.model.order_violation_energy(premise_order, hypothesis_order)
        
        # Separate margin loss based on actual label
        if label == 0:  # entailment
            loss = energy  # Minimize energy toward ~0.3
        elif label == 1:  # neutral - sandwich between neutral_margin and neutral_upper_bound
            lower_violation = torch.clamp(neutral_margin - energy, min=0)  # Penalty if energy < neutral_margin
            upper_violation = torch.clamp(energy - neutral_upper_bound, min=0)  # Penalty if energy > neutral_upper_bound
            loss = lower_violation + upper_violation  # Keep energy in [neutral_margin, neutral_upper_bound]
        else:  # contradiction (label == 2)
            loss = torch.clamp(contradiction_margin - energy, min=0)  # Ensure energy > contradiction_margin
        
        return loss, energy.item()
    
    def train_epoch(self, dataloader: DataLoader, neutral_margin: float = 1.0, 
                   contradiction_margin: float = 2.0, neutral_upper_bound: float = 1.5) -> float:
        """Train OrderEmbeddingModel for one epoch with separate margins"""
        self.model.train()
        total_loss = 0
        num_samples = 0
        
        for batch_idx, batch in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            batch_loss = 0
            batch_samples = 0
            
            for i in range(len(batch['premise_tokens'])):
                premise_tokens = batch['premise_tokens'][i]
                hypothesis_tokens = batch['hypothesis_tokens'][i]
                label = batch['label'][i].item()
                
                if premise_tokens.shape[0] < 2 or hypothesis_tokens.shape[0] < 2:
                    continue
                
                sample_loss, _ = self.compute_separate_margin_loss(
                    premise_tokens, hypothesis_tokens, label, 
                    neutral_margin, contradiction_margin, neutral_upper_bound
                )
                batch_loss += sample_loss
                batch_samples += 1

                # Memory management
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

                # Cleanup after each batch
                del batch_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Cleanup every 10 batches
            if batch_idx % 10 == 0:
                import gc
                gc.collect()
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def evaluate(self, dataloader: DataLoader, neutral_margin: float = 1.0, 
                contradiction_margin: float = 2.0, neutral_upper_bound: float = 1.5) -> Tuple[float, Dict]:
        """Evaluate OrderEmbeddingModel with separate margins"""
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
                    
                    sample_loss, energy = self.compute_separate_margin_loss(
                        premise_tokens, hypothesis_tokens, label,
                        neutral_margin, contradiction_margin, neutral_upper_bound
                    )
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




def plot_order_embedding_training_progress(trainer: OrderEmbeddingTrainerSeparateMargins, output_dir: str):
    """Plot comprehensive training progress for order embedding model with separate margins"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Training and validation losses
    epochs_order = range(1, len(trainer.train_losses) + 1)
    ax1.plot(epochs_order, trainer.train_losses, 'b-', label='Training Loss', linewidth=2)
    if trainer.val_losses:
        ax1.plot(epochs_order, trainer.val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Separate Margin Loss')
    ax1.set_title('Order Embedding Model: Training Progress (Separate Margins)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Energy rankings convergence
    if trainer.energy_rankings:
        entail_means = []
        neutral_means = []
        contra_means = []
        
        for ranking in trainer.energy_rankings:
            entail_means.append(ranking.get('entailment', {}).get('mean', 0))
            neutral_means.append(ranking.get('neutral', {}).get('mean', 0))
            contra_means.append(ranking.get('contradiction', {}).get('mean', 0))
        
        epochs_val = range(1, len(entail_means) + 1)
        ax2.plot(epochs_val, entail_means, 'g-', label='Entailment', linewidth=3, marker='o')
        ax2.plot(epochs_val, neutral_means, 'b-', label='Neutral', linewidth=3, marker='s')
        ax2.plot(epochs_val, contra_means, 'r-', label='Contradiction', linewidth=3, marker='^')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Order Violation Energy')
        ax2.set_title('Token-Level Order Violation Energy Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add target zones
        ax2.axhspan(0, 1.0, alpha=0.2, color='green', label='Entailment Zone')
        ax2.axhspan(1.0, 1.5, alpha=0.2, color='blue', label='Neutral Zone') 
        ax2.axhspan(1.8, max(max(contra_means) if contra_means else 2.5, 2.5), alpha=0.2, color='red', label='Contradiction Zone')
        
        # Check final ordering
        final_entail = entail_means[-1] if entail_means else 0
        final_neutral = neutral_means[-1] if neutral_means else 0
        final_contra = contra_means[-1] if contra_means else 0
        
        if final_entail < 1.0 and 1.0 <= final_neutral <= 1.5 and final_contra > 1.8:
            ax2.text(0.02, 0.98, '✅ Perfect Separation:\nE<1.0, 1.0≤N≤1.5, C>1.8', 
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        elif final_entail < final_neutral < final_contra:
            ax2.text(0.02, 0.98, '✅ Correct ordering:\nEntailment < Neutral < Contradiction', 
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        else:
            ax2.text(0.02, 0.98, '❌ Incorrect ordering', 
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Plot 3: Class separation gaps
    if trainer.energy_rankings and len(trainer.energy_rankings) > 1:
        entail_means = [r.get('entailment', {}).get('mean', 0) for r in trainer.energy_rankings]
        neutral_means = [r.get('neutral', {}).get('mean', 0) for r in trainer.energy_rankings]
        contra_means = [r.get('contradiction', {}).get('mean', 0) for r in trainer.energy_rankings]
        
        entail_neutral_gap = [n - e for e, n in zip(entail_means, neutral_means)]
        neutral_contra_gap = [c - n for n, c in zip(neutral_means, contra_means)]
        
        epochs_val = range(1, len(entail_neutral_gap) + 1)
        ax3.plot(epochs_val, entail_neutral_gap, 'cyan', label='Neutral - Entailment Gap', linewidth=2)
        ax3.plot(epochs_val, neutral_contra_gap, 'magenta', label='Contradiction - Neutral Gap', linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Energy Gap')
        ax3.set_title('Order Model: Class Separation Gaps')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Within-class consistency (standard deviations)
    if trainer.energy_rankings:
        entail_stds = [r.get('entailment', {}).get('std', 0) for r in trainer.energy_rankings]
        neutral_stds = [r.get('neutral', {}).get('std', 0) for r in trainer.energy_rankings]
        contra_stds = [r.get('contradiction', {}).get('std', 0) for r in trainer.energy_rankings]
        
        epochs_val = range(1, len(entail_stds) + 1)
        ax4.plot(epochs_val, entail_stds, 'g--', label='Entailment Std', linewidth=2)
        ax4.plot(epochs_val, neutral_stds, 'b--', label='Neutral Std', linewidth=2)
        ax4.plot(epochs_val, contra_stds, 'r--', label='Contradiction Std', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Energy Standard Deviation')
        ax4.set_title('Order Model: Within-Class Consistency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = Path(output_dir) / f"mnli_order_embedding_separate_margins_training_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training progress plots saved to: {plot_path}")
    
    return timestamp


def train_order_embedding_separate_margins(processed_data_path: str, output_dir: str, 
                                         epochs: int = 50, batch_size: int = 1020, 
                                         random_seed: int = 42, neutral_margin: float = 1.0,
                                         contradiction_margin: float = 2.0, 
                                         neutral_upper_bound: float = 1.5):
    """Train OrderEmbeddingModel with separate margins using your original training style"""
    
    print("=" * 80)
    print("TRAINING ORDER EMBEDDING MODEL WITH SEPARATE MARGINS")
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
    
    # Train order model with separate margins
    order_model = OrderEmbeddingModel(hidden_size=768)
    order_trainer = OrderEmbeddingTrainerSeparateMargins(order_model)
    
    print(f"Training OrderEmbeddingModel - Separate Margin Loss")
    print(f"Neutral margin: {neutral_margin}, Upper bound: {neutral_upper_bound}, Contradiction margin: {contradiction_margin}")
    
    best_val_loss = float('inf')
    patience = 8
    patience_counter = 0
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = order_trainer.train_epoch(train_loader, neutral_margin, contradiction_margin, neutral_upper_bound)
        val_loss, energy_stats = order_trainer.evaluate(val_loader, neutral_margin, contradiction_margin, neutral_upper_bound)
        order_trainer.scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if energy_stats:
            print("Energy Rankings:")
            for label, stats in energy_stats.items():
                print(f"  {label}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            
            # Check target zones
            entail_mean = energy_stats.get('entailment', {}).get('mean', float('inf'))
            neutral_mean = energy_stats.get('neutral', {}).get('mean', float('inf'))
            contra_mean = energy_stats.get('contradiction', {}).get('mean', float('inf'))
            
            zones_correct = (entail_mean < 1.0 and 
                           1.0 <= neutral_mean <= 1.5 and 
                           contra_mean > 1.8)
            if zones_correct:
                print("  ✅ Perfect target zones achieved!")
            elif entail_mean < neutral_mean < contra_mean:
                print("  ✅ Correct ordering maintained")
            else:
                print("  ❌ Ordering needs improvement")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save order model
            order_model_path = Path(output_dir) / "mnli_order_embedding_model_separate_margins.pt"
            torch.save({
                'model_state_dict': order_model.state_dict(),
                'training_stats': {
                    'train_losses': order_trainer.train_losses,
                    'val_losses': order_trainer.val_losses,
                    'energy_rankings': order_trainer.energy_rankings,
                },
                'training_config': {
                    'neutral_margin': neutral_margin,
                    'contradiction_margin': contradiction_margin,
                    'neutral_upper_bound': neutral_upper_bound,
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
    
    # Generate comprehensive plots
    timestamp = plot_order_embedding_training_progress(order_trainer, output_dir)
    
    # Generate analysis summary
    analysis_file = Path(output_dir) / f"mnli_separate_margins_analysis_{timestamp}.txt"
    with open(analysis_file, 'w') as f:
        f.write("ORDER EMBEDDING MODEL WITH SEPARATE MARGINS ANALYSIS\n")
        f.write("="*60 + "\n\n")
        
        f.write("TRAINING CONFIGURATION:\n")
        f.write("-"*25 + "\n")
        f.write(f"Neutral margin (lower bound): {neutral_margin}\n")
        f.write(f"Neutral upper bound: {neutral_upper_bound}\n")
        f.write(f"Contradiction margin: {contradiction_margin}\n")
        f.write(f"Target zones:\n")
        f.write(f"  Entailment: < {neutral_margin}\n")
        f.write(f"  Neutral: [{neutral_margin}, {neutral_upper_bound}]\n")
        f.write(f"  Contradiction: > {contradiction_margin}\n\n")
        
        if order_trainer.energy_rankings:
            f.write("FINAL PERFORMANCE:\n")
            f.write("-"*20 + "\n")
            final_ranking = order_trainer.energy_rankings[-1]
            for label, stats in final_ranking.items():
                f.write(f"{label}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})\n")
            
            # Check final zones
            entail_mean = final_ranking.get('entailment', {}).get('mean', float('inf'))
            neutral_mean = final_ranking.get('neutral', {}).get('mean', float('inf'))
            contra_mean = final_ranking.get('contradiction', {}).get('mean', float('inf'))
            
            f.write(f"\nZONE ANALYSIS:\n")
            f.write("-"*15 + "\n")
            f.write(f"Entailment in target zone (< {neutral_margin}): {'✅' if entail_mean < neutral_margin else '❌'}\n")
            f.write(f"Neutral in target zone ([{neutral_margin}, {neutral_upper_bound}]): {'✅' if neutral_margin <= neutral_mean <= neutral_upper_bound else '❌'}\n")
            f.write(f"Contradiction in target zone (> {contradiction_margin}): {'✅' if contra_mean > contradiction_margin else '❌'}\n")
            
            if entail_mean < neutral_margin and neutral_margin <= neutral_mean <= neutral_upper_bound and contra_mean > contradiction_margin:
                f.write("🎯 PERFECT SEPARATION ACHIEVED!\n")
            elif entail_mean < neutral_mean < contra_mean:
                f.write("✅ CORRECT ORDERING MAINTAINED\n")
            else:
                f.write("❌ ORDERING NEEDS IMPROVEMENT\n")
        
        f.write(f"\nBest validation loss: {best_val_loss:.6f}\n")
        f.write(f"Training completed at epoch: {len(order_trainer.train_losses)}\n")
    
    print(f"Analysis summary saved to: {analysis_file}")
    
    print("\n" + "="*80)
    print("ORDER EMBEDDING MODEL WITH SEPARATE MARGINS TRAINING COMPLETE!")
    print("✅ Separate margin loss: Enhanced neutral class separation")
    print("✅ Target energy zones: E<1.0, 1.0≤N≤1.5, C>1.8")
    print("✅ Comprehensive analysis and plots generated")
    print("Next: Use this model for improved point cloud clustering")
    print("="*80)
    
    return order_model, order_trainer


def main():
    """Train both models separately"""
    
    processed_data_path = "/vol/bitbucket/ahb24/tda_entailment_new/mnli_train_sbert_tokens.pkl"
    output_dir = "MSc_Topology_Codebase/phd_method/models/separate_models/"
    os.makedirs(output_dir, exist_ok=True)
    
    if not Path(processed_data_path).exists():
        print(f"Processed data not found at: {processed_data_path}")
        print("Please run sbert_token_extractor.py first")
        return
    
    # Step 1: Train OrderEmbeddingModel
    order_model, order_trainer = train_order_embedding_separate_margins(
        processed_data_path, output_dir, epochs=100, batch_size=1020, random_seed=42
    )
    
    plot_order_embedding_training_progress(order_trainer, output_dir)
    
    print("\n" + "="*80)
    print("SEPARATE MARGINS ORDER EMBEDDING TRAINING COMPLETE!")
    print("Next: Update point_cloud_clustering_test.py to use new model")
    print("="*80)
    
    return order_model


if __name__ == "__main__":
    main()