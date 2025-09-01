"""
Independet asymmetry model receiving sbert tokens as input, not order emebddings for energy calculations
AsymmetryTrainer: Asymmetric loss for directional patterns
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
from order_asymmetry_models import TokenLevelEntailmentDataset
from tqdm import tqdm
from independent_order_model import OrderEmbeddingModel


class AsymmetryTransformModel(nn.Module):
    """Asymmetry transformation model"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size

        #FOR V3
        # self.asymmetric_transform = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU()
        # )

        # Process SBERT tokens directly instead of order embeddings (FOR V2)
        self.asymmetric_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, sbert_tokens: torch.Tensor) -> torch.Tensor:
        """
        Transform SBERT tokens to asymmetric space
        
        Args:
            sbert_tokens: Original SBERT token embeddings [num_tokens, 768]
        Returns:
            asymmetric_features: Transformed tokens [num_tokens, 768]
        """

        #V2/V3 just use this
        return self.asymmetric_transform(sbert_tokens)
    
    def compute_asymmetric_energy(self, premise_sbert: torch.Tensor, hypothesis_sbert: torch.Tensor) -> torch.Tensor:
        """
        Compute asymmetric energy between premise and hypothesis SBERT embeddings
        """
        premise_asym = self.forward(premise_sbert)
        hypothesis_asym = self.forward(hypothesis_sbert)
        
        # Simple asymmetric energy: difference in norms
        premise_energy = torch.norm(premise_asym, dim=1).mean()
        hypothesis_energy = torch.norm(hypothesis_asym, dim=1).mean()
        
        return torch.abs(premise_energy - hypothesis_energy)

    def order_violation_energy(self, u_tokens: torch.Tensor, v_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute order violation energy in asymmetric space (needed for forward/backward energies)
        """
        u_asym = self.forward(u_tokens).mean(0, keepdim=True)  # [1, 768]
        v_asym = self.forward(v_tokens).mean(0, keepdim=True)  # [1, 768]
        
        # Order violation: max(0, v - u) components (u=premise, v=hypothesis)
        violation = torch.clamp(v_asym - u_asym, min=0)
        return torch.norm(violation, dim=-1)



class AsymmetryTrainer:
    """Trainer for AsymmetryTransformModel using SBERT tokens directly"""
    
    def __init__(self, asymmetry_model: AsymmetryTransformModel):
        self.asymmetry_model = asymmetry_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.asymmetry_model.to(self.device)
        
        # Only train asymmetry model
        self.optimizer = optim.Adam(self.asymmetry_model.parameters(), lr=1e-5, weight_decay=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=8, factor=0.5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.asymmetry_stats = []
        
        print(f"AsymmetryTrainer initialized on {self.device}")
        print("Training asymmetry model directly on SBERT tokens")
    
    def compute_asymmetric_loss(self, premise_tokens: torch.Tensor, hypothesis_tokens: torch.Tensor,
                              label_str: str) -> Tuple[torch.Tensor, Dict]:
        """
        Corrected asymmetric loss - neutral has lowest asymmetric energy
        """
        premise_tokens = premise_tokens.to(self.device)
        hypothesis_tokens = hypothesis_tokens.to(self.device)
        
        # Compute all energies from asymmetric space
        forward_energy = self.asymmetry_model.order_violation_energy(premise_tokens, hypothesis_tokens).squeeze()
        backward_energy = self.asymmetry_model.order_violation_energy(hypothesis_tokens, premise_tokens).squeeze()
        asymmetric_energy = self.asymmetry_model.compute_asymmetric_energy(premise_tokens, hypothesis_tokens).squeeze()

        asymmetric_loss = torch.tensor(0.0, device=self.device)  # Initialize as tensor
        
        # CORRECTED loss function based on your actual data patterns
        if label_str == 'entailment':
            # For entailment: enforce low forward energy, higher backward energy        
            asymmetric_loss += F.mse_loss(forward_energy, torch.tensor(0.0, device=self.device))
            asymmetric_loss += F.relu(torch.tensor(2.0, device=self.device) - backward_energy) # Encourage backward_e > 1.0
            
        elif label_str == 'neutral':
            # For neutral: enforce symmetric relationship (similar forward/backward energies)
            # Target medium energy levels
            target_energy = torch.tensor(1.0, device=self.device)
            asymmetric_loss += F.mse_loss(forward_energy, target_energy)
            asymmetric_loss += F.mse_loss(backward_energy, target_energy)
            # Penalty for high asymmetry
            asymmetric_loss += 0.5 * torch.abs(forward_energy - backward_energy)
            
        elif label_str == 'contradiction':
            # For contradiction: enforce high forward energy and high asymmetry
            asymmetric_loss += F.relu(torch.tensor(2.0, device=self.device) - forward_energy)  # Encourage forward_e > 2.0
            # Encourage high asymmetric energy (directional opposition)
            asymmetric_loss += F.relu(torch.tensor(1.0, device=self.device) - asymmetric_energy)  # Encourage asym_e > 1.0
            # Allow variable backward energy (contradiction can be asymmetric)
        
        stats = {
            'forward_energy': forward_energy.item(),
            'backward_energy': backward_energy.item(),
            'asymmetric_energy': asymmetric_energy.item()
        }
        
        return asymmetric_loss, stats
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train AsymmetryTransformModel for one epoch"""
        self.asymmetry_model.train()
        
        total_loss = 0
        num_samples = 0

        pbar = tqdm(dataloader, desc="Training", leave=False)
        
        for batch in pbar:
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

                pbar.set_postfix({'batch_loss': f'{batch_loss.item():.4f}'})

        pbar.close()
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """Evaluate AsymmetryTransformModel"""
        self.asymmetry_model.eval()
        
        total_loss = 0
        all_stats = {'entailment': [], 'neutral': [], 'contradiction': []}
        num_samples = 0

        pbar = tqdm(dataloader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for batch in pbar:
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

                    pbar.set_postfix({'val_loss': f'{batch_loss.item()/batch_samples:.4f}'})
        
        pbar.close()

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




def train_asymmetry_model_only(processed_data_path: str, output_dir: str,
                              epochs: int = 80, batch_size: int = 1020, random_seed: int = 42):
    """Train new AsymmetryTransformModel independently on SBERT tokens"""
    
    print("=" * 80)
    print("TRAINING NEW ASYMMETRY TRANSFORM MODEL")
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
    
    # Train asymmetry model independently
    asymmetry_model = AsymmetryTransformModel(hidden_size=768)
    asymmetry_trainer = AsymmetryTrainer(asymmetry_model)  # No order model needed
    
    print(f"Training AsymmetryTransformModel directly on SBERT tokens")
    print("Target: Entailment=HIGH asymmetric, Neutral=LOW asymmetric, Contradiction=MEDIUM")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    
    best_val_loss = float('inf')
    patience = 15  # Increased patience for longer training
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
            asymmetry_model_path = Path(output_dir) / "mnli_asymmetry_transform_model_(match_SNLI_v2).pt"
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
            
            print(f"✅ New asymmetry model saved: {asymmetry_model_path}")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Generate training plots
    plot_new_asymmetry_training(asymmetry_trainer, output_dir)
    
    return asymmetry_model, asymmetry_trainer

def plot_new_asymmetry_training(asymmetry_trainer: AsymmetryTrainer, output_dir: str):
    """Plot training progress for the new asymmetry model"""
    
    from datetime import datetime
    import matplotlib.pyplot as plt
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Training and validation loss
    epochs = range(1, len(asymmetry_trainer.train_losses) + 1)
    ax1.plot(epochs, asymmetry_trainer.train_losses, 'purple', label='Training Loss', linewidth=2)
    if asymmetry_trainer.val_losses:
        ax1.plot(epochs, asymmetry_trainer.val_losses, 'orange', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Asymmetric Loss')
    ax1.set_title('New Asymmetry Model: Training Progress (SBERT Input)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Forward vs Backward energy patterns
    if asymmetry_trainer.asymmetry_stats:
        entail_forward = [s.get('entailment', {}).get('forward_energy_mean', 0) for s in asymmetry_trainer.asymmetry_stats]
        entail_backward = [s.get('entailment', {}).get('backward_energy_mean', 0) for s in asymmetry_trainer.asymmetry_stats]
        
        neutral_forward = [s.get('neutral', {}).get('forward_energy_mean', 0) for s in asymmetry_trainer.asymmetry_stats]
        neutral_backward = [s.get('neutral', {}).get('backward_energy_mean', 0) for s in asymmetry_trainer.asymmetry_stats]
        
        contra_forward = [s.get('contradiction', {}).get('forward_energy_mean', 0) for s in asymmetry_trainer.asymmetry_stats]
        contra_backward = [s.get('contradiction', {}).get('backward_energy_mean', 0) for s in asymmetry_trainer.asymmetry_stats]
        
        epochs_val = range(1, len(entail_forward) + 1)
        
        # Forward energies (solid lines)
        ax2.plot(epochs_val, entail_forward, 'g-', label='Entailment Forward', linewidth=2)
        ax2.plot(epochs_val, neutral_forward, 'b-', label='Neutral Forward', linewidth=2)
        ax2.plot(epochs_val, contra_forward, 'r-', label='Contradiction Forward', linewidth=2)
        
        # Backward energies (dashed lines)
        ax2.plot(epochs_val, entail_backward, 'g--', label='Entailment Backward', linewidth=2)
        ax2.plot(epochs_val, neutral_backward, 'b--', label='Neutral Backward', linewidth=2)
        ax2.plot(epochs_val, contra_backward, 'r--', label='Contradiction Backward', linewidth=2)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Energy')
        ax2.set_title('New Asymmetry Model: Forward vs Backward Energy Patterns')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Asymmetric energy convergence (main target)
        entail_asym = [s.get('entailment', {}).get('asymmetric_energy_mean', 0) for s in asymmetry_trainer.asymmetry_stats]
        neutral_asym = [s.get('neutral', {}).get('asymmetric_energy_mean', 0) for s in asymmetry_trainer.asymmetry_stats]
        contra_asym = [s.get('contradiction', {}).get('asymmetric_energy_mean', 0) for s in asymmetry_trainer.asymmetry_stats]
        
        ax3.plot(epochs_val, entail_asym, 'g-', label='Entailment', linewidth=3, marker='o')
        ax3.plot(epochs_val, neutral_asym, 'b-', label='Neutral', linewidth=3, marker='s')
        ax3.plot(epochs_val, contra_asym, 'r-', label='Contradiction', linewidth=3, marker='^')
        
        # Add target lines
        ax3.axhline(y=3.5, color='green', linestyle=':', alpha=0.7, label='Entailment Target (3.5)')
        ax3.axhline(y=2.5, color='blue', linestyle=':', alpha=0.7, label='Neutral Target (2.5)')
        ax3.axhline(y=2.8, color='red', linestyle=':', alpha=0.7, label='Contradiction Target (2.8)')
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Asymmetric Energy')
        ax3.set_title('New Asymmetry Model: Target Learning (HIGH→LOW→MED)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Energy separation analysis
        if len(entail_asym) > 1:
            # Gap between entailment and neutral (should be positive and growing)
            entail_neutral_gap = [e - n for e, n in zip(entail_asym, neutral_asym)]
            # Gap between contradiction and neutral (should be smaller positive)
            contra_neutral_gap = [c - n for c, n in zip(contra_asym, neutral_asym)]
            
            ax4.plot(epochs_val, entail_neutral_gap, 'cyan', label='Entailment - Neutral Gap', linewidth=2, marker='o')
            ax4.plot(epochs_val, contra_neutral_gap, 'magenta', label='Contradiction - Neutral Gap', linewidth=2, marker='s')
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.axhline(y=1.0, color='green', linestyle=':', alpha=0.7, label='Target Entail-Neutral Gap (1.0)')
            ax4.axhline(y=0.3, color='red', linestyle=':', alpha=0.7, label='Target Contra-Neutral Gap (0.3)')
            
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Energy Gap')
            ax4.set_title('New Asymmetry Model: Class Separation Gaps')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Add success annotation
            if len(entail_neutral_gap) > 0:
                final_entail_gap = entail_neutral_gap[-1]
                final_contra_gap = contra_neutral_gap[-1]
                
                if final_entail_gap > 0.7 and final_contra_gap > 0.1 and final_entail_gap > final_contra_gap:
                    success_text = '✅ Good separation:\nEntailment > Contradiction > Neutral'
                    color = 'lightgreen'
                else:
                    success_text = '❌ Insufficient separation'
                    color = 'lightcoral'
                
                ax4.text(0.02, 0.98, success_text, transform=ax4.transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = Path(output_dir) / f"mnli__asymmetry_(match_SNLI_v2)_training_progress_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training plots saved to: {plot_path}")
    
    # Generate summary report
    generate_training_summary(asymmetry_trainer, output_dir, timestamp)

def generate_training_summary(asymmetry_trainer: AsymmetryTrainer, output_dir: str, timestamp: str):
    """Generate detailed training summary"""
    
    summary_file = Path(output_dir) / f"mnli_asymmetry_training_(match_SNLI_v2)_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("NEW ASYMMETRY MODEL TRAINING SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write("MODEL ARCHITECTURE:\n")
        f.write("-"*18 + "\n")
        f.write("Input: SBERT tokens directly (not order embeddings)\n")
        f.write("Architecture: Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm → ReLU → Linear\n")
        f.write("Output: Asymmetric features in same 768D space\n\n")
        
        f.write("TRAINING TARGETS:\n")
        f.write("-"*15 + "\n")
        f.write("Entailment: HIGH asymmetric energy (target: 3.5)\n")
        f.write("Neutral: LOW asymmetric energy (target: 2.5)\n")
        f.write("Contradiction: MEDIUM asymmetric energy (target: 2.8)\n")
        f.write("Rationale: Entailment = strong directional, Neutral = balanced, Contradiction = conflicted\n\n")
        
        f.write("TRAINING PROGRESS:\n")
        f.write("-"*16 + "\n")
        f.write(f"Total epochs: {len(asymmetry_trainer.train_losses)}\n")
        f.write(f"Final train loss: {asymmetry_trainer.train_losses[-1]:.4f}\n")
        if asymmetry_trainer.val_losses:
            f.write(f"Final val loss: {asymmetry_trainer.val_losses[-1]:.4f}\n")
        
        if asymmetry_trainer.asymmetry_stats:
            f.write("\nFINAL ENERGY PATTERNS:\n")
            f.write("-"*21 + "\n")
            final_stats = asymmetry_trainer.asymmetry_stats[-1]
            for label, stats in final_stats.items():
                f.write(f"{label}:\n")
                f.write(f"  Forward energy: {stats.get('forward_energy_mean', 0):.3f}\n")
                f.write(f"  Backward energy: {stats.get('backward_energy_mean', 0):.3f}\n")
                f.write(f"  Asymmetric energy: {stats.get('asymmetric_energy_mean', 0):.3f}\n")
                f.write(f"  Samples: {stats.get('count', 0)}\n\n")
            
            # Analyze success
            entail_asym = final_stats.get('entailment', {}).get('asymmetric_energy_mean', 0)
            neutral_asym = final_stats.get('neutral', {}).get('asymmetric_energy_mean', 0)
            contra_asym = final_stats.get('contradiction', {}).get('asymmetric_energy_mean', 0)
            
            f.write("SUCCESS ANALYSIS:\n")
            f.write("-"*15 + "\n")
            if entail_asym > neutral_asym and contra_asym > neutral_asym:
                f.write("✅ CORRECT PATTERN: Entailment & Contradiction > Neutral\n")
                if entail_asym > contra_asym:
                    f.write("✅ IDEAL: Entailment > Contradiction > Neutral\n")
                else:
                    f.write("⚠️  PARTIAL: Contradiction ≥ Entailment > Neutral\n")
            else:
                f.write("❌ INCORRECT PATTERN: Neutral not lowest\n")
        
        f.write("\nNEXT STEPS:\n")
        f.write("-"*10 + "\n")
        f.write("1. Update point_cloud_clustering_test.py to use new model\n")
        f.write("2. Test updated stratification with corrected energy patterns\n")
        f.write("3. Compare clustering performance against old asymmetry model\n")
        f.write("4. Monitor forward/backward energy patterns for neutral detection\n")
    
    print(f"Training summary saved to: {summary_file}")

def main():
    """Train new asymmetry model only"""
    
    processed_data_path = "/vol/bitbucket/ahb24/tda_entailment_new/mnli_train_sbert_tokens.pkl"
    output_dir = "MSc_Topology_Codebase/phd_method/models/separate_models/"
    os.makedirs(output_dir, exist_ok=True)
    
    if not Path(processed_data_path).exists():
        print(f"Processed data not found at: {processed_data_path}")
        return
    
    # Train new asymmetry model with longer training
    asymmetry_model, asymmetry_trainer = train_asymmetry_model_only(
        processed_data_path, output_dir, epochs=80, batch_size=1020, random_seed=42
    )
    
    print("\n" + "="*80)
    print("NEW ASYMMETRY MODEL TRAINING COMPLETE!")
    print("✅ Trained for up to 50 epochs with early stopping")
    print("✅ Now uses SBERT tokens directly (not order embeddings)")
    print("✅ Corrected loss: Entailment=HIGH, Neutral=LOW, Contradiction=MEDIUM")
    print("✅ Includes forward/backward energy computation")
    print("✅ Training plots and summary generated")
    print("="*80)
    
    return asymmetry_model


if __name__ == "__main__":
    main()


