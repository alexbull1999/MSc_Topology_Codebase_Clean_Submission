import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple
import json
import random

def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set generator for random_split to be reproducible
    return torch.Generator().manual_seed(seed)


class OrderEmbeddingModel(nn.Module):
    """Enhanced Order embedding model with asymmetry loss for better neutral/contradiction separation
    Following Vendrov et al. (2015) methodology with additional asymmetric relationship modeling
    """

    def __init__(self, bert_dim: int = 768, order_dim: int = 50, asymmetry_weight: float = 0.2):
        """Initialise enhanced order embedding model
        Args:
            bert_dim: Dimension of BERT input embeddings
            order_dim: Order embedding dimension (using 50D as per Vendrov et al.)
            asymmetry_weight: Weight for asymmetric loss component
        """
        super().__init__()
        self.bert_dim = bert_dim
        self.order_dim = order_dim
        self.asymmetry_weight = asymmetry_weight

        # Enhanced projection with batch normalization and dropout for reduced variance
        self.to_order_space = nn.Sequential(
            nn.Linear(bert_dim, order_dim * 2),
            nn.BatchNorm1d(order_dim * 2),  # Added for standardization
            nn.ReLU(),
            nn.Dropout(0.3),               # Increased from 0.3 to reduce overfitting
            nn.Linear(order_dim * 2, order_dim),
            nn.BatchNorm1d(order_dim),     # Final batch norm for consistent outputs
            nn.ReLU()  # Ensures non-negative coordinates for reversed product order
        )

        # Additional layer for asymmetric relationship modeling
        self.asymmetric_projection = nn.Sequential(
            nn.Linear(order_dim, order_dim),
            nn.BatchNorm1d(order_dim),     # Added batch norm
            nn.Tanh(),  # Allow negative values for directional encoding
            nn.Dropout(0.2)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, bert_embeddings: torch.Tensor) -> torch.Tensor:
        """Map BERT embeddings to order embedding space
        Args:
            bert_embeddings: Input BERT embeddings [batch_size, bert_dim]
        Returns:
            Order embeddings [batch_size, order_dim] with non-negative coordinates
        """
        return self.to_order_space(bert_embeddings)

    def get_asymmetric_features(self, order_embeddings: torch.Tensor) -> torch.Tensor:
        """Get asymmetric features for directional relationship modeling
        Args:
            order_embeddings: Order embeddings [batch_size, order_dim]
        Returns:
            Asymmetric features [batch_size, order_dim]
        """
        return self.asymmetric_projection(order_embeddings)

    def order_violation_energy(self, u_emb: torch.Tensor, v_emb: torch.Tensor) -> torch.Tensor:
        """Computing order violation energy, following Vendrov et al. (2015) methodology
        E(u, v) = ||max(0, v-u)||^2
        
        Args:
            u_emb: Order embeddings for u [batch_size, order_dim]
            v_emb: Order embeddings for v [batch_size, order_dim]
        Returns:
            Violation energies [batch_size]
        """
        # Element-wise max(0, v-u)
        violation = torch.clamp(v_emb - u_emb, min=0)
        # L2 norm squared
        energy = torch.norm(violation, dim=-1, p=2) ** 2
        return energy

    def asymmetric_energy(self, premise_emb: torch.Tensor, hypothesis_emb: torch.Tensor) -> torch.Tensor:
        """Compute asymmetric relationship energy using asymmetric features
        Args:
            premise_emb: Premise order embeddings [batch_size, order_dim]
            hypothesis_emb: Hypothesis order embeddings [batch_size, order_dim]
        Returns:
            Asymmetric energies [batch_size]
        """
        # Get asymmetric features
        premise_asym = self.get_asymmetric_features(premise_emb)
        hypothesis_asym = self.get_asymmetric_features(hypothesis_emb)
        
        # Compute directional difference
        directional_diff = premise_asym - hypothesis_asym
        
        # L2 norm of directional difference
        asymmetric_energy = torch.norm(directional_diff, dim=-1, p=2)
        return asymmetric_energy

    def compute_bidirectional_energies(self, premise_emb: torch.Tensor, hypothesis_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute both forward and backward energies for asymmetry analysis
        Args:
            premise_emb: Premise order embeddings
            hypothesis_emb: Hypothesis order embeddings
        Returns:
            Dictionary containing forward, backward, and asymmetric energies
        """
        # Standard order violation energies
        forward_energy = self.order_violation_energy(premise_emb, hypothesis_emb)
        backward_energy = self.order_violation_energy(hypothesis_emb, premise_emb)
        
        # Asymmetric relationship energy
        asym_energy = self.asymmetric_energy(premise_emb, hypothesis_emb)
        
        # Asymmetry measure: difference between forward and backward
        asymmetry_measure = torch.abs(forward_energy - backward_energy)
        
        return {
            'forward_energy': forward_energy,
            'backward_energy': backward_energy,
            'asymmetric_energy': asym_energy,
            'asymmetry_measure': asymmetry_measure
        }


class EntailmentDataset(Dataset):
    """Dataset for entailment training"""
    def __init__(self, processed_data: Dict):
        """Initialize dataset from processed embeddings
        Args:
            processed_data: output from text_processing.py
        """
        self.premise_embeddings = processed_data['premise_embeddings']
        self.hypothesis_embeddings = processed_data['hypothesis_embeddings']
        self.labels = processed_data['labels']

        # Convert labels to integers
        self.label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        self.numeric_labels = torch.tensor([self.label_to_idx[label] for label in self.labels])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'premise_emb': self.premise_embeddings[idx],
            'hypothesis_emb': self.hypothesis_embeddings[idx],
            'label': self.numeric_labels[idx],
            'label_str': self.labels[idx]
        }


class OrderEmbeddingTrainer:
    """Enhanced trainer for Order Embedding Model with asymmetric loss"""

    def __init__(self, model: OrderEmbeddingModel, device='cuda' if torch.cuda.is_available() else 'cpu', lr: float = 1e-3,
                l1_lambda: float = 1e-4, consistency_weight: float = 0.2, separation_weight: float=0.2):
        """
        Initialize the trainer.
        Args:
            model: Enhanced order embedding model
            device: device to train on
        """
        self.model = model.to(device)
        self.device = device
        self.l1_lambda = l1_lambda
        self.consistency_weight = consistency_weight
        self.separation_weight = separation_weight
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-3) #was 1e-3 for weight decay
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=15, factor=0.3
        )

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.energy_rankings = []
        self.asymmetry_stats = []

    def compute_l1_regularization(self) -> torch.Tensor:
        """Compute L1 regularization term for all model parameters"""
        l1_loss = 0
        for param in self.model.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return l1_loss

    def compute_consistency_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Penalize high variance within classes to reduce standard deviations
        
        Args:
            embeddings: Order embeddings [batch_size, order_dim]
            labels: Class labels [batch_size]
        Returns:
            Consistency loss that penalizes within-class variance
        """
        consistency_loss = 0
        num_classes = 0
        
        for label_idx in [0, 1, 2]:  # entailment, neutral, contradiction
            mask = (labels == label_idx)
            if mask.sum() > 1:  # Need at least 2 samples to compute variance
                class_embs = embeddings[mask]
                # Compute variance across samples for each dimension, then average
                class_var = torch.var(class_embs, dim=0, unbiased=False).mean()
                consistency_loss += class_var
                num_classes += 1
        
        return consistency_loss / max(num_classes, 1)

    def compute_asymmetric_loss(self, premise_embs: torch.Tensor, hypothesis_embs: torch.Tensor, 
                               labels: torch.Tensor, label_strs: List[str]) -> torch.Tensor:
        """Compute asymmetric loss to help distinguish neutral vs contradiction
        
        Key insight:
        - Entailment: Low forward energy (premise → hypothesis), high backward energy
        - Neutral: Medium energy in both directions (symmetric)
        - Contradiction: High forward energy, variable backward energy (asymmetric)
        
        Args:
            premise_embs: Premise BERT embeddings
            hypothesis_embs: Hypothesis BERT embeddings
            labels: Numeric labels
            label_strs: String labels for detailed analysis
            
        Returns:
            Asymmetric loss term
        """
        # Get order embeddings
        premise_order = self.model(premise_embs)
        hypothesis_order = self.model(hypothesis_embs)
        
        # Compute bidirectional energies
        energy_dict = self.model.compute_bidirectional_energies(premise_order, hypothesis_order)
        
        asymmetric_loss = 0.0
        batch_size = len(labels)
        
        for i, label_str in enumerate(label_strs):
            forward_e = energy_dict['forward_energy'][i]
            backward_e = energy_dict['backward_energy'][i]
            asym_e = energy_dict['asymmetric_energy'][i]
            
            if label_str == 'entailment':
                # For entailment: enforce low forward energy, higher backward energy
                asymmetric_loss += F.mse_loss(forward_e, torch.tensor(0.0, device=self.device))
                asymmetric_loss += F.relu(1.0 - backward_e)  # Encourage backward_e > 1.0
                
            elif label_str == 'neutral':
                # For neutral: enforce symmetric relationship (similar forward/backward energies)
                # Target medium energy levels
                target_energy = 1.0
                asymmetric_loss += F.mse_loss(forward_e, torch.tensor(target_energy, device=self.device))
                asymmetric_loss += F.mse_loss(backward_e, torch.tensor(target_energy, device=self.device))
                # Penalty for high asymmetry
                asymmetric_loss += 0.5 * torch.abs(forward_e - backward_e)
                
            elif label_str == 'contradiction':
                # For contradiction: enforce high forward energy and high asymmetry
                asymmetric_loss += F.relu(1.5 - forward_e)  # Encourage forward_e > 1.5
                # Encourage high asymmetric energy (directional opposition)
                asymmetric_loss += F.relu(1.0 - asym_e)  # Encourage asym_e > 1.0
                # Allow variable backward energy (contradiction can be asymmetric)
        
        return asymmetric_loss / batch_size


    def compute_enhanced_3way_loss_with_separation(self, premise_embs: torch.Tensor, hypothesis_embs: torch.Tensor, 
                                                  labels: torch.Tensor, margin: float = 1.5) -> torch.Tensor:
        """Enhanced 3-way loss with explicit separation penalties between all class pairs
        
        Args:
            premise_embs: Premise BERT embeddings
            hypothesis_embs: Hypothesis BERT embeddings  
            labels: Class labels
            margin: Base margin for separation
        Returns:
            Enhanced loss with separation penalties
        """
        premise_order = self.model(premise_embs)
        hypothesis_order = self.model(hypothesis_embs)
        energies = self.model.order_violation_energy(premise_order, hypothesis_order)
        
        # Standard 3-way loss
        loss = 0
        for target_label in [0, 1, 2]:
            target_mask = (labels == target_label)
            if target_mask.sum() == 0:
                continue
                
            target_energies = energies[target_mask]
            
            # Minimize energy for this class
            loss += target_energies.mean()
            
            # Maximize separation from other classes with stronger penalties
            for other_label in [0, 1, 2]:
                if other_label != target_label:
                    other_mask = (labels == other_label)
                    if other_mask.sum() > 0:
                        other_energies = energies[other_mask]
                        # Enhanced separation penalty - stronger than original
                        separation = other_energies.mean() - target_energies.mean()
                        loss += 2.0 * torch.clamp(margin - separation, min=0)  # Increased weight from 1.0 to 2.0
        
        return loss


    def compute_enhanced_loss(self, premise_embs: torch.Tensor, hypothesis_embs: torch.Tensor, 
                        labels: torch.Tensor, label_strs: List[str], margin: float = 1.5) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Compute enhanced loss with proper 3-class distinction (FIXED - no nested loops)
        Args:
            premise_embs: Premise BERT embeddings
            hypothesis_embs: Hypothesis BERT embeddings
            labels: Label indices (0=entailment, 1=neutral, 2=contradiction)
            label_strs: String labels
            margin: Base margin for class separation
        Returns:
            Total loss, standard energies, and energy statistics
        """
        # Get order embeddings
        premise_order = self.model(premise_embs)
        hypothesis_order = self.model(hypothesis_embs)

        # Compute standard violation energies (forward direction)
        energies = self.model.order_violation_energy(premise_order, hypothesis_order)

        # Create masks for each class
        entailment_mask = (labels == 0)
        neutral_mask = (labels == 1)
        contradiction_mask = (labels == 2)

        # Base 3-class max-margin loss with different energy targets
        total_loss = 0.0
    
        # Entailment pairs: minimize energy (target ~0)
        if entailment_mask.any():
            entailment_energies = energies[entailment_mask]
            entailment_loss = entailment_energies.mean()
            total_loss += entailment_loss
    
        # Neutral pairs: medium energy (target ~margin)
        if neutral_mask.any():
            neutral_energies = energies[neutral_mask]
            target_neutral = margin
            # Loss if energy is too low (< margin/2) or too high (> 2*margin)
            neutral_loss = (
                torch.clamp(margin/2 - neutral_energies, min=0).mean() +  # Push up if too low
                torch.clamp(neutral_energies - 2*margin, min=0).mean()    # Push down if too high
            )
            total_loss += neutral_loss
    
        # Contradiction pairs: high energy (target ~2*margin)
        if contradiction_mask.any():
            contradiction_energies = energies[contradiction_mask]
            target_contradiction = 2 * margin
            # Loss if energy is too low (should be > 1.5*margin)
            contradiction_loss = torch.clamp(1.5*margin - contradiction_energies, min=0).mean()
            total_loss += contradiction_loss
    
        # Enhanced ranking losses with stronger separation penalties
        ranking_loss = 0.0
        separation_margin = 0.8  # Increased from 0.5
    
        # Entailment should have lower energy than neutral (vectorized)
        if entailment_mask.any() and neutral_mask.any():
            ent_mean = energies[entailment_mask].mean()
            neut_mean = energies[neutral_mask].mean()
            ranking_loss += torch.clamp(ent_mean - neut_mean + separation_margin, min=0)
    
        # Neutral should have lower energy than contradiction (vectorized)
        if neutral_mask.any() and contradiction_mask.any():
            neut_mean = energies[neutral_mask].mean()
            cont_mean = energies[contradiction_mask].mean()
            ranking_loss += torch.clamp(neut_mean - cont_mean + separation_margin, min=0)
    
        # Entailment should have lower energy than contradiction (vectorized)
        if entailment_mask.any() and contradiction_mask.any():
            ent_mean = energies[entailment_mask].mean()
            cont_mean = energies[contradiction_mask].mean()
            ranking_loss += torch.clamp(ent_mean - cont_mean + separation_margin, min=0)
    
        # Combine base losses
        standard_loss = total_loss + 0.5 * ranking_loss
    
        # NEW: Enhanced 3-way loss with separation penalty
        separation_loss = self.compute_enhanced_3way_loss_with_separation(premise_embs, hypothesis_embs, labels, margin)
    
        # NEW: Consistency loss (reduces within-class variance)
        consistency_loss = self.compute_consistency_loss(premise_order, labels) + \
                          self.compute_consistency_loss(hypothesis_order, labels)
        
        # Asymmetric loss (existing)
        asymmetric_loss = self.compute_asymmetric_loss(premise_embs, hypothesis_embs, labels, label_strs)
        
        # NEW: L1 regularization
        l1_loss = self.compute_l1_regularization()

        # Combined loss with all components
        total_loss = (standard_loss + 
                     self.separation_weight * separation_loss +
                     self.consistency_weight * consistency_loss +
                     self.model.asymmetry_weight * asymmetric_loss +
                     self.l1_lambda * l1_loss)

        # Compute energy statistics for monitoring
        energy_stats = self._compute_energy_statistics(premise_order, hypothesis_order, label_strs)

        return total_loss, energies, energy_stats

    def _compute_energy_statistics(self, premise_order: torch.Tensor, hypothesis_order: torch.Tensor, 
                                 label_strs: List[str]) -> Dict:
        """Compute detailed energy statistics for monitoring"""
        energy_dict = self.model.compute_bidirectional_energies(premise_order, hypothesis_order)
        
        stats = {}
        for label in ['entailment', 'neutral', 'contradiction']:
            label_mask = [i for i, l in enumerate(label_strs) if l == label]
            if label_mask:
                stats[label] = {
                    'forward_energy': energy_dict['forward_energy'][label_mask].mean().item(),
                    'backward_energy': energy_dict['backward_energy'][label_mask].mean().item(),
                    'asymmetric_energy': energy_dict['asymmetric_energy'][label_mask].mean().item(),
                    'asymmetry_measure': energy_dict['asymmetry_measure'][label_mask].mean().item(),
                }
        
        return stats

    def train_epoch(self, dataloader: DataLoader, margin: float = 1.5) -> float:
        """Train for one epoch with enhanced loss"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            premise_embs = batch['premise_emb'].to(self.device)
            hypothesis_embs = batch['hypothesis_emb'].to(self.device)
            labels = batch['label'].to(self.device)
            label_strs = batch['label_str']

            self.optimizer.zero_grad()

            loss, _, _ = self.compute_enhanced_loss(premise_embs, hypothesis_embs, labels, label_strs, margin)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    def evaluate(self, dataloader: DataLoader, margin=1.5) -> Tuple[float, Dict]:
        """Evaluate model and compute enhanced energy rankings"""
        self.model.eval()
        total_loss = 0
        all_energies = {'entailment': [], 'neutral': [], 'contradiction': []}
        all_asymmetry_stats = {'entailment': [], 'neutral': [], 'contradiction': []}

        with torch.no_grad():
            for batch in dataloader:
                premise_embs = batch['premise_emb'].to(self.device)
                hypothesis_embs = batch['hypothesis_emb'].to(self.device)
                labels = batch['label'].to(self.device)
                label_strs = batch['label_str']

                loss, energies, energy_stats = self.compute_enhanced_loss(premise_embs, hypothesis_embs, labels, label_strs, margin)
                total_loss += loss.item()

                # Collect energies by label
                for i, label_str in enumerate(label_strs):
                    all_energies[label_str].append(energies[i].item())

                # Collect asymmetry statistics
                for label, stats in energy_stats.items():
                    if label not in all_asymmetry_stats:
                        all_asymmetry_stats[label] = []
                    all_asymmetry_stats[label].append(stats)

        avg_loss = total_loss / len(dataloader)
        self.val_losses.append(avg_loss)

        # Compute mean energies by label
        energy_summary = {}
        for label, energies in all_energies.items():
            if energies:
                energy_summary[label] = {
                    'mean': np.mean(energies),
                    'std': np.std(energies),
                    'count': len(energies)
                }

        # Store asymmetry statistics
        self.asymmetry_stats.append(all_asymmetry_stats)
        
        self.energy_rankings.append(energy_summary)
        return avg_loss, energy_summary

def train_order_embeddings(processed_data_path: str, output_dir: str = "models/",
                           epochs: int = 50, batch_size: int = 32, order_dim: int = 50,
                           asymmetry_weight: float = 0.2, lr: float = 1e-3, margin: float = 1.5, 
                           l1_lambda: float = 1e-4, consistency_weight: float = 0.2, 
                           separation_weight: float = 0.2, random_seed: int = 42):
    """Train enhanced order embedding model with asymmetric loss"""

    generator = set_random_seed(random_seed)

    print(f"Loading data from {processed_data_path}")
    processed_data = torch.load(processed_data_path)

    # Create dataset and dataloaders
    dataset = EntailmentDataset(processed_data)

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training on {train_size} samples, validating on {val_size} samples")

    # Initialize enhanced model and trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OrderEmbeddingModel(bert_dim=768, order_dim=order_dim, asymmetry_weight=asymmetry_weight)
    trainer = OrderEmbeddingTrainer(model, device, lr=lr)

    print(f"Training on {device} with asymmetry_weight={asymmetry_weight}")
    print(f"  - Order dim: {order_dim}")
    print(f"  - Asymmetry weight: {asymmetry_weight}")
    print(f"  - L1 lambda: {l1_lambda}")
    print(f"  - Consistency weight: {consistency_weight}")
    print(f"  - Separation weight: {separation_weight}")


    # Training loop
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader, margin=margin)

        # Validate
        val_loss, energy_stats = trainer.evaluate(val_loader, margin=margin)
        trainer.scheduler.step(val_loss)

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1} / {epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if energy_stats:
                print("  Energy Rankings:")
                for label, stats in energy_stats.items():
                    print(f"    {label}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'bert_dim': 768,
                    'order_dim': order_dim,
                    'asymmetry_weight': asymmetry_weight,
                    'l1_lambda': l1_lambda,
                    'consistency_weight': consistency_weight,
                },
                'training_stats': {
                    'train_losses': trainer.train_losses,
                    'val_losses': trainer.val_losses,
                    'energy_rankings': trainer.energy_rankings,
                    'asymmetry_stats': trainer.asymmetry_stats,
                },
                'best_val_loss': best_val_loss,
                'epoch': epoch,
            }, os.path.join(output_dir, "enhanced_order_embeddings_snli_SBERT_full_3way.pt"))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print("Training completed")
    return model, trainer


def validate_energy_rankings(trainer: OrderEmbeddingTrainer) -> bool:
    """Validate that energy rankings follow the expected pattern with asymmetry analysis"""
    if not trainer.energy_rankings:
        print("No energy rankings found")
        return False

    final_rankings = trainer.energy_rankings[-1]
    print("Final Energy Rankings:")
    
    entail_mean = final_rankings.get('entailment', {}).get('mean', float('inf'))
    neutral_mean = final_rankings.get('neutral', {}).get('mean', float('inf'))
    contra_mean = final_rankings.get('contradiction', {}).get('mean', float('inf'))

    print(f"    Entailment: {entail_mean:.4f}")
    print(f"    Neutral: {neutral_mean:.4f}")
    print(f"    Contradiction: {contra_mean:.4f}")

    # Check for proper ranking
    ranking_correct = entail_mean < neutral_mean and entail_mean < contra_mean
    binary_separation = abs(entail_mean - min(neutral_mean, contra_mean)) > 0.5

    print(f"Binary separation (entailment vs others): {binary_separation}")
    print(f"Neutral vs Contradiction gap: {abs(neutral_mean - contra_mean):.4f}")

    if ranking_correct and binary_separation:
        print("Energy rankings show good entailment separation!")
    else:
        print("Energy rankings may need adjustment")

    # NEW: Calculate gap-to-std ratio for assessment
    entail_std = final_rankings.get('entailment', {}).get('std', 1.0)
    neutral_std = final_rankings.get('neutral', {}).get('std', 1.0)
    contra_std = final_rankings.get('contradiction', {}).get('std', 1.0)
    
    # Calculate average gap / average std
    gap1 = neutral_mean - entail_mean
    gap2 = contra_mean - neutral_mean
    avg_gap = (gap1 + gap2) / 2
    avg_std = (entail_std + neutral_std + contra_std) / 3
    gap_to_std_ratio = avg_gap / avg_std if avg_std > 0 else 0
    
    print(f"\nGap-to-Std Analysis:")
    print(f"  Average gap: {avg_gap:.4f}")
    print(f"  Average std: {avg_std:.4f}")
    print(f"  Gap-to-std ratio: {gap_to_std_ratio:.3f}")
    print(f"  Target ratio for Phase 2: ≥2.2")

    # Print asymmetry statistics if available
    if trainer.asymmetry_stats:
        print("\nAsymmetry Analysis:")
        final_asym = trainer.asymmetry_stats[-1]
        for label, stats_list in final_asym.items():
            if stats_list:
                avg_stats = {}
                for key in stats_list[0].keys():
                    avg_stats[key] = np.mean([s[key] for s in stats_list])
                
                print(f"  {label}:")
                print(f"    Forward Energy: {avg_stats['forward_energy']:.4f}")
                print(f"    Backward Energy: {avg_stats['backward_energy']:.4f}")
                print(f"    Asymmetric Energy: {avg_stats['asymmetric_energy']:.4f}")
                print(f"    Asymmetry Measure: {avg_stats['asymmetry_measure']:.4f}")

    return ranking_correct and binary_separation

def plot_training_progress(trainer: OrderEmbeddingTrainer, save_path: str = "plots/"):
    """Plot enhanced training progress including asymmetry analysis"""
    os.makedirs(save_path, exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

    # Plot training/validation loss
    epochs = range(1, len(trainer.train_losses) + 1)
    ax1.plot(epochs, trainer.train_losses, 'b-', label="Training Loss")
    if trainer.val_losses:
        ax1.plot(epochs, trainer.val_losses, 'r-', label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot energy rankings over time
    if trainer.energy_rankings:
        entail_means = []
        neutral_means = []
        contra_means = []

        for ranking in trainer.energy_rankings:
            entail_means.append(ranking.get('entailment', {}).get('mean', 0))
            neutral_means.append(ranking.get('neutral', {}).get('mean', 0))
            contra_means.append(ranking.get('contradiction', {}).get('mean', 0))

        epochs_val = range(1, len(entail_means) + 1)
        ax2.plot(epochs_val, entail_means, 'g-', label="Entailment", linewidth=2)
        ax2.plot(epochs_val, neutral_means, 'b-', label="Neutral", linewidth=2)
        ax2.plot(epochs_val, contra_means, 'r-', label="Contradiction", linewidth=2)

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Mean Order Violation Energy")
        ax2.set_title("Energy Rankings Over Training")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Plot asymmetry measures if available
    if trainer.asymmetry_stats:
        try:
            entail_asym = []
            neutral_asym = []
            contra_asym = []

            for asym_epoch in trainer.asymmetry_stats:
                for label, stats_list in asym_epoch.items():
                    if stats_list:
                        avg_asym = np.mean([s['asymmetry_measure'] for s in stats_list])
                        if label == 'entailment':
                            entail_asym.append(avg_asym)
                        elif label == 'neutral':
                            neutral_asym.append(avg_asym)
                        elif label == 'contradiction':
                            contra_asym.append(avg_asym)

            epochs_asym = range(1, len(entail_asym) + 1)
            ax3.plot(epochs_asym, entail_asym, 'g-', label="Entailment", linewidth=2)
            ax3.plot(epochs_asym, neutral_asym, 'b-', label="Neutral", linewidth=2)
            ax3.plot(epochs_asym, contra_asym, 'r-', label="Contradiction", linewidth=2)
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Asymmetry Measure")
            ax3.set_title("Asymmetry Measures Over Training")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        except Exception as e:
            ax3.text(0.5, 0.5, f"Asymmetry plot error: {str(e)}", transform=ax3.transAxes, ha='center')

    # Plot forward vs backward energy comparison
    if trainer.asymmetry_stats and trainer.asymmetry_stats[-1]:
        try:
            labels = []
            forward_energies = []
            backward_energies = []

            final_asym = trainer.asymmetry_stats[-1]
            for label, stats_list in final_asym.items():
                if stats_list:
                    labels.append(label)
                    forward_energies.append(np.mean([s['forward_energy'] for s in stats_list]))
                    backward_energies.append(np.mean([s['backward_energy'] for s in stats_list]))

            x = np.arange(len(labels))
            width = 0.35

            ax4.bar(x - width/2, forward_energies, width, label='Forward Energy', alpha=0.8)
            ax4.bar(x + width/2, backward_energies, width, label='Backward Energy', alpha=0.8)
            ax4.set_xlabel('Relationship Type')
            ax4.set_ylabel('Energy')
            ax4.set_title('Forward vs Backward Energy Comparison')
            ax4.set_xticks(x)
            ax4.set_xticklabels(labels)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        except Exception as e:
            ax4.text(0.5, 0.5, f"Energy comparison plot error: {str(e)}", transform=ax4.transAxes, ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'enhanced_order_embedding_snli_SBERT_full_3way.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Enhanced training plots saved to {save_path}")

def test_order_embeddings():
    """Test enhanced order embeddings"""
    processed_data_path = "data/processed/snli_full_standard_SBERT.pt"
    if not os.path.exists(processed_data_path):
        print(f"Processed data not found at {processed_data_path}")
        return

    model, trainer = train_order_embeddings(
        processed_data_path=processed_data_path,
        epochs=100,
        batch_size=32,
        order_dim=75,
        asymmetry_weight=1.9,  # Adjust this parameter (was 0.2)
        random_seed=42
    )

    # Validate rankings
    ranking_correct = validate_energy_rankings(trainer)

    # Plot progress
    plot_training_progress(trainer)

    if ranking_correct:
        print("✓ Success: Enhanced order embeddings working!")
    else:
        print("⚠ Warning: Energy rankings may need further tuning")

    return model, trainer

if __name__ == "__main__":
    test_order_embeddings()
