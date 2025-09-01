import torch
import torch.nn as nn
import torch.optim as optim
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
    """Order embedding model, following Vendrov et al. (2015) methodology
        Maps BERT embeddings to non-negative, asymmetric order embedding space of 50-D
    """

    def __init__(self, bert_dim: int = 768, order_dim: int = 50):
        """Initialise order embedding model
        Args:
            bert_dim: Dimension of BERT input embeddings
            order_dim: Order embedding dimension (using 50D as per Vendrov et al.)
        """
        super().__init__()
        self.bert_dim = bert_dim
        self.order_dim = order_dim

        #Project BERT embeddings to order space
        self.to_order_space = nn.Sequential(
            nn.Linear(bert_dim, order_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3), #was 0.1
            nn.Linear(order_dim * 2, order_dim),
            nn.ReLU() # Ensures non-negative coordinates for reversed product order
        )

        #Initialize weights
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

    def order_violation_energy(self, u_emb: torch.Tensor, v_emb: torch.Tensor) -> torch.Tensor:
        """Computing order violation energy, following Vendrov et al. (2015) methodology, using equation:
        E(u, v) = ||max(0, v-u)||^2
        For entailment: u ⪯ v (premise ⪯ hypothesis), so E should be 0
        For violation: E > 0

        Args:
            u_emb: Order embeddings for u [batch_size, order_dim]
            v_emb: Order embeddings for v [batch_size, order_dim]
        Returns:
            Violation energies [batch_size]
        """
        #Element-wise max(0, v-u)
        violation = torch.clamp(v_emb - u_emb, min=0)
        #L2 norm squared
        energy = torch.norm(violation, dim=-1, p=2) ** 2
        return energy

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

        #Convert labels to integers
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
    """Trainer for Order Embedding Model"""

    def __init__(self, model: OrderEmbeddingModel, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the trainer.
        Args:
            model: Order embedding model
            device: device to train on
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=15, factor=0.3
        ) #patience was 5 for small

        #Training history
        self.train_losses = []
        self.val_losses = []
        self.energy_rankings = []

    def compute_loss(self, premise_embs: torch.Tensor, hypothesis_embs: torch.Tensor, labels: torch.Tensor,
                     margin: float = 1.0): #was 1
        """Compute max-margin loss following Vendrov et al. 2015
        For entailment pairs: minimize E(premise, hypothesis)
        For non-entailment pairs: ensure E(premise, hypothesis) > margin
        Args:
            premise_embs: Order embeddings for premises
            hypothesis_embs: Order embeddings for hypothesis
            labels: Label indices (0=entailment, 1=neutral, 2=contradiction)
            margin: Margin for non-entailment pairs
        Returns:
            Loss value
        """

        # Get order embeddings
        premise_order = self.model(premise_embs)
        hypothesis_order = self.model(hypothesis_embs)

        #Compute violation energies
        energies = self.model.order_violation_energy(premise_order, hypothesis_order)

        #Separate entailment and non-entailment
        entailment_mask = (labels == 0)
        non_entailment_mask = (labels > 0)

        #Loss for entailment pairs - minimize energy
        entailment_loss = energies[entailment_mask].mean() if entailment_mask.any() else 0

        #Loss for non-entailment pairs - ensure energy > margin
        if non_entailment_mask.any():
            non_entailment_energies = energies[non_entailment_mask]
            non_entailment_loss = torch.clamp(margin-non_entailment_energies, min=0).mean()
        else:
            non_entailment_loss = 0

        total_loss = entailment_loss + non_entailment_loss
        return total_loss, energies

    def train_epoch(self, dataloader: DataLoader, margin: float=1.0) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            premise_embs = batch['premise_emb'].to(self.device)
            hypothesis_embs = batch['hypothesis_emb'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            loss, _ = self.compute_loss(premise_embs, hypothesis_embs, labels, margin)
            loss.backward()

            #Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """Evaluate model and compute energy rankings"""
        self.model.eval()
        total_loss = 0
        all_energies = {'entailment': [], 'neutral': [], 'contradiction': []}

        with torch.no_grad():
            for batch in dataloader:
                premise_embs = batch['premise_emb'].to(self.device)
                hypothesis_embs = batch['hypothesis_emb'].to(self.device)
                labels = batch['label'].to(self.device)
                label_strs = batch['label_str']

                loss, energies = self.compute_loss(premise_embs, hypothesis_embs, labels)
                total_loss += loss.item()

                #Collect energies by label
                for i, label_str in enumerate(label_strs):
                    all_energies[label_str].append(energies[i].item())

        avg_loss = total_loss / len(dataloader)
        self.val_losses.append(avg_loss)

        #Compute mean energies by label
        energy_stats = {}
        for label, energies in all_energies.items():
            if energies:
                energy_stats[label] = {
                    'mean': np.mean(energies),
                    'std': np.std(energies),
                    'count': len(energies)
                }
        self.energy_rankings.append(energy_stats)
        return avg_loss, energy_stats

def train_order_embeddings(processed_data_path: str, output_dir: str = "models/",
                           epochs: int = 50, batch_size: int = 32, order_dim: int = 50,
                           random_seed: int=42):
    """Train order embedding model on processed dataset"""

    generator = set_random_seed(random_seed)

    print(f"Loading data from {processed_data_path}")
    processed_data = torch.load(processed_data_path)

    #Create dataset and dataloaders
    dataset = EntailmentDataset(processed_data)

    #Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training on {train_size} sample, validating on {val_size} samples")

    #Initialize model and trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OrderEmbeddingModel(bert_dim=768, order_dim=order_dim)
    trainer = OrderEmbeddingTrainer(model, device)

    print(f"Training on {device}")

    #Training loop
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader)

        #Validate:
        val_loss, energy_stats = trainer.evaluate(val_loader)
        trainer.scheduler.step(val_loss)

        #Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1} / {epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if energy_stats:
                print("  Energy Rankings:")
                for label, stats in energy_stats.items():
                    print(f"    {label}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        #Early stopping
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
                },
                'training_stats': {
                    'train_losses': trainer.train_losses,
                    'val_losses': trainer.val_losses,
                    'energy_rankings': trainer.energy_rankings,
                },
                'best_val_loss': best_val_loss,
                'epoch': epoch,

            }, os.path.join(output_dir, "order_embeddings_snli_10k_tests.pt"))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print("Training completed")
    return model, trainer


def validate_energy_rankings(trainer: OrderEmbeddingTrainer) -> bool:
    """
    Validate that energy rankings follow the expected pattern, namely:
    entailment < neutral < contradiction

    Args:
        trainer: Trained model trainer

    Returns:
        True if rankings are correct
    """
    if not trainer.energy_rankings:
        print("No energy rankings found")
        return False

    final_rankings = trainer.energy_rankings[-1]
    print("Final Rankings:")
    entail_mean = final_rankings.get('entailment', {}).get('mean', float('inf'))
    neutral_mean = final_rankings.get('neutral', {}).get('mean', float('inf'))
    contra_mean = final_rankings.get('contradiction', {}).get('mean', float('inf'))

    print(f"    Entailment: {entail_mean:.4f}")
    print(f"    Neutral: {neutral_mean:.4f}")
    print(f"    Contradiction: {contra_mean:.4f}")

    ranking_correct = entail_mean < neutral_mean < contra_mean
    if ranking_correct:
        print("Energy Rankings correct!")
    else:
        print("Energy rankings incorrect - approach may need adjusting")

    return ranking_correct

def plot_training_progress(trainer: OrderEmbeddingTrainer, save_path: str = "plots/"):
    """Plot training progress and energy rankings"""
    os.makedirs(save_path, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    #Plot training/validation loss
    epochs = range(1, len(trainer.train_losses) + 1)
    ax1.plot(epochs, trainer.train_losses, 'b-', label="Training Loss")
    if trainer.val_losses:
        ax1.plot(epochs, trainer.val_losses, 'r-', label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    #Plot energy rankings over time
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

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'order_embedding_training_snli_10k_tests.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training plots saved to {save_path}")

def test_order_embeddings():
    """Test order embeddings"""
    processed_data_path = "data/processed/snli_10k_subset_balanced.pt"
    if not os.path.exists(processed_data_path):
        print(f"Processed data not found at {processed_data_path}")
        return

    model, trainer = train_order_embeddings(
        processed_data_path=processed_data_path,
        epochs=80, #larger for large toy dataset
        batch_size=32,
        order_dim=50, #Smaller for toy dataset (was 50)
        random_seed=42
    )

    #Validate rankings
    ranking_correct = validate_energy_rankings(trainer)

    #Plot progress
    plot_training_progress(trainer)

    if ranking_correct:
        print("Success: Order embeddings working!")
    else:
        print("Warning: Energy rankings not as expected")

    return model, trainer

if __name__ == "__main__":
    test_order_embeddings()




