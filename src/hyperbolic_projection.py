import torch
import torch.nn as nn
import geoopt
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional
import json
import random

#Import existing order embedding classes
from order_embeddings import OrderEmbeddingModel, EntailmentDataset

def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.Generator().manual_seed(seed)


class HyperbolicProjector(nn.Module):
    """
    Projects order embeddings to hyperbolic space (Poincaré ball), bridging between Euclidean order embeddings
    and hyperbolic entailment cone analysis
    """

    def __init__(self, order_dim: int=50, hyperbolic_dim: int=30):
        """
        Initialize hyperbolic projector
        Args:
            order_dim: Input order embedding dimensions
            hyperbolic_dim: Dimension of hyperbolic space
        """
        super().__init__()
        self.order_dim = order_dim
        self.hyperbolic_dim = hyperbolic_dim

        #Poincaré ball manifold
        self.ball = geoopt.PoincareBall()

        #Euclidean projection layer (reduces dimension if needed)
        self.euclidean_projection = nn.Sequential(
            nn.Linear(order_dim, hyperbolic_dim),
            nn.Tanh() #constrains range to [-1, 1] range
        )

        # Scaling factor to ensure point stay inside unit ball
        self.scale_factor = 0.8 # Conservative scaling to avoid boundary issues

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with careful scaling for hyperbolic projection"""
        for module in self.euclidean_projection:
            if isinstance(module, nn.Linear):
                # Small initialization crucial for hyperbolic stability
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)

    def forward(self, order_embeddings: torch.Tensor) -> torch.Tensor:
        """Project order embeddings to Poincaré ball
        Args:
            order_embeddings: [batch_size, order_dim] Order embeddings
        Returns:
            hyperbolic_embeddings: [batch_size, hyperbolic_dim] points in Poincaré ball
        """
        batch_size = order_embeddings.shape[0]

        #Step 1: Project to lower dimension (if needed)
        euclidean_projected = self.euclidean_projection(order_embeddings)

        #Step 2: Scale to ensure points stay inside unit ball
        scaled = euclidean_projected * self.scale_factor

        #Step 3: Map to Poincaré ball using exponential map from origin, ensuring all point lie within unit ball
        hyperbolic_embeddings = self.ball.expmap0(scaled)

        #Verification: ensure all points are inside unit ball
        euclidean_norms = torch.norm(hyperbolic_embeddings, dim=-1)
        assert torch.all(euclidean_norms<1.0), f"Some points outside unit ball! Max norm: {euclidean_norms.max()}"

        return hyperbolic_embeddings

    def hyperbolic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute hyperbolic distances between points in Poincaré ball
        Args:
            x, y: Points in Poincaré ball [batch_size, hyperbolic_dim]
        Returns:
            distances: [batch_size], hyperbolic distances
        """
        return self.ball.dist(x, y)

    def get_hyperbolic_norms(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Get hyperbolic norms (distance from origin)"""
        origin = torch.zeros_like(embeddings[0:1]) # Single point at origin
        origin = origin.expand_as(embeddings) # Expand to match batch size
        return self.ball.dist(origin, embeddings)

class HyperbolicOrderEmbeddingPipeline:
    """Complete pipeline: BERT -> Order Embeddings -> Hyperbolic Embeddings"""

    def __init__(self,
                 order_model_path: str = "models/enhanced_order_embeddings_snli_10k_asymmetry.pt",
                 hyperbolic_dim: int = 30,
                 device: str = 'auto', random_seed: int=42):
        """Initialize the complete pipeline"""

        set_random_seed(random_seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hyperbolic_dim = hyperbolic_dim

        #Load trained order embedding model
        self.order_model = self._load_order_model(order_model_path)

        #Initialize hyperbolic projector
        order_dim = self.order_model.order_dim
        self.hyperbolic_projector = HyperbolicProjector(
            order_dim=order_dim,
            hyperbolic_dim=hyperbolic_dim,
        ).to(self.device)

        print(f"Pipeline initialized: {order_dim}D → {hyperbolic_dim}D hyperbolic")
        print(f"Running on: {self.device}")

    def _load_order_model(self, model_path: str) -> OrderEmbeddingModel:
        """Load order embeddings model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File {model_path} not found")

        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint['model_config']

        model = OrderEmbeddingModel(
            bert_dim=model_config['bert_dim'],
            order_dim=model_config['order_dim']
        )

        #Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        print(f"Loaded order model (val_loss: {checkpoint['best_val_loss']:.4f})")
        return model

    def bert_to_hyperbolic(self, bert_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bert_embeddings: [batch_size, hyperbolic_dim] BERT embeddings
        Returns:
            hyperbolic_embeddings: [batch_size, hyperbolic_dim] points in poincaré ball
        """
        with torch.no_grad():
            order_embeddings = self.order_model(bert_embeddings)
            hyperbolic_embeddings = self.hyperbolic_projector(order_embeddings)
        return hyperbolic_embeddings

    def compute_hyperbolic_energies(self,
                                    premise_bert: torch.Tensor,
                                    hypothesis_bert: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute various distance/energy measures in hyperbolic space
        Args:
            premise_bert: [batch_size, 768] premise BERT embeddings
            hypothesis_bert: [batch_size, 768] hypothesis BERT embeddings

        Returns:
            Dictionary with hyperbolic distance metrics
        """

        # Map to hyperbolic space
        premise_hyp = self.bert_to_hyperbolic(premise_bert)
        hypothesis_hyp = self.bert_to_hyperbolic(hypothesis_bert)

        #Compute hyperbolic distances
        hyperbolic_distances = self.hyperbolic_projector.hyperbolic_distance(premise_hyp, hypothesis_hyp)

        #Compute order violation energies in hyperbolic space
        order_premise = self.order_model(premise_bert)
        order_hypothesis = self.order_model(hypothesis_bert)
        order_energies = self.order_model.order_violation_energy(order_premise, order_hypothesis)

        #Distance from origin (hierarchy level indicator)
        premise_norms = self.hyperbolic_projector.get_hyperbolic_norms(premise_hyp)
        hypothesis_norms = self.hyperbolic_projector.get_hyperbolic_norms(hypothesis_hyp)

        return {
            'hyperbolic_distances': hyperbolic_distances,
            'order_energies': order_energies,
            'premise_norms': premise_norms,
            'hypothesis_norms': hypothesis_norms,
            'premise_hyperbolic': premise_hyp,
            'hypothesis_hyperbolic': hypothesis_hyp,
        }

    def compute_hyperbolic_energies_batch(self, premise_bert: torch.Tensor,
                                          hypothesis_bert: torch.Tensor,
                                          batch_size: int = 1000) -> Dict[str, torch.Tensor]:
        """
        Batch version of compute_hyperbolic_energies for large datasets
        """
        n_samples = premise_bert.shape[0]

        if n_samples <= batch_size:
            # Small enough to process all at once
            return self.compute_hyperbolic_energies(premise_bert, hypothesis_bert)

        # Process in batches
        print(f"Processing {n_samples} samples in batches of {batch_size} on {self.device}")

        all_results = {
            'hyperbolic_distances': [],
            'order_energies': [],
            'premise_norms': [],
            'hypothesis_norms': [],
            'premise_hyperbolic': [],
            'hypothesis_hyperbolic': []
        }

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)

            premise_batch = premise_bert[i:end_idx].to(self.device)
            hypothesis_batch = hypothesis_bert[i:end_idx].to(self.device)

            with torch.no_grad():
                batch_results = self.compute_hyperbolic_energies(premise_batch, hypothesis_batch)

                # Collect results (move to CPU to save GPU memory)
                for key in all_results.keys():
                    all_results[key].append(batch_results[key].cpu())

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed batch {i // batch_size + 1}/{(n_samples - 1) // batch_size + 1}")

        # Concatenate all results
        final_results = {}
        for key in all_results.keys():
            final_results[key] = torch.cat(all_results[key], dim=0)

        return final_results



def safe_tensor_to_float(tensor_or_float):
    """Safely convert tensor or float to Python float"""
    if hasattr(tensor_or_float, 'item'):
        return float(tensor_or_float.item())
    elif isinstance(tensor_or_float, (int, float)):
        return float(tensor_or_float)
    else:
        try:
            return float(tensor_or_float)
        except:
            print("ERROR CONVERTING TENSOR TO FLOAT; RETURNED 0.0")
            return 0.0

def test_hyperbolic_projection():
    """Test hyperbolic projection on toy dataset"""

    set_random_seed(42)

    processed_data_path = "data/processed/snli_10k_subset_balanced.pt"
    model_path = "models/enhanced_order_embeddings_snli_10k_asymmetry.pt"

    if not os.path.exists(processed_data_path):
        print(f"Processed data not found at {processed_data_path}")
        print("Please run text_processing.py first!")
        return

    if not os.path.exists(model_path):
        print(f"Order model not found at {model_path}")
        print("Please run order_embeddings.py first!")
        return

    #Initialize pipeline
    pipeline = HyperbolicOrderEmbeddingPipeline(
        order_model_path=model_path,
        hyperbolic_dim=30, #20 for smaller toy datasets
        random_seed=42
    )

    processed_data = torch.load(processed_data_path)
    dataset = EntailmentDataset(processed_data)

    #Test on a batch
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    batch = next(iter(dataloader))
    premise_embs = batch['premise_emb'].to(pipeline.device)
    hypothesis_embs = batch['hypothesis_emb'].to(pipeline.device)
    labels = batch['label']
    label_strs = batch['label_str']

    print(f"Testing on batch: {len(premise_embs)} examples")

    results = pipeline.compute_hyperbolic_energies(premise_embs, hypothesis_embs)

    # DEBUG: Print first few examples to see what's happening
    print("\nDEBUG: First 5 examples:")
    for i in range(min(5, len(label_strs))):
        label = label_strs[i]
        print(f"Example {i} ({label}):")
        print(f"  Premise norm: {safe_tensor_to_float(results['premise_norms'][i]):.4f}")
        print(f"  Hypothesis norm: {safe_tensor_to_float(results['hypothesis_norms'][i]):.4f}")
        print(f"  Order energy: {safe_tensor_to_float(results['order_energies'][i]):.4f}")
        print()

    # # DEBUG: Check if there's a systematic pattern (The flip in large toy, was actually due to nature of synthetic toy dataset and how premise/hypothesis pairs are flipped)
    # print("DEBUG: Checking for systematic swapping...")
    # entail_indices = [i for i, label in enumerate(label_strs) if label == 'entailment']
    # neutral_indices = [i for i, label in enumerate(label_strs) if label == 'neutral']
    #
    # if entail_indices and neutral_indices:
    #     print(
    #         f"Entailment example 0 - Premise: {safe_tensor_to_float(results['premise_norms'][entail_indices[0]]):.4f}, Hypothesis: {safe_tensor_to_float(results['hypothesis_norms'][entail_indices[0]]):.4f}")
    #     print(
    #         f"Neutral example 0 - Premise: {safe_tensor_to_float(results['premise_norms'][neutral_indices[0]]):.4f}, Hypothesis: {safe_tensor_to_float(results['hypothesis_norms'][neutral_indices[0]]):.4f}")


    #Analyze results by label
    hyperbolic_stats = {}
    for i, label_str in enumerate(label_strs):
        if label_str not in hyperbolic_stats:
            hyperbolic_stats[label_str] = {
                'hyperbolic_distances': [],
                'order_energies': [],
                'premise_norms': [],
                'hypothesis_norms': [],
            }

        #Convert all tensors to python scalars
        hyp_dist = results['hyperbolic_distances'][i]
        order_energy = results['order_energies'][i]
        premise_norm = results['premise_norms'][i]
        hypothesis_norm = results['hypothesis_norms'][i]

        hyperbolic_stats[label_str]['hyperbolic_distances'].append(safe_tensor_to_float(hyp_dist))
        hyperbolic_stats[label_str]['order_energies'].append(safe_tensor_to_float(order_energy))
        hyperbolic_stats[label_str]['premise_norms'].append(safe_tensor_to_float(premise_norm))
        hyperbolic_stats[label_str]['hypothesis_norms'].append(safe_tensor_to_float(hypothesis_norm))

    #Print stats
    print("Hyperbolic Projection Results:")
    print("-" * 60)
    for label, stats in hyperbolic_stats.items():
        print(f"\n{label.upper()}:")
        print(f"  Hyperbolic Distance:  {np.mean(stats['hyperbolic_distances']):.4f} ± {np.std(stats['hyperbolic_distances']):.4f}")
        print(f"  Order Energy:         {np.mean(stats['order_energies']):.4f} ± {np.std(stats['order_energies']):.4f}")
        print(f"  Premise Norm:         {np.mean(stats['premise_norms']):.4f} ± {np.std(stats['premise_norms']):.4f}")
        print(f"  Hypothesis Norm:      {np.mean(stats['hypothesis_norms']):.4f} ± {np.std(stats['hypothesis_norms']):.4f}")

    print("Validation Checks:")
    #Check 1 : All points inside unit ball
    all_premise_norms = results['premise_norms']
    all_hypothesis_norms = results['hypothesis_norms']
    # Convert to float for comparison
    premise_norms_float = torch.tensor([safe_tensor_to_float(x) for x in all_premise_norms])
    hypothesis_norms_float = torch.tensor([safe_tensor_to_float(x) for x in all_hypothesis_norms])
    max_norm = max(premise_norms_float.max().item(), hypothesis_norms_float.max().item())

    if max_norm < 1.0:
        print(f"All points inside unit ball (max norm: {max_norm:.4f})")
    else:
        print(f"Some points outside unit ball (max norm: {max_norm:.4f})")

    #Check 2: Order energy hierarchy maintained
    entail_energies = hyperbolic_stats.get('entailment', {}).get('order_energies', [])
    neutral_energies = hyperbolic_stats.get('neutral', {}).get('order_energies', [])
    contra_energies = hyperbolic_stats.get('contradiction', {}).get('order_energies', [])

    if entail_energies and neutral_energies and contra_energies:
        entail_mean = np.mean(entail_energies)
        neutral_mean = np.mean(neutral_energies)
        contra_mean = np.mean(contra_energies)

        if entail_mean < neutral_mean < contra_mean:
            print("Order energy hierarchy maintained in hyperbolic space")
        else:
            print("Order energy hierarchy changed after hyperbolic projection")

    #Check 3: Hyperbolic distances make sense
    hyp_distances = results['hyperbolic_distances']
    hyp_distances_float = [safe_tensor_to_float(x) for x in hyp_distances]
    min_dist = min(hyp_distances_float)
    max_dist = max(hyp_distances_float)
    print(f"Hyperbolic distance range: {min_dist:.4f} - {max_dist:.4f}")

    return pipeline, results, hyperbolic_stats

def visualise_hyperbolic_embeddings(pipeline, results, hyperbolic_stats, save_path="plots/"):
    """Create visualisations of hyperbolic embeddings"""
    os.makedirs(save_path, exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    #Plot 1: Hyperbolic distances by label
    labels = ['entailment', 'neutral', 'contradiction']
    distances = [np.mean(hyperbolic_stats[label]['hyperbolic_distances']) for label in labels]
    colours = ['green', 'blue', 'red']

    ax1.bar(labels, distances, color=colours, alpha=0.7)
    ax1.set_title('Mean Hyperbolic Distances by Label')
    ax1.set_ylabel('Mean Hyperbolic Distance')
    ax1.grid(True, alpha=0.3)

    #plot 2: order energies vs hyperbolic distances
    all_distances = []
    all_energies = []
    all_labels = []

    for label, stats in hyperbolic_stats.items():
        all_distances.extend(stats['hyperbolic_distances'])
        all_energies.extend(stats['order_energies'])
        all_labels.extend([label] * len(stats['hyperbolic_distances']))

    for i, label in enumerate(labels):
        mask = [l == label for l in all_labels]
        x = [all_distances[j] for j, m in enumerate(mask) if m]
        y = [all_energies[j] for j, m in enumerate(mask) if m]
        ax2.scatter(x, y, label=label, alpha=0.7, color=colours[i])

    ax2.set_xlabel('Hyperbolic Distance')
    ax2.set_ylabel('Order Violation Energy')
    ax2.set_title('Order Energy vs Hyperbolic Distance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    #Plot 3: Norms distribution
    for i, label in enumerate(labels):
        premise_norms = hyperbolic_stats[label]['premise_norms']
        hypothesis_norms = hyperbolic_stats[label]['hypothesis_norms']
        ax3.hist(premise_norms + hypothesis_norms, alpha=0.5, label=f'{label}',
                 bins=20, color=colours[i])

    ax3.set_xlabel('Distance from Origin (Hyperbolic Norm)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Hyperbolic Norms')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Energy comparison (order vs hyperbolic distance)
    energy_means = [np.mean(hyperbolic_stats[label]['order_energies']) for label in labels]
    distance_means = [np.mean(hyperbolic_stats[label]['hyperbolic_distances']) for label in labels]

    x_pos = np.arange(len(labels))
    width = 0.35

    ax4_twin = ax4.twinx()
    bars1 = ax4.bar(x_pos - width / 2, energy_means, width, label='Order Energy', color='orange', alpha=0.7)
    bars2 = ax4_twin.bar(x_pos + width / 2, distance_means, width, label='Hyperbolic Distance', color='purple',
                         alpha=0.7)

    ax4.set_xlabel('Label')
    ax4.set_ylabel('Order Energy', color='orange')
    ax4_twin.set_ylabel('Hyperbolic Distance', color='purple')
    ax4.set_title('Order Energy vs Hyperbolic Distance by Label')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'hyperbolic_projection_analysis_snli_10k_asymmetry.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Hyperbolic visualization saved to {save_path}")

if __name__ == "__main__":
    pipeline, results, stats = test_hyperbolic_projection()

    visualise_hyperbolic_embeddings(pipeline, results, stats)




