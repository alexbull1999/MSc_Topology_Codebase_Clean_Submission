import torch
import torch.nn as nn
import geoopt
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional
import json
import random

# Import existing order embedding classes - Updated import
from .order_embeddings_asymmetry import OrderEmbeddingModel, EntailmentDataset

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
                 order_model_path: str = "models/enhanced_order_embeddings_snli_10k_tests.pt",  # Updated default path
                 hyperbolic_dim: int = 30,
                 device: str = 'auto', random_seed: int=42):
        """Initialize the complete pipeline"""

        set_random_seed(random_seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hyperbolic_dim = hyperbolic_dim

        #Load trained order embedding model
        self.order_model = self._load_enhanced_order_model(order_model_path)  # Updated method name

        #Initialize hyperbolic projector
        order_dim = self.order_model.order_dim
        self.hyperbolic_projector = HyperbolicProjector(
            order_dim=order_dim,
            hyperbolic_dim=hyperbolic_dim,
        ).to(self.device)

        print(f"Pipeline initialized: {order_dim}D → {hyperbolic_dim}D hyperbolic")
        print(f"Running on: {self.device}")
        print(f"Enhanced model with asymmetry_weight: {getattr(self.order_model, 'asymmetry_weight', 'N/A')}")

    def _load_enhanced_order_model(self, model_path: str) -> OrderEmbeddingModel:
        """Load enhanced order embeddings model with asymmetric components"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File {model_path} not found")

        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint['model_config']

        # Create enhanced model with asymmetry_weight parameter
        asymmetry_weight = model_config.get('asymmetry_weight', 0.2)  # Default to 0.2 if not found
        
        model = OrderEmbeddingModel(
            bert_dim=model_config['bert_dim'],
            order_dim=model_config['order_dim'],
            asymmetry_weight=asymmetry_weight  # Include asymmetry_weight parameter
        )

        #Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        print(f"Loaded enhanced order model (val_loss: {checkpoint['best_val_loss']:.4f})")
        print(f"Model asymmetry_weight: {asymmetry_weight}")
        return model

    def bert_to_hyperbolic(self, bert_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bert_embeddings: [batch_size, bert_dim] BERT embeddings
        Returns:
            hyperbolic_embeddings: [batch_size, hyperbolic_dim] points in poincaré ball
        """
        with torch.no_grad():
            order_embeddings = self.order_model(bert_embeddings)
            hyperbolic_embeddings = self.hyperbolic_projector(order_embeddings)
        return hyperbolic_embeddings

    def compute_enhanced_hyperbolic_energies(self,
                                           premise_bert: torch.Tensor,
                                           hypothesis_bert: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute enhanced distance/energy measures including asymmetric features
        Args:
            premise_bert: [batch_size, 768] premise BERT embeddings
            hypothesis_bert: [batch_size, 768] hypothesis BERT embeddings

        Returns:
            Dictionary with enhanced hyperbolic distance metrics
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

        # Compute enhanced asymmetric energies
        asymmetric_energies_dict = self.order_model.compute_bidirectional_energies(order_premise, order_hypothesis)

        #Distance from origin (hierarchy level indicator)
        premise_norms = self.hyperbolic_projector.get_hyperbolic_norms(premise_hyp)
        hypothesis_norms = self.hyperbolic_projector.get_hyperbolic_norms(hypothesis_hyp)

        # Enhanced result dictionary with asymmetric features
        return {
            'hyperbolic_distances': hyperbolic_distances,
            'order_energies': order_energies,
            'premise_norms': premise_norms,
            'hypothesis_norms': hypothesis_norms,
            'premise_hyperbolic': premise_hyp,
            'hypothesis_hyperbolic': hypothesis_hyp,
            # New asymmetric features
            'forward_energies': asymmetric_energies_dict['forward_energy'],
            'backward_energies': asymmetric_energies_dict['backward_energy'],
            'asymmetric_energies': asymmetric_energies_dict['asymmetric_energy'],
            'asymmetry_measures': asymmetric_energies_dict['asymmetry_measure'],
        }

    # Keep the old method for backward compatibility
    def compute_hyperbolic_energies(self, premise_bert: torch.Tensor, hypothesis_bert: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Backward compatibility method - calls enhanced version"""
        return self.compute_enhanced_hyperbolic_energies(premise_bert, hypothesis_bert)

    def compute_hyperbolic_energies_batch(self, premise_bert: torch.Tensor,
                                          hypothesis_bert: torch.Tensor,
                                          batch_size: int = 1000) -> Dict[str, torch.Tensor]:
        """
        Batch version of compute_enhanced_hyperbolic_energies for large datasets
        """
        n_samples = premise_bert.shape[0]

        if n_samples <= batch_size:
            # Small enough to process all at once
            return self.compute_enhanced_hyperbolic_energies(premise_bert, hypothesis_bert)

        # Process in batches
        print(f"Processing {n_samples} samples in batches of {batch_size} on {self.device}")

        all_results = {
            'hyperbolic_distances': [],
            'order_energies': [],
            'premise_norms': [],
            'hypothesis_norms': [],
            'premise_hyperbolic': [],
            'hypothesis_hyperbolic': [],
            'forward_energies': [],
            'backward_energies': [],
            'asymmetric_energies': [],
            'asymmetry_measures': []
        }

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)

            premise_batch = premise_bert[i:end_idx].to(self.device)
            hypothesis_batch = hypothesis_bert[i:end_idx].to(self.device)

            with torch.no_grad():
                batch_results = self.compute_enhanced_hyperbolic_energies(premise_batch, hypothesis_batch)

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

def test_enhanced_hyperbolic_projection():
    """Test enhanced hyperbolic projection with asymmetric features"""

    set_random_seed(42)

    processed_data_path = "data/processed/snli_10k_subset_balanced.pt"
    
    # Try different possible model filenames
    possible_model_paths = [
        "models/enhanced_order_embeddings_snli_10k_tests.pt"
    ]
    
    model_path = None
    for path in possible_model_paths:
        if os.path.exists(path):
            model_path = path
            print(f"Found model at: {path}")
            break
    
    if model_path is None:
        print("No enhanced order model found. Tried:")
        for path in possible_model_paths:
            print(f"  - {path}")
        print("Please run enhanced order_embeddings.py first!")
        return None, None, None

    #Initialize pipeline
    pipeline = HyperbolicOrderEmbeddingPipeline(
        order_model_path=model_path,
        hyperbolic_dim=30,
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

    print(f"Testing enhanced pipeline on batch: {len(premise_embs)} examples")

    results = pipeline.compute_enhanced_hyperbolic_energies(premise_embs, hypothesis_embs)

    # Enhanced DEBUG: Print first few examples with asymmetric features
    print("\nDEBUG: First 5 examples with asymmetric features:")
    for i in range(min(5, len(label_strs))):
        label = label_strs[i]
        print(f"Example {i} ({label}):")
        print(f"  Premise norm: {safe_tensor_to_float(results['premise_norms'][i]):.4f}")
        print(f"  Hypothesis norm: {safe_tensor_to_float(results['hypothesis_norms'][i]):.4f}")
        print(f"  Order energy: {safe_tensor_to_float(results['order_energies'][i]):.4f}")
        print(f"  Forward energy: {safe_tensor_to_float(results['forward_energies'][i]):.4f}")
        print(f"  Backward energy: {safe_tensor_to_float(results['backward_energies'][i]):.4f}")
        print(f"  Asymmetric energy: {safe_tensor_to_float(results['asymmetric_energies'][i]):.4f}")
        print(f"  Asymmetry measure: {safe_tensor_to_float(results['asymmetry_measures'][i]):.4f}")
        print()

    #Analyze enhanced results by label
    hyperbolic_stats = {}
    for i, label_str in enumerate(label_strs):
        if label_str not in hyperbolic_stats:
            hyperbolic_stats[label_str] = {
                'hyperbolic_distances': [],
                'order_energies': [],
                'premise_norms': [],
                'hypothesis_norms': [],
                'forward_energies': [],
                'backward_energies': [],
                'asymmetric_energies': [],
                'asymmetry_measures': [],
            }

        #Convert all tensors to python scalars
        hyp_dist = results['hyperbolic_distances'][i]
        order_energy = results['order_energies'][i]
        premise_norm = results['premise_norms'][i]
        hypothesis_norm = results['hypothesis_norms'][i]
        forward_energy = results['forward_energies'][i]
        backward_energy = results['backward_energies'][i]
        asymmetric_energy = results['asymmetric_energies'][i]
        asymmetry_measure = results['asymmetry_measures'][i]

        hyperbolic_stats[label_str]['hyperbolic_distances'].append(safe_tensor_to_float(hyp_dist))
        hyperbolic_stats[label_str]['order_energies'].append(safe_tensor_to_float(order_energy))
        hyperbolic_stats[label_str]['premise_norms'].append(safe_tensor_to_float(premise_norm))
        hyperbolic_stats[label_str]['hypothesis_norms'].append(safe_tensor_to_float(hypothesis_norm))
        hyperbolic_stats[label_str]['forward_energies'].append(safe_tensor_to_float(forward_energy))
        hyperbolic_stats[label_str]['backward_energies'].append(safe_tensor_to_float(backward_energy))
        hyperbolic_stats[label_str]['asymmetric_energies'].append(safe_tensor_to_float(asymmetric_energy))
        hyperbolic_stats[label_str]['asymmetry_measures'].append(safe_tensor_to_float(asymmetry_measure))

    #Print enhanced stats
    print("Enhanced Hyperbolic Projection Results:")
    print("-" * 80)
    for label, stats in hyperbolic_stats.items():
        print(f"\n{label.upper()}:")
        print(f"  Hyperbolic Distance:  {np.mean(stats['hyperbolic_distances']):.4f} ± {np.std(stats['hyperbolic_distances']):.4f}")
        print(f"  Order Energy:         {np.mean(stats['order_energies']):.4f} ± {np.std(stats['order_energies']):.4f}")
        print(f"  Forward Energy:       {np.mean(stats['forward_energies']):.4f} ± {np.std(stats['forward_energies']):.4f}")
        print(f"  Backward Energy:      {np.mean(stats['backward_energies']):.4f} ± {np.std(stats['backward_energies']):.4f}")
        print(f"  Asymmetric Energy:    {np.mean(stats['asymmetric_energies']):.4f} ± {np.std(stats['asymmetric_energies']):.4f}")
        print(f"  Asymmetry Measure:    {np.mean(stats['asymmetry_measures']):.4f} ± {np.std(stats['asymmetry_measures']):.4f}")
        print(f"  Premise Norm:         {np.mean(stats['premise_norms']):.4f} ± {np.std(stats['premise_norms']):.4f}")
        print(f"  Hypothesis Norm:      {np.mean(stats['hypothesis_norms']):.4f} ± {np.std(stats['hypothesis_norms']):.4f}")

    print("\nValidation Checks:")
    #Check 1 : All points inside unit ball
    all_premise_norms = results['premise_norms']
    all_hypothesis_norms = results['hypothesis_norms']
    # Convert to float for comparison
    premise_norms_float = torch.tensor([safe_tensor_to_float(x) for x in all_premise_norms])
    hypothesis_norms_float = torch.tensor([safe_tensor_to_float(x) for x in all_hypothesis_norms])
    max_norm = max(premise_norms_float.max().item(), hypothesis_norms_float.max().item())

    if max_norm < 1.0:
        print(f"✓ All points inside unit ball (max norm: {max_norm:.4f})")
    else:
        print(f"⚠ Some points outside unit ball (max norm: {max_norm:.4f})")

    #Check 2: Order energy hierarchy maintained
    entail_energies = hyperbolic_stats.get('entailment', {}).get('order_energies', [])
    neutral_energies = hyperbolic_stats.get('neutral', {}).get('order_energies', [])
    contra_energies = hyperbolic_stats.get('contradiction', {}).get('order_energies', [])

    if entail_energies and neutral_energies and contra_energies:
        entail_mean = np.mean(entail_energies)
        neutral_mean = np.mean(neutral_energies)
        contra_mean = np.mean(contra_energies)

        if entail_mean < neutral_mean < contra_mean:
            print("✓ Order energy hierarchy maintained in hyperbolic space")
        else:
            print("⚠ Order energy hierarchy changed after hyperbolic projection")

    #Check 3: Asymmetry patterns
    print("\nAsymmetry Pattern Analysis:")
    for label in ['entailment', 'neutral', 'contradiction']:
        if label in hyperbolic_stats:
            forward_mean = np.mean(hyperbolic_stats[label]['forward_energies'])
            backward_mean = np.mean(hyperbolic_stats[label]['backward_energies'])
            asymmetry_mean = np.mean(hyperbolic_stats[label]['asymmetry_measures'])
            
            print(f"  {label}: Forward={forward_mean:.4f}, Backward={backward_mean:.4f}, Asymmetry={asymmetry_mean:.4f}")

    #Check 4: Hyperbolic distances make sense
    hyp_distances = results['hyperbolic_distances']
    hyp_distances_float = [safe_tensor_to_float(x) for x in hyp_distances]
    min_dist = min(hyp_distances_float)
    max_dist = max(hyp_distances_float)
    print(f"✓ Hyperbolic distance range: {min_dist:.4f} - {max_dist:.4f}")

    return pipeline, results, hyperbolic_stats

def visualise_enhanced_hyperbolic_embeddings(pipeline, results, hyperbolic_stats, save_path="plots/"):
    """Create enhanced visualisations of hyperbolic embeddings with asymmetric features"""
    os.makedirs(save_path, exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    #Plot 1: Enhanced energy comparison
    labels = ['entailment', 'neutral', 'contradiction']
    forward_means = [np.mean(hyperbolic_stats[label]['forward_energies']) for label in labels]
    backward_means = [np.mean(hyperbolic_stats[label]['backward_energies']) for label in labels]
    asymmetric_means = [np.mean(hyperbolic_stats[label]['asymmetric_energies']) for label in labels]
    
    x_pos = np.arange(len(labels))
    width = 0.25
    
    ax1.bar(x_pos - width, forward_means, width, label='Forward Energy', alpha=0.8)
    ax1.bar(x_pos, backward_means, width, label='Backward Energy', alpha=0.8)
    ax1.bar(x_pos + width, asymmetric_means, width, label='Asymmetric Energy', alpha=0.8)
    
    ax1.set_xlabel('Relationship Type')
    ax1.set_ylabel('Energy')
    ax1.set_title('Enhanced Energy Analysis by Label')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    #plot 2: Asymmetry measures
    asymmetry_measures = [np.mean(hyperbolic_stats[label]['asymmetry_measures']) for label in labels]
    colours = ['green', 'blue', 'red']

    ax2.bar(labels, asymmetry_measures, color=colours, alpha=0.7)
    ax2.set_title('Asymmetry Measures by Label')
    ax2.set_ylabel('Mean Asymmetry Measure')
    ax2.grid(True, alpha=0.3)

    #Plot 3: Forward vs Backward energy scatter
    all_forward = []
    all_backward = []
    all_labels = []

    for label, stats in hyperbolic_stats.items():
        all_forward.extend(stats['forward_energies'])
        all_backward.extend(stats['backward_energies'])
        all_labels.extend([label] * len(stats['forward_energies']))

    for i, label in enumerate(labels):
        mask = [l == label for l in all_labels]
        x = [all_forward[j] for j, m in enumerate(mask) if m]
        y = [all_backward[j] for j, m in enumerate(mask) if m]
        ax3.scatter(x, y, label=label, alpha=0.7, color=colours[i])

    # Add diagonal line for reference (perfect symmetry)
    max_energy = max(max(all_forward), max(all_backward))
    ax3.plot([0, max_energy], [0, max_energy], 'k--', alpha=0.5, label='Perfect Symmetry')
    
    ax3.set_xlabel('Forward Energy')
    ax3.set_ylabel('Backward Energy')
    ax3.set_title('Forward vs Backward Energy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Hyperbolic distance vs Order energy with asymmetry color coding
    all_distances = []
    all_order_energies = []
    all_asymmetry = []
    all_labels_scatter = []

    for label, stats in hyperbolic_stats.items():
        all_distances.extend(stats['hyperbolic_distances'])
        all_order_energies.extend(stats['order_energies'])
        all_asymmetry.extend(stats['asymmetry_measures'])
        all_labels_scatter.extend([label] * len(stats['hyperbolic_distances']))

    # Create scatter plot with asymmetry as color
    scatter = ax4.scatter(all_distances, all_order_energies, c=all_asymmetry, 
                         alpha=0.6, cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Asymmetry Measure')
    
    ax4.set_xlabel('Hyperbolic Distance')
    ax4.set_ylabel('Order Energy')
    ax4.set_title('Order Energy vs Hyperbolic Distance\n(Color = Asymmetry)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'enhanced_hyperbolic_projection_analysis_snli_10k.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Enhanced hyperbolic visualization saved to {save_path}")

# Backward compatibility function
def test_hyperbolic_projection():
    """Backward compatibility function - calls enhanced version"""
    return test_enhanced_hyperbolic_projection()

def visualise_hyperbolic_embeddings(pipeline, results, hyperbolic_stats, save_path="plots/"):
    """Backward compatibility function - calls enhanced version"""
    return visualise_enhanced_hyperbolic_embeddings(pipeline, results, hyperbolic_stats, save_path)

if __name__ == "__main__":
    try:
        pipeline, results, stats = test_enhanced_hyperbolic_projection()
        if pipeline is not None and results is not None and stats is not None:
            visualise_enhanced_hyperbolic_embeddings(pipeline, results, stats)
        else:
            print("Test failed - check file paths and dependencies")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()