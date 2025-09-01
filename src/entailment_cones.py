"""
Implementation of Hyperbolic Entailment Cones from the Hyperbolic Projection of our Order Embeddings (in hyperbolic_projection.py)
following the methodology described in Ganea et al. (2018) "Hyperbolic Entailment Cones for Learning Hierarchical Embeddings"
"""

import torch
import torch.nn as nn
import geoopt
import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import os
from sympy.codegen.ast import RuntimeError_
from hyperbolic_projection import HyperbolicOrderEmbeddingPipeline, safe_tensor_to_float
from order_embeddings import OrderEmbeddingModel, EntailmentDataset
from hyperbolic_projection import set_random_seed
from text_processing import TextToEmbedding


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HyperbolicEntailmentCones:
    """
    Implementation of hyperbolic entailment cones following Ganea et al.
    This class creates entailment cones in hyperbolic space, and computes cone violation energies
    """
    def __init__(self, K: float=0.1, epsilon: float=0.1):
        """
        Initialisation
        Args:
            K: Cone aperture parameter (0.1 from paper)
            epsilon: Exclusion radius around origin (0.1 from paper)
        """
        self.K = K
        self.epsilon = epsilon
        self.ball = geoopt.PoincareBall()
        self.device = get_device()
        print(f"Hyperbolic Entailment Cones using device: {self.device}")

    def cone_aperture(self, premise_embedding: torch.Tensor) -> torch.Tensor:
        """Compute cone aperture using Ganea et al. equation 26
         ψ(x) = arcsin(K(1 - ||x||²)/||x||)
         Args:
             premise_embedding: Point in Poincaré ball [batch_size, dim]
        Returns:
            Cone aperture angles [batch_size]
        Note: We use Euclidean norm (not hyperbolic norm) because:
        - Points in Poincaré ball are represented as Euclidean vectors with norm < 1
        - The formula uses Euclidean distance from origin to determine hierarchy level
        - This is consistent with Ganea et al.'s original implementation
        """

        norm = torch.norm(premise_embedding, dim=-1, keepdim=True)

        #Ensure not too close to origin (outside epsilon ball)
        norm = torch.clamp(norm, min=self.epsilon)

        #Cone aperture formula
        numerator = self.K * (1 - norm**2)
        aperture_arg = numerator/norm

        #clamp to valid arcsin domain [-1, 1]
        aperture_arg = torch.clamp(aperture_arg, min=-1.0, max=1.0)
        aperture = torch.arcsin(aperture_arg)

        return aperture.squeeze(-1)

    def compute_xi_angle(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        """
        Compute angle Ξ(x,y) from Ganea et al. Equation 28:
        This is the angle between the half-lines (xy and (0x
        Args:
            premise: Premise embeddings in Poincaré ball [batch_size, dim]
            hypothesis: Hypothesis embeddings in Poincaré ball [batch_size, dim]
        Returns:
            Angles Ξ(premise, hypothesis) [batch_size]
        """

        #Compute required norms and dot products
        premise_norm = torch.norm(premise, dim=-1) #||x||
        hypothesis_norm = torch.norm(hypothesis, dim=-1) #||y||
        diff_norm = torch.norm(premise - hypothesis, dim=-1) #||x-y||
        dot_product = torch.sum(premise * hypothesis, dim=-1) # ⟨x,y⟩

        # Numerator: ⟨x,y⟩(1 + ||x||²) - ||x||²(1 + ||y||²)
        numerator = (dot_product * (1 + premise_norm**2) - premise_norm**2 * (1 + hypothesis_norm**2))

        # Denominator: ||x|| · ||x-y|| · √(1 + ||x||²||y||² - 2⟨x,y⟩)
        sqrt_term = torch.sqrt(1+premise_norm**2 * hypothesis_norm**2 - 2 * dot_product)
        denominator = premise_norm * diff_norm * sqrt_term

        # Avoid division by zero
        denominator = torch.clamp(denominator, min=1e-8)

        #Compute cosine of angle
        cos_inverse_xi = numerator / denominator
        inverse_xi = torch.arccos(cos_inverse_xi)
        return np.pi - inverse_xi

    def cone_membership_energy(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        """ Compute cone violation energy using Ganea et al. equation 33, adapted from Vendrov et al.
                E(u,v) = max(0, Ξ(u,v) - ψ(u))
                If E ≈ 0: hypothesis is inside premise's cone (entailment)
                If E > 0: hypothesis is outside premise's cone (no entailment)

        Args:
            premise: Premise embeddings [batch_size, dim]
            hypothesis: Hypothesis embeddings [batch_size, dim]
        Returns:
            Cone violation energies [batch_size]
        """
        #Compute angle between premise and hypothesis
        xi_angle = self.compute_xi_angle(premise, hypothesis)

        #Compute cone aperture for premise
        aperture = self.cone_aperture(premise)

        # #DEBUG
        # print(f"DEBUG: Xi angle range: [{xi_angle.min():.4f}, {xi_angle.max():.4f}]")
        # print(f"DEBUG: Aperture range: [{aperture.min():.4f}, {aperture.max():.4f}]")
        # print(f"DEBUG: Premise norm range: [{torch.norm(premise, dim=-1).min():.4f}, {torch.norm(premise, dim=-1).max():.4f}]")

        #Cone violation energy: max(0, angle-aperture)
        violation_energy = torch.relu(xi_angle - aperture)

        return violation_energy

    def cone_violation_energy_batch(self, premises, hypotheses, batch_size=1000):
        """
        Compute cone violation energies in batches with automatic GPU/CPU handling
        Args:
            premises: Premise embeddings [n_samples, dim]
            hypotheses: Hypothesis embeddings [n_samples, dim]
            batch_size: Batch size for processing
        Returns:
            Cone violation energies [n_samples]
        """
        # Convert to tensors and move to device
        if not isinstance(premises, torch.Tensor):
            premises = torch.tensor(premises, dtype=torch.float32)
        if not isinstance(hypotheses, torch.Tensor):
            hypotheses = torch.tensor(hypotheses, dtype=torch.float32)

        premises = premises.to(self.device)
        hypotheses = hypotheses.to(self.device)

        n_samples = premises.shape[0]
        all_violations = []

        print(f"Computing cone violations for {n_samples} samples on {self.device}")

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)

            # Get batch
            premise_batch = premises[i:end_idx]
            hypothesis_batch = hypotheses[i:end_idx]

            # Use your existing cone_membership_energy method
            with torch.no_grad():
                batch_violations = self.cone_membership_energy(premise_batch, hypothesis_batch)
                all_violations.append(batch_violations.cpu())

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed batch {i // batch_size + 1}/{(n_samples - 1) // batch_size + 1}")

        return torch.cat(all_violations, dim=0)

    def create_entailment_cone(self, premise_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create entailment cone structure for a premise embedding
        Args:
            premise_embedding: Single premise point in Poincaré ball
        Returns:
            Dictionary with cone properties:
            - apex: cone apex (premise point)
            - aperture: cone aperture angle
            - axis: cone central axis direction
        """
        if premise_embedding.dim() == 1:
            premise_embedding = premise_embedding.unsqueeze(0)

        aperture = self.cone_aperture(premise_embedding)
        norm = torch.norm(premise_embedding, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=self.epsilon)

        #Central axis = radial direction from origin through premise
        axis = premise_embedding / norm

        return {
            'apex': premise_embedding.squeeze(0),
            'aperture': aperture.squeeze(0),
            'axis': axis.squeeze(0)
        }


class HyperbolicConeEmbeddingPipeline:
    def __init__(self, model_path: str = "models/order_embeddings_snli_10k.pt",
                 K: float=0.01, epsilon: float=0.1): #changed k from 0.1 (with toy data) to 0.01 with SNLI
        """
        Initialize the complete pipeline
        Args:
            model_path: Path to trained order embeddings
            K: cone aperture parameter
            epsilon: exclusion radius around origin
        """

        self.model_path = model_path
        self.hyperbolic_pipeline = None
        self.cone_computer = HyperbolicEntailmentCones(K=K, epsilon=epsilon)

        try:
            self.hyperbolic_pipeline = HyperbolicOrderEmbeddingPipeline(model_path)
            print("Successfuly loaded hyperbolic projection of order embeddings")
        except Exception as e:
            print(f"Could not load hyperbolic projection pipeline: {e}")

    def compute_cone_energies(self, premise_bert: torch.Tensor, hypothesis_bert: torch.Tensor) -> Dict[str, torch.Tensor]:
        """ Compute cone violation energies for premise-hypothesis pairs
        Args:
            premise_bert: BERT embeddings for premises [batch_size, 768]
            hypothesis_bert: BERT embeddings for hypotheses [batch_size, 768]
        Returns:
            Dictionary with:
            - cone_energies: Cone violation energies
            - order_energies: Order violation energies (for comparison)
            - hyperbolic_distances: Hyperbolic distances 9for comparison)
            - premise_hyperbolic: Premise embeddings in hyperbolic space
            - hypothesis_hyperbolic: Hypothesis embeddings in hyperbolic space
            - premise norms: premise norms
            - hypothesis norms: hypothesis norms
        """
        if self.hyperbolic_pipeline is None:
            raise RuntimeError("Hyperbolic pipeline not loaded. Cannot compute cone energies")

        if self.hyperbolic_pipeline.device.type == 'cuda':
            print("Using GPU batch processing for hyperbolic energies")
            results = self.hyperbolic_pipeline.compute_hyperbolic_energies_batch(premise_bert, hypothesis_bert)
        else:
            print("Using CPU single-pass processing for hyperbolic energies")
            results = self.hyperbolic_pipeline.compute_hyperbolic_energies(premise_bert, hypothesis_bert)

        premises_hyp = results['premise_hyperbolic']
        hypotheses_hyp = results['hypothesis_hyperbolic']

        if self.cone_computer.device.type == 'cuda':
            print("Using GPU batch processing for cone energies")
            #Compute cone violation energies
            cone_energies = self.cone_computer.cone_violation_energy_batch(premises_hyp, hypotheses_hyp)
        else:
            print("Using CPU single-pass processing for cone energies")
            cone_energies = self.cone_computer.cone_membership_energy(premises_hyp, hypotheses_hyp)

        #Return results
        return {
            'cone_energies': cone_energies,
            'order_energies': results['order_energies'],
            'hyperbolic_distances': results['hyperbolic_distances'],
            'premise_hyperbolic': premises_hyp,
            'hypothesis_hyperbolic': hypotheses_hyp,
            'premise_norms': results['premise_norms'],
            'hypothesis_norms': results['hypothesis_norms']
        }

    def validate_cone_hierarchy(self, premise_bert: torch.Tensor, hypothesis_bert: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Validate that cone energies follow expected hierarchy
        Args:
            premise_bert: BERT embeddings for premises [batch_size, 768]
            hypothesis_bert: BERT embeddings for hypotheses [batch_size, 768]
            labels: Labels tensor where 0=entailment, 1=neutral, 2=contradiction

        Returns:
            Dictionary with mean energies for each category
        """
        results = {}
        all_energies = self.compute_cone_energies(premise_bert, hypothesis_bert)

        #Group by label
        for label_idx, label_name in enumerate(['entailment', 'neutral', 'contradiction']):
            mask = (labels == label_idx)
            if mask.sum() > 0:
                cone_energies_subset = all_energies['cone_energies'][mask]
                order_energies_subset = all_energies['order_energies'][mask]
                hyperbolic_distances_subset = all_energies['hyperbolic_distances'][mask]

                results[f"{label_name}_cone_energy"] = safe_tensor_to_float(cone_energies_subset.mean())
                results[f"{label_name}_order_energy"] = safe_tensor_to_float(order_energies_subset.mean())
                results[f"{label_name}_hyperbolic_distance"] = safe_tensor_to_float(hyperbolic_distances_subset.mean())

        return results

def test_cone_implementation():
    set_random_seed(42)
    print("Testing hyperbolic cone implementation")
    pipeline = HyperbolicConeEmbeddingPipeline()

    processed_data_path = "data/processed/snli_10k_subset_balanced.pt"
    if not os.path.exists(processed_data_path):
        return RuntimeError("Processed data not found! Error")

    from torch.utils.data import DataLoader

    processed_data = torch.load(processed_data_path)
    dataset = EntailmentDataset(processed_data)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False) #Change this for real datasets that are too large to do in one batch

    batch = next(iter(dataloader))
    premise_embs = batch['premise_emb'].to(pipeline.hyperbolic_pipeline.device)
    hypothesis_embs = batch['hypothesis_emb'].to(pipeline.hyperbolic_pipeline.device)
    labels = batch['label']
    label_strs = batch['label_str']

    print(f"Testing on real dataset: {len(premise_embs)} examples")

    # Compute cone energies
    results = pipeline.compute_cone_energies(premise_embs, hypothesis_embs)

    # Analyze by label
    stats_by_label = {}
    for i, label_str in enumerate(label_strs):
        if label_str not in stats_by_label:
            stats_by_label[label_str] = {
                'cone_energies': [],
                'order_energies': [],
                'hyperbolic_distances': []
            }

        stats_by_label[label_str]['cone_energies'].append(safe_tensor_to_float(results['cone_energies'][i]))
        stats_by_label[label_str]['order_energies'].append(safe_tensor_to_float(results['order_energies'][i]))
        stats_by_label[label_str]['hyperbolic_distances'].append(safe_tensor_to_float(results['hyperbolic_distances'][i]))

    print("Results:")
    print("-" * 60)
    for label, stats in stats_by_label.items():
        print(f"\n{label.upper()}:")
        print(f"  Cone Energy:          {np.mean(stats['cone_energies']):.4f} ± {np.std(stats['cone_energies']):.4f}")
        print(f"  Order Energy:         {np.mean(stats['order_energies']):.4f} ± {np.std(stats['order_energies']):.4f}")
        print(f"  Hyperbolic Distance:  {np.mean(stats['hyperbolic_distances']):.4f} ± {np.std(stats['hyperbolic_distances']):.4f}")

    # Validate hierarchy
    ent_cone = np.mean(stats_by_label.get('entailment', {}).get('cone_energies', [float('inf')]))
    neu_cone = np.mean(stats_by_label.get('neutral', {}).get('cone_energies', [float('inf')]))
    con_cone = np.mean(stats_by_label.get('contradiction', {}).get('cone_energies', [float('inf')]))

    hierarchy_valid = ent_cone < neu_cone < con_cone
    print(f"\nCone Energy Hierarchy: {'VALID' if hierarchy_valid else 'INVALID'}")
    print(f"   {ent_cone:.4f} < {neu_cone:.4f} < {con_cone:.4f}")

    return pipeline, results, stats_by_label


if __name__ == "__main__":
    # Run basic tests
    test_results = test_cone_implementation()

    # # Try to test with real pipeline if available
    # try:
    #     pipeline = HyperbolicConeEmbeddingPipeline()
    #     print("\nSuccessfully created complete cone embedding pipeline!")
    # except Exception as e:
    #     print(f"\nCould not create full pipeline: {e}")
    #     print("Basic cone mathematics implementation is working correctly.")















