"""
Enhanced Implementation of Hyperbolic Entailment Cones with Asymmetric Features
following the methodology described in Ganea et al. (2018) "Hyperbolic Entailment Cones for Learning Hierarchical Embeddings"
Updated to work with enhanced asymmetric order embeddings
"""

import torch
import torch.nn as nn
import geoopt
import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import os
from .hyperbolic_projection_asymmetry import HyperbolicOrderEmbeddingPipeline, safe_tensor_to_float, set_random_seed
from .order_embeddings_asymmetry import OrderEmbeddingModel, EntailmentDataset
from .text_processing import TextToEmbedding


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HyperbolicEntailmentCones:
    """
    Enhanced implementation of hyperbolic entailment cones following Ganea et al.
    This class creates entailment cones in hyperbolic space, and computes cone violation energies
    Now includes asymmetric relationship modeling
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
        print(f"Enhanced Hyperbolic Entailment Cones using device: {self.device}")

    def cone_aperture(self, premise_embedding: torch.Tensor) -> torch.Tensor:
        """Compute cone aperture using Ganea et al. equation 26
         ψ(x) = arcsin(K(1 - ||x||²)/||x||)
         Args:
             premise_embedding: Point in Poincaré ball [batch_size, dim]
        Returns:
            Cone aperture angles [batch_size]
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

        #Cone violation energy: max(0, angle-aperture)
        violation_energy = torch.relu(xi_angle - aperture)

        return violation_energy

    def bidirectional_cone_energies(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute both forward and backward cone energies for asymmetric analysis
        Args:
            premise: Premise embeddings [batch_size, dim]
            hypothesis: Hypothesis embeddings [batch_size, dim]
        Returns:
            Dictionary with forward, backward, and asymmetric cone energies
        """
        # Forward cone energy: premise → hypothesis
        forward_cone_energy = self.cone_membership_energy(premise, hypothesis)
        
        # Backward cone energy: hypothesis → premise
        backward_cone_energy = self.cone_membership_energy(hypothesis, premise)
        
        # Asymmetric cone energy measure
        cone_asymmetry = torch.abs(forward_cone_energy - backward_cone_energy)
        
        return {
            'forward_cone_energy': forward_cone_energy,
            'backward_cone_energy': backward_cone_energy,
            'cone_asymmetry': cone_asymmetry
        }

    def cone_violation_energy_batch(self, premises, hypotheses, batch_size=1000):
        """
        Compute cone violation energies in batches with automatic GPU/CPU handling
        Args:
            premises: Premise embeddings [n_samples, dim]
            hypotheses: Hypothesis embeddings [n_samples, dim]
            batch_size: Batch size for processing
        Returns:
            Dictionary with all cone energy measurements
        """
        # Convert to tensors and move to device
        if not isinstance(premises, torch.Tensor):
            premises = torch.tensor(premises, dtype=torch.float32)
        if not isinstance(hypotheses, torch.Tensor):
            hypotheses = torch.tensor(hypotheses, dtype=torch.float32)

        premises = premises.to(self.device)
        hypotheses = hypotheses.to(self.device)

        n_samples = premises.shape[0]
        all_results = {
            'cone_energies': [],
            'forward_cone_energies': [],
            'backward_cone_energies': [],
            'cone_asymmetries': []
        }

        # print(f"Computing enhanced cone violations for {n_samples} samples on {self.device}")

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)

            # Get batch
            premise_batch = premises[i:end_idx]
            hypothesis_batch = hypotheses[i:end_idx]

            # Compute both standard and bidirectional cone energies
            with torch.no_grad():
                # Standard cone energy (forward)
                batch_cone_violations = self.cone_membership_energy(premise_batch, hypothesis_batch)
                
                # Enhanced bidirectional energies
                batch_bidirectional = self.bidirectional_cone_energies(premise_batch, hypothesis_batch)
                
                # Store results
                all_results['cone_energies'].append(batch_cone_violations.cpu())
                all_results['forward_cone_energies'].append(batch_bidirectional['forward_cone_energy'].cpu())
                all_results['backward_cone_energies'].append(batch_bidirectional['backward_cone_energy'].cpu())
                all_results['cone_asymmetries'].append(batch_bidirectional['cone_asymmetry'].cpu())

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed batch {i // batch_size + 1}/{(n_samples - 1) // batch_size + 1}")

        # Concatenate all results
        final_results = {}
        for key in all_results.keys():
            final_results[key] = torch.cat(all_results[key], dim=0)

        return final_results

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


class EnhancedHyperbolicConeEmbeddingPipeline:
    def __init__(self, model_path: str = None, K: float=0.01, epsilon: float=0.1):
        """
        Initialize the enhanced pipeline with asymmetric features
        Args:
            model_path: Path to trained enhanced order embeddings (auto-detect if None)
            K: cone aperture parameter
            epsilon: exclusion radius around origin
        """

        # Auto-detect enhanced model if not specified
        if model_path is None:
            possible_paths = [
                "models/enhanced_order_embeddings_snli_10k_asymmetry.pt"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"Auto-detected enhanced model: {path}")
                    break
            
            if model_path is None:
                raise FileNotFoundError("No enhanced order embedding model found!")

        self.model_path = model_path
        self.hyperbolic_pipeline = None
        self.cone_computer = HyperbolicEntailmentCones(K=K, epsilon=epsilon)

        try:
            self.hyperbolic_pipeline = HyperbolicOrderEmbeddingPipeline(model_path)
            print("Successfully loaded enhanced hyperbolic projection of order embeddings")
        except Exception as e:
            print(f"Could not load hyperbolic projection pipeline: {e}")
            raise

    def compute_enhanced_cone_energies(self, premise_bert: torch.Tensor, hypothesis_bert: torch.Tensor) -> Dict[str, torch.Tensor]:
        """ Compute enhanced cone violation energies with asymmetric features
        Args:
            premise_bert: BERT embeddings for premises [batch_size, 768]
            hypothesis_bert: BERT embeddings for hypotheses [batch_size, 768]
        Returns:
            Dictionary with enhanced cone and order energy measurements
        """
        if self.hyperbolic_pipeline is None:
            raise RuntimeError("Hyperbolic pipeline not loaded. Cannot compute cone energies")

        # Get enhanced hyperbolic results (includes asymmetric features)
        if self.hyperbolic_pipeline.device.type == 'cuda':
            # print("Using GPU batch processing for enhanced hyperbolic energies")
            hyperbolic_results = self.hyperbolic_pipeline.compute_hyperbolic_energies_batch(premise_bert, hypothesis_bert)
        else:
            print("Using CPU single-pass processing for enhanced hyperbolic energies")
            hyperbolic_results = self.hyperbolic_pipeline.compute_enhanced_hyperbolic_energies(premise_bert, hypothesis_bert)

        premises_hyp = hyperbolic_results['premise_hyperbolic']
        hypotheses_hyp = hyperbolic_results['hypothesis_hyperbolic']

        # Compute enhanced cone violation energies
        if self.cone_computer.device.type == 'cuda':
            # print("Using GPU batch processing for enhanced cone energies")
            cone_results = self.cone_computer.cone_violation_energy_batch(premises_hyp, hypotheses_hyp)
        else:
            print("Using CPU single-pass processing for enhanced cone energies")
            cone_results = {
                'cone_energies': self.cone_computer.cone_membership_energy(premises_hyp, hypotheses_hyp),
                **self.cone_computer.bidirectional_cone_energies(premises_hyp, hypotheses_hyp)
            }

        # Combine all results
        enhanced_results = {
            # Standard cone energies
            'cone_energies': cone_results['cone_energies'],
            
            # Enhanced bidirectional cone energies
            'forward_cone_energies': cone_results.get('forward_cone_energies', cone_results['cone_energies']),
            'backward_cone_energies': cone_results.get('backward_cone_energies', torch.zeros_like(cone_results['cone_energies'])),
            'cone_asymmetries': cone_results.get('cone_asymmetries', torch.zeros_like(cone_results['cone_energies'])),
            
            # Enhanced order energies from hyperbolic pipeline
            'order_energies': hyperbolic_results['order_energies'],
            'forward_energies': hyperbolic_results.get('forward_energies', hyperbolic_results['order_energies']),
            'backward_energies': hyperbolic_results.get('backward_energies', torch.zeros_like(hyperbolic_results['order_energies'])),
            'asymmetric_energies': hyperbolic_results.get('asymmetric_energies', torch.zeros_like(hyperbolic_results['order_energies'])),
            'asymmetry_measures': hyperbolic_results.get('asymmetry_measures', torch.zeros_like(hyperbolic_results['order_energies'])),
            
            # Geometric features
            'hyperbolic_distances': hyperbolic_results['hyperbolic_distances'],
            'premise_hyperbolic': premises_hyp,
            'hypothesis_hyperbolic': hypotheses_hyp,
            'premise_norms': hyperbolic_results['premise_norms'],
            'hypothesis_norms': hyperbolic_results['hypothesis_norms']
        }

        return enhanced_results

    # Backward compatibility
    def compute_cone_energies(self, premise_bert: torch.Tensor, hypothesis_bert: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Backward compatibility method - calls enhanced version"""
        return self.compute_enhanced_cone_energies(premise_bert, hypothesis_bert)

    def validate_enhanced_cone_hierarchy(self, premise_bert: torch.Tensor, hypothesis_bert: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Validate that enhanced cone energies follow expected hierarchy
        Args:
            premise_bert: BERT embeddings for premises [batch_size, 768]
            hypothesis_bert: BERT embeddings for hypotheses [batch_size, 768]
            labels: Labels tensor where 0=entailment, 1=neutral, 2=contradiction

        Returns:
            Dictionary with mean energies for each category
        """
        results = {}
        all_energies = self.compute_enhanced_cone_energies(premise_bert, hypothesis_bert)

        #Group by label
        for label_idx, label_name in enumerate(['entailment', 'neutral', 'contradiction']):
            mask = (labels == label_idx)
            if mask.sum() > 0:
                # Standard energies
                cone_energies_subset = all_energies['cone_energies'][mask]
                order_energies_subset = all_energies['order_energies'][mask]
                hyperbolic_distances_subset = all_energies['hyperbolic_distances'][mask]
                
                # Enhanced asymmetric features
                forward_cone_subset = all_energies['forward_cone_energies'][mask]
                backward_cone_subset = all_energies['backward_cone_energies'][mask]
                cone_asymmetry_subset = all_energies['cone_asymmetries'][mask]
                forward_order_subset = all_energies['forward_energies'][mask]
                backward_order_subset = all_energies['backward_energies'][mask]
                order_asymmetry_subset = all_energies['asymmetry_measures'][mask]

                results[f"{label_name}_cone_energy"] = safe_tensor_to_float(cone_energies_subset.mean())
                results[f"{label_name}_order_energy"] = safe_tensor_to_float(order_energies_subset.mean())
                results[f"{label_name}_hyperbolic_distance"] = safe_tensor_to_float(hyperbolic_distances_subset.mean())
                
                # Enhanced metrics
                results[f"{label_name}_forward_cone"] = safe_tensor_to_float(forward_cone_subset.mean())
                results[f"{label_name}_backward_cone"] = safe_tensor_to_float(backward_cone_subset.mean())
                results[f"{label_name}_cone_asymmetry"] = safe_tensor_to_float(cone_asymmetry_subset.mean())
                results[f"{label_name}_forward_order"] = safe_tensor_to_float(forward_order_subset.mean())
                results[f"{label_name}_backward_order"] = safe_tensor_to_float(backward_order_subset.mean())
                results[f"{label_name}_order_asymmetry"] = safe_tensor_to_float(order_asymmetry_subset.mean())

        return results


def test_enhanced_cone_implementation():
    set_random_seed(42)
    print("Testing enhanced hyperbolic cone implementation with asymmetric features")
    
    try:
        pipeline = EnhancedHyperbolicConeEmbeddingPipeline()
    except Exception as e:
        print(f"Failed to create enhanced pipeline: {e}")
        return None

    processed_data_path = "data/processed/snli_10k_subset_balanced.pt"
    if not os.path.exists(processed_data_path):
        print("Processed data not found!")
        return None

    from torch.utils.data import DataLoader

    processed_data = torch.load(processed_data_path)
    dataset = EntailmentDataset(processed_data)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    batch = next(iter(dataloader))
    premise_embs = batch['premise_emb'].to(pipeline.hyperbolic_pipeline.device)
    hypothesis_embs = batch['hypothesis_emb'].to(pipeline.hyperbolic_pipeline.device)
    labels = batch['label']
    label_strs = batch['label_str']

    print(f"Testing enhanced pipeline on real dataset: {len(premise_embs)} examples")

    # Compute enhanced cone energies
    results = pipeline.compute_enhanced_cone_energies(premise_embs, hypothesis_embs)

    # Analyze by label with enhanced features
    stats_by_label = {}
    for i, label_str in enumerate(label_strs):
        if label_str not in stats_by_label:
            stats_by_label[label_str] = {
                'cone_energies': [],
                'forward_cone_energies': [],
                'backward_cone_energies': [],
                'cone_asymmetries': [],
                'order_energies': [],
                'forward_energies': [],
                'backward_energies': [],
                'asymmetry_measures': [],
                'hyperbolic_distances': []
            }

        # Standard features
        stats_by_label[label_str]['cone_energies'].append(safe_tensor_to_float(results['cone_energies'][i]))
        stats_by_label[label_str]['order_energies'].append(safe_tensor_to_float(results['order_energies'][i]))
        stats_by_label[label_str]['hyperbolic_distances'].append(safe_tensor_to_float(results['hyperbolic_distances'][i]))
        
        # Enhanced asymmetric features
        stats_by_label[label_str]['forward_cone_energies'].append(safe_tensor_to_float(results['forward_cone_energies'][i]))
        stats_by_label[label_str]['backward_cone_energies'].append(safe_tensor_to_float(results['backward_cone_energies'][i]))
        stats_by_label[label_str]['cone_asymmetries'].append(safe_tensor_to_float(results['cone_asymmetries'][i]))
        stats_by_label[label_str]['forward_energies'].append(safe_tensor_to_float(results['forward_energies'][i]))
        stats_by_label[label_str]['backward_energies'].append(safe_tensor_to_float(results['backward_energies'][i]))
        stats_by_label[label_str]['asymmetry_measures'].append(safe_tensor_to_float(results['asymmetry_measures'][i]))

    print("Enhanced Cone Implementation Results:")
    print("-" * 80)
    for label, stats in stats_by_label.items():
        print(f"\n{label.upper()}:")
        print(f"  Cone Energy:           {np.mean(stats['cone_energies']):.4f} ± {np.std(stats['cone_energies']):.4f}")
        print(f"  Forward Cone Energy:   {np.mean(stats['forward_cone_energies']):.4f} ± {np.std(stats['forward_cone_energies']):.4f}")
        print(f"  Backward Cone Energy:  {np.mean(stats['backward_cone_energies']):.4f} ± {np.std(stats['backward_cone_energies']):.4f}")
        print(f"  Cone Asymmetry:        {np.mean(stats['cone_asymmetries']):.4f} ± {np.std(stats['cone_asymmetries']):.4f}")
        print(f"  Order Energy:          {np.mean(stats['order_energies']):.4f} ± {np.std(stats['order_energies']):.4f}")
        print(f"  Forward Order Energy:  {np.mean(stats['forward_energies']):.4f} ± {np.std(stats['forward_energies']):.4f}")
        print(f"  Backward Order Energy: {np.mean(stats['backward_energies']):.4f} ± {np.std(stats['backward_energies']):.4f}")
        print(f"  Order Asymmetry:       {np.mean(stats['asymmetry_measures']):.4f} ± {np.std(stats['asymmetry_measures']):.4f}")
        print(f"  Hyperbolic Distance:   {np.mean(stats['hyperbolic_distances']):.4f} ± {np.std(stats['hyperbolic_distances']):.4f}")

    # Validate enhanced hierarchy
    print("\nEnhanced Hierarchy Validation:")
    
    # Standard cone energy hierarchy
    ent_cone = np.mean(stats_by_label.get('entailment', {}).get('cone_energies', [float('inf')]))
    neu_cone = np.mean(stats_by_label.get('neutral', {}).get('cone_energies', [float('inf')]))
    con_cone = np.mean(stats_by_label.get('contradiction', {}).get('cone_energies', [float('inf')]))

    hierarchy_valid = ent_cone < neu_cone < con_cone
    print(f"Standard Cone Energy Hierarchy: {'VALID' if hierarchy_valid else 'INVALID'}")
    print(f"   {ent_cone:.4f} < {neu_cone:.4f} < {con_cone:.4f}")
    
    # Asymmetry pattern analysis
    print("\nAsymmetry Pattern Analysis:")
    for label in ['entailment', 'neutral', 'contradiction']:
        if label in stats_by_label:
            cone_asym = np.mean(stats_by_label[label]['cone_asymmetries'])
            order_asym = np.mean(stats_by_label[label]['asymmetry_measures'])
            print(f"  {label}: Cone Asymmetry={cone_asym:.4f}, Order Asymmetry={order_asym:.4f}")

    return pipeline, results, stats_by_label


# Backward compatibility
def test_cone_implementation():
    """Backward compatibility function - calls enhanced version"""
    return test_enhanced_cone_implementation()


if __name__ == "__main__":
    # Run enhanced tests
    test_results = test_enhanced_cone_implementation()
    
    if test_results is not None:
        print("\nEnhanced cone implementation test completed successfully!")
    else:
        print("\nEnhanced cone implementation test failed!")