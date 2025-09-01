"""
Point Cloud Clustering Test (Updated for Separate Models)
Tests clustering accuracy using separate trained order embedding and asymmetry models
"""

import os
import sys
import json
import numpy as np
import torch
import random
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from persim import PersistenceImager
from itertools import permutations
import matplotlib.pyplot as plt
from gph.python import ripser_parallel
from sklearn.metrics.pairwise import pairwise_distances

# Import the separate models
from order_asymmetry_models import OrderEmbeddingModel
#from order_asymmetry_models import AsymmetryTransformModel
from independent_asymmetry_model import AsymmetryTransformModel
from hyperbolic_token_projector import TokenLevelHyperbolicProjector


@dataclass
class ClusteringResult:
    """Results for point cloud clustering test"""
    order_model_name: str
    asymmetry_model_name: str
    clustering_accuracy: float
    silhouette_score: float
    adjusted_rand_score: float
    num_samples: int
    success: bool  # True if accuracy > 70%
    ph_dim_values: Dict[str, List[float]]  # class_name -> list of ph_dim values
    ph_dim_stats: Dict[str, Dict[str, float]]  # class_name -> {mean, std, min, max}
    point_cloud_stats: Dict[str, Dict[str, float]]  # Point cloud size statistics


class SeparateModelPointCloudGenerator:
    """Generate point clouds using separate trained order embedding and asymmetry models"""
    
    def __init__(self, order_model_path: str, asymmetry_model_path: str, hyperbolic_model_path: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading separate models...")
        print(f"Order model: {order_model_path}")
        print(f"Asymmetry model: {asymmetry_model_path}")
        
        # Load order embedding model
        order_checkpoint = torch.load(order_model_path, map_location=self.device)
        self.order_model = OrderEmbeddingModel(hidden_size=768)
        self.order_model.load_state_dict(order_checkpoint['model_state_dict'])
        self.order_model.to(self.device)
        self.order_model.eval()
        
        # Load asymmetry transform model
        asymmetry_checkpoint = torch.load(asymmetry_model_path, map_location=self.device)
        self.asymmetry_model = AsymmetryTransformModel(hidden_size=768)
        self.asymmetry_model.load_state_dict(asymmetry_checkpoint['model_state_dict'])
        self.asymmetry_model.to(self.device)
        self.asymmetry_model.eval()

        hyperbolic_checkpoint = torch.load(hyperbolic_model_path, map_location=self.device)
        self.hyperbolic_model = TokenLevelHyperbolicProjector()
        self.hyperbolic_model.load_state_dict(hyperbolic_checkpoint['projector_state_dict'])
        self.hyperbolic_model.to(self.device)
        self.hyperbolic_model.eval()
        
        print(f"Both models loaded on {self.device}")
        print(f"Order model training stats: Best val loss = {order_checkpoint.get('best_val_loss', 'N/A')}")
        print(f"Asymmetry model training stats: Best val loss = {asymmetry_checkpoint.get('best_val_loss', 'N/A')}")
    
    def generate_point_cloud_variations(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate point cloud variations using separate models
        
        Args:
            tokens: SBERT token embeddings [num_tokens, 768]
            
        Returns:
            List of point clouds: [original_tokens, order_embeddings, asymmetric_features]
        """
        point_clouds = []
        
        with torch.no_grad():
            tokens = tokens.to(self.device)
            
            # 1. Original SBERT tokens (semantic baseline)
            point_clouds.append(tokens.cpu().clone())
            
            # 2. Order embeddings (hierarchical structure)
            order_embeddings = self.order_model(tokens)
            point_clouds.append(order_embeddings.cpu().clone())
            
            # 3. Asymmetric features (directional relationships)
            asymmetric_features = self.asymmetry_model(order_embeddings)  # Takes order embeddings as input
            point_clouds.append(asymmetric_features.cpu().clone())

            #4. Hyperbolic features - Don't seem to help       
            hyperbolic_features = self.hyperbolic_model(order_embeddings)
            point_clouds.append(hyperbolic_features.cpu().clone())
        
        return point_clouds
    
    def generate_premise_hypothesis_point_cloud(self, premise_tokens: torch.Tensor, 
                                               hypothesis_tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Generate combined point cloud from premise-hypothesis pair with detailed statistics
        """
        # Generate premise point clouds
        premise_clouds = self.generate_point_cloud_variations(premise_tokens)
        
        # Generate hypothesis point clouds
        hypothesis_clouds = self.generate_point_cloud_variations(hypothesis_tokens)

        #ENERGY WEIGHTED
        energy_weighted_cloud = self._generate_refined_energy_weighted_features(premise_tokens, hypothesis_tokens)

        #DIRECTIONAL
        directional_cloud = self._generate_enhanced_directional_separation(premise_tokens, hypothesis_tokens)
                
        # Use the existing successful general stratification
        separation_stratified_cloud = self._generate_perfect_separation_stratified_features(premise_tokens, hypothesis_tokens)

        multi_boundary_cloud = self._generate_multi_signal_stratified_features(premise_tokens, hypothesis_tokens)
   
        # Combine all point clouds
        all_clouds = premise_clouds + hypothesis_clouds + [energy_weighted_cloud, directional_cloud, separation_stratified_cloud, multi_boundary_cloud]

        # Check total size and optimize if needed
        total_points = sum(cloud.shape[0] for cloud in all_clouds)
        target_max = 500  # Sweet spot from handover docs
        
        if total_points > target_max:
            print(f"Point cloud optimization: {total_points} -> {target_max}")
            all_clouds = self._optimize_point_cloud_size(all_clouds, target_max)

        combined_cloud = torch.cat(all_clouds, dim=0)

        has_hyperbolic = len(premise_clouds) == 4  # Check if hyperbolic was added
        # print(f"Has hyperbolic is {has_hyperbolic}")
        
        # Generate detailed statistics
        stats = {
            'premise_original_points': premise_clouds[0].shape[0],
            'premise_order_points': premise_clouds[1].shape[0],
            'premise_asymmetric_points': premise_clouds[2].shape[0],
            'premise_total_points': sum(cloud.shape[0] for cloud in premise_clouds),
            
            'hypothesis_original_points': hypothesis_clouds[0].shape[0],
            'hypothesis_order_points': hypothesis_clouds[1].shape[0], 
            'hypothesis_asymmetric_points': hypothesis_clouds[2].shape[0],
            'hypothesis_total_points': sum(cloud.shape[0] for cloud in hypothesis_clouds),
            
            'combined_total_points': combined_cloud.shape[0],
            'sufficient_for_phd': combined_cloud.shape[0] >= 100
        }

        if has_hyperbolic:
            stats['premise_hyperbolic_points'] = premise_clouds[3].shape[0]
            stats['hypothesis_hyperbolic_points'] = hypothesis_clouds[3].shape[0]
        
        return combined_cloud, stats

    def analyze_model_outputs(self, premise_tokens: torch.Tensor, hypothesis_tokens: torch.Tensor) -> Dict:
        """
        Analyze the outputs of both models for debugging/validation
        """
        analysis = {}
        
        with torch.no_grad():
            premise_tokens = premise_tokens.to(self.device)
            hypothesis_tokens = hypothesis_tokens.to(self.device)
            
            # Order model analysis
            premise_order = self.order_model(premise_tokens)
            hypothesis_order = self.order_model(hypothesis_tokens)
            
            order_energy = self.order_model.order_violation_energy(premise_order, hypothesis_order)
            
            analysis['order_model'] = {
                'premise_order_range': [premise_order.min().item(), premise_order.max().item()],
                'hypothesis_order_range': [hypothesis_order.min().item(), hypothesis_order.max().item()],
                'order_violation_energy': order_energy.item(),
                'premise_order_mean': premise_order.mean().item(),
                'hypothesis_order_mean': hypothesis_order.mean().item()
            }
            
            # Asymmetry model analysis
            premise_asym = self.asymmetry_model(premise_order)
            hypothesis_asym = self.asymmetry_model(hypothesis_order)
            
            asymmetric_energy = self.asymmetry_model.compute_asymmetric_energy(premise_order, hypothesis_order)
            
            analysis['asymmetry_model'] = {
                'premise_asym_range': [premise_asym.min().item(), premise_asym.max().item()],
                'hypothesis_asym_range': [hypothesis_asym.min().item(), hypothesis_asym.max().item()],
                'asymmetric_energy': asymmetric_energy.item(),
                'premise_asym_mean': premise_asym.mean().item(),
                'hypothesis_asym_mean': hypothesis_asym.mean().item()
            }
            
            # Combined analysis
            analysis['combined'] = {
                'forward_energy': order_energy.item(),
                'backward_energy': self.order_model.order_violation_energy(hypothesis_order, premise_order).item(),
                'asymmetry_measure': abs(analysis['order_model']['order_violation_energy'] - 
                                       self.order_model.order_violation_energy(hypothesis_order, premise_order).item())
            }
        
        return analysis


    def _generate_enhanced_directional_separation(self, premise_tokens: torch.Tensor, hypothesis_tokens: torch.Tensor) -> torch.Tensor:
        """
        Generate enhanced directional separation features optimized for Bray-Curtis distance
        """
        with torch.no_grad():
            premise_tokens = premise_tokens.to(self.device)
            hypothesis_tokens = hypothesis_tokens.to(self.device)
            
            # Get order and asymmetric embeddings
            premise_order = self.order_model(premise_tokens)
            hypothesis_order = self.order_model(hypothesis_tokens)
            premise_asym = self.asymmetry_model(premise_order)
            hypothesis_asym = self.asymmetry_model(hypothesis_order)
            
            directional_points = []
            
            # 1. Global directional vector (premise centroid → hypothesis centroid)
            premise_centroid = premise_order.mean(dim=0)
            hypothesis_centroid = hypothesis_order.mean(dim=0)
            global_direction = hypothesis_centroid - premise_centroid
            
            # 2. Asymmetric energy scaling (leverages your perfect asymmetric patterns)
            asymmetric_energy = self.asymmetry_model.compute_asymmetric_energy(premise_order, hypothesis_order)
            
            # 3. Create multiple directional points with different asymmetric scalings
            for scale_factor in [0.5, 1.0, 1.5, 2.0]:  # Multiple scales for topological richness
                # Scale directional vector by asymmetric energy
                scaled_direction = global_direction * (asymmetric_energy.item() * scale_factor)
                
                # Create points along the directional vector
                directional_point_forward = premise_centroid + scaled_direction
                directional_point_backward = hypothesis_centroid - scaled_direction
                
                directional_points.extend([directional_point_forward, directional_point_backward])
            
            # 4. Token-level directional features (subset to avoid explosion)
            # Use top 5 tokens by asymmetric magnitude from each side
            premise_asym_norms = torch.norm(premise_asym, dim=-1)
            hypothesis_asym_norms = torch.norm(hypothesis_asym, dim=-1)
            
            top_premise_idx = torch.topk(premise_asym_norms, k=min(5, len(premise_asym_norms))).indices
            top_hypothesis_idx = torch.topk(hypothesis_asym_norms, k=min(5, len(hypothesis_asym_norms))).indices
            
            for i in top_premise_idx:
                for j in top_hypothesis_idx:
                    # Directional vector between high-asymmetry tokens
                    token_direction = hypothesis_asym[j] - premise_asym[i]
                    # Scale by local asymmetric strength
                    local_asym_strength = (premise_asym_norms[i] + hypothesis_asym_norms[j]) / 2
                    scaled_token_direction = token_direction * local_asym_strength
                    
                    directional_points.append(premise_asym[i] + scaled_token_direction)
            
            return torch.stack(directional_points).cpu()


    def _generate_perfect_separation_stratified_features(self, premise_tokens: torch.Tensor, hypothesis_tokens: torch.Tensor) -> torch.Tensor:
        """
        Stratified features optimized for the PERFECT energy separation achieved by new order model
        
        New Energy Zones:
        - Entailment: ~0.4
        - Neutral: ~0.8-1.5 
        - Contradiction: ~1.8+
        """
        with torch.no_grad():
            premise_tokens = premise_tokens.to(self.device)
            hypothesis_tokens = hypothesis_tokens.to(self.device)
            
            premise_order = self.order_model(premise_tokens)
            hypothesis_order = self.order_model(hypothesis_tokens)
            
            # Use forward energy to determine exact zone
            forward_energy = self.order_model.order_violation_energy(
                premise_order.mean(0, keepdim=True), hypothesis_order.mean(0, keepdim=True)
            ).item()
            
            stratified_points = []
            centroid = (premise_order.mean(0) + hypothesis_order.mean(0)) / 2
            
            # ZONE 1: Entailment (< 1.0) - ULTRA TIGHT clusters WAS 0.5
            if forward_energy < 0.5:
                # print(f"    ENTAILMENT zone detected (energy={forward_energy:.3f})")
                target_distances = [0.01, 0.02, 0.03, 0.04, 0.05]  # Extremely tight
                for dist in target_distances:
                    for i in range(15):  # Dense for maximum cohesion
                        direction = torch.randn(768, device=self.device)
                        direction = direction / torch.norm(direction) * dist
                        stratified_points.append(centroid + direction)
            
            # ZONE 2: Neutral ([1.0, 1.5]) - DISTINCTIVE middle pattern. WAS 0.5 to 1.5
            elif 0.5 <= forward_energy <= 1.5:
                # print(f"    NEUTRAL zone detected (energy={forward_energy:.3f})")
                # Create STRUCTURED neutral signature instead of random
                
                # Ring pattern at moderate distance (unique to neutral)
                ring_radius = 0.8
                num_ring_points = 16
                
                for i in range(num_ring_points):
                    angle = 2 * np.pi * i / num_ring_points
                    
                    # Create two orthogonal directions in 768D space
                    direction1 = hypothesis_order.mean(0) - premise_order.mean(0)
                    direction1 = direction1 / torch.norm(direction1)
                    
                    # Random orthogonal direction
                    direction2 = torch.randn(768, device=self.device)
                    direction2 = direction2 - torch.dot(direction2, direction1) * direction1
                    direction2 = direction2 / torch.norm(direction2)
                    
                    # Point on ring
                    ring_point = centroid + ring_radius * (np.cos(angle) * direction1 + np.sin(angle) * direction2)
                    stratified_points.append(ring_point)
                
                # Add central cluster for stability
                for i in range(8):
                    noise = torch.randn(768, device=self.device) * 0.1
                    stratified_points.append(centroid + noise)
            
            # ZONE 3: Contradiction (> 2.0) - MAXIMUM spread WAS 1.5
            elif forward_energy > 1.6:
                # print(f"    CONTRADICTION zone detected (energy={forward_energy:.3f})")
                target_distances = [3.5, 4.0, 4.5, 5.0, 5.5]  # Even more spread
                for dist in target_distances:
                    for i in range(5):  # Sparse for maximum spread
                        direction = torch.randn(768, device=self.device)
                        direction = direction / torch.norm(direction) * dist
                        stratified_points.append(centroid + direction)
            
            # EDGE CASE: Energy in gap zones (shouldn't happen with perfect separation)
            else:
                # print(f"    EDGE CASE: energy={forward_energy:.3f} in gap zone")
                # Default moderate pattern
                target_distances = [1.0, 1.5, 2.0]
                for dist in target_distances:
                    for i in range(6):
                        direction = torch.randn(768, device=self.device)
                        direction = direction / torch.norm(direction) * dist
                        stratified_points.append(centroid + direction)
            
            if stratified_points:
                return torch.stack(stratified_points).cpu()
            else:
                return torch.empty(0, 768)


    def _generate_multi_signal_stratified_features(self, premise_tokens: torch.Tensor, hypothesis_tokens: torch.Tensor) -> torch.Tensor:
        """
        Use BOTH forward energy AND asymmetric energy - with ORIGINAL boundaries
        """
        with torch.no_grad():
            premise_tokens = premise_tokens.to(self.device)
            hypothesis_tokens = hypothesis_tokens.to(self.device)
            
            premise_order = self.order_model(premise_tokens)
            hypothesis_order = self.order_model(hypothesis_tokens)
            
            forward_energy = self.order_model.order_violation_energy(
                premise_order.mean(0, keepdim=True), hypothesis_order.mean(0, keepdim=True)
            ).item()
            
            asymmetric_energy = self.asymmetry_model.compute_asymmetric_energy(premise_order, hypothesis_order).item()
            
            stratified_points = []
            centroid = (premise_order.mean(0) + hypothesis_order.mean(0)) / 2
            
            # Multi-signal classification with ORIGINAL 0.5/1.5 boundaries
            if forward_energy < 0.5 and asymmetric_energy > 0.3:  # Strong entailment
                target_distances = [0.01, 0.02, 0.03]
                point_count = 25
            elif forward_energy < 0.5:  # Weak entailment (low asymmetric)
                target_distances = [0.05, 0.1, 0.15]
                point_count = 15
            elif forward_energy > 1.5 and asymmetric_energy > 0.8:  # Strong contradiction
                target_distances = [5.0, 6.0, 7.0]
                point_count = 8
            elif forward_energy > 1.5:  # Weak contradiction
                target_distances = [3.0, 4.0, 5.0]
                point_count = 10
            else:  # Neutral (0.5 ≤ forward_energy ≤ 1.5)
                target_distances = [0.4, 0.6, 0.8, 1.0]
                point_count = 12
            
            for dist in target_distances:
                for i in range(point_count):
                    direction = torch.randn(768, device=self.device)
                    direction = direction / torch.norm(direction) * dist
                    stratified_points.append(centroid + direction)
            
            if stratified_points:
                return torch.stack(stratified_points).cpu()
            else:
                return torch.empty(0, 768)

    def _generate_refined_energy_weighted_features(self, premise_tokens: torch.Tensor, hypothesis_tokens: torch.Tensor) -> torch.Tensor:
        """
        Refined version of energy weighting - only use high-confidence energy predictions
        """
        with torch.no_grad():
            premise_tokens = premise_tokens.to(self.device)
            hypothesis_tokens = hypothesis_tokens.to(self.device)
            
            premise_order = self.order_model(premise_tokens)
            hypothesis_order = self.order_model(hypothesis_tokens)
            
            energy_weighted_points = []
            
            for i in range(premise_order.shape[0]):
                for j in range(hypothesis_order.shape[0]):
                    energy = self.order_model.order_violation_energy(
                        premise_order[i:i+1], hypothesis_order[j:j+1]
                    )
                    
                    # REFINEMENT: Only use tokens with "confident" energy predictions
                    energy_confidence = abs(energy.item() - 0.8)  # Distance from neutral zone
                    
                    if energy_confidence > 0.3:  # Only confident predictions
                        energy_weight = torch.sigmoid(energy)
                        weighted_point = energy_weight * hypothesis_order[j] + (1 - energy_weight) * premise_order[i]
                        energy_weighted_points.append(weighted_point)
            
            # Ensure minimum points if filtering was too aggressive
            if len(energy_weighted_points) < 20:
                # Fall back to top-K most confident pairs
                all_pairs = []
                for i in range(premise_order.shape[0]):
                    for j in range(hypothesis_order.shape[0]):
                        energy = self.order_model.order_violation_energy(
                            premise_order[i:i+1], hypothesis_order[j:j+1]
                        )
                        confidence = abs(energy.item() - 0.8)
                        all_pairs.append((i, j, energy.item(), confidence))
                
                # Sort by confidence and take top 30
                all_pairs.sort(key=lambda x: x[3], reverse=True)
                energy_weighted_points = []
                
                for i, j, energy, _ in all_pairs[:30]:
                    energy_weight = torch.sigmoid(torch.tensor(energy))
                    weighted_point = energy_weight * hypothesis_order[j] + (1 - energy_weight) * premise_order[i]
                    energy_weighted_points.append(weighted_point)
            
            return torch.stack(energy_weighted_points).cpu()



    def _optimize_point_cloud_size(self, all_clouds: List[torch.Tensor], target_max: int) -> List[torch.Tensor]:
        """
        Intelligent point cloud size optimization preserving most important points
        """
        total_points = sum(cloud.shape[0] for cloud in all_clouds)
        reduction_needed = total_points - target_max
        
        # Priority order (based on your success rates)
        # Keep original 6 clouds intact, reduce enhancement clouds proportionally
        cloud_priorities = [1.0] * 6 + [0.9, 0.8, 0.7, 0.6]  # Original clouds + enhancements
        
        optimized_clouds = []
        
        for i, cloud in enumerate(all_clouds):
            if i < 6:  # Keep original premise/hypothesis clouds intact
                optimized_clouds.append(cloud)
            else:
                # Reduce enhancement clouds if needed
                priority = cloud_priorities[min(i, len(cloud_priorities)-1)]
                reduction_factor = max(0.7, 1.0 - (reduction_needed / total_points) / priority)
                
                keep_points = int(cloud.shape[0] * reduction_factor)
                if keep_points < cloud.shape[0]:
                    indices = torch.randperm(cloud.shape[0])[:keep_points]
                    optimized_clouds.append(cloud[indices])
                    reduction_needed -= (cloud.shape[0] - keep_points)
                else:
                    optimized_clouds.append(cloud)
        
        return optimized_clouds


class SeparateModelClusteringValidator:
    """
    Validator for point cloud clustering using separate trained models
    """
    
    def __init__(self, val_data_path: str,
                 order_model_path: str=None,
                 asymmetry_model_path: str=None,
                 hyperbolic_model_path: str=None,
                 output_dir: str = "phd_method/individual_phd_clustering_results/",
                 seed: int = 42):
        
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load validation data
        print(f"Loading validation data: {val_data_path}")
        with open(val_data_path, 'rb') as f:
            self.val_data = pickle.load(f)
        
        print(f"Loaded {len(self.val_data['labels'])} validation samples")
        
        # Initialize point cloud generator with separate models
        if order_model_path:
            self.point_cloud_generator = SeparateModelPointCloudGenerator(
                order_model_path, asymmetry_model_path, hyperbolic_model_path
            )
        
        # Clustering parameters
        self.samples_per_class = 100  # 100 samples per class for robust clustering
        self.min_points_for_phd = 200  # Minimum points for reliable PHD
        
        print(f"Separate model clustering validator initialized")
        print(f"Samples per class: {self.samples_per_class}")
    
    def ph_dim_and_diagrams_from_distance_matrix(self, dm: np.ndarray,
                                 min_points: int = 50,
                                 max_points: int = 1000,
                                 point_jump: int = 25,
                                 h_dim: int = 0,
                                 alpha: float = 1.0) -> Tuple[float, List]:
        """
        FIXED VERSION: Compute persistence on FULL point cloud, not subsampled
        """
        assert dm.ndim == 2 and dm.shape[0] == dm.shape[1]
        
        # DON'T SUBSAMPLE - use the full distance matrix for topology!
        print(f"Computing persistence on full {dm.shape[0]} point cloud...")
    
        # Compute persistence diagrams on FULL point cloud with H1 dimension
        full_diagrams = ripser_parallel(dm, maxdim=1, n_threads=-1, metric="precomputed")['dgms']
        
        print(f"  H0 features: {len(full_diagrams[0])}")
        print(f"  H1 features: {len(full_diagrams[1])}")
        
        # For PH-dimension, still need to subsample (that's what PH-dim measures)
        # But for clustering, we want the FULL topology
        test_n = range(min_points, min(max_points, dm.shape[0]), point_jump)
        lengths = []
        
        for points_number in test_n:
            if points_number >= dm.shape[0]:
                break
                
            sample_indices = np.random.choice(dm.shape[0], points_number, replace=False)
            dist_matrix = dm[sample_indices, :][:, sample_indices]
            
            # Compute persistence diagrams - this is for PH-DIM calculation
            sub_diagrams = ripser_parallel(dist_matrix, maxdim=0, n_threads=-1, metric="precomputed")['dgms']
            
            # Extract specific dimension for PH-dim calculation
            d = sub_diagrams[h_dim]
            d = d[d[:, 1] < np.inf]
            lengths.append(np.power((d[:, 1] - d[:, 0]), alpha).sum())
        
        if len(lengths) < 2:
            ph_dimension = 0.0
        else:
            lengths = np.array(lengths)
            
            # Compute PH dimension
            x = np.log(np.array(list(test_n[:len(lengths)])))
            y = np.log(lengths)
            N = len(x)
            
            if N < 2:
                ph_dimension = 0.0
            else:
                m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
                ph_dimension = alpha / (1 - m) if m != 1 else 0.0
        
        # Return FULL topology diagrams (not subsampled ones!)
        return ph_dimension, full_diagrams
    
    def persistence_diagrams_to_images(self, all_diagrams: List) -> List[np.ndarray]:
        """Convert persistence diagrams to standardized images with robust error handling"""
        
        # First, analyze the actual range of data that exists
        all_birth_times = []
        all_death_times = []
        all_lifespans = []
        valid_diagrams_count = 0
        
        print("Analyzing persistence diagram ranges...")
        
        for diagram_idx, diagrams in enumerate(all_diagrams):
            if diagrams is None:
                continue
            if isinstance(diagrams, (list, tuple)) and len(diagrams) == 0:
                continue
            if isinstance(diagrams, np.ndarray) and diagrams.size == 0:
                continue

            #IF not including H2 reduce back to 2    
            for dim in range(min(2, len(diagrams))):
                diagram = diagrams[dim]
                
                # Handle different possible diagram formats
                if isinstance(diagram, np.ndarray) and diagram.size > 0:
                    # Check if it's 1D (empty) or 2D (has features)
                    if diagram.ndim == 1:
                        print(f"  Diagram {diagram_idx}, dim {dim}: 1D array (no features)")
                        continue
                    elif diagram.ndim == 2 and diagram.shape[1] >= 2:
                        # Valid 2D diagram with birth/death columns
                        finite_mask = np.isfinite(diagram).all(axis=1)
                        finite_diagram = diagram[finite_mask]
                        
                        if len(finite_diagram) > 0:
                            all_birth_times.extend(finite_diagram[:, 0])
                            all_death_times.extend(finite_diagram[:, 1])
                            lifespans = finite_diagram[:, 1] - finite_diagram[:, 0]
                            all_lifespans.extend(lifespans)
                            valid_diagrams_count += 1
                            print(f"  Diagram {diagram_idx}, dim {dim}: {len(finite_diagram)} finite features")
                        else:
                            print(f"  Diagram {diagram_idx}, dim {dim}: No finite features")
                    else:
                        print(f"  Diagram {diagram_idx}, dim {dim}: Unexpected shape {diagram.shape}")
                else:
                    print(f"  Diagram {diagram_idx}, dim {dim}: Empty or invalid")
        
        print(f"Found {valid_diagrams_count} valid diagrams with {len(all_lifespans)} total finite features")
        
        if len(all_lifespans) == 0:
            print("❌ No finite features found across all diagrams!")
            return []
        
        # Calculate actual data ranges
        min_birth = np.min(all_birth_times)
        max_birth = np.max(all_birth_times)
        min_life = np.min(all_lifespans)
        max_life = np.max(all_lifespans)
        
        print(f"Actual persistence ranges:")
        print(f"  Birth times: {min_birth:.4f} - {max_birth:.4f}")
        print(f"  Lifespans: {min_life:.4f} - {max_life:.4f}")
        
        # Use data-driven ranges with padding
        birth_padding = max(0.01, (max_birth - min_birth) * 0.1)
        life_padding = max(0.001, (max_life - min_life) * 0.1)
        
        birth_range = (max(0, min_birth - birth_padding), max_birth + birth_padding)
        pers_range = (max(0.001, min_life - life_padding), max_life + life_padding)
        
        print(f"Adjusted persistence image parameters:")
        print(f"  birth_range: ({birth_range[0]:.4f}, {birth_range[1]:.4f})")
        print(f"  pers_range: ({pers_range[0]:.4f}, {pers_range[1]:.4f})")
        
        # Create persistence imager with correct scale
        # pixel_size = max(0.005, (pers_range[1] - pers_range[0]) / 50)
        # sigma = max(0.005, (pers_range[1] - pers_range[0]) / 30)

        #USING BEST CONFIGS FROM PARAMETER SEARCH
        pixel_size = max(0.001, (pers_range[1] - pers_range[0]) / 138.9)  # Changed from /50 to /100
        sigma = max(0.001, (pers_range[1] - pers_range[0]) / 82.6)        # Changed from /30 to /60
        target_resolution = 30  # Changed from 20 to 25
        
        pimgr = PersistenceImager(
            pixel_size=pixel_size,
            birth_range=birth_range,
            pers_range=pers_range,
            kernel_params={'sigma': sigma}
        )
        
        print(f"PersistenceImager config: pixel_size={pixel_size:.4f}, sigma={sigma:.4f}")
        
        persistence_images = []
        successful_conversions = 0
        
        for diagram_idx, diagrams in enumerate(all_diagrams):
            if diagrams is None:
                continue
            if isinstance(diagrams, (list, tuple)) and len(diagrams) == 0:
                continue
            if isinstance(diagrams, np.ndarray) and diagrams.size == 0:
                continue

                
            # combined_image = np.zeros((20, 20))
            combined_image = np.zeros((target_resolution, target_resolution))
            has_content = False
            
            # Process H0 and H1 diagrams with robust handling -- reduce to 2 if not including H2
            for dim in range(min(2, len(diagrams))):
                diagram = diagrams[dim]
                
                # Robust diagram handling
                if isinstance(diagram, np.ndarray) and diagram.size > 0:
                    if diagram.ndim == 1:
                        # 1D array means no features
                        continue
                    elif diagram.ndim == 2 and diagram.shape[1] >= 2:
                        # Valid 2D diagram
                        finite_mask = np.isfinite(diagram).all(axis=1)
                        finite_diagram = diagram[finite_mask]
                        
                        if len(finite_diagram) > 0:
                            try:
                                img = pimgr.transform([finite_diagram])[0]
                                
                                # Resize if needed
                                # if img.shape != (20, 20):
                                if img.shape != (target_resolution, target_resolution):
                                    from scipy.ndimage import zoom
                                    # zoom_factors = (20 / img.shape[0], 20 / img.shape[1])
                                    zoom_factors = (target_resolution / img.shape[0], target_resolution / img.shape[1])
                                    img = zoom(img, zoom_factors)
                                
                                combined_image += img
                                has_content = True
                                
                            except Exception as e:
                                print(f"    Failed to convert diagram {diagram_idx}, dim {dim}: {e}")
                                continue
            
            # Only add if image has content
            if has_content and combined_image.max() > 0:
                combined_image = combined_image / combined_image.max()
                persistence_images.append(combined_image.flatten())
                successful_conversions += 1
            else:
                print(f"    Diagram {diagram_idx}: No content after processing")
        
        print(f"\nPersistence image conversion results:")
        print(f"  Successful: {successful_conversions}/{len(all_diagrams)}")
        print(f"  Success rate: {successful_conversions/len(all_diagrams)*100:.1f}%" if all_diagrams else "N/A")
        
        return persistence_images


    def compute_distance_matrix(self, point_cloud: torch.Tensor, metric: str = 'braycurtis') -> np.ndarray:
        """Compute distance matrix for point cloud"""
        
        point_cloud_np = point_cloud.numpy()
        
        distance_matrix = pairwise_distances(point_cloud_np, metric=metric)
        return distance_matrix

    #WAS 40
    def filter_samples_by_token_count(self, samples: List[Dict], min_combined_tokens: int = 0) -> List[Dict]:
        """
        Pre-filter samples to ensure sufficient tokens for 200+ point clouds
        
        Args:
            samples: List of sample dictionaries
            min_combined_tokens: Minimum combined tokens needed 
                            (67 tokens × 3 transforms = 201 points)
        
        Returns:
            Filtered samples with sufficient token counts
        """
        filtered_samples = []
        
        for sample in samples:
            premise_tokens = sample['premise_tokens'].shape[0]
            hypothesis_tokens = sample['hypothesis_tokens'].shape[0]
            combined_tokens = premise_tokens + hypothesis_tokens
            
            if combined_tokens >= min_combined_tokens:
                filtered_samples.append(sample)
        
        print(f"Token filtering: {len(filtered_samples)}/{len(samples)} samples have ≥{min_combined_tokens} tokens")
        
        return filtered_samples

    def generate_fixed_samples_by_class(self) -> Dict[str, List[Dict]]:
        """Generate fixed sample indices for each class"""
        
        print("Generating fixed sample indices...")
        
        # Organize data by class
        class_data = {'entailment': [], 'neutral': [], 'contradiction': []}
        
        for i, label in enumerate(self.val_data['labels']):
            class_data[label].append({
                'index': i,
                'premise_tokens': self.val_data['premise_tokens'][i],
                'hypothesis_tokens': self.val_data['hypothesis_tokens'][i],
                'label': label
            })
        
        # Sample fixed indices for each class
        fixed_samples = {}
        
        for class_name, class_samples in class_data.items():
            print(f"  {class_name}: {len(class_samples)} total samples available")

            filtered_samples = self.filter_samples_by_token_count(class_samples)
            
            if len(filtered_samples) < self.samples_per_class:
                print(f"Warning: Only {len(filtered_samples)} samples with sufficient tokens")
                selected_samples = filtered_samples
            else:
                # Set seed for reproducible sampling
                class_seeds = {'entailment': 42, 'neutral': 142, 'contradiction': 242}
                np.random.seed(class_seeds[class_name])
                selected_indices = np.random.choice(
                    len(filtered_samples), self.samples_per_class, replace=False
                )
                selected_samples = [filtered_samples[i] for i in selected_indices]
            
            fixed_samples[class_name] = selected_samples
            print(f"    Selected {len(selected_samples)} samples")

            # Show token statistics for selected samples
            token_counts = [
                s['premise_tokens'].shape[0] + s['hypothesis_tokens'].shape[0] 
                for s in selected_samples
            ]
            print(f"Token count stats: {np.mean(token_counts):.0f} ± {np.std(token_counts):.0f} "
                f"(range: {np.min(token_counts)}-{np.max(token_counts)})")
        
        return fixed_samples

    def generate_maximum_samples_by_class(self) -> Dict[str, List[Dict]]:
        """Generate maximum available samples for each class for more robust clustering"""
        
        print("Generating maximum sample indices...")
        
        # Organize data by class
        class_data = {'entailment': [], 'neutral': [], 'contradiction': []}
        
        for i, label in enumerate(self.val_data['labels']):
            class_data[label].append({
                'index': i,
                'premise_tokens': self.val_data['premise_tokens'][i],
                'hypothesis_tokens': self.val_data['hypothesis_tokens'][i],
                'label': label
            })
        
        # Use ALL filtered samples for each class
        max_samples = {}
        
        for class_name, class_samples in class_data.items():
            print(f"  {class_name}: {len(class_samples)} total samples available")

            filtered_samples = self.filter_samples_by_token_count(class_samples)
            
            # Use ALL filtered samples (no subsampling)
            max_samples[class_name] = filtered_samples
            
            print(f"    Using ALL {len(filtered_samples)} samples after token filtering")

            # Show token statistics for all samples
            token_counts = [
                s['premise_tokens'].shape[0] + s['hypothesis_tokens'].shape[0] 
                for s in filtered_samples
            ]
            print(f"Token count stats: {np.mean(token_counts):.0f} ± {np.std(token_counts):.0f} "
                f"(range: {np.min(token_counts)}-{np.max(token_counts)})")
        
        return max_samples

    
    def validate_separate_model_clustering(self) -> ClusteringResult:
        """
        Main validation function - test point cloud clustering with separate models
        """
        print("\n" + "="*80)
        print("SEPARATE MODEL POINT CLOUD CLUSTERING VALIDATION")
        print("="*80)
        
        # Generate fixed samples
        # fixed_samples = self.generate_fixed_samples_by_class()

        #Generate max samples
        max_samples = self.generate_maximum_samples_by_class()

        # Diagnose a few samples from each class
        # for class_name, samples in fixed_samples.items():
        for class_name, samples in max_samples.items():
            if samples:
                print(f"\n{'='*60}")
                print(f"DIAGNOSING {class_name.upper()} TOPOLOGY")
                print('='*60)
                
                # Diagnose first 2 samples
                for i in range(min(2, len(samples))):
                    sample = samples[i]
                    premise_tokens = sample['premise_tokens'] 
                    hypothesis_tokens = sample['hypothesis_tokens']
                    
                    point_cloud, stats = self.point_cloud_generator.generate_premise_hypothesis_point_cloud(
                        premise_tokens, hypothesis_tokens
                    )
                    
                    self.diagnose_topological_complexity(
                        point_cloud, 
                        f"{class_name}_sample_{i+1}"
                    )
        
        print(f"\n{'='*60}")
        print("DIAGNOSIS COMPLETE")
        print('='*60)
            
        # Initialize collection variables
        all_persistence_diagrams = []
        sample_labels = []
        ph_dim_values = {'entailment': [], 'neutral': [], 'contradiction': []}
        point_cloud_stats = {'entailment': [], 'neutral': [], 'contradiction': []}
        model_analysis = {'entailment': [], 'neutral': [], 'contradiction': []}
        
        # Process each class
        for class_idx, class_name in enumerate(['entailment', 'neutral', 'contradiction']):
            print(f"\nProcessing {class_name} samples...")
            
            # class_samples = fixed_samples[class_name]
            class_samples = max_samples[class_name]

            for sample_idx, sample_data in enumerate(class_samples):
                premise_tokens = sample_data['premise_tokens']
                hypothesis_tokens = sample_data['hypothesis_tokens']
                
                print(f"  Sample {sample_idx+1}: P={premise_tokens.shape[0]} tokens, H={hypothesis_tokens.shape[0]} tokens")
                
                # Generate point cloud using separate models
                point_cloud, stats = self.point_cloud_generator.generate_premise_hypothesis_point_cloud(
                    premise_tokens, hypothesis_tokens
                )
                
                # Analyze model outputs for this sample
                analysis = self.point_cloud_generator.analyze_model_outputs(premise_tokens, hypothesis_tokens)
                model_analysis[class_name].append(analysis)
                
                # Record statistics
                point_cloud_stats[class_name].append(stats)
                
                print(f"    Point cloud breakdown:")
                print(f"      Premise: {stats['premise_original_points']} + {stats['premise_order_points']} + {stats['premise_asymmetric_points']} = {stats['premise_total_points']}")
                print(f"      Hypothesis: {stats['hypothesis_original_points']} + {stats['hypothesis_order_points']} + {stats['hypothesis_asymmetric_points']} = {stats['hypothesis_total_points']}")
                print(f"      Combined: {stats['combined_total_points']} points")
                print(f"      Sufficient for PHD: {'✅' if stats['sufficient_for_phd'] else '❌'}")
                
                print(f"    Model analysis:")
                print(f"      Order violation energy: {analysis['order_model']['order_violation_energy']:.4f}")
                print(f"      Asymmetric energy: {analysis['asymmetry_model']['asymmetric_energy']:.4f}")
                
                # Skip if insufficient points
                if not stats['sufficient_for_phd']:
                    print(f"    ⚠️ Skipping due to insufficient points")
                    continue
                
                # Compute distance matrix
                distance_matrix = self.compute_distance_matrix(point_cloud)
                
                # Compute PHD and persistence diagrams
                ph_dim, diagrams = self.ph_dim_and_diagrams_from_distance_matrix(
                    distance_matrix,
                    min_points=50,  # ← CHANGED: Use same params as debug
                    max_points=min(200, point_cloud.shape[0]),  # ← CHANGED: Same as debug
                    point_jump=25   # ← CHANGED: Same as debug
                )
                
                # Store PH-dimension
                ph_dim_values[class_name].append(ph_dim)
                print(f"    PH-dimension: {ph_dim:.2f}")
                
                # ← FIX: Actually collect the diagrams!
                all_persistence_diagrams.append(diagrams)
                sample_labels.append(class_idx)
                print(f"    ✅ Added diagrams to collection (total: {len(all_persistence_diagrams)})")

        # ← FIX: Move this OUTSIDE the loops
        print(f"\nCollected {len(all_persistence_diagrams)} diagram sets for clustering")

        # Convert all diagrams to persistence images
        persistence_images = self.persistence_diagrams_to_images(all_persistence_diagrams)
                
        # Use persistence images for clustering
        if len(persistence_images) > 0:
            # Expand labels to match number of images generated
            images_per_sample = len(persistence_images) // len(all_persistence_diagrams) if all_persistence_diagrams else 1
            expanded_labels = []
            for label in sample_labels:
                expanded_labels.extend([label] * images_per_sample)
            sample_labels = expanded_labels
        
        print(f"Generated {len(persistence_images)} persistence images for clustering")
        
        # Perform clustering analysis
        if len(persistence_images) > 0:
            print("Performing clustering analysis...")
            accuracy, sil_score, ari_score = self.perform_clustering_analysis(
                persistence_images, sample_labels  # ← FIX: Use correct variable name
            )
        else:
            print("❌ No persistence images generated - cannot perform clustering")
            accuracy, sil_score, ari_score = 0.0, 0.0, 0.0
    
        
        # Calculate comprehensive statistics
        ph_dim_stats = {}
        comprehensive_point_stats = {}
        
        for class_name in ['entailment', 'neutral', 'contradiction']:
            # PH-dimension stats
            ph_dims = ph_dim_values[class_name]
            if ph_dims:
                ph_dim_stats[class_name] = {
                    'mean': float(np.mean(ph_dims)),
                    'std': float(np.std(ph_dims)),
                    'min': float(np.min(ph_dims)),
                    'max': float(np.max(ph_dims))
                }
            
            # Comprehensive point cloud stats
            class_stats = point_cloud_stats[class_name]
            if class_stats:
                combined_points = [s['combined_total_points'] for s in class_stats]
                premise_points = [s['premise_total_points'] for s in class_stats]
                hypothesis_points = [s['hypothesis_total_points'] for s in class_stats]
                
                comprehensive_point_stats[class_name] = {
                    'combined_mean': float(np.mean(combined_points)),
                    'combined_std': float(np.std(combined_points)),
                    'premise_mean': float(np.mean(premise_points)),
                    'hypothesis_mean': float(np.mean(hypothesis_points)),
                    'sufficient_rate': float(np.mean([s['sufficient_for_phd'] for s in class_stats]))
                }
        
        # Create result
        result = ClusteringResult(
            order_model_name="order_embedding_model.pt",
            asymmetry_model_name="new_independent_asymmetry_transform_model_v2.pt",
            clustering_accuracy=accuracy,
            silhouette_score=sil_score,
            adjusted_rand_score=ari_score,
            # num_samples=sum(len(samples) for samples in fixed_samples.values()),
            num_samples=sum(len(samples) for samples in max_samples.values()),
            success=(accuracy > 0.7),
            ph_dim_values=ph_dim_values,
            ph_dim_stats=ph_dim_stats,
            point_cloud_stats=comprehensive_point_stats
        )
        
        # Print comprehensive results
        print(f"\n" + "="*80)
        print("SEPARATE MODEL CLUSTERING RESULTS")
        print("="*80)
        print(f"Clustering Accuracy: {accuracy:.3f}")
        print(f"Silhouette Score: {sil_score:.3f}")
        print(f"Adjusted Rand Index: {ari_score:.3f}")
        print(f"Success (>70%): {'🎉 YES' if result.success else '❌ NO'}")
        
        print(f"\nPH-Dimension Statistics:")
        for class_name, stats in ph_dim_stats.items():
            print(f"  {class_name}: {stats['mean']:.2f} ± {stats['std']:.2f}")
        
        print(f"\nPoint Cloud Statistics:")
        for class_name, stats in comprehensive_point_stats.items():
            print(f"  {class_name}:")
            print(f"    Combined: {stats['combined_mean']:.0f} ± {stats['combined_std']:.0f} points")
            print(f"    Premise: {stats['premise_mean']:.0f}, Hypothesis: {stats['hypothesis_mean']:.0f}")
            print(f"    Sufficient for PHD: {stats['sufficient_rate']*100:.0f}%")
        
        # Analyze model performance patterns
        self.analyze_model_performance_patterns(model_analysis)

        self.debug_neutral_detection()

        self.debug_discover_optimal_three_energy_patterns()
        
        return result


    def perform_clustering_analysis(self, persistence_images: List[np.ndarray], 
                               sample_labels: List[int]) -> Tuple[float, float, float]:
        """
        Perform clustering analysis on persistence images
        Based on the successful methodology from phdim_clustering.py
        """
        
        if len(persistence_images) == 0:
            print("No persistence images available for clustering")
            raise 
        
        print(f"Clustering {len(persistence_images)} persistence images...")
        
        # Convert to numpy array
        X = np.array(persistence_images)
        y_true = np.array(sample_labels)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"True labels distribution: {np.bincount(y_true)}")
        
        # Perform k-means clustering (3 clusters for entailment/neutral/contradiction)
        n_clusters = len(np.unique(y_true))
        print(f"Performing k-means with {n_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10, init='k-means++', max_iter=300, tol=1e-4)
        y_pred = kmeans.fit_predict(X)
        
        print(f"Predicted labels distribution: {np.bincount(y_pred)}")
        
        # Calculate clustering metrics
        # For accuracy, we need to find the best label permutation
        best_accuracy = 0.0
        best_permutation = None
        
        # Try all possible label permutations to find best match
        for perm in permutations(range(n_clusters)):
            # Map predicted labels using this permutation
            mapped_pred = np.array([perm[label] for label in y_pred])
            accuracy = np.mean(mapped_pred == y_true)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_permutation = perm
        
        # Use best permutation for final predicted labels
        final_pred = np.array([best_permutation[label] for label in y_pred])

        # ADD THE FAILURE ANALYSIS HERE
        print("\n" + "="*50)
        print("CLUSTERING FAILURE ANALYSIS")
        print("="*50)
        confusion_analysis = self.analyze_clustering_failures(persistence_images, y_true, final_pred)
        
        # Calculate additional metrics
        try:
            silhouette = silhouette_score(X, final_pred)
        except:
            print("SILHOUETTE CALCULATION FAILED!")
            silhouette = 0.0
        
        try:
            ari = adjusted_rand_score(y_true, final_pred)
        except:
            print("ARI CALCULATION FAILED!")
            ari = 0.0
        
        print(f"Clustering results:")
        print(f"  Best accuracy: {best_accuracy:.3f}")
        print(f"  Silhouette score: {silhouette:.3f}")
        print(f"  Adjusted Rand Index: {ari:.3f}")
        print(f"  Best permutation: {best_permutation}")
        
        # Create confusion matrix analysis
        print(f"\nConfusion analysis:")
        for true_class in range(n_clusters):
            true_mask = y_true == true_class
            pred_for_true = final_pred[true_mask]
            class_names = ['entailment', 'neutral', 'contradiction']
            
            if true_class < len(class_names):
                print(f"  True {class_names[true_class]}: predicted as {np.bincount(pred_for_true, minlength=n_clusters)}")
        
        return best_accuracy, silhouette, ari

    def analyze_model_performance_patterns(self, model_analysis: Dict[str, List[Dict]]):
        """Analyze how well the separate models learned their respective patterns"""
        
        print(f"\n" + "="*60)
        print("SEPARATE MODEL PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Analyze order model patterns
        print("Order Model (Vendrov Max-Margin Loss):")
        order_energies = {}
        for class_name, analyses in model_analysis.items():
            if analyses:
                energies = [a['order_model']['order_violation_energy'] for a in analyses]
                order_energies[class_name] = np.mean(energies)
                print(f"  {class_name}: {np.mean(energies):.4f} ± {np.std(energies):.4f}")
        
        # Check order ranking
        if len(order_energies) == 3:
            entail_energy = order_energies.get('entailment', float('inf'))
            neutral_energy = order_energies.get('neutral', float('inf'))
            contra_energy = order_energies.get('contradiction', float('inf'))
            
            if entail_energy < neutral_energy < contra_energy:
                print("  ✅ ORDER MODEL SUCCESS: Correct energy ranking!")
            else:
                print("  ❌ ORDER MODEL ISSUE: Incorrect energy ranking")
        
        # Analyze asymmetry model patterns
        print("\nAsymmetry Model (Directional Pattern Loss):")
        asymmetry_energies = {}
        for class_name, analyses in model_analysis.items():
            if analyses:
                asym_energies = [a['asymmetry_model']['asymmetric_energy'] for a in analyses]
                forward_energies = [a['combined']['forward_energy'] for a in analyses]
                backward_energies = [a['combined']['backward_energy'] for a in analyses]
                
                asymmetry_energies[class_name] = {
                    'asymmetric': np.mean(asym_energies),
                    'forward': np.mean(forward_energies),
                    'backward': np.mean(backward_energies)
                }
                
                print(f"  {class_name}:")
                print(f"    Asymmetric energy: {np.mean(asym_energies):.4f} ± {np.std(asym_energies):.4f}")
                print(f"    Forward energy: {np.mean(forward_energies):.4f}")
                print(f"    Backward energy: {np.mean(backward_energies):.4f}")
        
        # Check asymmetry patterns
        if len(asymmetry_energies) == 3:
            entail_forward = asymmetry_energies.get('entailment', {}).get('forward', float('inf'))
            contra_forward = asymmetry_energies.get('contradiction', {}).get('forward', float('inf'))
            neutral_asym = asymmetry_energies.get('neutral', {}).get('asymmetric', float('inf'))
            contra_asym = asymmetry_energies.get('contradiction', {}).get('asymmetric', float('inf'))
            
            print(f"\nAsymmetry Pattern Analysis:")
            if entail_forward < contra_forward:
                print("  ✅ Forward energy: Entailment < Contradiction")
            else:
                print("  ❌ Forward energy: Incorrect pattern")
            
            if contra_asym > neutral_asym:
                print("  ✅ Asymmetric energy: Contradiction > Neutral") 
            else:
                print("  ❌ Asymmetric energy: Incorrect pattern")


    def save_comprehensive_results(self, result: ClusteringResult):
        """Save comprehensive clustering results with model analysis"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_data = {
            'timestamp': timestamp,
            'methodology': 'separate_order_and_asymmetry_models',
            'models': {
                'order_model': result.order_model_name,
                'asymmetry_model': result.asymmetry_model_name
            },
            'clustering_performance': {
                'accuracy': float(result.clustering_accuracy),
                'silhouette_score': float(result.silhouette_score),
                'adjusted_rand_score': float(result.adjusted_rand_score),
                'success': bool(result.success)
            },
            'sample_info': {
                'total_samples': int(result.num_samples),
                'samples_per_class': self.samples_per_class
            },
            'ph_dim_analysis': {
                'values': {k: [float(x) for x in v] for k, v in result.ph_dim_values.items()},
                'statistics': {
                    k: {stat_k: float(stat_v) for stat_k, stat_v in stat_dict.items()} 
                    for k, stat_dict in result.ph_dim_stats.items()
                }
            },
            'point_cloud_analysis': {
                k: {stat_k: float(stat_v) for stat_k, stat_v in stat_dict.items()} 
                for k, stat_dict in result.point_cloud_stats.items()
            }
        }
        
        # Save JSON results
        results_file = self.output_dir / f"separate_model_clustering_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save comprehensive summary report
        summary_file = self.output_dir / f"separate_model_clustering_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("SEPARATE MODEL POINT CLOUD CLUSTERING VALIDATION RESULTS\n")
            f.write("="*60 + "\n\n")
            
            f.write("METHODOLOGY:\n")
            f.write("-" * 12 + "\n")
            f.write("1. OrderEmbeddingModel: Trained with Vendrov max-margin loss\n")
            f.write("2. AsymmetryTransformModel: Trained with asymmetric directional loss\n")
            f.write("3. Point cloud generation: SBERT + Order + Asymmetric features\n")
            f.write("4. Clustering: Persistence images → k-means\n\n")
            
            f.write("MODELS USED:\n")
            f.write("-" * 12 + "\n")
            f.write(f"Order Model: {result.order_model_name}\n")
            f.write(f"Asymmetry Model: {result.asymmetry_model_name}\n\n")
            
            f.write("CLUSTERING PERFORMANCE:\n")
            f.write("-" * 22 + "\n")
            f.write(f"Accuracy: {result.clustering_accuracy:.3f}\n")
            f.write(f"Silhouette Score: {result.silhouette_score:.3f}\n") 
            f.write(f"Adjusted Rand Index: {result.adjusted_rand_score:.3f}\n")
            f.write(f"Success (>70%): {'YES' if result.success else 'NO'}\n\n")
            
            f.write("PH-DIMENSION STATISTICS:\n")
            f.write("-" * 24 + "\n")
            for class_name, stats in result.ph_dim_stats.items():
                f.write(f"{class_name}: {stats['mean']:.2f} ± {stats['std']:.2f} "
                       f"(range: {stats['min']:.2f}-{stats['max']:.2f})\n")
            
            f.write("\nPOINT CLOUD STATISTICS:\n")
            f.write("-" * 23 + "\n")
            for class_name, stats in result.point_cloud_stats.items():
                f.write(f"{class_name}:\n")
                f.write(f"  Combined: {stats['combined_mean']:.0f} ± {stats['combined_std']:.0f} points\n")
                f.write(f"  Premise: {stats['premise_mean']:.0f} points\n")
                f.write(f"  Hypothesis: {stats['hypothesis_mean']:.0f} points\n")
                f.write(f"  Sufficient for PHD: {stats['sufficient_rate']*100:.0f}%\n\n")
            
            if result.success:
                f.write("IMPLICATIONS:\n")
                f.write("-" * 12 + "\n")
                f.write("✅ Separate model training approach successful!\n")
                f.write("✅ Order embeddings create topologically distinct point clouds\n")
                f.write("✅ Asymmetric features enhance directional discrimination\n")
                f.write("✅ Individual premise-hypothesis pairs are topologically classifiable\n")
                f.write("✅ Validates extension from global to local topological analysis\n")
            else:
                f.write("NEXT STEPS:\n")
                f.write("-" * 11 + "\n")
                f.write("- Check model training convergence (energy rankings)\n")
                f.write("- Analyze point cloud generation statistics\n")
                f.write("- Consider adjusting model architectures or training parameters\n")
                f.write("- Experiment with different point cloud aggregation strategies\n")
        
        print(f"\nComprehensive results saved to:")
        print(f"  JSON: {results_file}")
        print(f"  Summary: {summary_file}")

    def diagnose_topological_complexity(self, point_cloud: torch.Tensor, sample_name: str = ""):
        """
        Diagnose why point clouds have simple topology (PH-dim = 0)
        """
        point_cloud_np = point_cloud.numpy()
        n_points = point_cloud_np.shape[0]
        
        print(f"\n--- Topological Diagnostic: {sample_name} ---")
        print(f"Points: {n_points}")
        
        # 1. Point spread analysis
        centroid = np.mean(point_cloud_np, axis=0)
        distances_from_centroid = np.linalg.norm(point_cloud_np - centroid, axis=1)
        
        print(f"Point spread:")
        print(f"  Mean distance from centroid: {np.mean(distances_from_centroid):.4f}")
        print(f"  Std distance from centroid: {np.std(distances_from_centroid):.4f}")
        print(f"  Min/Max: {np.min(distances_from_centroid):.4f} / {np.max(distances_from_centroid):.4f}")
        
        # 2. Pairwise distance analysis
        distance_matrix = self.compute_distance_matrix(point_cloud)
        upper_tri = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        
        print(f"Pairwise distances:")
        print(f"  Mean: {np.mean(upper_tri):.4f}")
        print(f"  Std: {np.std(upper_tri):.4f}")
        print(f"  Min/Max: {np.min(upper_tri):.4f} / {np.max(upper_tri):.4f}")
        
        # 3. Check for clustering (why no holes?)
        print(f"Distance distribution:")
        print(f"  <1.0: {np.sum(upper_tri < 1.0) / len(upper_tri) * 100:.1f}%")
        print(f"  1.0-5.0: {np.sum((upper_tri >= 1.0) & (upper_tri < 5.0)) / len(upper_tri) * 100:.1f}%") 
        print(f"  >5.0: {np.sum(upper_tri >= 5.0) / len(upper_tri) * 100:.1f}%")
        
        # 4. Quick persistence check
        try:
            diagrams = ripser_parallel(distance_matrix, maxdim=1, n_threads=1, metric="precomputed")['dgms']
            h1_diagram = diagrams[1]
            finite_h1 = h1_diagram[np.isfinite(h1_diagram[:, 1])]
            
            print(f"Persistence analysis:")
            print(f"  H1 features (total): {len(h1_diagram)}")
            print(f"  H1 features (finite): {len(finite_h1)}")
            if len(finite_h1) > 0:
                lifespans = finite_h1[:, 1] - finite_h1[:, 0]
                print(f"  H1 lifespans: {np.mean(lifespans):.4f} ± {np.std(lifespans):.4f}")
                print(f"  Longest lifespan: {np.max(lifespans):.4f}")
            else:
                print(f"  No finite H1 features → explains PH-dim = 0")
        except Exception as e:
            print(f"  Persistence computation failed: {e}")

    def debug_full_persistence_pipeline(self):
        """Debug the entire persistence pipeline step by step"""
        
        print("\n" + "="*80)
        print("DEBUGGING FULL PERSISTENCE PIPELINE")
        print("="*80)
        
        # Test with just 3 samples (1 per class)
        fixed_samples = self.generate_fixed_samples_by_class()
        
        debug_samples = []
        for class_name in ['entailment', 'neutral', 'contradiction']:
            if fixed_samples[class_name]:
                debug_samples.append((class_name, fixed_samples[class_name][0]))
        
        all_diagrams = []
        
        for class_name, sample_data in debug_samples:
            print(f"\n--- DEBUGGING {class_name.upper()} SAMPLE ---")
            
            premise_tokens = sample_data['premise_tokens']
            hypothesis_tokens = sample_data['hypothesis_tokens']
            
            print(f"1. Token info: P={premise_tokens.shape[0]}, H={hypothesis_tokens.shape[0]}")
            
            # Generate point cloud
            point_cloud, stats = self.point_cloud_generator.generate_premise_hypothesis_point_cloud(
                premise_tokens, hypothesis_tokens
            )
            
            print(f"2. Point cloud: {stats['combined_total_points']} points")
            
            # Compute distance matrix
            distance_matrix = self.compute_distance_matrix(point_cloud)
            print(f"3. Distance matrix: {distance_matrix.shape}")
            print(f"   Distance range: {np.min(distance_matrix):.4f} - {np.max(distance_matrix):.4f}")
            
            # Test direct ripser call (like your diagnostic)
            try:
                print("4. Testing direct ripser call...")
                direct_diagrams = ripser_parallel(distance_matrix, maxdim=1, n_threads=1, metric="precomputed")['dgms']
                print(f"   Direct ripser success: H0={len(direct_diagrams[0])}, H1={len(direct_diagrams[1])}")
                
                # Check for finite features
                h1_diagram = direct_diagrams[1]
                if len(h1_diagram) > 0 and h1_diagram.ndim == 2:
                    finite_h1 = h1_diagram[np.isfinite(h1_diagram).all(axis=1)]
                    print(f"   Finite H1 features: {len(finite_h1)}")
                    if len(finite_h1) > 0:
                        lifespans = finite_h1[:, 1] - finite_h1[:, 0]
                        print(f"   H1 lifespans: {np.min(lifespans):.6f} - {np.max(lifespans):.6f}")
                
                all_diagrams.append(direct_diagrams)
                
            except Exception as e:
                print(f"   ❌ Direct ripser failed: {e}")
                continue
            
            # Test your fixed ph_dim function
            try:
                print("5. Testing fixed ph_dim function...")
                ph_dim, ph_diagrams = self.ph_dim_and_diagrams_from_distance_matrix(
                    distance_matrix, min_points=50, max_points=200, point_jump=25
                )
                print(f"   PH-dim function success: PH-dim={ph_dim:.2f}")
                print(f"   Returned diagrams: H0={len(ph_diagrams[0])}, H1={len(ph_diagrams[1])}")
                
            except Exception as e:
                print(f"   ❌ PH-dim function failed: {e}")
        
        # Test persistence image conversion
        if all_diagrams:
            print(f"\n6. Testing persistence image conversion on {len(all_diagrams)} diagrams...")
            try:
                persistence_images = self.persistence_diagrams_to_images(all_diagrams)
                print(f"   ✅ Generated {len(persistence_images)} persistence images")
                return len(persistence_images) > 0
            except Exception as e:
                print(f"   ❌ Persistence image conversion failed: {e}")
                return False
        else:
            print("6. ❌ No diagrams to convert to images")
            return False


    def analyze_clustering_failures(self, persistence_images, sample_labels, y_pred_final):
        """
        Analyze which samples are being misclassified to guide next optimizations
        """
        misclassified_indices = np.where(sample_labels != y_pred_final)[0]
        
        print(f"Misclassified samples: {len(misclassified_indices)}/{len(sample_labels)}")
        
        # Group misclassifications by type
        confusion_analysis = {}
        class_names = ['entailment', 'neutral', 'contradiction']
        
        for true_class in range(3):
            for pred_class in range(3):
                if true_class != pred_class:
                    mask = (sample_labels == true_class) & (y_pred_final == pred_class)
                    count = np.sum(mask)
                    if count > 0:
                        confusion_analysis[f"{class_names[true_class]}_as_{class_names[pred_class]}"] = count
        
        print("Confusion breakdown:")
        for error_type, count in confusion_analysis.items():
            print(f"  {error_type}: {count}")
        
        return confusion_analysis


    def debug_neutral_detection(self):
        """
        Debug the neutral detection logic to see if it's working correctly
        """
        print("\n" + "="*60)
        print("DEBUGGING NEUTRAL DETECTION LOGIC")
        print("="*60)
        
        # Test on a few samples from each class
        fixed_samples = self.generate_maximum_samples_by_class()
        
        detection_stats = {
            'entailment': {'detected_as_neutral': 0, 'total': 0},
            'neutral': {'detected_as_neutral': 0, 'total': 0},
            'contradiction': {'detected_as_neutral': 0, 'total': 0}
        }
        
        energy_stats = {
            'entailment': {'forward': [], 'backward': [], 'gap': [], 'asymm': []},
            'neutral': {'forward': [], 'backward': [], 'gap': [], 'asymm': []},
            'contradiction': {'forward': [], 'backward': [], 'gap': [], 'asymm': []}
        }
        
        for class_name, samples in fixed_samples.items():
            for i, sample in enumerate(samples):  # Test first 20 of each class
                premise_tokens = sample['premise_tokens']
                hypothesis_tokens = sample['hypothesis_tokens']
                        
                # Get actual energies
                with torch.no_grad():
                    premise_tokens_gpu = premise_tokens.to(self.point_cloud_generator.device)
                    hypothesis_tokens_gpu = hypothesis_tokens.to(self.point_cloud_generator.device)
                    
                    premise_order = self.point_cloud_generator.order_model(premise_tokens_gpu)
                    hypothesis_order = self.point_cloud_generator.order_model(hypothesis_tokens_gpu)
                    
                    forward_energy = self.point_cloud_generator.order_model.order_violation_energy(
                        premise_order.mean(0, keepdim=True), hypothesis_order.mean(0, keepdim=True)
                    ).item()
                    
                    backward_energy = self.point_cloud_generator.order_model.order_violation_energy(
                        hypothesis_order.mean(0, keepdim=True), premise_order.mean(0, keepdim=True)
                    ).item()
                    
                    energy_gap = abs(forward_energy - backward_energy)

                    asymm_energy = self.point_cloud_generator.asymmetry_model.compute_asymmetric_energy(
                        premise_order.mean(0, keepdim=True), hypothesis_order.mean(0, keepdim=True)
                    ).item()
                
                energy_stats[class_name]['forward'].append(forward_energy)
                energy_stats[class_name]['backward'].append(backward_energy)
                energy_stats[class_name]['gap'].append(energy_gap)
                energy_stats[class_name]['asymm'].append(asymm_energy)
        
        # Print energy analysis
        print("\nEnergy Analysis:")
        for class_name, energies in energy_stats.items():
            avg_forward = np.mean(energies['forward'])
            avg_backward = np.mean(energies['backward'])
            avg_gap = np.mean(energies['gap'])
            avg_asymm = np.mean(energies['asymm'])
            
            print("FROM ORDER MODEL")
            print(f"  {class_name}:")
            print(f"    Forward: {avg_forward:.3f}")
            print(f"    Backward: {avg_backward:.3f}")
            print(f"    Gap: {avg_gap:.3f}")
            print(f"    Low forward (<0.6): {np.sum(np.array(energies['forward']) < 0.6)}/{len(energies['forward'])}")
            print(f"    Low backward (<0.8): {np.sum(np.array(energies['backward']) < 0.8)}/{len(energies['backward'])}")
            forward_array = np.array(energies['forward'])
            mid_forward_mask = (forward_array > 0.6) & (forward_array <= 1.4)
            print(f"    Mid forward (0.6< x <=1.4): {np.sum(mid_forward_mask)}/{len(energies['forward'])}")
            backward_array = np.array(energies['backward'])
            gap_array = np.array(energies['gap'])
            low_backward_low_gap_mask = (backward_array < 0.8) & (gap_array <= 0.5)
            print(f"    Low backward and low gap (<0.8 backward, < 0.5 gap): {np.sum(low_backward_low_gap_mask)}/{len(energies['gap'])}")
            print(f"    Symmetric (<0.2): {np.sum(np.array(energies['gap']) < 0.2)}/{len(energies['gap'])}")
            print("FROM ASYMMETRY MODEL")
            print(f"    Asymmetric: {avg_asymm:.3f}")
            print(f"    Asymmetric (<0.4): {np.sum(np.array(energies['asymm']) < 0.4)}/{len(energies['asymm'])}")
            asymm_array = np.array(energies['asymm'])


    def debug_discover_optimal_three_energy_patterns(self):
        """
        Discover optimal patterns using forward, backward, AND asymmetric energies
        """
        fixed_samples = self.generate_maximum_samples_by_class()
        
        all_data = []
        
        # Collect data from all samples
        for class_name, samples in fixed_samples.items():
            print(f"\n=== COLLECTING {class_name.upper()} DATA ===")
            
            for i in range(min(100, len(samples))):
                sample = samples[i]
                premise_tokens = sample['premise_tokens']
                hypothesis_tokens = sample['hypothesis_tokens']

                with torch.no_grad():
                    premise_tokens_gpu = premise_tokens.to(self.point_cloud_generator.device)
                    hypothesis_tokens_gpu = hypothesis_tokens.to(self.point_cloud_generator.device)
                    
                    premise_order = self.point_cloud_generator.order_model(premise_tokens_gpu)
                    hypothesis_order = self.point_cloud_generator.order_model(hypothesis_tokens_gpu)
                    
                    forward_energy = self.point_cloud_generator.order_model.order_violation_energy(
                        premise_order.mean(0, keepdim=True), hypothesis_order.mean(0, keepdim=True)
                        ).item()
                    
                    backward_energy = self.point_cloud_generator.order_model.order_violation_energy(
                        hypothesis_order.mean(0, keepdim=True), premise_order.mean(0, keepdim=True)
                        ).item()
                    
                    gap_energy = abs(forward_energy - backward_energy)

                    asymmetric_energy = self.point_cloud_generator.asymmetry_model.compute_asymmetric_energy(
                        premise_order.mean(0, keepdim=True), hypothesis_order.mean(0, keepdim=True)
                        ).item()
                    
                    all_data.append({
                        'class': class_name,
                        'forward': forward_energy,
                        'backward': backward_energy,
                        'gap': gap_energy,
                        'asymmetric': asymmetric_energy
                    })
        
        # Statistical analysis
        print(f"\n=== COMPREHENSIVE ENERGY STATISTICS ===")
        
        entailment_data = [d for d in all_data if d['class'] == 'entailment']
        neutral_data = [d for d in all_data if d['class'] == 'neutral']
        contradiction_data = [d for d in all_data if d['class'] == 'contradiction']
        
        for energy_type in ['forward', 'backward', 'gap', 'asymmetric']:
            print(f"\n{energy_type.upper()} Energy Statistics:")
            for class_name, class_data in [('entailment', entailment_data), ('neutral', neutral_data), ('contradiction', contradiction_data)]:
                values = [d[energy_type] for d in class_data]
                print(f"{class_name}: mean={np.mean(values):.3f}, std={np.std(values):.3f}, median={np.median(values):.3f}")
                print(f"  25%={np.percentile(values, 25):.3f}, 75%={np.percentile(values, 75):.3f}, range=[{np.min(values):.3f}, {np.max(values):.3f}]")
        
        # Test different combinations of energy boundaries
        print(f"\n=== MULTI-ENERGY BOUNDARY OPTIMIZATION ===")
        
        best_accuracy = 0
        best_params = None
        
        # Simpler grid search with key boundaries
        forward_boundaries = [(0.4, 1.4), (0.5, 1.5), (0.6, 1.6)]
        backward_boundaries = [(0.8, 1.2), (0.9, 1.3), (1.0, 1.4)]
        gap_boundaries = [(0.4, 0.8), (0.5, 0.9), (0.6, 1.0)]
        asymmetric_boundaries = [(0.4, 0.8), (0.5, 0.9), (0.6, 1.0)]
        
        for f_low, f_high in forward_boundaries:
            for b_low, b_high in backward_boundaries:
                for g_low, g_high in gap_boundaries:
                    for a_low, a_high in asymmetric_boundaries:
                        
                        correct_predictions = 0
                        class_correct = {'entailment': 0, 'neutral': 0, 'contradiction': 0}
                        class_total = {'entailment': 0, 'neutral': 0, 'contradiction': 0}
                        
                        for sample in all_data:
                            f = sample['forward']
                            b = sample['backward']
                            g = sample['gap']
                            a = sample['asymmetric']
                            true_class = sample['class']
                            
                            class_total[true_class] += 1
                            
                            # Multi-energy classification logic
                            if f < f_low and g > g_low and a > a_low:  # Entailment
                                predicted = 'entailment'
                            elif f > f_high and b > b_high and a > a_high:  # Contradiction
                                predicted = 'contradiction'
                            elif f_low <= f <= f_high and b < b_high and a < a_low:  # Neutral
                                predicted = 'neutral'
                            else:
                                predicted = 'unknown'
                            
                            if predicted == true_class:
                                correct_predictions += 1
                                class_correct[true_class] += 1
                        
                        accuracy = correct_predictions / len(all_data)
                        
                        # Only consider if all classes have reasonable performance (>30%)
                        neutral_acc = class_correct['neutral'] / class_total['neutral']
                        entail_acc = class_correct['entailment'] / class_total['entailment'] 
                        contra_acc = class_correct['contradiction'] / class_total['contradiction']
                        
                        if accuracy > best_accuracy and min(neutral_acc, entail_acc, contra_acc) > 0.3:
                            best_accuracy = accuracy
                            best_params = {
                                'f_low': f_low, 'f_high': f_high,
                                'b_low': b_low, 'b_high': b_high,
                                'g_low': g_low, 'g_high': g_high,
                                'a_low': a_low, 'a_high': a_high,
                                'accuracy': accuracy,
                                'neutral_acc': neutral_acc,
                                'entail_acc': entail_acc,
                                'contra_acc': contra_acc
                            }
        
        print(f"\n=== OPTIMAL MULTI-ENERGY BOUNDARIES ===")
        if best_params:
            print(f"Best overall accuracy: {best_params['accuracy']:.3f}")
            print(f"Per-class accuracy: Entail={best_params['entail_acc']:.3f}, Neutral={best_params['neutral_acc']:.3f}, Contra={best_params['contra_acc']:.3f}")
            print(f"\nOptimal Rules:")
            print(f"Entailment: forward < {best_params['f_low']} AND gap > {best_params['g_low']} AND asymmetric > {best_params['a_low']}")
            print(f"Neutral: {best_params['f_low']} ≤ forward ≤ {best_params['f_high']} AND backward < {best_params['b_high']} AND asymmetric < {best_params['a_low']}")
            print(f"Contradiction: forward > {best_params['f_high']} AND backward > {best_params['b_high']} AND asymmetric > {best_params['a_high']}")
        
        return best_params, all_data


    def optimize_persistence_image_parameters(self, all_diagrams: List, sample_labels: List[int]) -> Dict:
        """
        Systematic optimization of persistence image parameters
        Tests different combinations to find optimal clustering performance
        """
        
        print("\n" + "="*80)
        print("PERSISTENCE IMAGE PARAMETER OPTIMIZATION")
        print("="*80)
        
        # Calculate data ranges once (same as your original method)
        all_birth_times = []
        all_death_times = []
        all_lifespans = []
        
        for diagrams in all_diagrams:
            if diagrams is None:
                continue
            for dim in range(min(3, len(diagrams))):
                diagram = diagrams[dim]
                if isinstance(diagram, np.ndarray) and diagram.size > 0:
                    if diagram.ndim == 2 and diagram.shape[1] >= 2:
                        finite_mask = np.isfinite(diagram).all(axis=1)
                        finite_diagram = diagram[finite_mask]
                        if len(finite_diagram) > 0:
                            all_birth_times.extend(finite_diagram[:, 0])
                            all_death_times.extend(finite_diagram[:, 1])
                            lifespans = finite_diagram[:, 1] - finite_diagram[:, 0]
                            all_lifespans.extend(lifespans)
        
        if len(all_lifespans) == 0:
            print("No valid features found for optimization")
            return {}
        
        min_birth = np.min(all_birth_times)
        max_birth = np.max(all_birth_times)
        min_life = np.min(all_lifespans)
        max_life = np.max(all_lifespans)
        
        birth_padding = max(0.01, (max_birth - min_birth) * 0.1)
        life_padding = max(0.001, (max_life - min_life) * 0.1)
        
        birth_range = (max(0, min_birth - birth_padding), max_birth + birth_padding)
        pers_range = (max(0.001, min_life - life_padding), max_life + life_padding)
        
        print(f"Data ranges: birth={birth_range}, persistence={pers_range}")
        
        # Define parameter search space
        base_pixel_size = (pers_range[1] - pers_range[0]) / 50
        base_sigma = (pers_range[1] - pers_range[0]) / 30
        
        # Parameter combinations to test
        param_configs = [
            # Current default
            {"name": "current_default", "pixel_divisor": 138.9, "sigma_divisor": 82.6, "resolution": 30},
            
            # Finer resolution
            {"name": "fine_resolution", "pixel_divisor": 100, "sigma_divisor": 60, "resolution": 25},
            {"name": "very_fine", "pixel_divisor": 150, "sigma_divisor": 90, "resolution": 35},
            
            # Coarser resolution  
            {"name": "coarse_resolution", "pixel_divisor": 25, "sigma_divisor": 15, "resolution": 15},
            
            # Different sigma values (smoothing)
            {"name": "sharp_features", "pixel_divisor": 50, "sigma_divisor": 60, "resolution": 20},
            {"name": "smooth_features", "pixel_divisor": 50, "sigma_divisor": 20, "resolution": 20},
            
            # High resolution
            {"name": "high_res", "pixel_divisor": 75, "sigma_divisor": 45, "resolution": 35},
            
            # Literature-inspired (common Persim settings)
            {"name": "literature_std", "pixel_divisor": 40, "sigma_divisor": 25, "resolution": 20},
            {"name": "literature_fine", "pixel_divisor": 80, "sigma_divisor": 50, "resolution": 25},
        ]
        
        results = []
        
        for config in param_configs:
            print(f"\nTesting configuration: {config['name']}")
            
            # Calculate parameters
            pixel_size = max(0.001, (pers_range[1] - pers_range[0]) / config["pixel_divisor"])
            sigma = max(0.001, (pers_range[1] - pers_range[0]) / config["sigma_divisor"])
            resolution = config["resolution"]
            
            print(f"  pixel_size: {pixel_size:.4f}")
            print(f"  sigma: {sigma:.4f}")
            print(f"  resolution: {resolution}x{resolution}")
            
            try:
                # Generate persistence images with these parameters
                persistence_images = self.generate_persistence_images_with_params(
                    all_diagrams, birth_range, pers_range, pixel_size, sigma, resolution
                )
                
                if len(persistence_images) == 0:
                    print(f"  ❌ No images generated")
                    continue
                
                # Test clustering performance
                accuracy, sil_score, ari_score = self.test_clustering_performance(
                    persistence_images, sample_labels
                )
                
                result = {
                    "config_name": config['name'],
                    "pixel_size": pixel_size,
                    "sigma": sigma,
                    "resolution": resolution,
                    "accuracy": accuracy,
                    "silhouette": sil_score,
                    "ari": ari_score,
                    "num_images": len(persistence_images)
                }
                
                results.append(result)
                
                print(f"  Results: Acc={accuracy:.3f}, Sil={sil_score:.3f}, ARI={ari_score:.3f}")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                continue
        
        # Analyze results
        if results:
            print(f"\n" + "="*60)
            print("PARAMETER OPTIMIZATION RESULTS")
            print("="*60)
            
            # Sort by accuracy
            results.sort(key=lambda x: x['accuracy'], reverse=True)
            
            print(f"{'Config':<20} {'Accuracy':<10} {'Silhouette':<12} {'ARI':<8} {'Images':<8}")
            print("-" * 70)
            
            for result in results:
                print(f"{result['config_name']:<20} {result['accuracy']:<10.3f} "
                    f"{result['silhouette']:<12.3f} {result['ari']:<8.3f} {result['num_images']:<8}")
            
            best_result = results[0]
            print(f"\n🏆 BEST CONFIGURATION: {best_result['config_name']}")
            print(f"   Accuracy: {best_result['accuracy']:.3f}")
            print(f"   Pixel size: {best_result['pixel_size']:.4f}")
            print(f"   Sigma: {best_result['sigma']:.4f}")
            print(f"   Resolution: {best_result['resolution']}x{best_result['resolution']}")
            
            # Check for improvement
            current_accuracy = next((r['accuracy'] for r in results if r['config_name'] == 'current_default'), 0)
            improvement = best_result['accuracy'] - current_accuracy
            
            if improvement > 0.01:  # 1% improvement threshold
                print(f"🎉 IMPROVEMENT FOUND: +{improvement:.3f} ({improvement*100:.1f}%)")
            else:
                print(f"📊 Marginal change: {improvement:+.3f}")
        
        return results


    def generate_persistence_images_with_params(self, all_diagrams: List, birth_range: tuple, 
                                            pers_range: tuple, pixel_size: float, 
                                            sigma: float, resolution: int) -> List[np.ndarray]:
        """Generate persistence images with specific parameters"""
        
        pimgr = PersistenceImager(
            pixel_size=pixel_size,
            birth_range=birth_range,
            pers_range=pers_range,
            kernel_params={'sigma': sigma}
        )
        
        persistence_images = []
        
        for diagrams in all_diagrams:
            if diagrams is None:
                continue
            
            combined_image = np.zeros((resolution, resolution))
            has_content = False
            
            # Process H0, H1, H2 diagrams
            for dim in range(min(3, len(diagrams))):
                diagram = diagrams[dim]
                
                if isinstance(diagram, np.ndarray) and diagram.size > 0:
                    if diagram.ndim == 2 and diagram.shape[1] >= 2:
                        finite_mask = np.isfinite(diagram).all(axis=1)
                        finite_diagram = diagram[finite_mask]
                        
                        if len(finite_diagram) > 0:
                            try:
                                img = pimgr.transform([finite_diagram])[0]
                                
                                # Resize to target resolution
                                if img.shape != (resolution, resolution):
                                    from scipy.ndimage import zoom
                                    zoom_factors = (resolution / img.shape[0], resolution / img.shape[1])
                                    img = zoom(img, zoom_factors)
                                
                                combined_image += img
                                has_content = True
                                
                            except Exception as e:
                                continue
            
            if has_content and combined_image.max() > 0:
                combined_image = combined_image / combined_image.max()
                persistence_images.append(combined_image.flatten())
        
        return persistence_images


    def test_clustering_performance(self, persistence_images: List[np.ndarray], 
                                sample_labels: List[int]) -> Tuple[float, float, float]:
        """Test clustering performance with given persistence images"""
        
        if len(persistence_images) == 0:
            return 0.0, 0.0, 0.0
        
        X = np.array(persistence_images)
        y_true = np.array(sample_labels)
        
        # Ensure we have same number of labels as images
        if len(y_true) != len(X):
            # Adjust labels if needed (same logic as main function)
            images_per_sample = len(X) // len(y_true) if len(y_true) > 0 else 1
            expanded_labels = []
            for label in y_true:
                expanded_labels.extend([label] * images_per_sample)
            y_true = np.array(expanded_labels[:len(X)])
        
        n_clusters = len(np.unique(y_true))
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        y_pred = kmeans.fit_predict(X)
        
        # Find best label permutation
        best_accuracy = 0.0
        for perm in permutations(range(n_clusters)):
            mapped_pred = np.array([perm[label] for label in y_pred])
            accuracy = np.mean(mapped_pred == y_true)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_permutation = perm
        
        final_pred = np.array([best_permutation[label] for label in y_pred])
        
        try:
            silhouette = silhouette_score(X, final_pred)
        except:
            silhouette = 0.0
        
        try:
            ari = adjusted_rand_score(y_true, final_pred)
        except:
            ari = 0.0
        
        return best_accuracy, silhouette, ari


    def run_persistence_optimization(self):
        """
        Main function to run persistence image parameter optimization
        Add this to your existing clustering validation workflow
        """
        
        print("Running persistence image parameter optimization...")
        
        # Generate samples (same as existing method)
        # fixed_samples = self.generate_fixed_samples_by_class()
        max_samples = self.generate_maximum_samples_by_class()
        
        # Collect diagrams
        all_persistence_diagrams = []
        sample_labels = []
        
        for class_idx, class_name in enumerate(['entailment', 'neutral', 'contradiction']):
            # class_samples = fixed_samples[class_name]
            class_samples = max_samples[class_name]

            for sample_data in class_samples:
                premise_tokens = sample_data['premise_tokens']
                hypothesis_tokens = sample_data['hypothesis_tokens']
                
                point_cloud, stats = self.point_cloud_generator.generate_premise_hypothesis_point_cloud(
                    premise_tokens, hypothesis_tokens
                )
                
                if not stats['sufficient_for_phd']:
                    continue
                
                distance_matrix = self.compute_distance_matrix(point_cloud)
                ph_dim, diagrams = self.ph_dim_and_diagrams_from_distance_matrix(distance_matrix)
                
                all_persistence_diagrams.append(diagrams)
                sample_labels.append(class_idx)
        
        # Run optimization
        optimization_results = self.optimize_persistence_image_parameters(
            all_persistence_diagrams, sample_labels
        )
        
        return optimization_results




def main():
    """Run separate model point cloud clustering validation"""
    
    # Paths for separate models
    order_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/order_embedding_model_separate_margins.pt"
    asymmetry_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/new_independent_asymmetry_transform_model_v2.pt"
    hyperbolic_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/best_hyperbolic_projector.pt"
    val_data_path = "/vol/bitbucket/ahb24/tda_entailment_new/snli_val_sbert_tokens.pkl"
    output_dir = "MSc_Topology_Codebase/phd_method/clustering_results/"
    os.makedirs(output_dir, exist_ok=True)

    
    # Check if files exist
    if not Path(order_model_path).exists():
        print(f"Order model not found at: {order_model_path}")
        print("Please run separate_model_trainers.py first")
        return
    
    if not Path(asymmetry_model_path).exists():
        print(f"Asymmetry model not found at: {asymmetry_model_path}")
        print("Please run separate_model_trainers.py first")
        return
    
    if not Path(val_data_path).exists():
        print(f"Validation data not found at: {val_data_path}")
        print("Please run sbert_token_extractor.py first")
        return
    
    # Run validation
    validator = SeparateModelClusteringValidator(
        order_model_path=order_model_path,
        asymmetry_model_path=asymmetry_model_path,
        hyperbolic_model_path=hyperbolic_model_path,
        val_data_path=val_data_path,
        output_dir=output_dir,
        seed=42
    )
    
    validator.debug_full_persistence_pipeline()

    result = validator.validate_separate_model_clustering()
    validator.save_comprehensive_results(result)
    
    print("\n" + "="*80)
    print("SEPARATE MODEL CLUSTERING VALIDATION COMPLETED!")
    
    if result.success:
        print("🎉 SUCCESS: Achieved >70% clustering accuracy!")
        print("This validates that:")
        print("  ✅ Separate model training approach works")
        print("  ✅ Order embeddings create hierarchical structure in point clouds")
        print("  ✅ Asymmetry models add directional discrimination")
        print("  ✅ Individual premise-hypothesis pairs are topologically distinct")
        print("  ✅ Token-level processing enables rich point cloud generation")
    else:
        print("❌ Did not achieve 70% clustering threshold")
        print("Analysis points to consider:")
        print("  - Check if models trained to convergence")
        print("  - Verify point cloud generation produces sufficient points")
        print("  - Review energy ranking patterns in training plots")
        print("  - Consider alternative aggregation strategies")
    
    print(f"\nDetailed analysis available in: {output_dir}")
    print("="*80)

    # optimization_results = validator.run_persistence_optimization()


# def test_sbert_clustering_comparison():
#     """
#     Test clustering using raw SBERT embeddings for comparison with topological approach
#     """
#     print("=" * 80)
#     print("SBERT CLUSTERING COMPARISON")
#     print("=" * 80)

#     val_data_path = "/vol/bitbucket/ahb24/tda_entailment_new/mnli_val_mismatched_sbert_tokens.pkl"

    
#     # Load same data as topological clustering
#     validator = SeparateModelClusteringValidator(
#         val_data_path=val_data_path,
#         order_model_path=None,  # Won't be used
#         asymmetry_model_path=None, 
#         hyperbolic_model_path=None,
#         seed=42
#     )
    
#     # Generate samples (using existing filtering)
#     max_samples = validator.generate_maximum_samples_by_class()
    
#     print("Extracting SBERT embeddings for clustering...")
#     all_embeddings = []
#     sample_labels = []
    
#     for class_idx, class_name in enumerate(['entailment', 'neutral', 'contradiction']):
#         class_samples = max_samples[class_name]
#         print(f"Processing {class_name}: {len(class_samples)} samples")
        
#         for sample_data in class_samples:
#             premise_tokens = sample_data['premise_tokens']
#             hypothesis_tokens = sample_data['hypothesis_tokens']
            
#             # Create embedding representation
#             premise_mean = torch.mean(premise_tokens, dim=0)  # [768]
#             hypothesis_mean = torch.mean(hypothesis_tokens, dim=0)  # [768]
            
#             # Different strategies for combining premise/hypothesis:
            
#             # Strategy 1: Concatenation
#             # combined_embedding = torch.cat([premise_mean, hypothesis_mean])  # [1536]
            
#             # Strategy 2: Difference + Product (richer representation)  
#             combined_embedding = torch.cat([
#                 premise_mean, 
#                 hypothesis_mean,
#                 premise_mean - hypothesis_mean,
#                 premise_mean * hypothesis_mean
#             ])  # [3072]
            
#             all_embeddings.append(combined_embedding.numpy())
#             sample_labels.append(class_idx)
    
#     # Convert to clustering format
#     X_embeddings = np.array(all_embeddings)
#     y_true = np.array(sample_labels)
    
#     print(f"SBERT clustering data: {X_embeddings.shape[0]} samples, {X_embeddings.shape[1]} dimensions")
    
#     n_clusters = 3
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#     y_pred = kmeans.fit_predict(X_embeddings)
    
#     # Find best label permutation
#     best_accuracy = 0.0
#     for perm in permutations(range(n_clusters)):
#         mapped_pred = np.array([perm[label] for label in y_pred])
#         accuracy = np.mean(mapped_pred == y_true)
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_permutation = perm
    
#     final_pred = np.array([best_permutation[label] for label in y_pred])
    
#     # Calculate metrics
#     try:
#         silhouette = silhouette_score(X_embeddings, final_pred)
#         ari = adjusted_rand_score(y_true, final_pred)
#     except:
#         silhouette, ari = 0.0, 0.0
    
#     print(f"\n" + "=" * 60)
#     print("SBERT CLUSTERING RESULTS")
#     print("=" * 60)
#     print(f"SBERT Clustering Accuracy: {best_accuracy:.3f}")
#     print(f"Silhouette Score: {silhouette:.3f}")
#     print(f"Adjusted Rand Index: {ari:.3f}")
    
#     # Compare with topological clustering
#     print(f"\nComparison:")
#     print(f"Topological Clustering: 70.2%")
#     print(f"SBERT Clustering: {best_accuracy:.1%}")
#     improvement = best_accuracy - 0.702
#     print(f"Difference: {improvement:+.3f}")

#     # Print detailed confusion matrix analysis
#     print(f"\nDetailed confusion analysis:")
#     class_names = ['entailment', 'neutral', 'contradiction']
    
#     for true_class in range(n_clusters):
#         true_mask = y_true == true_class
#         pred_for_true = final_pred[true_mask]
#         pred_counts = np.bincount(pred_for_true, minlength=n_clusters)
        
#         print(f"  True {class_names[true_class]} ({np.sum(true_mask)} samples):")
#         for pred_class in range(n_clusters):
#             count = pred_counts[pred_class]
#             percentage = (count / np.sum(true_mask)) * 100 if np.sum(true_mask) > 0 else 0
#             print(f"    → predicted as {class_names[pred_class]}: {count} ({percentage:.1f}%)")
    
#     # Additional detailed breakdown
#     print(f"\nPer-class accuracy breakdown:")
#     for class_idx, class_name in enumerate(class_names):
#         true_mask = y_true == class_idx
#         correct_pred = final_pred[true_mask] == class_idx
#         class_accuracy = np.mean(correct_pred) if np.sum(true_mask) > 0 else 0
#         print(f"  {class_name}: {class_accuracy:.3f} ({np.sum(correct_pred)}/{np.sum(true_mask)})")
    
#     # Misclassification analysis
#     misclassified_indices = np.where(y_true != final_pred)[0]
#     print(f"\nMisclassification analysis:")
#     print(f"  Total misclassified: {len(misclassified_indices)}/{len(y_true)} ({len(misclassified_indices)/len(y_true)*100:.1f}%)")
    
#     # Group misclassifications by type
#     confusion_breakdown = {}
#     for true_class in range(3):
#         for pred_class in range(3):
#             if true_class != pred_class:
#                 mask = (y_true == true_class) & (final_pred == pred_class)
#                 count = np.sum(mask)
#                 if count > 0:
#                     confusion_breakdown[f"{class_names[true_class]}_as_{class_names[pred_class]}"] = count
    
#     print("  Confusion breakdown:")
#     for error_type, count in confusion_breakdown.items():
#         print(f"    {error_type}: {count}")
    
#     # Compare with topological clustering
#     print(f"\n" + "=" * 60)
#     print("COMPARISON WITH TOPOLOGICAL CLUSTERING")
#     print("=" * 60)
#     print(f"Topological Clustering: 70.2%")
#     print(f"SBERT Clustering: {best_accuracy:.1%}")
#     improvement = best_accuracy - 0.702
#     print(f"Difference: {improvement:+.3f} ({improvement*100:+.1f} percentage points)")
    
#     if best_accuracy > 0.702:
#         print("📊 SBERT embeddings cluster better than topological features")
#     else:
#         print("🔬 Topological features provide better clustering structure")
    
#     return {
#         'sbert_clustering_accuracy': best_accuracy,
#         'sbert_silhouette': silhouette,
#         'sbert_ari': ari,
#         'vs_topological_improvement': improvement
#     }

def test_sbert_clustering_comparison():
    """
    Test clustering using SBERT sentence embeddings with InferSent concatenation strategy
    """
    print("=" * 80)
    print("SBERT + INFERSENT CONCATENATION CLUSTERING COMPARISON")
    print("=" * 80)

    # Load SBERT sentence embeddings
    sbert_data_path = "MSc_Topology_Codebase/data/processed/mnli_full_SBERT_validation_mismatched.pt"
    sbert_data = torch.load(sbert_data_path)
    
    print(f"Loaded SBERT data with keys: {sbert_data.keys()}")
    
    # Extract embeddings and labels
    premise_embeddings = sbert_data['premise_embeddings']  # [N, 768]
    hypothesis_embeddings = sbert_data['hypothesis_embeddings']  # [N, 768]
    labels = sbert_data['labels']  # [N]
    
    print(f"Premise embeddings shape: {premise_embeddings.shape}")
    print(f"Hypothesis embeddings shape: {hypothesis_embeddings.shape}")
    print(f"Labels shape: {len(labels) if isinstance(labels, list) else labels.shape}")
    
    # Apply InferSent's concatenation strategy
    print("Applying InferSent concatenation strategy...")
    all_embeddings = []
    sample_labels = []
    
    # Convert labels to indices if they're strings
    label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    
    for i in range(len(labels)):
        premise_emb = premise_embeddings[i]  # [768]
        hypothesis_emb = hypothesis_embeddings[i]  # [768]
        
        # InferSent concatenation strategy: [u, v, |u-v|, u*v]
        combined_embedding = torch.cat([
            premise_emb,                           # [768]
            hypothesis_emb,                        # [768]
            torch.abs(premise_emb - hypothesis_emb),  # [768] - absolute difference
            premise_emb * hypothesis_emb           # [768] - element-wise product
        ])  # [3072]
        
        all_embeddings.append(combined_embedding.numpy())
        
        # Convert label to index
        if isinstance(labels[i], str):
            sample_labels.append(label_to_idx[labels[i]])
        else:
            sample_labels.append(labels[i])
    
    # Convert to clustering format
    X_embeddings = np.array(all_embeddings)
    y_true = np.array(sample_labels)
    
    print(f"InferSent-style embeddings: {X_embeddings.shape[0]} samples, {X_embeddings.shape[1]} dimensions")
    
    # Rest of clustering code remains the same...
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X_embeddings)
    
    # Find best label permutation
    best_accuracy = 0.0
    for perm in permutations(range(n_clusters)):
        mapped_pred = np.array([perm[label] for label in y_pred])
        accuracy = np.mean(mapped_pred == y_true)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_permutation = perm
    
    final_pred = np.array([best_permutation[label] for label in y_pred])
    
    # Calculate metrics
    try:
        silhouette = silhouette_score(X_embeddings, final_pred)
        ari = adjusted_rand_score(y_true, final_pred)
    except:
        silhouette, ari = 0.0, 0.0
    
    print(f"\n" + "=" * 60)
    print("SBERT + INFERSENT CONCATENATION RESULTS")
    print("=" * 60)
    print(f"Clustering Accuracy: {best_accuracy:.3f}")
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Adjusted Rand Index: {ari:.3f}")
    
    # Compare with topological clustering
    print(f"\n" + "=" * 60)
    print("COMPARISON WITH TOPOLOGICAL CLUSTERING")
    print("=" * 60)
    print(f"Topological Clustering: 70.2%")
    print(f"SBERT + InferSent Strategy: {best_accuracy:.1%}")
    improvement = best_accuracy - 0.702
    print(f"Difference: {improvement:+.3f} ({improvement*100:+.1f} percentage points)")
    
    return {
        'sbert_infersent_accuracy': best_accuracy,
        'sbert_infersent_silhouette': silhouette,
        'sbert_infersent_ari': ari,
        'vs_topological_improvement': improvement
    }



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--sbert_baseline":
        test_sbert_clustering_comparison()
    else:
        main()

        