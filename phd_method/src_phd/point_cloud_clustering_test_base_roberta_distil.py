"""
Point Cloud Clustering Test (Updated for Separate Models)
Tests clustering accuracy using separate trained order embedding and asymmetry models
"""

import os
import sys
import json
import numpy as np
import torch
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
from order_asymmetry_models import OrderEmbeddingModel, AsymmetryTransformModel
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
    
    def __init__(self, order_model_path: str, asymmetry_model_path: str, hyperbolic_model_path: str, 
        distil_order_path: str, distil_asymmetry_path: str, device: str = None):
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

        #Load hyperbolic projector
        hyperbolic_checkpoint = torch.load(hyperbolic_model_path, map_location=self.device)
        self.hyperbolic_model = TokenLevelHyperbolicProjector()
        self.hyperbolic_model.load_state_dict(hyperbolic_checkpoint['projector_state_dict'])
        self.hyperbolic_model.to(self.device)
        self.hyperbolic_model.eval()

        #distil order
        distil_order_checkpoint = torch.load(distil_order_path, map_location=self.device)
        self.distil_order_model = OrderEmbeddingModel(hidden_size=768)
        self.distil_order_model.load_state_dict(distil_order_checkpoint['model_state_dict'])
        self.distil_order_model.to(self.device)
        self.distil_order_model.eval()

        distil_asymmetry_checkpoint = torch.load(distil_asymmetry_path, map_location=self.device)
        self.distil_asymmetry_model = AsymmetryTransformModel(hidden_size=768)
        self.distil_asymmetry_model.load_state_dict(distil_asymmetry_checkpoint['model_state_dict'])
        self.distil_asymmetry_model.to(self.device)
        self.distil_asymmetry_model.eval()
        
        print(f"Both models loaded on {self.device}")
        print(f"Order model training stats: Best val loss = {order_checkpoint.get('best_val_loss', 'N/A')}")
        print(f"Asymmetry model training stats: Best val loss = {asymmetry_checkpoint.get('best_val_loss', 'N/A')}")
    
    def generate_point_cloud_variations(self, base_tokens: torch.Tensor, distil_tokens: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate point cloud variations using separate models
        
        Args:
            tokens: SBERT token embeddings [num_tokens, 768]
            
        Returns:
            List of point clouds: [original_tokens, order_embeddings, asymmetric_features]
        """
        point_clouds = []
        
        with torch.no_grad():
            base_tokens = base_tokens.to(self.device)
            
            # 1. Original SBERT tokens (semantic baseline)
            point_clouds.append(base_tokens.cpu().clone())
            
            # 2. Order embeddings (hierarchical structure)
            order_embeddings = self.order_model(base_tokens)
            point_clouds.append(order_embeddings.cpu().clone())
            
            # 3. Asymmetric features (directional relationships)
            asymmetric_features = self.asymmetry_model(order_embeddings)  # Takes order embeddings as input
            point_clouds.append(asymmetric_features.cpu().clone())

            # #4. Hyperbolic features            
            hyperbolic_features = self.hyperbolic_model(order_embeddings)
            point_clouds.append(hyperbolic_features.cpu().clone())

            distil_tokens = distil_tokens.to(self.device)

            point_clouds.append(distil_tokens.cpu().clone())
            
            distil_order_embeddings = self.distil_order_model(distil_tokens)
            point_clouds.append(distil_order_embeddings.cpu().clone())

            distil_asymmetric_features = self.distil_asymmetry_model(distil_order_embeddings)
            point_clouds.append(distil_asymmetric_features.cpu().clone())
        
        return point_clouds
    
    def generate_premise_hypothesis_point_cloud(self, premise_tokens: torch.Tensor, hypothesis_tokens: torch.Tensor, 
        distil_premise_tokens: torch.Tensor, distil_hypothesis_tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Generate combined point cloud from premise-hypothesis pair with detailed statistics
        """
        # Generate premise point clouds
        premise_clouds = self.generate_point_cloud_variations(premise_tokens, distil_premise_tokens)
        
        # Generate hypothesis point clouds
        hypothesis_clouds = self.generate_point_cloud_variations(hypothesis_tokens, distil_hypothesis_tokens)

        # #ENERGY WEIGHTED
        # energy_weighted_cloud = self._generate_energy_weighted_features(premise_tokens, hypothesis_tokens)
        # distil_energy_weighted_cloud = self._generate_energy_weighted_features(distil_premise_tokens, distil_hypothesis_tokens)

        # #DIRECTIONAL
        # directional_cloud = self._generate_enhanced_directional_separation(premise_tokens, hypothesis_tokens)
        # distil_directional_cloud = self._generate_enhanced_directional_separation(distil_premise_tokens, distil_hypothesis_tokens)


        #COSINE ONLY HELPS
        # angular_features_cloud = self._generate_normalized_angular_features(premise_tokens, hypothesis_tokens)
        
        # Combine all point clouds
        # all_clouds = premise_clouds + hypothesis_clouds + [energy_weighted_cloud, directional_cloud, distil_energy_weighted_cloud, distil_directional_cloud]

        all_clouds = premise_clouds + hypothesis_clouds
        combined_cloud = torch.cat(all_clouds, dim=0)

        # has_hyperbolic = len(premise_clouds) == 4  # Check if hyperbolic was added
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

        # if has_hyperbolic:
        #     stats['premise_hyperbolic_points'] = premise_clouds[3].shape[0]
        #     stats['hypothesis_hyperbolic_points'] = hypothesis_clouds[3].shape[0]
        
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


    def _generate_energy_weighted_features(self, premise_tokens: torch.Tensor, hypothesis_tokens: torch.Tensor) -> torch.Tensor:
        """
        Generate energy-weighted point cloud leveraging perfect energy rankings
        """
        with torch.no_grad():
            premise_tokens = premise_tokens.to(self.device)
            hypothesis_tokens = hypothesis_tokens.to(self.device)
            
            # Get order embeddings
            premise_order = self.order_model(premise_tokens)
            hypothesis_order = self.order_model(hypothesis_tokens)
            
            energy_weighted_points = []
            
            # Create energy-weighted combinations of premise-hypothesis token pairs
            for i in range(premise_order.shape[0]):
                for j in range(hypothesis_order.shape[0]):
                    # Compute order violation energy for this token pair
                    energy = self.order_model.order_violation_energy(
                        premise_order[i:i+1], hypothesis_order[j:j+1]
                    )
                    
                    # Convert energy to weight (sigmoid for smooth [0,1] range)
                    energy_weight = torch.sigmoid(energy)
                    
                    # Energy-weighted combination: high energy = more hypothesis influence
                    weighted_point = energy_weight * hypothesis_order[j] + (1 - energy_weight) * premise_order[i]
                    energy_weighted_points.append(weighted_point)
            
            return torch.stack(energy_weighted_points).cpu()

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



    def _generate_normalized_angular_features(self, premise_tokens: torch.Tensor, hypothesis_tokens: torch.Tensor) -> torch.Tensor:
        """
        Generate normalized angular features optimized for cosine distance (Only benefit COSINE not Braycurtis)
        """
        with torch.no_grad():
            premise_tokens = premise_tokens.to(self.device)
            hypothesis_tokens = hypothesis_tokens.to(self.device)
            
            # Get order embeddings
            premise_order = self.order_model(premise_tokens)
            hypothesis_order = self.order_model(hypothesis_tokens)
            
            # L2 normalize all embeddings for cosine optimization
            premise_normalized = torch.nn.functional.normalize(premise_order, p=2, dim=-1)
            hypothesis_normalized = torch.nn.functional.normalize(hypothesis_order, p=2, dim=-1)
            
            angular_points = []
            
            # Create angular relationship features between normalized premise/hypothesis tokens
            for i in range(premise_normalized.shape[0]):
                for j in range(hypothesis_normalized.shape[0]):
                    # Cosine similarity between normalized embeddings
                    cosine_sim = torch.cosine_similarity(
                        premise_normalized[i:i+1], hypothesis_normalized[j:j+1], dim=-1
                    )
                    
                    # Angular difference vector (optimized for cosine space)
                    angular_diff = premise_normalized[i] - hypothesis_normalized[j]
                    angular_diff_normalized = torch.nn.functional.normalize(angular_diff.unsqueeze(0), p=2, dim=-1).squeeze(0)
                    
                    # Cosine-weighted angular feature (scales by similarity)
                    cosine_weighted_feature = cosine_sim.item() * angular_diff_normalized
                    
                    angular_points.append(cosine_weighted_feature)
            
            return torch.stack(angular_points).cpu()


class SeparateModelClusteringValidator:
    """
    Validator for point cloud clustering using separate trained models
    """
    
    def __init__(self, 
                 order_model_path: str,
                 asymmetry_model_path: str,
                 hyperbolic_model_path: str,
                 distil_order_path: str,
                 distil_asymmetry_path: str,
                 val_data_path: str,
                 distil_val_path: str,
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

        print(f"Loading DISTIL validation data: {distil_val_path}")
        with open(distil_val_path, 'rb') as f:
            self.distil_val_data = pickle.load(f)
        
        print(f"Loaded {len(self.distil_val_data['labels'])} validation samples")

        assert len(self.val_data['labels']) == len(self.distil_val_data['labels']), "Dataset sizes don't match!"
        
        # Initialize point cloud generator with separate models
        self.point_cloud_generator = SeparateModelPointCloudGenerator(
            order_model_path, asymmetry_model_path, hyperbolic_model_path, distil_order_path, distil_asymmetry_path
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
            
            # Compute persistence diagrams
            sub_diagrams = ripser_parallel(dist_matrix, maxdim=1, n_threads=-1, metric="precomputed")['dgms']
            
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
    
    def persistence_diagrams_to_images(self, all_diagrams: List, track_indices: bool = False):
        """Convert persistence diagrams to standardized images with robust error handling"""
        
        successful_indices = [] if track_indices else None

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
        pixel_size = max(0.005, (pers_range[1] - pers_range[0]) / 50)
        sigma = max(0.005, (pers_range[1] - pers_range[0]) / 30)
        
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

                
            combined_image = np.zeros((20, 20))
            has_content = False
            
            # Process H0 and H1 diagrams with robust handling
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
                                if img.shape != (20, 20):
                                    from scipy.ndimage import zoom
                                    zoom_factors = (20 / img.shape[0], 20 / img.shape[1])
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
                if track_indices:
                    successful_indices.append(diagram_idx)
                successful_conversions += 1
            else:
                print(f"    Diagram {diagram_idx}: No content after processing")
        
        print(f"\nPersistence image conversion results:")
        print(f"  Successful: {successful_conversions}/{len(all_diagrams)}")
        print(f"  Success rate: {successful_conversions/len(all_diagrams)*100:.1f}%" if all_diagrams else "N/A")
        
        if track_indices:
            return persistence_images, successful_indices
        else:
            return persistence_images

    def compute_distance_matrix(self, point_cloud: torch.Tensor, metric: str = 'braycurtis') -> np.ndarray:
        """Compute distance matrix for point cloud"""
        
        point_cloud_np = point_cloud.numpy()
        
        distance_matrix = pairwise_distances(point_cloud_np, metric=metric)
        return distance_matrix


    def filter_samples_by_token_count(self, samples: List[Dict], min_combined_tokens: int = 40) -> List[Dict]:
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

            distil_premise_tokens = sample['distil_premise_tokens'].shape[0]
            distil_hypothesis_tokens = sample['distil_hypothesis_tokens'].shape[0]
            distil_combined = distil_premise_tokens + distil_hypothesis_tokens
            
            if combined_tokens >= min_combined_tokens and distil_combined >= min_combined_tokens:
                filtered_samples.append(sample)
        
        print(f"Token filtering: {len(filtered_samples)}/{len(samples)} samples have ≥{min_combined_tokens} tokens")
        
        return filtered_samples

    def generate_fixed_samples_by_class(self) -> Dict[str, List[Dict]]:
        """Generate fixed sample indices for each class"""
        
        print("Generating fixed sample indices...")
        
        # Organize data by class
        class_data = {'entailment': [], 'neutral': [], 'contradiction': []}
        
        for i, label in enumerate(self.val_data['labels']):
            assert label == self.distil_val_data['labels'][i], f"Label mismatch at index {i}"

            class_data[label].append({
                'index': i,
                'premise_tokens': self.val_data['premise_tokens'][i],
                'hypothesis_tokens': self.val_data['hypothesis_tokens'][i],
                'distil_premise_tokens': self.distil_val_data['premise_tokens'][i],  # NEW
                'distil_hypothesis_tokens': self.distil_val_data['hypothesis_tokens'][i],
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

    def convert_diagrams_with_label_tracking(self, diagram_label_pairs):
        """Convert diagrams to images while tracking successful conversions"""
    
        # Extract diagrams and labels from pairs
        all_diagrams = [pair[0] for pair in diagram_label_pairs]
        all_labels = [pair[1] for pair in diagram_label_pairs]
    
        # Convert diagrams and track which ones succeed
        persistence_images, successful_indices = self.persistence_diagrams_to_images(all_diagrams, track_indices=True)
    
        # Get labels for successful conversions only
        successful_labels = [all_labels[i] for i in successful_indices]
    
        print(f"Label tracking: {len(all_diagrams)} diagrams → {len(persistence_images)} images → {len(successful_labels)} labels")
    
        return persistence_images, successful_labels

    
    def validate_separate_model_clustering(self) -> ClusteringResult:
        """
        Main validation function - test point cloud clustering with separate models
        """
        print("\n" + "="*80)
        print("SEPARATE MODEL POINT CLOUD CLUSTERING VALIDATION")
        print("="*80)
        
        # Generate fixed samples
        fixed_samples = self.generate_fixed_samples_by_class()

        # Diagnose a few samples from each class
        for class_name, samples in fixed_samples.items():
            if samples:
                print(f"\n{'='*60}")
                print(f"DIAGNOSING {class_name.upper()} TOPOLOGY")
                print('='*60)
                
                # Diagnose first 2 samples
                for i in range(min(2, len(samples))):
                    sample = samples[i]
                    premise_tokens = sample['premise_tokens'] 
                    hypothesis_tokens = sample['hypothesis_tokens']
                    distil_premise_tokens = sample['distil_premise_tokens']
                    distil_hypothesis_tokens = sample['distil_hypothesis_tokens']
                    
                    point_cloud, stats = self.point_cloud_generator.generate_premise_hypothesis_point_cloud(
                        premise_tokens, hypothesis_tokens, distil_premise_tokens, distil_hypothesis_tokens
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
            
            class_samples = fixed_samples[class_name]
            
            for sample_idx, sample_data in enumerate(class_samples):
                premise_tokens = sample_data['premise_tokens']
                hypothesis_tokens = sample_data['hypothesis_tokens']
                distil_premise_tokens = sample_data['distil_premise_tokens']  # NEW
                distil_hypothesis_tokens = sample_data['distil_hypothesis_tokens']
                
                print(f"  Sample {sample_idx+1}: P={premise_tokens.shape[0]} tokens, H={hypothesis_tokens.shape[0]} tokens")
                
                # Generate point cloud using separate models
                point_cloud, stats = self.point_cloud_generator.generate_premise_hypothesis_point_cloud(
                    premise_tokens, hypothesis_tokens, distil_premise_tokens, distil_hypothesis_tokens
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
                all_persistence_diagrams.append((diagrams, class_idx))
                print(f"    ✅ Added diagrams to collection (total: {len(all_persistence_diagrams)})")

        # ← FIX: Move this OUTSIDE the loops
        print(f"\nCollected {len(all_persistence_diagrams)} diagram sets for clustering")

        # Convert all diagrams to persistence images
        persistence_images, sample_labels = self.convert_diagrams_with_label_tracking(all_persistence_diagrams)

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
            asymmetry_model_name="asymmetry_transform_model.pt",
            clustering_accuracy=accuracy,
            silhouette_score=sil_score,
            adjusted_rand_score=ari_score,
            num_samples=sum(len(samples) for samples in fixed_samples.values()),
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
        sample_labels = [int(label) for label in sample_labels]
        y_true = np.array(sample_labels, dtype=int)
        
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



def main():
    """Run separate model point cloud clustering validation"""
    
    # Paths for separate models
    order_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/order_embedding_model.pt"
    asymmetry_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/asymmetry_transform_model.pt"
    hyperbolic_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/best_hyperbolic_projector.pt"
    val_data_path="/vol/bitbucket/ahb24/tda_entailment_new/snli_val_sbert_tokens.pkl"
    distil_order_path = "MSc_Topology_Codebase/phd_method/models/separate_models/order_embedding_model_all_distilroberta_v1.pt"
    distil_asymmetry_path = "MSc_Topology_Codebase/phd_method/models/separate_models/asymmetry_transform_model_all_distilroberta_v1.pt"
    distil_val_path = "/vol/bitbucket/ahb24/tda_entailment_new/snli_val_sbert_tokens_all_distilroberta_v1.pkl"
    output_dir = "MSc_Topology_Codebase/phd_method/clustering_results_base_distil/"
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
        distil_order_path=distil_order_path,
        distil_asymmetry_path=distil_asymmetry_path,
        distil_val_path=distil_val_path,
        output_dir=output_dir,
        seed=42
    )
    
    # validator.debug_full_persistence_pipeline()
    # validator.debug_full_persistence_pipeline()

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


def test_separate_model_point_generation():
    """Test point cloud generation with separate models (for debugging)"""
    
    print("="*80)
    print("TESTING SEPARATE MODEL POINT CLOUD GENERATION")
    print("="*80)
    
    # Test with sample data
    order_model_path = "phd_method/models/separate_models/order_embedding_model_all_distilroberta_v1.pt"
    asymmetry_model_path = "phd_method/models/separate_models/asymmetry_transform_model_all_distilroberta_v1.pt"
    
    if not (Path(order_model_path).exists() and Path(asymmetry_model_path).exists()):
        print("Models not found - please train them first")
        return
    
    # Create generator
    generator = SeparateModelPointCloudGenerator(order_model_path, asymmetry_model_path)
    
    # Test with dummy token data
    test_cases = [
        ("entailment", torch.randn(25, 768), torch.randn(20, 768)),
        ("neutral", torch.randn(30, 768), torch.randn(15, 768)),
        ("contradiction", torch.randn(20, 768), torch.randn(25, 768))
    ]
    
    for label, premise_tokens, hypothesis_tokens in test_cases:
        print(f"\n--- Testing {label.upper()} ---")
        print(f"Premise tokens: {premise_tokens.shape}")
        print(f"Hypothesis tokens: {hypothesis_tokens.shape}")
        
        # Generate point cloud
        point_cloud, stats = generator.generate_premise_hypothesis_point_cloud(
            premise_tokens, hypothesis_tokens
        )
        
        # Analyze model outputs
        analysis = generator.analyze_model_outputs(premise_tokens, hypothesis_tokens)
        
        print(f"Point cloud breakdown:")
        print(f"  Premise: {stats['premise_original_points']} + {stats['premise_order_points']} + {stats['premise_asymmetric_points']} = {stats['premise_total_points']}")
        print(f"  Hypothesis: {stats['hypothesis_original_points']} + {stats['hypothesis_order_points']} + {stats['hypothesis_asymmetric_points']} = {stats['hypothesis_total_points']}")
        print(f"  Combined: {stats['combined_total_points']} points")
        print(f"  Sufficient: {'✅' if stats['sufficient_for_phd'] else '❌'}")
        
        print(f"Model analysis:")
        print(f"  Order violation energy: {analysis['order_model']['order_violation_energy']:.4f}")
        print(f"  Asymmetric energy: {analysis['asymmetry_model']['asymmetric_energy']:.4f}")
        print(f"  Forward-backward asymmetry: {analysis['combined']['asymmetry_measure']:.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_separate_model_point_generation()
    else:
        main()

        