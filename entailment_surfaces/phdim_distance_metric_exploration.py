"""
Step 1.1: Optimal Distance Metric Discovery for Entailment Surface Learning
Works with pre-processed outputs from text_processing.py and order_embeddings_asymmetry.py

Prerequisites:
1. Run src/text_processing.py on 500k dataset to get BERT embeddings
2. Run src/order_embeddings_asymmetry.py to train order embeddings 
3. This script loads those outputs and tests distance metrics for surface learning

Usage:
    python step_1_1_surface_analysis_with_preprocessed.py \
        --bert_data path/to/processed_bert_data.pt \
        --order_model path/to/trained_order_model.pt \
        --results_dir results/step_1_1_analysis
"""


import torch
import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import pdist, squareform, mahalanobis
from scipy.stats import multivariate_normal
import pickle
import time
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phd_method.src_phd.topology import ph_dim_from_distance_matrix, fast_ripser, calculate_ph_dim

from src.order_embeddings_asymmetry import OrderEmbeddingModel

class SurfaceDistanceMetricAnalyzer:
    """
    Step 1.1: Surface Distance Metric Discovery using pre-processed embeddings
    
    Tests comprehensive set of distance metrics on different embedding spaces
    to find optimal metrics for entailment surface learning.
    """
    
    def __init__(self, 
                 bert_data_path: str,
                 order_model_path: str,
                 results_dir: str = 'entailment_surfaces/results/surface_analysis',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 seed: int = 42):
        """
        Initialize analyzer with pre-processed data paths
        
        Args:
            bert_data_path: Path to processed BERT embeddings (.pt file from text_processing.py)
            order_model_path: Path to trained order embedding model (.pt file from order_embeddings_asymmetry.py)
            results_dir: Directory to save analysis results
            device: Computing device
            seed: Random seed
        """
        self.bert_data_path = bert_data_path
        self.order_model_path = order_model_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = seed
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Comprehensive distance metrics for surface learning
        self.distance_metrics = [
            # Standard metrics
            'euclidean',        # L2 norm
            'manhattan',        # L1 norm  
            'chebyshev',        # L∞ norm
            'cosine',          # Cosine distance
            
            # Minkowski family
            'minkowski_3',     # L3 norm
            'minkowski_4',     # L4 norm
            
            # Other metrics
            'canberra',        # Weighted Manhattan
            'braycurtis',      # Normalized Manhattan
            
            # Custom metrics (implemented separately)
            'hyperbolic',      # Hyperbolic distance (if in hyperbolic space)
            'order_violation', # Order embedding violation energy
        ]

         # Embedding spaces for surface analysis (CORRECTED - only relational spaces)
        self.embedding_spaces = [
            'bert_concat',          # Concatenated [premise||hypothesis] - joint representation
            'bert_difference',      # Premise - Hypothesis (relationship vector)
            'order_concat',         # Concatenated order embeddings [order_premise||order_hypothesis]
            'order_difference',     # Order premise - hypothesis (order relationship)
            'order_violations',     # Order violation energies (inherently relational)
            'hyperbolic_concat',    # Concatenated hyperbolic embeddings
            'hyperbolic_distances', # Direct hyperbolic distances between P-H pairs (1D)
            'cone_energies',        # Entailment cone violation energies (inherently relational)
            'cone_features',        # Multiple cone-related features
        ]

        # PH-Dim parameters
        self.phd_params = {
            'min_points': 200,
            'max_points': 1000,
            'point_jump': 50,
            'h_dim': 0,
            'alpha': 1.0,
            'seed': seed
        }

        print(f"Surface Distance Metric Analyzer initialized")
        print(f"FOCUS: Testing distance metrics for premise-hypothesis pair relationships")
        print(f"BERT data: {bert_data_path}")
        print(f"Order model: {order_model_path}")
        print(f"Distance metrics: {len(self.distance_metrics)} total")
        print(f"Relational embedding spaces: {len(self.embedding_spaces)} total")
        print(f"Each space represents premise-hypothesis pair relationships")

        # Load data
        self._load_preprocessed_data()

    
    def _load_preprocessed_data(self):
        """Load pre-processed BERT embeddings and order model"""
        print("Loading pre-processed data...")

        # Load BERT embeddings
        if not os.path.exists(self.bert_data_path):
            raise FileNotFoundError(f"BERT data not found: {self.bert_data_path}")

        print(f"Loading BERT embeddings from {self.bert_data_path}")
        self.bert_data = torch.load(self.bert_data_path, map_location=self.device)

        print(f"BERT data loaded:")
        print(f"  Premise embeddings: {self.bert_data['premise_embeddings'].shape}")
        print(f"  Hypothesis embeddings: {self.bert_data['hypothesis_embeddings'].shape}")
        print(f"  Labels: {len(self.bert_data['labels'])}")
        print(f"  Label distribution: {self.bert_data['metadata']['label_counts']}")
        
        # Load order embedding model
        if not os.path.exists(self.order_model_path):
            raise FileNotFoundError(f"Order model not found: {self.order_model_path}")
        
        print(f"Loading order model from {self.order_model_path}")
        checkpoint = torch.load(self.order_model_path, map_location=self.device)

        model_config = checkpoint['model_config']
        self.order_model = OrderEmbeddingModel(
            bert_dim=model_config['bert_dim'],
            order_dim=model_config['order_dim'],
            asymmetry_weight=model_config.get('asymmetry_weight', 0.2)
        )
        self.order_model.load_state_dict(checkpoint['model_state_dict'])
        self.order_model.to(self.device)
        self.order_model.eval()
        
        print(f"Order model loaded (validation loss: {checkpoint.get('best_val_loss', 'N/A')})")

        # Initialize hyperbolic and cone pipeline

        try:
            from src.hyperbolic_projection_asymmetry import HyperbolicOrderEmbeddingPipeline
            self.hyperbolic_pipeline = HyperbolicOrderEmbeddingPipeline(self.order_model_path)
            print("Hyperbolic pipeline loaded successfully")
        except Exception as e:
            print(f"Could not load hyperbolic pipeline: {e}")
            raise
       
        try:
            from src.entailment_cones_asymmetry import EnhancedHyperbolicConeEmbeddingPipeline
            self.cone_pipeline = EnhancedHyperbolicConeEmbeddingPipeline(self.order_model_path)
            print("Enhanced cone pipeline loaded successfully")
        except Exception as e:
            print(f"Could not load cone pipeline: {e}")
            raise        


    def extract_all_embedding_spaces(self, max_samples_per_class: int = None) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract all embedding spaces from pre-processed data
        
        Args:
            max_samples_per_class: Limit samples per class (None for all)
            
        Returns:
            Dict mapping space names to class embeddings
        """
        print("Extracting all embedding spaces...")

        # Organize data by class
        premise_embs = self.bert_data['premise_embeddings']
        hypothesis_embs = self.bert_data['hypothesis_embeddings']
        labels = self.bert_data['labels']

        # Group by class
        data_by_class = {'entailment': {}, 'neutral': {}, 'contradiction': {}}

        for label in data_by_class.keys():
            mask = torch.tensor([l == label for l in labels])
            indices = torch.where(mask)[0]
            
            if max_samples_per_class and len(indices) > max_samples_per_class:
                # Random sampling
                perm = torch.randperm(len(indices))[:max_samples_per_class]
                indices = indices[perm]
            
            data_by_class[label] = {
                'premise_bert': premise_embs[indices],
                'hypothesis_bert': hypothesis_embs[indices],
                'indices': indices
            }
            
            print(f"  {label}: {len(indices)} samples")

        # STEP 1: Generate EUCLIDEAN order embeddings (before hyperbolic projection)
        print("Computing EUCLIDEAN order embeddings...")
        with torch.no_grad():
            for label in data_by_class.keys():
                premise_bert = data_by_class[label]['premise_bert']
                hypothesis_bert = data_by_class[label]['hypothesis_bert']
                
                # Compute Euclidean order embeddings using the order model
                premise_order_euclidean = self.order_model(premise_bert)
                hypothesis_order_euclidean = self.order_model(hypothesis_bert)
                
                # Compute Euclidean order violation energies
                euclidean_order_violations = self.order_model.order_violation_energy(
                    premise_order_euclidean, hypothesis_order_euclidean
                )
                
                data_by_class[label].update({
                    'premise_order_euclidean': premise_order_euclidean,
                    'hypothesis_order_euclidean': hypothesis_order_euclidean,
                    'euclidean_order_violations': euclidean_order_violations
                })

        # STEP 2: Generate HYPERBOLIC features (enhanced cone pipeline as single source of truth)
        if self.cone_pipeline:
            print("Computing ALL HYPERBOLIC features from enhanced cone pipeline...")
            with torch.no_grad():
                for label in data_by_class.keys():
                    premise_bert = data_by_class[label]['premise_bert']
                    hypothesis_bert = data_by_class[label]['hypothesis_bert']
                    
                    try:
                        # ONE call gets everything: hyperbolic order, cone, and geometric features
                        enhanced_results = self.cone_pipeline.compute_enhanced_cone_energies(
                            premise_bert, hypothesis_bert
                        )
                        
                        # Store ALL hyperbolic features from enhanced cone pipeline
                        data_by_class[label].update({
                            # HYPERBOLIC order embeddings (after hyperbolic projection)
                            'premise_order_hyperbolic': enhanced_results['premise_hyperbolic'],
                            'hypothesis_order_hyperbolic': enhanced_results['hypothesis_hyperbolic'],
                            
                            # HYPERBOLIC order energies (computed in hyperbolic space)
                            'hyperbolic_order_energies': enhanced_results['order_energies'],
                            'forward_order_energies': enhanced_results.get('forward_energies', enhanced_results['order_energies']),
                            'backward_order_energies': enhanced_results.get('backward_energies', torch.zeros_like(enhanced_results['order_energies'])),
                            'asymmetric_order_energies': enhanced_results.get('asymmetric_energies', torch.zeros_like(enhanced_results['order_energies'])),
                            
                            # Hyperbolic geometric features
                            'hyperbolic_distances': enhanced_results['hyperbolic_distances'],
                            
                            # Cone energies (computed in hyperbolic space)
                            'cone_energies': enhanced_results['cone_energies'],
                            'forward_cone_energies': enhanced_results.get('forward_cone_energies', enhanced_results['cone_energies']),
                            'backward_cone_energies': enhanced_results.get('backward_cone_energies', torch.zeros_like(enhanced_results['cone_energies'])),
                            'cone_asymmetries': enhanced_results.get('cone_asymmetries', torch.zeros_like(enhanced_results['cone_energies'])),
                            
                            # Combined enhanced features (hyperbolic)
                            'enhanced_cone_features': torch.cat([
                                enhanced_results['cone_energies'].unsqueeze(1),
                                enhanced_results.get('forward_cone_energies', enhanced_results['cone_energies']).unsqueeze(1),
                                enhanced_results.get('backward_cone_energies', torch.zeros_like(enhanced_results['cone_energies'])).unsqueeze(1),
                                enhanced_results['order_energies'].unsqueeze(1),
                                enhanced_results.get('asymmetric_energies', torch.zeros_like(enhanced_results['order_energies'])).unsqueeze(1)
                            ], dim=1)
                        })
                        
                    except Exception as e:
                        print(f"Error computing enhanced cone features for {label}: {e}")
                        raise
        else:
            print("Enhanced cone pipeline not available - hyperbolic features will be skipped")

        
        # Extract all embedding spaces (CORRECTED - only relational embeddings)
        all_embeddings = {}

        for space in self.embedding_spaces:
            print(f"Extracting {space}...")
            space_embeddings = {}
            
            for label in data_by_class.keys():
                if space == 'bert_concat':
                    # Joint representation: [premise||hypothesis]
                    space_embeddings[label] = torch.cat([
                        data_by_class[label]['premise_bert'],
                        data_by_class[label]['hypothesis_bert']
                    ], dim=1)
                    
                elif space == 'bert_difference':
                    # Relationship vector: premise - hypothesis
                    space_embeddings[label] = data_by_class[label]['premise_bert'] - data_by_class[label]['hypothesis_bert']
                    
                elif space == 'euclidean_order_concat':
                    # Concatenated EUCLIDEAN order embeddings
                    if 'premise_order_euclidean' in data_by_class[label]:
                        space_embeddings[label] = torch.cat([
                            data_by_class[label]['premise_order_euclidean'],
                            data_by_class[label]['hypothesis_order_euclidean']
                        ], dim=1)
                    else:
                        print(f"    Euclidean order embeddings not available for {space}")
                        continue
                        
                elif space == 'euclidean_order_difference':
                    # EUCLIDEAN order relationship vector
                    if 'premise_order_euclidean' in data_by_class[label]:
                        space_embeddings[label] = data_by_class[label]['premise_order_euclidean'] - data_by_class[label]['hypothesis_order_euclidean']
                    else:
                        print(f"    Euclidean order embeddings not available for {space}")
                        continue
                        
                elif space == 'euclidean_order_violations':
                    # EUCLIDEAN order violation energies
                    if 'euclidean_order_violations' in data_by_class[label]:
                        space_embeddings[label] = data_by_class[label]['euclidean_order_violations'].unsqueeze(1)
                    else:
                        print(f"    Euclidean order violations not available for {space}")
                        continue
                        
                elif space == 'hyperbolic_order_concat':
                    # Concatenated HYPERBOLIC order embeddings
                    if 'premise_order_hyperbolic' in data_by_class[label] and data_by_class[label]['premise_order_hyperbolic'] is not None:
                        space_embeddings[label] = torch.cat([
                            data_by_class[label]['premise_order_hyperbolic'],
                            data_by_class[label]['hypothesis_order_hyperbolic']
                        ], dim=1)
                    else:
                        print(f"    Hyperbolic order embeddings not available for {space}")
                        continue
                        
                elif space == 'hyperbolic_order_difference':
                    # HYPERBOLIC order relationship vector  
                    if 'premise_order_hyperbolic' in data_by_class[label] and data_by_class[label]['premise_order_hyperbolic'] is not None:
                        space_embeddings[label] = data_by_class[label]['premise_order_hyperbolic'] - data_by_class[label]['hypothesis_order_hyperbolic']
                    else:
                        print(f"    Hyperbolic order embeddings not available for {space}")
                        continue
                        
                elif space == 'hyperbolic_order_energies':
                    # Standard HYPERBOLIC order violation energies
                    if 'hyperbolic_order_energies' in data_by_class[label] and data_by_class[label]['hyperbolic_order_energies'] is not None:
                        space_embeddings[label] = data_by_class[label]['hyperbolic_order_energies'].unsqueeze(1)
                    else:
                        print(f"    Hyperbolic order energies not available for {space}")
                        continue
                        
                elif space == 'forward_order_energies':
                    # Forward HYPERBOLIC order violation energies
                    if 'forward_order_energies' in data_by_class[label] and data_by_class[label]['forward_order_energies'] is not None:
                        space_embeddings[label] = data_by_class[label]['forward_order_energies'].unsqueeze(1)
                    else:
                        print(f"    Forward order energies not available for {space}")
                        continue
                        
                elif space == 'backward_order_energies':
                    # Backward HYPERBOLIC order violation energies
                    if 'backward_order_energies' in data_by_class[label] and data_by_class[label]['backward_order_energies'] is not None:
                        space_embeddings[label] = data_by_class[label]['backward_order_energies'].unsqueeze(1)
                    else:
                        print(f"    Backward order energies not available for {space}")
                        continue
                        
                elif space == 'asymmetric_order_energies':
                    # Asymmetric HYPERBOLIC order measures
                    if 'asymmetric_order_energies' in data_by_class[label] and data_by_class[label]['asymmetric_order_energies'] is not None:
                        space_embeddings[label] = data_by_class[label]['asymmetric_order_energies'].unsqueeze(1)
                    else:
                        print(f"    Asymmetric order energies not available for {space}")
                        continue
                        
                elif space == 'hyperbolic_distances':
                    if 'hyperbolic_distances' in data_by_class[label] and data_by_class[label]['hyperbolic_distances'] is not None:
                        # Direct hyperbolic distances between premise-hypothesis pairs
                        space_embeddings[label] = data_by_class[label]['hyperbolic_distances'].unsqueeze(1)
                    else:
                        print(f"    Hyperbolic distances not available for {space}")
                        continue
                        
                elif space == 'cone_energies':
                    if 'cone_energies' in data_by_class[label] and data_by_class[label]['cone_energies'] is not None:
                        # Standard cone violation energies
                        space_embeddings[label] = data_by_class[label]['cone_energies'].unsqueeze(1)
                    else:
                        print(f"    Cone energies not available for {space}")
                        continue
                        
                elif space == 'forward_cone_energies':
                    if 'forward_cone_energies' in data_by_class[label] and data_by_class[label]['forward_cone_energies'] is not None:
                        # Forward cone violation energies
                        space_embeddings[label] = data_by_class[label]['forward_cone_energies'].unsqueeze(1)
                    else:
                        print(f"    Forward cone energies not available for {space}")
                        continue
                        
                elif space == 'backward_cone_energies':
                    if 'backward_cone_energies' in data_by_class[label] and data_by_class[label]['backward_cone_energies'] is not None:
                        # Backward cone violation energies
                        space_embeddings[label] = data_by_class[label]['backward_cone_energies'].unsqueeze(1)
                    else:
                        print(f"    Backward cone energies not available for {space}")
                        continue
                        
                elif space == 'cone_asymmetries':
                    if 'cone_asymmetries' in data_by_class[label] and data_by_class[label]['cone_asymmetries'] is not None:
                        # Cone asymmetry measures
                        space_embeddings[label] = data_by_class[label]['cone_asymmetries'].unsqueeze(1)
                    else:
                        print(f"    Cone asymmetries not available for {space}")
                        continue
                        
                elif space == 'enhanced_cone_features':
                    if 'enhanced_cone_features' in data_by_class[label] and data_by_class[label]['enhanced_cone_features'] is not None:
                        # Combined enhanced features (multi-dimensional)
                        space_embeddings[label] = data_by_class[label]['enhanced_cone_features']
                    else:
                        print(f"    Enhanced cone features not available for {space}")
                        continue
            
            all_embeddings[space] = space_embeddings

            # Print shapes for available embeddings
            if space_embeddings:  # Only print if we have embeddings for this space
                for label, embs in space_embeddings.items():
                    print(f"  {space} {label}: {embs.shape}")


        return all_embeddings


    def compute_hyperbolic_distance_matrix(self, embeddings_np: np.ndarray) -> np.ndarray:
        """
        Compute hyperbolic distance matrix for embeddings in Poincaré ball
        
        For points in Poincaré ball, hyperbolic distance is:
        d(x,y) = acosh(1 + 2*||x-y||²/((1-||x||²)(1-||y||²)))
        """

        n_samples = embeddings_np.shape[0]
        distances = np.zeros((n_samples, n_samples))
        
        # Ensure embeddings are in unit ball (clamp to avoid numerical issues)
        norms = np.linalg.norm(embeddings_np, axis=1)
        max_norm = 0.99  # Stay away from boundary
        scale_factor = np.where(norms > max_norm, max_norm / norms, 1.0)
        embeddings_scaled = embeddings_np * scale_factor[:, np.newaxis]
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                x, y = embeddings_scaled[i], embeddings_scaled[j]
                
                # Compute norms
                norm_x_sq = np.dot(x, x)
                norm_y_sq = np.dot(y, y)
                
                # Euclidean distance squared
                diff = x - y
                euclidean_dist_sq = np.dot(diff, diff)
                
                # Hyperbolic distance formula
                numerator = 2 * euclidean_dist_sq
                denominator = (1 - norm_x_sq) * (1 - norm_y_sq)
                
                if denominator > 1e-10:  # Avoid division by zero
                    ratio = 1 + numerator / denominator
                    if ratio >= 1:  # Ensure valid input to acosh
                        hyperbolic_dist = np.arccosh(ratio)
                    else:
                        hyperbolic_dist = 0  # Fallback for numerical issues
                else:
                    hyperbolic_dist = np.inf  # Points too close to boundary
                
                distances[i, j] = distances[j, i] = hyperbolic_dist


    def compute_order_violation_distance_matrix(self, embeddings_np: np.ndarray) -> np.ndarray:
        """
        Compute distance matrix based on order violation energies
        
        For order embeddings, distance between two points can be based on
        their violation energy differences
        """
        n_samples = embeddings_np.shape[0]
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                x, y = embeddings_np[i], embeddings_np[j]
                
                # If we have 1D violation energies, use absolute difference
                if embeddings_np.shape[1] == 1:
                    dist = abs(x[0] - y[0])
                else:
                    # For multi-dimensional, use a custom order-aware distance
                    # Compute element-wise max(0, x_i - y_i) for order violation
                    violation_x_to_y = np.maximum(0, x - y)
                    violation_y_to_x = np.maximum(0, y - x)
                    
                    # Distance is sum of violations in both directions
                    dist = np.sum(violation_x_to_y) + np.sum(violation_y_to_x)
                
                distances[i, j] = distances[j, i] = dist
        
        return distances

    
    def compute_distance_matrix_advanced(self, embeddings: torch.Tensor, metric: str) -> np.ndarray:
        """
        Compute distance matrix using advanced metrics
        
        Args:
            embeddings: Embedding tensor [n_samples, embed_dim]
            metric: Distance metric name
            
        Returns:
            Distance matrix [n_samples, n_samples]
        """
        embeddings_np = embeddings.detach().cpu().numpy()
        n_samples = embeddings_np.shape[0]
        
        # Standard sklearn metrics
        sklearn_metrics = [
            'euclidean', 'manhattan', 'chebyshev', 'cosine', 
            'correlation', 'braycurtis', 'canberra']
        
        if metric in sklearn_metrics:
            return pairwise_distances(embeddings_np, metric=metric)
        
        # Minkowski metrics
        elif metric == 'minkowski_3':
            return pairwise_distances(embeddings_np, metric='minkowski', p=3)
        elif metric == 'minkowski_4':
            return pairwise_distances(embeddings_np, metric='minkowski', p=4)

        else:
            raise ValueError(f"Unknown metric: {metric}")


    def compute_topology_analysis(self, embeddings: torch.Tensor, metric: str, 
                                 class_name: str, space_name: str) -> float:
        """
        Compute PH-Dim for surface topology analysis using appropriate topology function
        
        Args:
            embeddings: Class embeddings
            metric: Distance metric
            class_name: Entailment class name
            space_name: Embedding space name
            
        Returns:
            PH-Dim value
        """
        print(f"  Computing PH-Dim for {class_name} in {space_name} using {metric}")
        
        if len(embeddings) < self.phd_params['min_points']:
            print(f"    Warning: Only {len(embeddings)} samples, need ≥{self.phd_params['min_points']}")
            return np.nan
        
        try:
            embeddings_np = embeddings.detach().cpu().numpy()
            
            # Sklearn-supported metrics: use fast_ripser for efficiency
            sklearn_metrics = [
                'euclidean', 'manhattan', 'chebyshev', 'cosine', 
                'correlation', 'braycurtis', 'canberra', 'hamming'
            ]
            
            # Minkowski metrics
            minkowski_metrics = ['minkowski_3', 'minkowski_4']
            
            if metric in sklearn_metrics:
                # Use fast_ripser which supports sklearn metrics directly
                phd = fast_ripser(
                    embeddings_np,
                    min_points=self.phd_params['min_points'],
                    max_points=min(self.phd_params['max_points'], len(embeddings)),
                    point_jump=self.phd_params['point_jump'],
                    h_dim=self.phd_params['h_dim'],
                    alpha=self.phd_params['alpha'],
                    seed=self.phd_params['seed'],
                    metric=metric
                )
                
            elif metric in minkowski_metrics:
                # Handle Minkowski metrics with custom p values
                distance_matrix = self.compute_distance_matrix_advanced(embeddings, metric)
                
                phd = ph_dim_from_distance_matrix(
                    distance_matrix,
                    min_points=self.phd_params['min_points'],
                    max_points=min(self.phd_params['max_points'], len(embeddings)),
                    point_jump=self.phd_params['point_jump'],
                    h_dim=self.phd_params['h_dim'],
                    alpha=self.phd_params['alpha'],
                    seed=self.phd_params['seed']
                )
                
            elif metric in ['hyperbolic', 'order_violation']:
                phd = calculate_ph_dim(
                    embeddings_np,
                    min_points=self.phd_params['min_points'],
                    max_points=min(self.phd_params['max_points'], len(embeddings)),
                    point_jump=self.phd_params['point_jump'],
                    h_dim=self.phd_params['h_dim'],
                    metric=None,  # Let ripser handle raw points
                    alpha=self.phd_params['alpha'],
                    seed=self.phd_params['seed']
                    # Custom distance will be computed internally by ripser as needed
                )
                
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            # Check for valid result
            if np.any(np.isnan([phd])) or np.any(np.isinf([phd])):
                print(f"    Invalid PH-Dim result for {metric}, skipping")
                return np.nan
            
            print(f"    {class_name} PH-Dim: {phd:.4f}")
            return phd
            
        except Exception as e:
            print(f"    Error computing PH-Dim for {class_name}: {e}")
            return np.nan


    def compute_cross_class_surface_analysis(self, embeddings_by_class: Dict[str, torch.Tensor], 
                                           metric: str, space_name: str) -> Dict[str, float]:
        """
        STEP 2: Analyze cross-class distances for surface learning
        
        This measures distances BETWEEN different entailment classes to find
        metrics suitable for learning entailment surfaces.
        
        Args:
            embeddings_by_class: Embeddings organized by entailment class
            metric: Distance metric to test
            space_name: Embedding space name
            
        Returns:
            Dictionary with cross-class distance metrics
        """
        print(f"  Computing cross-class surface analysis for {space_name} using {metric}")
        
        # Check if we have all three classes
        required_classes = {'entailment', 'neutral', 'contradiction'}
        available_classes = set(embeddings_by_class.keys())
        
        if not required_classes.issubset(available_classes):
            print(f"    Missing classes for cross-class analysis: {required_classes - available_classes}")
            return {}
        
        try:
            # Convert to numpy for distance computation
            class_embeddings_np = {}
            for label, embeddings in embeddings_by_class.items():
                class_embeddings_np[label] = embeddings.detach().cpu().numpy()
            
            # 1. CENTROID DISTANCES - How far apart are class centers?
            centroids = {}
            for label, embs in class_embeddings_np.items():
                centroids[label] = np.mean(embs, axis=0)
            
            # Compute pairwise centroid distances using appropriate distance function
            centroid_distances = {}
            for label1 in required_classes:
                for label2 in required_classes:
                    if label1 != label2:
                        # Use our custom distance functions for consistency
                        if metric in ['euclidean', 'manhattan', 'chebyshev', 'cosine', 'correlation', 'braycurtis', 'canberra']:
                            dist = pairwise_distances([centroids[label1]], [centroids[label2]], metric=metric)[0, 0]
                        elif metric.startswith('minkowski'):
                            p_value = int(metric.split('_')[1])
                            dist = pairwise_distances([centroids[label1]], [centroids[label2]], metric='minkowski', p=p_value)[0, 0]
                        elif metric == 'hyperbolic':
                            # Use our hyperbolic distance function for centroids
                            try:
                                centroid_matrix = np.array([centroids[label1], centroids[label2]])
                                hyperbolic_dm = self.compute_hyperbolic_distance_matrix(centroid_matrix)
                                dist = hyperbolic_dm[0, 1]
                            except:
                                raise
                        elif metric == 'order_violation':
                            # Use our order violation distance function for centroids
                            try:
                                centroid_matrix = np.array([centroids[label1], centroids[label2]])
                                order_dm = self.compute_order_violation_distance_matrix(centroid_matrix)
                                dist = order_dm[0, 1]
                            except:
                                raise
                        else:
                            # Fallback to euclidean
                            print("Unknown metric error")
                            exit(1)
                        
                        centroid_distances[f'{label1}_to_{label2}'] = dist
            
            # 2. MINIMUM CROSS-CLASS DISTANCES - Closest points between classes
            min_distances = {}
            max_distances = {}
            
            for label1 in required_classes:
                for label2 in required_classes:
                    if label1 != label2:
                        # Sample subset for efficiency (cross-class distance computation is expensive)
                        sample_size = min(500, len(class_embeddings_np[label1]), len(class_embeddings_np[label2]))
                        
                        if sample_size < 10:
                            continue
                            
                        # Random sampling
                        idx1 = np.random.choice(len(class_embeddings_np[label1]), sample_size, replace=False)
                        idx2 = np.random.choice(len(class_embeddings_np[label2]), sample_size, replace=False)
                        
                        embs1_sample = class_embeddings_np[label1][idx1]
                        embs2_sample = class_embeddings_np[label2][idx2]
                        
                        # Compute cross-class distance matrix using our custom distance functions
                        if metric in ['euclidean', 'manhattan', 'chebyshev', 'cosine', 'correlation', 'braycurtis', 'canberra', 'hamming']:
                            cross_distances = pairwise_distances(embs1_sample, embs2_sample, metric=metric)
                        elif metric.startswith('minkowski'):
                            p_value = int(metric.split('_')[1])
                            cross_distances = pairwise_distances(embs1_sample, embs2_sample, metric='minkowski', p=p_value)
                        elif metric == 'hyperbolic':
                            # Use our hyperbolic distance implementation
                            try:
                                # Create combined matrix and extract cross-distances
                                combined = np.vstack([embs1_sample, embs2_sample])
                                full_dm = self.compute_hyperbolic_distance_matrix(combined)
                                n1 = len(embs1_sample)
                                cross_distances = full_dm[:n1, n1:]
                            except:
                                raise
                        elif metric == 'order_violation':
                            # Use our order violation distance implementation
                            try:
                                combined = np.vstack([embs1_sample, embs2_sample])
                                full_dm = self.compute_order_violation_distance_matrix(combined)
                                n1 = len(embs1_sample)
                                cross_distances = full_dm[:n1, n1:]
                            except:
                                raise
                        else:
                            # Fallback for unknown metrics
                            print("Unknown metric error!!!")
                            exit(1)
                        
                        min_distances[f'{label1}_to_{label2}'] = np.min(cross_distances)
                        max_distances[f'{label1}_to_{label2}'] = np.max(cross_distances)

            # 3. SURFACE SEPARATION QUALITY - Key metrics for surface learning
            
            # Entailment separation score - How well separated is entailment from others?
            entailment_separation = 0.0
            if 'entailment_to_neutral' in centroid_distances and 'entailment_to_contradiction' in centroid_distances:
                # Average distance from entailment to other classes
                avg_entailment_distance = (centroid_distances['entailment_to_neutral'] + 
                                         centroid_distances['entailment_to_contradiction']) / 2
                
                # Compare to within-class spread (approximate)
                entailment_spread = np.std(np.linalg.norm(class_embeddings_np['entailment'] - centroids['entailment'], axis=1))
                
                if entailment_spread > 0:
                    entailment_separation = avg_entailment_distance / entailment_spread

            # Surface gradient score - Is there a clear ordering: entailment < neutral < contradiction?
            surface_gradient = 0.0
            if ('entailment_to_neutral' in centroid_distances and 
                'entailment_to_contradiction' in centroid_distances and
                'neutral_to_contradiction' in centroid_distances):
                
                ent_to_neutral = centroid_distances['entailment_to_neutral']
                ent_to_contradiction = centroid_distances['entailment_to_contradiction']
                neutral_to_contradiction = centroid_distances['neutral_to_contradiction']
                
                # Ideal ordering: ent_to_neutral < ent_to_contradiction
                if ent_to_neutral < ent_to_contradiction:
                    surface_gradient += 1.0
                
                # Bonus if neutral is truly between entailment and contradiction
                if ent_to_neutral < neutral_to_contradiction < ent_to_contradiction:
                    surface_gradient += 0.5
            
            # 4. OVERALL SURFACE LEARNING SCORE
            surface_learning_score = (entailment_separation / 10.0 + surface_gradient) / 2.0
            
            results = {
                # Centroid distances
                'centroid_ent_to_neutral': centroid_distances.get('entailment_to_neutral', 0),
                'centroid_ent_to_contradiction': centroid_distances.get('entailment_to_contradiction', 0),
                'centroid_neutral_to_contradiction': centroid_distances.get('neutral_to_contradiction', 0),
                
                # Minimum cross-class distances
                'min_ent_to_neutral': min_distances.get('entailment_to_neutral', 0),
                'min_ent_to_contradiction': min_distances.get('entailment_to_contradiction', 0),
                
                # Surface learning metrics
                'entailment_separation': entailment_separation,
                'surface_gradient': surface_gradient,
                'surface_learning_score': surface_learning_score
            }
            
            print(f"    Entailment separation: {entailment_separation:.4f}")
            print(f"    Surface gradient: {surface_gradient:.4f}")
            print(f"    Surface learning score: {surface_learning_score:.4f}")
            
            return results
            
        except Exception as e:
            print(f"    Error in cross-class analysis: {e}")
            return {}


    def evaluate_surface_separation_quality(self, phd_scores: Dict[str, float]) -> Dict[str, float]:
        """Evaluate how well this metric separates entailment classes - CORRECTED: Focus on maximum class separation"""
        valid_scores = {k: v for k, v in phd_scores.items() if not np.isnan(v)}
        
        if len(valid_scores) < 2:
            return {
                'surface_quality': 0.0, 
                'separation_ratio': 0.0, 
                'entailment_simplicity': 0.0,
                'class_distinctiveness': 0.0,
                'overall_separation': 0.0
            }
        
        scores = list(valid_scores.values())
        
        # Basic separation metrics
        separation_ratio = max(scores) / min(scores) if min(scores) > 0 else 0.0
        min_max_diff = max(scores) - min(scores)
        std_dev = np.std(scores)
        
        # Entailment simplicity - test if entailment has simpler topology than others
        entailment_simplicity = 0.0
        if 'entailment' in valid_scores:
            entailment_phd = valid_scores['entailment']
            other_phds = [v for k, v in valid_scores.items() if k != 'entailment']
            if other_phds:
                entailment_simplicity = sum(1 for phd in other_phds if entailment_phd < phd) / len(other_phds)
        
        # Class distinctiveness - how well separated are ALL classes from each other
        class_distinctiveness = 0.0
        if len(valid_scores) >= 3:
            class_list = list(valid_scores.keys())
            pairwise_diffs = []
            
            for i in range(len(class_list)):
                for j in range(i+1, len(class_list)):
                    diff = abs(valid_scores[class_list[i]] - valid_scores[class_list[j]])
                    pairwise_diffs.append(diff)
            
            if pairwise_diffs:
                avg_pairwise_diff = np.mean(pairwise_diffs)
                max_possible_diff = max(scores) - min(scores)
                if max_possible_diff > 0:
                    class_distinctiveness = avg_pairwise_diff / max_possible_diff
        elif len(valid_scores) == 2:
            max_possible_diff = max(scores) - min(scores)
            class_distinctiveness = 1.0 if max_possible_diff > 0 else 0.0
        
        # Overall separation quality combines multiple factors
        overall_separation = (
            separation_ratio / 20.0 +      # Ratio component (normalized)
            min_max_diff / 10.0 +          # Absolute difference component  
            std_dev / 5.0 +                # Standard deviation component
            class_distinctiveness          # Pairwise distinctiveness
        ) / 4.0
        
        # Surface quality for backward compatibility
        surface_quality = (
            separation_ratio/20.0 + 
            min_max_diff/10.0 + 
            entailment_simplicity + 
            class_distinctiveness
        ) / 4.0
        
        return {
            'surface_quality': surface_quality,
            'separation_ratio': separation_ratio,
            'entailment_simplicity': entailment_simplicity,
            'class_distinctiveness': class_distinctiveness,
            'overall_separation': overall_separation,
            'min_max_diff': min_max_diff,
            'std_dev': std_dev
        }

    

    def run_comprehensive_analysis(self, max_samples_per_class: int = 20000):
        """Run comprehensive surface distance metric analysis"""
        print("="*80)
        print("COMPREHENSIVE SURFACE DISTANCE METRIC ANALYSIS")
        print("Testing all distance metrics across all embedding spaces")
        print("="*80)
        
        # Extract all embedding spaces
        all_embeddings = self.extract_all_embedding_spaces(max_samples_per_class)
        
        # Results storage
        all_results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Test each embedding space
        for space_name, space_embeddings in all_embeddings.items():
            print(f"\n{'='*60}")
            print(f"TESTING SPACE: {space_name}")
            print(f"{'='*60}")
            
            space_results = {}
            
            # Test each distance metric
            for metric in self.distance_metrics:
                print(f"\n--- Testing {metric} metric ---")
                
                metric_results = {}
                phd_scores = {}
                
                # STEP 1: Compute PH-Dim for each class (within-class topology)
                for class_name, embeddings in space_embeddings.items():
                    phd = self.compute_topology_analysis(
                        embeddings, metric, class_name, space_name
                    )
                    phd_scores[class_name] = phd
                    metric_results[f'phd_{class_name}'] = phd
                
                # Evaluate surface separation quality (from Step 1)
                surface_metrics = self.evaluate_surface_separation_quality(phd_scores)
                metric_results.update(surface_metrics)
                
                # STEP 2: Compute cross-class distances for surface learning
                cross_class_metrics = self.compute_cross_class_surface_analysis(
                    space_embeddings, metric, space_name
                )
                metric_results.update(cross_class_metrics)
                
                space_results[metric] = metric_results
                
                print(f"  STEP 1 - PH-Dim scores: {phd_scores}")
                print(f"  STEP 1 - Surface quality: {surface_metrics['surface_quality']:.4f}")
                print(f"  STEP 2 - Cross-class surface score: {cross_class_metrics.get('surface_learning_score', 0):.4f}")
                
                # Combined score for ranking
                combined_score = (surface_metrics['surface_quality'] + 
                                cross_class_metrics.get('surface_learning_score', 0)) / 2.0
                metric_results['combined_surface_score'] = combined_score
                print(f"  COMBINED surface learning potential: {combined_score:.4f}")
            
            all_results[space_name] = space_results
            
            # Save intermediate results
            results_file = self.results_dir / f"surface_analysis_{space_name}_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(space_results, f, indent=2, default=str)
            print(f"\nSaved {space_name} results to {results_file}")
        
        # Save complete results and generate comprehensive report
        final_results_file = self.results_dir / f"comprehensive_surface_analysis_{timestamp}.json"
        with open(final_results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        self._generate_simple_report(all_results, timestamp)
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE ANALYSIS COMPLETE")
        print(f"Results saved to: {final_results_file}")
        print(f"{'='*80}")
        
        return all_results


    def _generate_simple_report(self, results: Dict, timestamp: str):
        """Generate simple plain output report for analysis"""
        report_file = self.results_dir / f"simple_analysis_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("SURFACE DISTANCE METRIC ANALYSIS - PLAIN RESULTS\n")
            f.write("="*80 + "\n\n")
            
            for space_name, space_results in results.items():
                f.write(f"EMBEDDING SPACE: {space_name}\n")
                f.write("-" * 50 + "\n")
                
                for metric, metric_results in space_results.items():
                    f.write(f"\nMetric: {metric}\n")
                    
                    # PH-Dim results
                    f.write("  PH-Dim Analysis:\n")
                    for key, value in metric_results.items():
                        if key.startswith('phd_'):
                            f.write(f"    {key}: {value}\n")
                    
                    # Surface separation quality
                    f.write("  Surface Separation Quality:\n")
                    separation_keys = ['surface_quality', 'separation_ratio', 'entailment_simplicity', 
                                     'class_distinctiveness', 'overall_separation', 'min_max_diff', 'std_dev']
                    for key in separation_keys:
                        if key in metric_results:
                            f.write(f"    {key}: {metric_results[key]}\n")
                    
                    # Cross-class analysis
                    f.write("  Cross-Class Surface Analysis:\n")
                    cross_class_keys = ['centroid_ent_to_neutral', 'centroid_ent_to_contradiction', 
                                      'centroid_neutral_to_contradiction', 'min_ent_to_neutral', 
                                      'min_ent_to_contradiction', 'entailment_separation', 
                                      'class_separation_quality', 'surface_learning_score']
                    for key in cross_class_keys:
                        if key in metric_results:
                            f.write(f"    {key}: {metric_results[key]}\n")
                    
                    # Combined score
                    if 'combined_surface_score' in metric_results:
                        f.write(f"  Combined Surface Score: {metric_results['combined_surface_score']}\n")
                
                f.write("\n" + "="*80 + "\n\n")
        
        print(f"Simple report saved to: {report_file}")
    

def main():
    """Main execution function"""

    # Initialize analyzer
    analyzer = SurfaceDistanceMetricAnalyzer(
        bert_data_path="data/processed/snli_full_standard_BERT.pt",
        order_model_path="models/enhanced_order_embeddings_snli_full.pt",
        results_dir="entailment_surfaces/results",
        seed=42
    )
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    return results


if __name__ == "__main__":
    main()