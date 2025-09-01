"""
Latent Embeddings Clustering Validation Script

Tests k-means clustering performance on latent embeddings from trained models.
Compares to the original 100% clustering accuracy achieved with SBERT embeddings.
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
import json
from datetime import datetime
import time
from scipy.optimize import linear_sum_assignment
from contrastive_autoencoder_model_global import ContrastiveAutoencoder
from attention_autoencoder_model import AttentionAutoencoder
from data_loader_global import GlobalDataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from persim import PersistenceImager
from gph.python import ripser_parallel
from itertools import permutations
import matplotlib.pyplot as plt

# Add project paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from phd_method.src_phd.topology import ph_dim_from_distance_matrix


def ph_dim_and_diagrams_from_distance_matrix(dm: np.ndarray,
                                            min_points=200,
                                            max_points=1000,
                                            point_jump=50,
                                            h_dim=0,
                                            alpha: float = 1.,
                                            seed: int = 42) -> Tuple[float, List]:
    """
    Modified version of ph_dim_from_distance_matrix that returns BOTH 
    PH-dimension and persistence diagrams
    
    Returns:
        Tuple of (ph_dimension, list_of_persistence_diagrams)
    """
    assert dm.ndim == 2, dm
    assert dm.shape[0] == dm.shape[1], dm.shape

    #np.random.seed(seed) <--- REMOVE: this was causing the global seed state to change unintentionally
    test_n = range(min_points, max_points, point_jump)
    lengths = []
    all_diagrams = []  # Store all persistence diagrams

    for points_number in test_n:
        sample_indices = np.random.choice(dm.shape[0], points_number, replace=False)
        dist_matrix = dm[sample_indices, :][:, sample_indices]

        # Compute persistence diagrams
        diagrams = ripser_parallel(dist_matrix, maxdim=1, n_threads=-1, metric="precomputed")['dgms']
        all_diagrams.append(diagrams)  # Store the diagrams

        # Extract specific dimension for PH-dim calculation
        d = diagrams[h_dim]
        d = d[d[:, 1] < np.inf]
        lengths.append(np.power((d[:, 1] - d[:, 0]), alpha).sum())

    lengths = np.array(lengths)

    # Compute PH dimension
    x = np.log(np.array(list(test_n)))
    y = np.log(lengths)
    N = len(x)
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    
    ph_dimension = alpha / (1 - m)
    
    return ph_dimension, all_diagrams

def persistence_diagrams_to_images(all_diagrams: List) -> List[np.ndarray]:
    """
    Convert persistence diagrams to standardized images
    CORRECTED: Now matches the original implementation exactly
    """
    pimgr = PersistenceImager(
        pixel_size=0.5,
        birth_range=(0, 5),
        pers_range=(0, 5),
        kernel_params={'sigma': 0.3}
    )
    
    persistence_images = []
    
    for diagrams in all_diagrams:
        combined_image = np.zeros((20, 20))  # Fixed size
        
        # Process H0 and H1 diagrams
        for dim in range(min(2, len(diagrams))):  #Change 1 to 2 to include H1 and H0
            diagram = diagrams[dim]
            if len(diagram) > 0:
                # Filter infinite persistence
                finite_diagram = diagram[np.isfinite(diagram).all(axis=1)]
                if len(finite_diagram) > 0:
                    try:
                        if PersistenceImager is not None:
                            img = pimgr.transform([finite_diagram])[0]
                            # Resize to standard size if needed
                            if img.shape != (20, 20):
                                try:
                                    from scipy.ndimage import zoom
                                    zoom_factors = (20 / img.shape[0], 20 / img.shape[1])
                                    img = zoom(img, zoom_factors)
                                except ImportError:
                                    # Fallback: pad or crop to (20, 20)
                                    img = np.resize(img, (20, 20))
                            combined_image += img
                        else:
                            print("Warning something failed")
                    except Exception as e:
                        print(f"Warning: Failed to process diagram for dim {dim}: {e}")
                        continue
        
        # Normalize and flatten
        if combined_image.max() > 0:
            combined_image = combined_image / combined_image.max()
        
        persistence_images.append(combined_image.flatten())
    
    return persistence_images


class LatentPHDimensionClusteringValidator:
    """
    Test k-means clustering performance on PH-dimension values from latent embeddings
    Replicates the original phdim_clustering_validation_best_metrics.py methodology
    """
    
    def __init__(self, 
                 data_config: Dict,
                 n_clusters: int = 3,
                 samples_per_class: int = 10,
                 sample_size: int = 1000,
                 random_state: int = 42,
                 output_dir: str = 'clustering_validation_results'):
        """
        Initialize PH-dimension clustering validator
        
        Args:
            data_config: Configuration for data loading
            n_clusters: Number of clusters for k-means (3 for entailment classes)
            samples_per_class: Number of samples to extract per class (like original validation)
            sample_size: Size of each sample (points per sample)
            random_state: Random seed for reproducibility
            output_dir: Directory to save results
        """
        self.data_config = data_config
        self.n_clusters = n_clusters
        self.samples_per_class = samples_per_class
        self.sample_size = sample_size
        self.random_state = random_state
        self.output_dir = output_dir
        
        # PH-dimension parameters (matching original validation)
        self.ph_params = {
            'min_points': 200,
            'max_points': 1000,
            'point_jump': 50,
            'h_dim': 0,  # H0 dimension
            'alpha': 1.0
        }
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load test data for PH-dimension clustering validation"""
        print("Loading data...")
        
        # Create data loader
        data_loader = GlobalDataLoader(
            train_path=self.data_config['train_path'],
            val_path=self.data_config['val_path'],
            test_path=self.data_config['test_path'],
            embedding_type=self.data_config['embedding_type'],
            sample_size=self.data_config.get('sample_size', 5000)
        )
        
        # Load datasets
        train_dataset, val_dataset, test_dataset = data_loader.load_data()
        
        # Create data loaders
        train_loader, val_loader, test_loader = data_loader.get_dataloaders(
            batch_size=5000,  # Large batch to get full data
            balanced_sampling=False
        )
        
        # Store the data loader for extracting embeddings
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        print(f"Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    def extract_latent_embeddings_by_class(self, model_path: str) -> Dict[str, torch.Tensor]:
        """
        Extract latent embeddings organized by class from a trained model
        
        Args:
            model_path: Path to trained model checkpoint
            
        Returns:
            Dictionary mapping class names to latent embeddings
        """
        print(f"Extracting latent embeddings by class from {model_path}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model
        model = AttentionAutoencoder(
            input_dim=1536,  # Adjust based on your embedding dimension
            latent_dim=75,   # Adjust based on your latent dimension (for others = 75)
            hidden_dims=[1024, 768, 512, 256, 128]  # Adjust based on your architecture (for others = 1024, 768, 512, 256, 128)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # Extract embeddings by class
        class_embeddings = {'entailment': [], 'neutral': [], 'contradiction': []}
        class_names = ['entailment', 'neutral', 'contradiction']
        
        with torch.no_grad():
            # Use validation data
            for batch in self.val_loader:
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Get latent representations
                latent_features = model.encode(embeddings)
                
                # Group by class
                for class_idx, class_name in enumerate(class_names):
                    class_mask = (labels == class_idx)
                    if class_mask.sum() > 0:
                        class_embeddings[class_name].append(latent_features[class_mask].cpu())
        
        # Combine embeddings for each class
        final_embeddings = {}
        for class_name in class_names:
            if class_embeddings[class_name]:
                combined = torch.cat(class_embeddings[class_name], dim=0)
                final_embeddings[class_name] = combined
                print(f"  {class_name}: {combined.shape[0]} samples, dim: {combined.shape[1]}")
            else:
                print(f"  Warning: No {class_name} samples found")
                final_embeddings[class_name] = torch.empty((0, 75))  # Empty tensor
        
        return final_embeddings
    
    def compute_persistence_image_for_sample(self, embeddings: torch.Tensor) -> np.ndarray:
        """
        CORRECTED: Compute COMBINED H0+H1 persistence image for a sample of embeddings
        Following the exact original methodology
        
        Args:
            embeddings: Sample of embeddings [sample_size, latent_dim]
            
        Returns:
            Combined persistence image (H0 + H1) as flattened array
        """
        try:
            # Convert to numpy
            embeddings_np = embeddings.numpy()
            
            # Compute distance matrix
            distance_matrix = pairwise_distances(embeddings_np, metric='euclidean')

            import gc
            gc.collect()
            
            # Get PH-dim AND persistence diagrams (both H0 and H1)
            ph_dim, all_diagrams = ph_dim_and_diagrams_from_distance_matrix(
                distance_matrix,
                min_points=self.ph_params['min_points'],
                max_points=min(self.ph_params['max_points'], len(embeddings)),
                point_jump=self.ph_params['point_jump'],
                h_dim=self.ph_params['h_dim'],
                alpha=self.ph_params['alpha']
            )

            del distance_matrix
            gc.collect()
            
            # Convert to combined persistence image (H0 + H1)
            # CORRECTED: Use the exact same approach as original
            if len(all_diagrams) > 0:
                persistence_images = persistence_diagrams_to_images([all_diagrams[0]])
                if len(persistence_images) > 0:
                    return persistence_images[0]  # Return first (and only) image
                else:
                    print("WARNING ERROR ERROR")
            else:
                print("WARNING ERROR ERROR")
            
        except Exception as e:
            print(f"    Error computing persistence image: {e}")
            return None
    
    def extract_persistence_image_samples(self, class_embeddings: Dict[str, torch.Tensor], run_seed: int) -> Tuple[List[np.ndarray], List[int]]:
        """
        CORRECTED: Extract persistence images from multiple samples of each class
        
        Args:
            class_embeddings: Dictionary mapping class names to embeddings
            run_seed: Random seed for this run
            
        Returns:
            Tuple of (persistence_images, class_labels)
        """
        print(f"  Extracting persistence image samples (seed={run_seed})")
        
        # Set random seed for reproducible sampling
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        
        all_persistence_images = []
        sample_labels = []
        class_names = ['entailment', 'neutral', 'contradiction']
        
        # Extract samples from each class
        for class_idx, class_name in enumerate(class_names):
            embeddings = class_embeddings[class_name]
            
            if len(embeddings) < self.sample_size:
                print(f"    Warning: {class_name} has only {len(embeddings)} samples, need {self.sample_size}")
                continue
            
            print(f"    {class_name}: extracting {self.samples_per_class} samples of size {self.sample_size}")
            
            # Extract multiple samples from this class
            for sample_idx in range(self.samples_per_class):
                # Random sample without replacement
                indices = torch.randperm(len(embeddings))[:self.sample_size]
                sample_embeddings = embeddings[indices]
                
                # Compute combined H0+H1 persistence image for this sample
                persistence_image = self.compute_persistence_image_for_sample(sample_embeddings)
                
                if persistence_image is not None:
                    all_persistence_images.append(persistence_image)
                    sample_labels.append(class_idx)
                    print(f"      Sample {sample_idx+1}: SUCCESS (image shape: {persistence_image.shape})")
                else:
                    print(f"      Sample {sample_idx+1}: FAILED")
        
        print(f"  Extracted {len(all_persistence_images)} valid persistence images")
        return all_persistence_images, sample_labels
    
    def analyze_natural_clustering(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict:
        """
        Analyze how well k-means naturally discovered the true class structure
        WITHOUT post-hoc optimization - tests if clustering naturally works
        """
        # Create confusion matrix to see natural cluster-class correspondence
        n_clusters = len(np.unique(pred_labels))
        n_classes = len(np.unique(true_labels))
        
        confusion = np.zeros((n_classes, n_clusters), dtype=int)
        for true_idx in range(n_classes):
            for pred_idx in range(n_clusters):
                confusion[true_idx, pred_idx] = ((true_labels == true_idx) & (pred_labels == pred_idx)).sum()
        
        # Find the natural dominant class for each cluster
        natural_mapping = {}
        natural_accuracy_per_cluster = {}
        
        for cluster_idx in range(n_clusters):
            # Find which true class dominates this cluster
            dominant_class = np.argmax(confusion[:, cluster_idx])
            cluster_size = confusion[:, cluster_idx].sum()
            dominant_count = confusion[dominant_class, cluster_idx]
            
            natural_mapping[cluster_idx] = int(dominant_class)
            natural_accuracy_per_cluster[cluster_idx] = float(dominant_count / cluster_size) if cluster_size > 0 else 0.0
        
        # Calculate natural clustering accuracy
        natural_correct = 0
        total_samples = len(true_labels)
        
        for sample_idx in range(total_samples):
            true_class = true_labels[sample_idx]
            pred_cluster = pred_labels[sample_idx]
            
            # Check if this cluster's dominant class matches the true class
            if natural_mapping[pred_cluster] == true_class:
                natural_correct += 1
        
        natural_accuracy = natural_correct / total_samples
        
        return {
            'natural_accuracy': float(natural_accuracy),
            'confusion_matrix': confusion.tolist(),
            'natural_mapping': natural_mapping,
            'cluster_purities': natural_accuracy_per_cluster,
            'total_samples': int(total_samples)
        }
    
    def evaluate_persistence_image_clustering(self, class_embeddings: Dict[str, torch.Tensor], n_runs: int = 10) -> Dict:
        """
        CORRECTED: Evaluate k-means clustering performance on COMBINED H0+H1 persistence images
        
        Args:
            class_embeddings: Dictionary mapping class names to latent embeddings
            n_runs: Number of clustering runs for statistical significance
            
        Returns:
            Dictionary containing clustering metrics with statistical analysis
        """
        print(f"\nEvaluating persistence image clustering with statistical significance (n_runs={n_runs})")
        print("Computing COMBINED H0+H1 persistence images and clustering on them...")
        print(f"Each run uses {self.samples_per_class} samples per class ({self.samples_per_class * 3} total images per run)")
        
        # Storage for multiple runs
        all_natural_accuracies = []
        all_adjusted_rand_scores = []
        all_normalized_mutual_infos = []
        all_silhouette_scores = []
        all_image_info_by_run = []
        
        # Run multiple clustering attempts
        for run_idx in range(n_runs):
            print(f"\n  Persistence image clustering run {run_idx+1}/{n_runs}")
            
            # Extract persistence images for this run
            run_seed = self.random_state + run_idx
            persistence_images, labels = self.extract_persistence_image_samples(class_embeddings, run_seed)
            
            if len(persistence_images) < 6:  # Need at least 6 samples for meaningful clustering
                print(f"    Insufficient samples ({len(persistence_images)}), skipping run")
                continue
            
            # Convert to numpy arrays for clustering
            images_np = np.array(persistence_images)  # Shape: [n_samples, image_features]
            labels_np = np.array(labels)
            
            print(f"    Clustering {len(images_np)} persistence images with {images_np.shape[1]} features each")
            
            # Perform k-means clustering on persistence images
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=run_seed,
                n_init=10,
                max_iter=300
            )
            
            cluster_pred = kmeans.fit_predict(images_np)
            
            # Analyze natural clustering performance
            natural_analysis = self.analyze_natural_clustering(labels_np, cluster_pred)
            
            # Calculate additional metrics
            adjusted_rand = adjusted_rand_score(labels_np, cluster_pred)
            normalized_mutual_info = normalized_mutual_info_score(labels_np, cluster_pred)
            
            # Silhouette score (only if we have enough samples and more than 1 cluster)
            if len(np.unique(cluster_pred)) > 1 and len(images_np) > self.n_clusters:
                silhouette_avg = silhouette_score(images_np, cluster_pred)
            else:
                silhouette_avg = np.nan
            
            # Store results
            all_natural_accuracies.append(natural_analysis['natural_accuracy'])
            all_adjusted_rand_scores.append(adjusted_rand)
            all_normalized_mutual_infos.append(normalized_mutual_info)
            all_silhouette_scores.append(silhouette_avg)
            all_image_info_by_run.append({
                'n_images': len(persistence_images),
                'image_shape': images_np.shape,
                'labels': labels,
                'natural_mapping': natural_analysis['natural_mapping']
            })
            
            print(f"    Natural Accuracy: {natural_analysis['natural_accuracy']:.3f}")
            print(f"    Images processed: {len(persistence_images)} (shape: {images_np.shape})")
        
        # Calculate statistical summary
        natural_accuracies = np.array(all_natural_accuracies)
        adjusted_rands = np.array(all_adjusted_rand_scores)
        normalized_mis = np.array(all_normalized_mutual_infos)
        silhouettes = np.array([s for s in all_silhouette_scores if not np.isnan(s)])
        
        # Convert all numpy types to native Python types for JSON serialization
        results = {
            # Mean and std for all metrics
            'natural_clustering_accuracy_mean': float(np.mean(natural_accuracies)),
            'natural_clustering_accuracy_std': float(np.std(natural_accuracies)),
            'natural_clustering_accuracy_min': float(np.min(natural_accuracies)),
            'natural_clustering_accuracy_max': float(np.max(natural_accuracies)),
            'natural_clustering_accuracy_all_runs': [float(x) for x in natural_accuracies],
            
            'adjusted_rand_score_mean': float(np.mean(adjusted_rands)),
            'adjusted_rand_score_std': float(np.std(adjusted_rands)),
            
            'normalized_mutual_info_mean': float(np.mean(normalized_mis)),
            'normalized_mutual_info_std': float(np.std(normalized_mis)),
            
            'silhouette_score_mean': float(np.mean(silhouettes)) if len(silhouettes) > 0 else None,
            'silhouette_score_std': float(np.std(silhouettes)) if len(silhouettes) > 0 else None,
            
            # Statistical significance measures
            'n_runs': int(n_runs),
            'successful_runs': int(len(natural_accuracies)),
            'samples_per_class': int(self.samples_per_class),
            'sample_size': int(self.sample_size),
            'clustering_method': 'Combined H0+H1 persistence images',
            
            # Consistency analysis
            'accuracy_coefficient_of_variation': float(np.std(natural_accuracies) / np.mean(natural_accuracies)) if np.mean(natural_accuracies) > 0 else float('inf'),
            'runs_above_90_percent': int(np.sum(natural_accuracies >= 0.90)),
            'runs_above_95_percent': int(np.sum(natural_accuracies >= 0.95)),
            'runs_above_99_percent': int(np.sum(natural_accuracies >= 0.99)),
            
            # Image analysis
            'image_info_by_run': all_image_info_by_run
        }
        
        # Print statistical results
        print(f"\nStatistical Persistence Image Clustering Results:")
        print(f"  Method: Combined H0+H1 persistence images")
        print(f"  Successful runs: {results['successful_runs']}/{n_runs}")
        print(f"  Natural Clustering Accuracy: {results['natural_clustering_accuracy_mean']:.4f} ± {results['natural_clustering_accuracy_std']:.4f}")
        print(f"  Range: [{results['natural_clustering_accuracy_min']:.4f}, {results['natural_clustering_accuracy_max']:.4f}]")
        print(f"  Coefficient of Variation: {results['accuracy_coefficient_of_variation']:.4f}")
        print(f"  Runs ≥90%: {results['runs_above_90_percent']}/{results['successful_runs']}")
        print(f"  Runs ≥95%: {results['runs_above_95_percent']}/{results['successful_runs']}")
        print(f"  Runs ≥99%: {results['runs_above_99_percent']}/{results['successful_runs']}")
        
        # Assess clustering consistency
        if results['accuracy_coefficient_of_variation'] < 0.05:
            consistency = "EXCELLENT (Very consistent across runs)"
        elif results['accuracy_coefficient_of_variation'] < 0.10:
            consistency = "GOOD (Reasonably consistent)"
        elif results['accuracy_coefficient_of_variation'] < 0.20:
            consistency = "MODERATE (Some variation)"
        else:
            consistency = "POOR (High variation between runs)"
        
        print(f"  Clustering Consistency: {consistency}")
        results['clustering_consistency'] = consistency
        
        return results
    
    def test_model_persistence_clustering(self, model_path: str, model_name: str, n_clustering_runs: int = 10) -> Dict:
        """
        Test persistence image clustering performance for a single model
        """
        print(f"\n{'='*70}")
        print(f"TESTING PERSISTENCE IMAGE CLUSTERING: {model_name}")
        print(f"{'='*70}")
        print(f"CORRECTED Methodology: Extract samples -> Compute H0+H1 diagrams -> Create combined images -> Cluster")
        print(f"Statistical significance testing with {n_clustering_runs} clustering runs")
        
        try:
            # Extract latent embeddings by class
            class_embeddings = self.extract_latent_embeddings_by_class(model_path)
            
            # Evaluate persistence image clustering with statistical significance
            results = self.evaluate_persistence_image_clustering(class_embeddings, n_runs=n_clustering_runs)
            
            # Add model info
            results['model_name'] = model_name
            results['model_path'] = model_path
            results['success'] = True
            
            return results
            
        except Exception as e:
            print(f"Error testing model {model_name}: {e}")
            return {
                'model_name': model_name,
                'model_path': model_path,
                'success': False,
                'error': str(e)
            }
    
    def save_results(self, results, output_path: str):
        """Save results to JSON file with proper type conversion"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_data = {
            'timestamp': timestamp,
            'test_config': {
                'n_clusters': int(self.n_clusters),
                'samples_per_class': int(self.samples_per_class),
                'sample_size': int(self.sample_size),
                'random_state': int(self.random_state),
                'ph_params': {k: (int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v) 
                            for k, v in self.ph_params.items()},
                'methodology': 'CORRECTED: Extract samples -> Compute H0+H1 diagrams -> Create combined images -> Cluster'
            },
            'results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def print_summary(self, results: Dict):
        """Print summary of results"""
        print(f"\n{'='*80}")
        print("CORRECTED LATENT PERSISTENCE IMAGE CLUSTERING VALIDATION SUMMARY")
        print(f"{'='*80}")
        
        if not results.get('success', False):
            print(f"Model: {results.get('model_name', 'Unknown')}")
            print(f"Status: FAILED - {results.get('error', 'Unknown error')}")
            return
        
        print(f"Model: {results['model_name']}")
        print(f"Method: {results.get('clustering_method', 'Combined H0+H1 persistence images')}")
        print("-" * 50)
        print(f"Persistence Image Clustering Accuracy: {results['natural_clustering_accuracy_mean']:.4f} ± {results['natural_clustering_accuracy_std']:.4f}")
        print(f"Range: [{results['natural_clustering_accuracy_min']:.4f}, {results['natural_clustering_accuracy_max']:.4f}]")
        print(f"Successful runs: {results['successful_runs']}/{results['n_runs']}")
        print(f"Runs ≥90%: {results['runs_above_90_percent']}/{results['successful_runs']}")
        print(f"Runs ≥95%: {results['runs_above_95_percent']}/{results['successful_runs']}")  
        print(f"Runs ≥99%: {results['runs_above_99_percent']}/{results['successful_runs']}")
        print(f"Clustering Consistency: {results['clustering_consistency']}")
        print(f"Samples per class: {results['samples_per_class']}")
        print(f"Sample size: {results['sample_size']}")
        
        # Compare to original SBERT performance
        original_accuracy = 1.0  # 100% from original SBERT clustering
        mean_performance_ratio = results['natural_clustering_accuracy_mean'] / original_accuracy
        print(f"\nComparison to Original SBERT Persistence Image Clustering:")
        print(f"Original SBERT Combined H0+H1 Clustering: 100.00%")
        print(f"Current Model Mean Combined H0+H1 Clustering: {results['natural_clustering_accuracy_mean']*100:.2f}%")
        print(f"Performance Retention: {mean_performance_ratio*100:.2f}%")
        
        if mean_performance_ratio >= 0.95:
            performance_level = "EXCELLENT (Near-perfect persistence image clustering)"
        elif mean_performance_ratio >= 0.85:
            performance_level = "GOOD (Strong persistence image clustering)"
        elif mean_performance_ratio >= 0.75:
            performance_level = "MODERATE (Reasonable persistence image clustering)"
        else:
            performance_level = "POOR (Weak persistence image clustering)"
        
        print(f"Performance Level: {performance_level}")
        
        # Check if we achieved perfect clustering like original SBERT
        if results['runs_above_99_percent'] > 0:
            print(f"\n🎉 PERFECT PERSISTENCE IMAGE CLUSTERING ACHIEVED IN {results['runs_above_99_percent']} RUNS! 🎉")
            if results['runs_above_99_percent'] >= results['successful_runs'] * 0.8:
                print("This model consistently preserves the same persistence image clustering properties as original SBERT!")


    def test_multiple_models_persistence_clustering(self, models_config: List[Dict], n_clustering_runs: int = 10) -> Dict:
        """
        Test persistence image clustering performance for multiple models
    
        Args:
        models_config: List of dictionaries with 'model_path' and 'model_name' keys
        n_clustering_runs: Number of clustering runs per model for statistical significance
        
        Returns:
        Dictionary containing results for all models with comparative analysis
        """
        print(f"\n{'='*80}")
        print(f"MULTI-MODEL PERSISTENCE IMAGE CLUSTERING VALIDATION")
        print(f"{'='*80}")
        print(f"Testing {len(models_config)} models with {n_clustering_runs} clustering runs each")
        print("CORRECTED Methodology: Extract samples -> Compute H0+H1 diagrams -> Create combined images -> Cluster")
    
        all_results = {}
        successful_models = []
        failed_models = []
    
        # Test each model
        for i, model_config in enumerate(models_config):
            model_name = model_config['model_name']
            model_path = model_config['model_path']
        
            print(f"\n{'-'*60}")
            print(f"TESTING MODEL {i+1}/{len(models_config)}: {model_name}")
            print(f"{'-'*60}")
        
            try:
                # Test single model
                result = self.test_model_persistence_clustering(model_path, model_name, n_clustering_runs)
                all_results[model_name] = result
            
                if result.get('success', False):
                    successful_models.append(model_name)
                    print(f"✓ {model_name}: SUCCESS")
                else:
                    failed_models.append(model_name)
                    print(f"✗ {model_name}: FAILED")
                
            except Exception as e:
                print(f"✗ {model_name}: ERROR - {e}")
                all_results[model_name] = {
                    'model_name': model_name,
                    'model_path': model_path,
                    'success': False,
                    'error': str(e)
                }
                failed_models.append(model_name)
    
        # Generate comparative analysis
        comparative_results = self._generate_comparative_analysis(all_results, successful_models)
    
        # Combine results
        final_results = {
            'individual_results': all_results,
            'comparative_analysis': comparative_results,
            'summary': {
                'total_models': len(models_config),
                'successful_models': len(successful_models),
                'failed_models': len(failed_models),
                'success_rate': len(successful_models) / len(models_config),
                'successful_model_names': successful_models,
                'failed_model_names': failed_models
            }
        }
    
        return final_results

    def _generate_comparative_analysis(self, all_results: Dict, successful_models: List[str]) -> Dict:
        """
        Generate comparative analysis across all successful models
        """
        if len(successful_models) == 0:
            return {"error": "No successful models to compare"}
        
        print(f"\n{'='*60}")
        print("GENERATING COMPARATIVE ANALYSIS")
        print(f"{'='*60}")
        
        # Extract metrics from successful models
        accuracy_means = []
        accuracy_stds = []
        accuracy_maxes = []
        accuracy_mins = []
        consistency_scores = []
        runs_above_90 = []
        runs_above_95 = []
        runs_above_99 = []
        
        model_rankings = []
        
        for model_name in successful_models:
            result = all_results[model_name]
            accuracy_means.append(result['natural_clustering_accuracy_mean'])
            accuracy_stds.append(result['natural_clustering_accuracy_std'])
            accuracy_maxes.append(result['natural_clustering_accuracy_max'])
            accuracy_mins.append(result['natural_clustering_accuracy_min'])
            consistency_scores.append(result['accuracy_coefficient_of_variation'])
            runs_above_90.append(result['runs_above_90_percent'])
            runs_above_95.append(result['runs_above_95_percent'])
            runs_above_99.append(result['runs_above_99_percent'])
            
            # Create ranking score (higher is better)
            ranking_score = (
                result['natural_clustering_accuracy_mean'] * 0.4 +  # Mean accuracy (40%)
                (1 - result['accuracy_coefficient_of_variation']) * 0.3 +  # Consistency (30%)
                (result['runs_above_95_percent'] / result['successful_runs']) * 0.3  # Reliability (30%)
            )
            model_rankings.append((model_name, ranking_score))
        
        # Sort models by ranking
        model_rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Statistical analysis
        accuracy_means = np.array(accuracy_means)
        accuracy_stds = np.array(accuracy_stds)
        
        comparative_analysis = {
            # Overall statistics
            'overall_accuracy_mean': float(np.mean(accuracy_means)),
            'overall_accuracy_std': float(np.std(accuracy_means)),
            'best_model_accuracy': float(np.max(accuracy_means)),
            'worst_model_accuracy': float(np.min(accuracy_means)),
            'accuracy_range': float(np.max(accuracy_means) - np.min(accuracy_means)),
            
            # Model rankings
            'model_rankings': [{'model': name, 'score': float(score)} for name, score in model_rankings],
            'best_model': model_rankings[0][0] if model_rankings else None,
            'worst_model': model_rankings[-1][0] if model_rankings else None,
            
            # Performance categories
            'excellent_models': [name for name, score in model_rankings if all_results[name]['natural_clustering_accuracy_mean'] >= 0.95],
            'good_models': [name for name, score in model_rankings if 0.85 <= all_results[name]['natural_clustering_accuracy_mean'] < 0.95],
            'moderate_models': [name for name, score in model_rankings if 0.75 <= all_results[name]['natural_clustering_accuracy_mean'] < 0.85],
            'poor_models': [name for name, score in model_rankings if all_results[name]['natural_clustering_accuracy_mean'] < 0.75],
            
            # Consistency analysis
            'most_consistent_model': min(successful_models, key=lambda x: all_results[x]['accuracy_coefficient_of_variation']),
            'least_consistent_model': max(successful_models, key=lambda x: all_results[x]['accuracy_coefficient_of_variation']),
            
            # Perfect clustering analysis
            'models_with_perfect_runs': [name for name in successful_models if all_results[name]['runs_above_99_percent'] > 0],
            'total_perfect_runs': sum(all_results[name]['runs_above_99_percent'] for name in successful_models),
            
            # Comparison to SBERT baseline
            'models_above_sbert_threshold': [name for name in successful_models if all_results[name]['natural_clustering_accuracy_mean'] >= 0.90],
            'retention_scores': {name: all_results[name]['natural_clustering_accuracy_mean'] for name in successful_models}
        }
        
        return comparative_analysis

    def print_multi_model_summary(self, results: Dict):
        """Print comprehensive summary of multi-model results"""
        print(f"\n{'='*80}")
        print("MULTI-MODEL PERSISTENCE IMAGE CLUSTERING SUMMARY")
        print(f"{'='*80}")
        
        summary = results['summary']
        comparative = results['comparative_analysis']
        individual = results['individual_results']
        
        # Overall summary
        print(f"Models tested: {summary['total_models']}")
        print(f"Successful: {summary['successful_models']}")
        print(f"Failed: {summary['failed_models']}")
        print(f"Success rate: {summary['success_rate']*100:.1f}%")
        
        if summary['successful_models'] == 0:
            print("\nNo successful models to analyze.")
            return
        
        # Performance overview
        print(f"\n{'-'*50}")
        print("PERFORMANCE OVERVIEW")
        print(f"{'-'*50}")
        print(f"Best model accuracy: {comparative['best_model_accuracy']:.4f}")
        print(f"Worst model accuracy: {comparative['worst_model_accuracy']:.4f}")
        print(f"Overall mean accuracy: {comparative['overall_accuracy_mean']:.4f} ± {comparative['overall_accuracy_std']:.4f}")
        print(f"Performance range: {comparative['accuracy_range']:.4f}")
        
        # Model rankings
        print(f"\n{'-'*50}")
        print("MODEL RANKINGS")
        print(f"{'-'*50}")
        for i, ranking in enumerate(comparative['model_rankings'][:5]):  # Top 5
            model_name = ranking['model']
            score = ranking['score']
            accuracy = individual[model_name]['natural_clustering_accuracy_mean']
            consistency = individual[model_name]['clustering_consistency']
            perfect_runs = individual[model_name]['runs_above_99_percent']
            print(f"{i+1}. {model_name}")
            print(f"   Accuracy: {accuracy:.4f}, Consistency: {consistency}")
            print(f"   Perfect runs: {perfect_runs}, Ranking score: {score:.3f}")
        
        # Performance categories
        print(f"\n{'-'*50}")
        print("PERFORMANCE CATEGORIES")
        print(f"{'-'*50}")
        print(f"Excellent (≥95%): {len(comparative['excellent_models'])} models")
        for model in comparative['excellent_models']:
            print(f"  - {model}: {individual[model]['natural_clustering_accuracy_mean']:.4f}")
        
        print(f"Good (85-95%): {len(comparative['good_models'])} models")
        for model in comparative['good_models']:
            print(f"  - {model}: {individual[model]['natural_clustering_accuracy_mean']:.4f}")
        
        print(f"Moderate (75-85%): {len(comparative['moderate_models'])} models")
        print(f"Poor (<75%): {len(comparative['poor_models'])} models")
        
        # Perfect clustering analysis
        if comparative['models_with_perfect_runs']:
            print(f"\n🎉 MODELS WITH PERFECT CLUSTERING RUNS:")
            for model in comparative['models_with_perfect_runs']:
                perfect_runs = individual[model]['runs_above_99_percent']
                total_runs = individual[model]['successful_runs']
                print(f"  - {model}: {perfect_runs}/{total_runs} perfect runs")
            print(f"Total perfect runs across all models: {comparative['total_perfect_runs']}")
        
        # SBERT comparison
        sbert_threshold_models = comparative['models_above_sbert_threshold']
        print(f"\n{'-'*50}")
        print("SBERT BASELINE COMPARISON")
        print(f"{'-'*50}")
        print(f"Original SBERT persistence image clustering: 100.00%")
        print(f"Models achieving ≥90% (strong retention): {len(sbert_threshold_models)}")
        
        if comparative['best_model']:
            best_model = comparative['best_model']
            best_accuracy = individual[best_model]['natural_clustering_accuracy_mean']
            retention = best_accuracy / 1.0  # Compare to 100% SBERT
            print(f"Best model ({best_model}): {best_accuracy*100:.2f}% ({retention*100:.1f}% retention)")
        
        # Consistency analysis
        print(f"\n{'-'*50}")
        print("CONSISTENCY ANALYSIS")
        print(f"{'-'*50}")
        most_consistent = comparative['most_consistent_model']
        least_consistent = comparative['least_consistent_model']
        print(f"Most consistent: {most_consistent} ({individual[most_consistent]['clustering_consistency']})")
        print(f"Least consistent: {least_consistent} ({individual[least_consistent]['clustering_consistency']})")

    def save_multi_model_results(self, results: Dict, output_prefix: str = "multi_model_clustering"):
        """Save multi-model results with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_output_path = os.path.join(self.output_dir, f'{output_prefix}_detailed_{timestamp}.json')
        
        output_data = {
            'timestamp': timestamp,
            'test_config': {
                'n_clusters': int(self.n_clusters),
                'samples_per_class': int(self.samples_per_class),
                'sample_size': int(self.sample_size),
                'random_state': int(self.random_state),
                'ph_params': {k: (int(v) if isinstance(v, (np.integer, np.int64)) else 
                                float(v) if isinstance(v, (np.floating, np.float64)) else v) 
                            for k, v in self.ph_params.items()},
                'methodology': 'CORRECTED: Extract samples -> Compute H0+H1 diagrams -> Create combined images -> Cluster'
            },
            'results': results
        }
        
        with open(detailed_output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Save summary CSV for easy analysis
        summary_output_path = os.path.join(self.output_dir, f'{output_prefix}_summary_{timestamp}.csv')
        self._save_summary_csv(results, summary_output_path)
        
        print(f"\nDetailed results saved to: {detailed_output_path}")
        print(f"Summary CSV saved to: {summary_output_path}")

    def _save_summary_csv(self, results: Dict, csv_path: str):
        """Save a summary CSV of all models for easy analysis"""
        summary_data = []
        
        for model_name, result in results['individual_results'].items():
            if result.get('success', False):
                summary_data.append({
                    'model_name': model_name,
                    'accuracy_mean': result['natural_clustering_accuracy_mean'],
                    'accuracy_std': result['natural_clustering_accuracy_std'],
                    'accuracy_min': result['natural_clustering_accuracy_min'],
                    'accuracy_max': result['natural_clustering_accuracy_max'],
                    'coefficient_of_variation': result['accuracy_coefficient_of_variation'],
                    'consistency': result['clustering_consistency'],
                    'runs_above_90_percent': result['runs_above_90_percent'],
                    'runs_above_95_percent': result['runs_above_95_percent'],
                    'runs_above_99_percent': result['runs_above_99_percent'],
                    'successful_runs': result['successful_runs'],
                    'sbert_retention_percent': result['natural_clustering_accuracy_mean'] * 100
                })
            else:
                summary_data.append({
                    'model_name': model_name,
                    'accuracy_mean': 'FAILED',
                    'error': result.get('error', 'Unknown error')
                })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_path, index=False)



def main():
    """Main execution function"""
    
    # Data configuration
    data_config = {
        'train_path': 'data/processed/snli_full_standard_SBERT.pt',
        'val_path': 'data/processed/snli_full_standard_SBERT_validation.pt',
        'test_path': 'data/processed/snli_full_standard_SBERT_test.pt',
        'embedding_type': 'concat',
        'sample_size': 5000
    }

    test_mode = 'single'
    
    if test_mode == "single":
        # Model to test
        model_path = "entailment_surfaces/supervised_contrastive_autoencoder/experiments/FIXED_DECODERS/gw_topological_autoencoder_attention_20250727_180538/checkpoints/best_model.pt"
        model_name = "gromov-wasserstein_attention"
        
        # Test configuration
        output_dir = 'entailment_surfaces/supervised_contrastive_autoencoder/latent_clustering_PERSIM_validation_results'
        
        
        print("Latent Embeddings Clustering Validation")
        print("="*50)
        print(f"Model: {model_name}")
        print(f"Output directory: {output_dir}")
        print("Testing k-means clustering performance on latent embeddings")
        print("Comparing to original 100% SBERT clustering accuracy")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize validator
        validator = LatentPHDimensionClusteringValidator(
            data_config=data_config,
            n_clusters=3,
            random_state=42,
            output_dir=output_dir
        )
        
        # Test model clustering
        results = validator.test_model_persistence_clustering(model_path, model_name)
        
        # Save results
        output_path = os.path.join(output_dir, f'{model_name}_clustering_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        validator.save_results(results, output_path)
        
        # Print summary
        validator.print_summary(results)

    elif test_mode == "multi":
        # Multi-model testing (new functionality)
        models_config = [
            {
                'model_path': "entailment_surfaces/supervised_contrastive_autoencoder/experiments/FIXED_DECODERS/global_concat_contrastive_test_no_attention_20250724_203741/checkpoints/best_model.pt",
                'model_name': "contrastive+recon_no_attention"
            },
            {
                'model_path': "entailment_surfaces/supervised_contrastive_autoencoder/experiments/FIXED_DECODERS/pure_moor_topological_autoencoder_no_attention_20250724_222029/checkpoints/best_model.pt",
                'model_name': "moor+recon_no_attention"
            },
            {
                'model_path': "entailment_surfaces/supervised_contrastive_autoencoder/experiments/FIXED_DECODERS/moor_topo-contrastive_autoencoder_noattention_20250725_170549_30%topo-acc_70%classi-acc/checkpoints/checkpoint_epoch_50.pt", 
                'model_name': "moor_topo+contrastive_no_attention"
            }
        ]
        
        output_dir = 'entailment_surfaces/supervised_contrastive_autoencoder/latent_clustering_PERSIM_validation_results/cosine_dmatrix_h1h0'
        
        os.makedirs(output_dir, exist_ok=True)

        
        print("Multi-Model Clustering Validation")
        print("="*50)
        print(f"Testing {len(models_config)} models")
        
        # Initialize validator
        validator = LatentPHDimensionClusteringValidator(
            data_config=data_config,
            n_clusters=3,
            random_state=42,
            output_dir=output_dir
        )
        
        # Test all models
        results = validator.test_multiple_models_persistence_clustering(models_config, n_clustering_runs=10)
        
        # Save and print results
        validator.save_multi_model_results(results)
        validator.print_multi_model_summary(results)


if __name__ == "__main__":
    main()