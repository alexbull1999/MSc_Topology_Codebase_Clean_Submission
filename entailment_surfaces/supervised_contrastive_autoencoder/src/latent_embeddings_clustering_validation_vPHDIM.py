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

# Add project paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from phd_method.src_phd.topology import ph_dim_from_distance_matrix



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
        model = ContrastiveAutoencoder(
            input_dim=1536,  # Adjust based on your embedding dimension
            latent_dim=75,   # Adjust based on your latent dimension
            hidden_dims=[1024, 768, 512, 256, 128]  # Adjust based on your architecture
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
    
    def compute_ph_dimension_for_sample(self, embeddings: torch.Tensor) -> float:
        """
        Compute PH dimension for a sample of embeddings
        
        Args:
            embeddings: Sample of embeddings [sample_size, latent_dim]
            
        Returns:
            PH dimension value (H0)
        """
        try:
            # Convert to numpy
            embeddings_np = embeddings.numpy()
            
            # Compute distance matrix
            distance_matrix = pairwise_distances(embeddings_np, metric='euclidean')
            
            # Calculate PH dimension
            ph_dim = ph_dim_from_distance_matrix(
                distance_matrix,
                min_points=self.ph_params['min_points'],
                max_points=min(self.ph_params['max_points'], len(embeddings)),
                point_jump=self.ph_params['point_jump'],
                h_dim=self.ph_params['h_dim'],
                alpha=self.ph_params['alpha']
            )
            
            return ph_dim
            
        except Exception as e:
            print(f"    Error computing PH dimension: {e}")
            return np.nan
    
    def extract_ph_dimension_samples(self, class_embeddings: Dict[str, torch.Tensor], run_seed: int) -> Tuple[List[float], List[int]]:
        """
        Extract PH-dimension values from multiple samples of each class
        Following the original phdim_clustering_validation_best_metrics.py methodology
        
        Args:
            class_embeddings: Dictionary mapping class names to embeddings
            run_seed: Random seed for this run
            
        Returns:
            Tuple of (ph_dimension_values, class_labels)
        """
        print(f"  Extracting PH-dimension samples (seed={run_seed})")
        
        # Set random seed for reproducible sampling
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        
        ph_dim_values = []
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
                
                # Compute PH dimension for this sample
                ph_dim = self.compute_ph_dimension_for_sample(sample_embeddings)
                
                if not np.isnan(ph_dim):
                    ph_dim_values.append(ph_dim)
                    sample_labels.append(class_idx)
                    print(f"      Sample {sample_idx+1}: PH-dim = {ph_dim:.4f}")
                else:
                    print(f"      Sample {sample_idx+1}: FAILED")
        
        print(f"  Extracted {len(ph_dim_values)} valid PH-dimension values")
        return ph_dim_values, sample_labels
    
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
    
    def evaluate_ph_dimension_clustering(self, class_embeddings: Dict[str, torch.Tensor], n_runs: int = 10) -> Dict:
        """
        Evaluate k-means clustering performance on PH-dimension values with statistical significance
        
        Args:
            class_embeddings: Dictionary mapping class names to latent embeddings
            n_runs: Number of clustering runs for statistical significance
            
        Returns:
            Dictionary containing clustering metrics with statistical analysis
        """
        print(f"\nEvaluating PH-dimension clustering with statistical significance (n_runs={n_runs})")
        print("Computing PH-dimension values and clustering on them...")
        
        # Storage for multiple runs
        all_natural_accuracies = []
        all_adjusted_rand_scores = []
        all_normalized_mutual_infos = []
        all_silhouette_scores = []
        all_ph_values_by_run = []
        
        # Run multiple clustering attempts
        for run_idx in range(n_runs):
            print(f"\n  PH-dimension clustering run {run_idx+1}/{n_runs}")
            
            # Extract PH-dimension values for this run
            run_seed = self.random_state + run_idx
            ph_values, labels = self.extract_ph_dimension_samples(class_embeddings, run_seed)
            
            if len(ph_values) < 6:  # Need at least 6 samples for 3-class clustering
                print(f"    Insufficient samples ({len(ph_values)}), skipping run")
                continue
            
            # Convert to numpy arrays and reshape for clustering
            ph_values_np = np.array(ph_values).reshape(-1, 1)  # 1D feature vector
            labels_np = np.array(labels)
            
            # Perform k-means clustering on PH-dimension values
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=run_seed,
                n_init=10,
                max_iter=300
            )
            
            cluster_pred = kmeans.fit_predict(ph_values_np)
            
            # Analyze natural clustering performance
            natural_analysis = self.analyze_natural_clustering(labels_np, cluster_pred)
            
            # Calculate additional metrics
            adjusted_rand = adjusted_rand_score(labels_np, cluster_pred)
            normalized_mutual_info = normalized_mutual_info_score(labels_np, cluster_pred)
            
            # Silhouette score (only if we have enough samples and more than 1 cluster)
            if len(np.unique(cluster_pred)) > 1 and len(ph_values_np) > self.n_clusters:
                silhouette_avg = silhouette_score(ph_values_np, cluster_pred)
            else:
                silhouette_avg = np.nan
            
            # Store results
            all_natural_accuracies.append(natural_analysis['natural_accuracy'])
            all_adjusted_rand_scores.append(adjusted_rand)
            all_normalized_mutual_infos.append(normalized_mutual_info)
            all_silhouette_scores.append(silhouette_avg)
            all_ph_values_by_run.append({
                'ph_values': ph_values,
                'labels': labels,
                'natural_mapping': natural_analysis['natural_mapping']
            })
            
            print(f"    Natural Accuracy: {natural_analysis['natural_accuracy']:.3f}")
            print(f"    PH-values range: [{min(ph_values):.3f}, {max(ph_values):.3f}]")
        
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
            
            # Consistency analysis
            'accuracy_coefficient_of_variation': float(np.std(natural_accuracies) / np.mean(natural_accuracies)) if np.mean(natural_accuracies) > 0 else float('inf'),
            'runs_above_90_percent': int(np.sum(natural_accuracies >= 0.90)),
            'runs_above_95_percent': int(np.sum(natural_accuracies >= 0.95)),
            'runs_above_99_percent': int(np.sum(natural_accuracies >= 0.99)),
            
            # PH-dimension analysis
            'ph_values_by_run': all_ph_values_by_run
        }
        
        # Print statistical results
        print(f"\nStatistical PH-Dimension Clustering Results:")
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
    
    def test_model_ph_clustering(self, model_path: str, model_name: str, n_clustering_runs: int = 10) -> Dict:
        """
        Test PH-dimension clustering performance for a single model
        
        Args:
            model_path: Path to model checkpoint
            model_name: Name of the model for reporting
            n_clustering_runs: Number of clustering runs for statistical significance
            
        Returns:
            Dictionary containing clustering results
        """
        print(f"\n{'='*70}")
        print(f"TESTING PH-DIMENSION CLUSTERING: {model_name}")
        print(f"{'='*70}")
        print(f"Methodology: Extract samples -> Compute PH-dim -> Cluster on PH-dim values")
        print(f"Statistical significance testing with {n_clustering_runs} clustering runs")
        
        try:
            # Extract latent embeddings by class
            class_embeddings = self.extract_latent_embeddings_by_class(model_path)
            
            # Evaluate PH-dimension clustering with statistical significance
            results = self.evaluate_ph_dimension_clustering(class_embeddings, n_runs=n_clustering_runs)
            
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
    
    def test_multiple_models(self, model_configs: List[Dict], n_clustering_runs: int = 20) -> List[Dict]:
        """
        Test PH-dimension clustering performance across multiple models
        
        Args:
            model_configs: List of dicts with 'path' and 'name' keys
            n_clustering_runs: Number of clustering runs per model
            
        Returns:
            List of results for each model
        """
        all_results = []
        
        print(f"Testing PH-dimension clustering performance across {len(model_configs)} models")
        print(f"Statistical significance: {n_clustering_runs} clustering runs per model")
        print(f"Each model will be tested and results saved individually...")
        
        for i, config in enumerate(model_configs, 1):
            print(f"\n{'='*80}")
            print(f"TESTING MODEL {i}/{len(model_configs)}: {config['name']}")
            print(f"{'='*80}")
            
            if not os.path.exists(config['path']):
                print(f"Warning: Model not found: {config['path']}")
                error_result = {
                    'model_name': config['name'],
                    'model_path': config['path'],
                    'success': False,
                    'error': f'Model file not found: {config["path"]}'
                }
                all_results.append(error_result)
                continue
            
            try:
                results = self.test_model_ph_clustering(config['path'], config['name'], n_clustering_runs)
                all_results.append(results)
                
                # Save individual result immediately
                individual_output_path = os.path.join(
                    self.output_dir, 
                    f'{config["name"]}_ph_clustering_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                )
                self.save_results(results, individual_output_path)
                
                # Quick summary for this model
                if results.get('success', False):
                    mean_accuracy = results['natural_clustering_accuracy_mean']
                    std_accuracy = results['natural_clustering_accuracy_std']
                    runs_95 = results['runs_above_95_percent']
                    print(f"\n✅ {config['name']}: {mean_accuracy:.4f} ± {std_accuracy:.4f} PH-dim clustering accuracy")
                    print(f"   {runs_95}/{results['successful_runs']} runs achieved ≥95% accuracy")
                else:
                    print(f"\n❌ {config['name']}: FAILED")
                    
            except Exception as e:
                print(f"Error testing model {config['name']}: {e}")
                error_result = {
                    'model_name': config['name'],
                    'model_path': config['path'], 
                    'success': False,
                    'error': str(e)
                }
                all_results.append(error_result)
                
                # Save error result too
                individual_output_path = os.path.join(
                    self.output_dir, 
                    f'{config["name"]}_ph_clustering_ERROR_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                )
                self.save_results(error_result, individual_output_path)
        
        return all_results
    
    def save_results(self, results, output_path: str):
        """Save results to JSON file with proper type conversion"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Handle both single result dict and list of results
        if isinstance(results, dict):
            # Single model result
            output_data = {
                'timestamp': timestamp,
                'test_config': {
                    'n_clusters': int(self.n_clusters),
                    'samples_per_class': int(self.samples_per_class),
                    'sample_size': int(self.sample_size),
                    'random_state': int(self.random_state),
                    'ph_params': {k: (int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v) 
                                for k, v in self.ph_params.items()},
                    'methodology': 'Extract samples -> Compute PH-dim -> Cluster on PH-dim values'
                },
                'results': results
            }
        else:
            # Multiple model results
            output_data = {
                'timestamp': timestamp,
                'test_config': {
                    'n_clusters': int(self.n_clusters),
                    'samples_per_class': int(self.samples_per_class),
                    'sample_size': int(self.sample_size),
                    'random_state': int(self.random_state),
                    'ph_params': {k: (int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v) 
                                for k, v in self.ph_params.items()},
                    'methodology': 'Extract samples -> Compute PH-dim -> Cluster on PH-dim values'
                },
                'results': results
            }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def print_summary(self, results: Dict):
        """Print summary of results"""
        print(f"\n{'='*80}")
        print("LATENT PH-DIMENSION CLUSTERING VALIDATION SUMMARY")
        print(f"{'='*80}")
        
        if not results.get('success', False):
            print(f"Model: {results.get('model_name', 'Unknown')}")
            print(f"Status: FAILED - {results.get('error', 'Unknown error')}")
            return
        
        print(f"Model: {results['model_name']}")
        print(f"Methodology: PH-dimension clustering (like original validation)")
        print("-" * 50)
        print(f"PH-Dimension Clustering Accuracy: {results['natural_clustering_accuracy_mean']:.4f} ± {results['natural_clustering_accuracy_std']:.4f}")
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
        print(f"\nComparison to Original SBERT PH-Dimension Clustering:")
        print(f"Original SBERT PH-Dim Clustering: 100.00%")
        print(f"Current Model Mean PH-Dim Clustering: {results['natural_clustering_accuracy_mean']*100:.2f}%")
        print(f"Performance Retention: {mean_performance_ratio*100:.2f}%")
        
        if mean_performance_ratio >= 0.95:
            performance_level = "EXCELLENT (Near-perfect PH-dim clustering)"
        elif mean_performance_ratio >= 0.85:
            performance_level = "GOOD (Strong PH-dim clustering)"
        elif mean_performance_ratio >= 0.75:
            performance_level = "MODERATE (Reasonable PH-dim clustering)"
        else:
            performance_level = "POOR (Weak PH-dim clustering)"
        
        print(f"Performance Level: {performance_level}")
        
        # Check if we achieved perfect clustering like original SBERT
        if results['runs_above_99_percent'] > 0:
            print(f"\n🎉 PERFECT PH-DIM CLUSTERING ACHIEVED IN {results['runs_above_99_percent']} RUNS! 🎉")
            if results['runs_above_99_percent'] >= results['successful_runs'] * 0.8:
                print("This model consistently preserves the same PH-dimension clustering properties as original SBERT!")
        
        # Statistical significance assessment
        if results['accuracy_coefficient_of_variation'] < 0.05:
            print(f"\n📊 EXCELLENT STATISTICAL CONSISTENCY")
            print("PH-dimension clustering results are highly reliable across multiple runs")
        elif results['accuracy_coefficient_of_variation'] < 0.10:
            print(f"\n📊 GOOD STATISTICAL CONSISTENCY") 
            print("PH-dimension clustering results show reasonable stability across runs")


            
    def print_multiple_models_summary(self, all_results: List[Dict]):
        """Print comprehensive summary of all models tested"""
        print(f"\n{'='*80}")
        print("MULTIPLE MODELS CLUSTERING VALIDATION SUMMARY")
        print(f"{'='*80}")
        
        successful_models = [r for r in all_results if r.get('success', False)]
        failed_models = [r for r in all_results if not r.get('success', False)]
        
        print(f"Total Models Tested: {len(all_results)}")
        print(f"Successful Tests: {len(successful_models)}")
        print(f"Failed Tests: {len(failed_models)}")
        
        if successful_models:
            print(f"\n{'Model Performance Ranking:'}")
            print("-" * 50)
            
            # Sort by natural clustering accuracy
            sorted_models = sorted(successful_models, 
                                 key=lambda x: x['natural_clustering_accuracy'], 
                                 reverse=True)
            
            for i, result in enumerate(sorted_models, 1):
                accuracy = result['natural_clustering_accuracy']
                silhouette = result['silhouette_score']
                performance_vs_sbert = (accuracy / 1.0) * 100  # vs 100% SBERT
                
                print(f"{i:2d}. {result['model_name']}")
                print(f"     Natural Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                print(f"     vs SBERT: {performance_vs_sbert:.1f}%")
                print(f"     Silhouette: {silhouette:.3f}")
                
                if accuracy >= 0.99:
                    print(f"     🎉 PERFECT CLUSTERING!")
                elif accuracy >= 0.90:
                    print(f"     ⭐ EXCELLENT")
                elif accuracy >= 0.80:
                    print(f"     ✅ GOOD")
                else:
                    print(f"     ⚠️  NEEDS IMPROVEMENT")
                print()
        
        if failed_models:
            print(f"\nFailed Models:")
            print("-" * 20)
            for result in failed_models:
                print(f"❌ {result['model_name']}: {result.get('error', 'Unknown error')}")
        
        print(f"\nAll individual results saved to: {self.output_dir}")


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
    
    # Model to test
    model_path = "entailment_surfaces/supervised_contrastive_autoencoder/experiments/H0H1_signature_moor_lifted_autoencoder_no_attention_20250728_152242/checkpoints/best_model.pt"
    model_name = "moor_signatureh0+h1_lifted_no_attention"
    
    # Test configuration
    output_dir = 'entailment_surfaces/supervised_contrastive_autoencoder/latent_clustering_PHDIM_validation_results'
    
    
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
    results = validator.test_model_ph_clustering(model_path, model_name)
    
    # Save results
    output_path = os.path.join(output_dir, f'{model_name}_clustering_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    validator.save_results(results, output_path)
    
    # Print summary
    validator.print_summary(results)


if __name__ == "__main__":
    main()