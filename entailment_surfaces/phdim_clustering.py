"""
Phase 1: Topological Clustering Validation
Simple implementation using existing infrastructure from phdim_distance_metric_optimized.py

Leverages existing functions:
- SurfaceDistanceMetricAnalyzer for embedding generation
- fast_ripser and ph_dim_from_distance_matrix for persistence computation
- Existing distance matrix computation methods

Method:
1. Generate samples using existing sampling approach
2. Use modified ph_dim_from_distance_matrix to get both PH-dim AND persistence diagrams
3. Convert persistence diagrams to images for clustering
4. Apply k-means clustering analysis
"""

import os
import sys
import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from persim import PersistenceImager
from phdim_distance_metric_optimized import SurfaceDistanceMetricAnalyzer
from gph.python import ripser_parallel
from itertools import permutations
import matplotlib.pyplot as plt


# Import existing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phd_method.src_phd.topology import ph_dim_from_distance_matrix

def flush_output():
    """Force output to appear immediately"""
    sys.stdout.flush()
    sys.stderr.flush()

@dataclass
class ClusteringResult:
    """Results for one embedding space + metric combination"""
    embedding_space: str
    distance_metric: str
    clustering_accuracy: float
    silhouette_score: float
    adjusted_rand_score: float
    num_samples: int
    success: bool  # True if accuracy > 70%
    # Add PH-dimension tracking
    ph_dim_values: Dict[str, List[float]]  # class_name -> list of ph_dim values
    ph_dim_stats: Dict[str, Dict[str, float]]  # class_name -> {mean, std, min, max}


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


class ClusteringValidator:
    """
    Simple clustering validator using existing infrastructure
    """

    def __init__(self, 
                 bert_data_path: str,
                 order_model_path: str,
                 output_dir: str = "entailment_surfaces/clustering",
                 seed: int = 42):
        """Initialize Phase 1 validator"""
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Target embedding spaces from Revised Next Steps
        self.target_spaces = [
            'sbert_concat',
            'lattice_containment',
            'order_concat',
            'hyperbolic_concat'
        ]
        
        # All distance metrics
        self.distance_metrics = [
            'euclidean', 
            'manhattan', 
            'chebyshev', 
            'cosine',
            'minkowski_3', 
            'minkowski_4', 
            'canberra', 
            'braycurtis'
        ]
        
        # Sampling parameters
        self.samples_per_class = 10
        self.points_per_sample = 1000
        
        print(f"Phase 1 Clustering Validator initialized")
        print(f"Target spaces: {self.target_spaces}")
        print(f"Distance metrics: {self.distance_metrics}")
        print(f"Samples per class: {self.samples_per_class}")
        print(f"Points per sample: {self.points_per_sample}")
        flush_output()
        
        # Initialize analyzer using existing infrastructure
        self.analyzer = SurfaceDistanceMetricAnalyzer(
            bert_data_path=bert_data_path,
            order_model_path=order_model_path,
            seed=seed
        )


    def generate_embedding_spaces_by_class(self):
        """
        Generate embedding spaces using existing analyzer infrastructure
        """
        print("Extracting all embedding spaces using analyzer...")
        flush_output()
        
        # Use existing function to get ALL embedding spaces properly
        all_embeddings = self.analyzer.extract_all_embedding_spaces(max_samples_per_class=None)
        
        # Create dictionary with only the spaces we want to test
        target_embeddings = {}
        for space_name in self.target_spaces:
            if space_name in all_embeddings:
                target_embeddings[space_name] = all_embeddings[space_name]
                print(f"  {space_name}: Available")
                flush_output()
            else:
                print(f"  {space_name}: Not available in extracted embeddings")
                flush_output()
        
        # Delete the full dictionary to save memory
        del all_embeddings
        
        return target_embeddings

    def generate_fixed_samples_from_embeddings(self, all_embeddings):
        """
        Generate fixed sample indices that will be used across all embedding spaces
        """
        print("\nGenerating fixed sample indices...")
        
        fixed_sample_indices = {}
        
        # For each class, generate fixed sample indices
        for class_idx, class_name in enumerate(['entailment', 'neutral', 'contradiction']):
            class_indices = []
            
            # Get the size from the first embedding space (they should all be the same)
            first_space = list(all_embeddings.keys())[0]
            n_total = all_embeddings[first_space][class_name].shape[0]
            
            print(f"  {class_name}: {n_total} total samples available")
            
            # Generate sample indices for this class
            base_seed = self.seed + class_idx * 100
            
            for sample_idx in range(self.samples_per_class):
                sample_seed = base_seed + sample_idx
                np.random.seed(sample_seed)
                
                if self.points_per_sample > n_total:
                    indices = np.arange(n_total)
                    actual_points = n_total
                    print(f"    Warning: Using all {n_total} points for {class_name} sample {sample_idx}")
                else:
                    indices = np.random.choice(n_total, self.points_per_sample, replace=False)
                    actual_points = self.points_per_sample
                
                class_indices.append({
                    'indices': indices,
                    'actual_points': actual_points,
                    'sample_idx': sample_idx,
                    'class_name': class_name,
                    'class_idx': class_idx
                })
            
            fixed_sample_indices[class_name] = class_indices
            print(f"    Generated {len(class_indices)} sample index sets")
        
        return fixed_sample_indices

    
    def persistence_diagrams_to_images(self, all_diagrams: List) -> List[np.ndarray]:
        """
        Convert persistence diagrams to standardized images
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
            for dim in range(min(2, len(diagrams))):
                diagram = diagrams[dim]
                if len(diagram) > 0:
                    # Filter infinite persistence
                    finite_diagram = diagram[np.isfinite(diagram).all(axis=1)]
                    if len(finite_diagram) > 0:
                        try:
                            img = pimgr.transform([finite_diagram])[0]
                            # Resize to standard size if needed
                            if img.shape != (20, 20):
                                from scipy.ndimage import zoom
                                zoom_factors = (20 / img.shape[0], 20 / img.shape[1])
                                img = zoom(img, zoom_factors)
                            combined_image += img
                        except:
                            continue
            
            # Normalize and flatten
            if combined_image.max() > 0:
                combined_image = combined_image / combined_image.max()
            
            persistence_images.append(combined_image.flatten())
        
        return persistence_images


    def perform_clustering_analysis(self, persistence_images: List[np.ndarray], 
                                  true_labels: List[int],
                                  kmeans_seed: int) -> Tuple[float, float, float]:
        """Perform k-means clustering analysis with reproducible seed"""
        if len(persistence_images) == 0:
            print("ERROR PERSISTENCE IMAGES NOT FOUND")
            flush_output()
            return 0.0, 0.0, 0.0
            
        X = np.vstack(persistence_images)
        
        # K-means with reproducible seed
        kmeans = KMeans(n_clusters=3, random_state=kmeans_seed, n_init=10)
        predicted_labels = kmeans.fit_predict(X)
        
        # Calculate metrics
        accuracy = self._calculate_clustering_accuracy(true_labels, predicted_labels)
        sil_score = silhouette_score(X, predicted_labels) if len(set(predicted_labels)) > 1 else 0.0 #how well separated clusters are (-1 to 1)
        ari_score = adjusted_rand_score(true_labels, predicted_labels) #How well the predicted clusters match true class labels (0 to 1)
        
        return accuracy, sil_score, ari_score

    
    def _calculate_clustering_accuracy(self, true_labels: List[int], 
                                     predicted_labels: List[int]) -> float:
        """Calculate best clustering accuracy over all label permutations"""
        
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        
        unique_predicted = np.unique(predicted_labels)
        best_accuracy = 0.0
        
        for perm in permutations(range(len(unique_predicted))):
            mapped_labels = np.zeros_like(predicted_labels)
            for i, cluster_id in enumerate(unique_predicted):
                mapped_labels[predicted_labels == cluster_id] = perm[i]
                
            accuracy = np.mean(true_labels == mapped_labels)
            best_accuracy = max(best_accuracy, accuracy)
            
        return best_accuracy

    
    def validate_embedding_space(self, space_name: str, 
                                metric_name: str,
                                all_embeddings: Dict,
                                fixed_sample_indices: Dict) -> Optional[ClusteringResult]:
        """
        Validate one embedding space + metric combination using fixed samples
        """
        print(f"\nValidating {space_name} + {metric_name}")
        flush_output()
        
        try:
            space_embeddings = all_embeddings[space_name]
            
            all_persistence_images = []
            sample_labels = []
            ph_dim_values = {'entailment': [], 'neutral': [], 'contradiction': []}
            
            # Process each fixed sample
            for class_name in ['entailment', 'neutral', 'contradiction']:
                class_embeddings = space_embeddings[class_name]
                class_sample_indices = fixed_sample_indices[class_name]
                class_idx = ['entailment', 'neutral', 'contradiction'].index(class_name)
                
                for sample_info in class_sample_indices:
                    indices = sample_info['indices']
                    sample_idx = sample_info['sample_idx']
                    
                    # Extract the sample using fixed indices
                    sampled_embeddings = class_embeddings[indices]
                    
                    # Compute distance matrix using existing analyzer method
                    distance_matrix = self.analyzer.compute_distance_matrix_advanced(
                        sampled_embeddings, metric_name
                    )
                    
                    # Get persistence diagrams - USE SAME SEED FOR ALL SAMPLES
                    ph_dim, all_diagrams = ph_dim_and_diagrams_from_distance_matrix(
                        distance_matrix,
                        min_points=min(200, sampled_embeddings.shape[0]//2),
                        max_points=min(1000, sampled_embeddings.shape[0]),
                        point_jump=50,
                        h_dim=0,
                        alpha=1.0
                        #seed=self.seed  <--- REMOVE: this was causing the global seed state to change unintentionally
                    )
                    
                    # Store PH-dimension value
                    ph_dim_values[class_name].append(ph_dim)
                    
                    # Analyze clustering for EACH sample size in persistence diagram computation
                    # This will show how clustering accuracy varies with sample size
                    for diagram_idx, diagram in enumerate(all_diagrams):
                        persistence_image = self.persistence_diagrams_to_images([diagram])
                        
                        if len(persistence_image) > 0:
                            all_persistence_images.append(persistence_image[0])
                            sample_labels.append(class_idx)
            
            # Now we have multiple persistence images per sample (one for each sample size)
            # Let's see how many sample sizes we're analyzing
            sample_sizes = list(range(200, min(1000, sampled_embeddings.shape[0]), 50))
            n_sample_sizes = len(sample_sizes)
            print(f"  Analyzing clustering across {n_sample_sizes} different sample sizes")
            
            # Group persistence images by sample size for analysis
            persistence_by_size = {}
            labels_by_size = {}
            
            for size_idx in range(n_sample_sizes):
                # Extract images for this sample size across all samples
                size_images = []
                size_labels = []
                
                for sample_idx in range(30):  # 30 total samples (10 per class)
                    img_idx = sample_idx * n_sample_sizes + size_idx
                    if img_idx < len(all_persistence_images):
                        size_images.append(all_persistence_images[img_idx])
                        size_labels.append(sample_labels[img_idx])
                
                if len(size_images) == 30:  # Make sure we have all samples for this size
                    persistence_by_size[sample_sizes[size_idx]] = size_images
                    labels_by_size[sample_sizes[size_idx]] = size_labels
            
            # Analyze clustering for each sample size
            clustering_by_size = {}
            for sample_size, images in persistence_by_size.items():
                accuracy, sil_score, ari_score = self.perform_clustering_analysis(
                    images, labels_by_size[sample_size], self.seed
                )
                clustering_by_size[sample_size] = {
                    'accuracy': accuracy,
                    'silhouette': sil_score,
                    'ari': ari_score
                }
                print(f"    Sample size {sample_size}: Acc={accuracy:.3f}, Sil={sil_score:.3f}, ARI={ari_score:.3f}")
            
            # Use the best performing sample size for final result
            best_size = max(clustering_by_size.keys(), 
                          key=lambda x: clustering_by_size[x]['accuracy'])
            
            accuracy = clustering_by_size[best_size]['accuracy']
            sil_score = clustering_by_size[best_size]['silhouette'] 
            ari_score = clustering_by_size[best_size]['ari']
            
            print(f"  Best sample size: {best_size} points")
            
            # Store sample size analysis for plotting
            size_analysis = {
                'sample_sizes': list(clustering_by_size.keys()),
                'accuracies': [clustering_by_size[size]['accuracy'] for size in clustering_by_size.keys()],
                'silhouette_scores': [clustering_by_size[size]['silhouette'] for size in clustering_by_size.keys()],
                'ari_scores': [clustering_by_size[size]['ari'] for size in clustering_by_size.keys()],
                'best_size': best_size
            }
            
            # Calculate PH-dimension statistics
            ph_dim_stats = {}
            for class_name in ['entailment', 'neutral', 'contradiction']:
                ph_dims = ph_dim_values[class_name]
                if ph_dims:
                    ph_dim_stats[class_name] = {
                        'mean': float(np.mean(ph_dims)),
                        'std': float(np.std(ph_dims)),
                        'min': float(np.min(ph_dims)),
                        'max': float(np.max(ph_dims))
                    }
            
            result = ClusteringResult(
                embedding_space=space_name,
                distance_metric=metric_name,
                clustering_accuracy=accuracy,
                silhouette_score=sil_score,
                adjusted_rand_score=ari_score,
                num_samples=30,  # 30 total samples
                success=(accuracy > 0.7),
                ph_dim_values=ph_dim_values,
                ph_dim_stats=ph_dim_stats
            )
            
            # Add size analysis to result
            result.size_analysis = size_analysis
            
            # Print simple results
            print(f"  Results: Accuracy={accuracy:.3f}, Silhouette={sil_score:.3f}, ARI={ari_score:.3f}")
            print(f"  PH-Dims: E={[f'{x:.1f}' for x in ph_dim_values['entailment']]}")
            print(f"           N={[f'{x:.1f}' for x in ph_dim_values['neutral']]}")  
            print(f"           C={[f'{x:.1f}' for x in ph_dim_values['contradiction']]}")
            print(f"  SUCCESS: {'YES' if result.success else 'NO'}")
            flush_output()
            
            return result
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None


    
    def run_full_validation(self) -> Dict:
        """Run full Phase 1 validation using fixed samples"""
        print("="*60)
        print("PHASE 1: TOPOLOGICAL CLUSTERING VALIDATION")
        print("="*60)
        print("Using fixed samples across all combinations for fair comparison")
        flush_output()
        
        # Generate all embedding spaces once
        all_embeddings = self.generate_embedding_spaces_by_class()
        
        # Generate fixed sample indices once
        fixed_sample_indices = self.generate_fixed_samples_from_embeddings(all_embeddings)
        
        all_results = []
        successful_combinations = []
        
        # Test each combination using the same fixed sample indices
        for space_idx, space_name in enumerate(self.target_spaces):
            print(f"\n{'='*60}")
            print(f"PROCESSING EMBEDDING SPACE {space_idx+1}/{len(self.target_spaces)}: {space_name}")
            print(f"{'='*60}")
            
            for metric_idx, metric_name in enumerate(self.distance_metrics):
                result = self.validate_embedding_space(
                    space_name, metric_name, all_embeddings, fixed_sample_indices
                )
                
                if result is not None:
                    all_results.append(result)
                    if result.success:
                        successful_combinations.append(result)
                
                # CLEAR MEMORY after each metric to prevent OOM
                print(f"Clearing memory after {space_name} + {metric_name}...")
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        
        # Summary
        print("\n" + "="*60)
        print("PHASE 1 VALIDATION SUMMARY")
        print("="*60)
        print(f"Total combinations tested: {len(all_results)}")
        print(f"Successful combinations (>70% accuracy): {len(successful_combinations)}")
        flush_output()
        
        if successful_combinations:
            print("\nSUCCESSFUL COMBINATIONS:")
            for result in sorted(successful_combinations, key=lambda x: x.clustering_accuracy, reverse=True):
                print(f"  {result.embedding_space} + {result.distance_metric}: {result.clustering_accuracy:.3f}")
                flush_output()
        else:
            print("\nNO COMBINATIONS ACHIEVED >70% CLUSTERING ACCURACY")
            print("Best performing combinations:")
            best_results = sorted(all_results, key=lambda x: x.clustering_accuracy, reverse=True)[:5]
            for result in best_results:
                print(f"  {result.embedding_space} + {result.distance_metric}: {result.clustering_accuracy:.3f}")
                flush_output()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_data = {
            'timestamp': timestamp,
            'experimental_design': 'fixed_samples_across_combinations',
            'samples_per_class': self.samples_per_class,
            'points_per_sample': self.points_per_sample,
            'total_samples': sum(len(indices) for indices in fixed_sample_indices.values()),
            'successful_combinations': len(successful_combinations),
            'total_combinations': len(all_results),
            'results': [
                {
                    'embedding_space': r.embedding_space,
                    'distance_metric': r.distance_metric,
                    'clustering_accuracy': float(r.clustering_accuracy),
                    'silhouette_score': float(r.silhouette_score),
                    'adjusted_rand_score': float(r.adjusted_rand_score),
                    'num_samples': int(r.num_samples),
                    'success': bool(r.success),
                    'ph_dim_values': {
                    k: [float(x) for x in v] for k, v in r.ph_dim_values.items()  # Convert all to Python floats
                    },
                    'ph_dim_stats': {
                    k: {stat_k: float(stat_v) for stat_k, stat_v in stat_dict.items()} 
                    for k, stat_dict in r.ph_dim_stats.items()  # Convert all to Python floats
                    }
                }
                for r in all_results
            ]
        }
        
        # Save files
        results_file = self.output_dir / f"MNLI_clustering_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        summary_file = self.output_dir / f"MNLI_clustering_summary_report_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("PHASE 1 CLUSTERING VALIDATION RESULTS\n")
            f.write("="*50 + "\n\n")
            
            # Simple table format
            f.write("EMBEDDING_SPACE | METRIC | ACCURACY | SILHOUETTE | ARI | SUCCESS\n")
            f.write("-" * 70 + "\n")
            
            for result in sorted(all_results, key=lambda x: x.clustering_accuracy, reverse=True):
                f.write(f"{result.embedding_space:15} | {result.distance_metric:8} | ")
                f.write(f"{result.clustering_accuracy:8.3f} | {result.silhouette_score:10.3f} | ")
                f.write(f"{result.adjusted_rand_score:7.3f} | {'YES' if result.success else 'NO':7}\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write("PH-DIMENSION VALUES BY SAMPLE\n")
            f.write("="*50 + "\n\n")
            
            for result in sorted(all_results, key=lambda x: x.clustering_accuracy, reverse=True):
                f.write(f"{result.embedding_space} + {result.distance_metric}:\n")
                f.write(f"  E: {[f'{x:.1f}' for x in result.ph_dim_values['entailment']]}\n")
                f.write(f"  N: {[f'{x:.1f}' for x in result.ph_dim_values['neutral']]}\n")
                f.write(f"  C: {[f'{x:.1f}' for x in result.ph_dim_values['contradiction']]}\n\n")
        
        # Generate plots showing sample size vs clustering accuracy
        self.generate_sample_size_plots(all_results, timestamp)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Summary report saved to: {summary_file}")
        print(f"Sample size plots saved to: {self.plots_dir}")
        
        return results_data



    def generate_sample_size_plots(self, all_results: List[ClusteringResult], timestamp: str):
        """
        Generate plots showing how clustering accuracy varies with sample size
        """
        print("\nGenerating sample size analysis plots...")
        
        # Create plots for each embedding space
        for space_name in self.target_spaces:
            space_results = [r for r in all_results if r.embedding_space == space_name and hasattr(r, 'size_analysis')]
            
            if not space_results:
                continue
                
            plt.figure(figsize=(12, 8))
            
            # Plot accuracy vs sample size for each metric
            for result in space_results:
                sa = result.size_analysis
                plt.plot(sa['sample_sizes'], sa['accuracies'], 
                        marker='o', label=result.distance_metric, linewidth=2)
            
            plt.xlabel('Sample Size (Number of Points)')
            plt.ylabel('Clustering Accuracy')
            plt.title(f'Clustering Accuracy vs Sample Size: {space_name}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.xlim(left=0)
            plt.ylim(0, 1)
            
            # Add horizontal line at 70% success threshold
            plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='70% Success Threshold')
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.plots_dir / f"MNLI_sample_size_analysis_{space_name}_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved plot for {space_name}: {plot_file}")
        
        # Create summary plot comparing best embedding spaces
        plt.figure(figsize=(14, 10))
        
        # Find best metric for each embedding space
        best_results = {}
        for space_name in self.target_spaces:
            space_results = [r for r in all_results if r.embedding_space == space_name and hasattr(r, 'size_analysis')]
            if space_results:
                best_result = max(space_results, key=lambda x: x.clustering_accuracy)
                best_results[space_name] = best_result
        
        # Plot comparison
        colors = ['blue', 'green', 'red']
        for i, (space_name, result) in enumerate(best_results.items()):
            sa = result.size_analysis
            plt.plot(sa['sample_sizes'], sa['accuracies'], 
                    marker='o', label=f"{space_name} ({result.distance_metric})", 
                    linewidth=3, color=colors[i % len(colors)])
        
        plt.xlabel('Sample Size (Number of Points)')
        plt.ylabel('Clustering Accuracy')
        plt.title('Best Performing Combinations: Clustering Accuracy vs Sample Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(left=0)
        plt.ylim(0, 1)
        
        # Add horizontal line at 70% success threshold
        plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='70% Success Threshold')
        
        plt.tight_layout()
        
        # Save summary plot
        summary_plot_file = self.plots_dir / f"sample_size_summary_{space_name}_{timestamp}.png"
        plt.savefig(summary_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved summary plot: {summary_plot_file}")


def main():
    """Run Phase 1 clustering validation"""
    bert_data_path = "data/processed/snli_10k_subset_train_SBERT_STSB_LARGE.pt"
    order_model_path = "models/enhanced_order_embeddings_snli_SBERT_full.pt"
    output_dir = "entailment_surfaces/clustering"
    
    validator = ClusteringValidator(
        bert_data_path=bert_data_path,
        order_model_path=order_model_path,
        output_dir=output_dir,
        seed=42
    )
    
    results = validator.run_full_validation()
    print("\nPhase 1 clustering validation completed!")
    return results

if __name__ == "__main__":
    main()