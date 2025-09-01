"""
Phase 1 Statistical Validation: Test Top Performers with More Samples
Increase sample size for statistical significance testing of clustering accuracy
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy import stats
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import copy
warnings.filterwarnings('ignore')

# Import necessary components from the clustering validation module
from phdim_clustering import ClusteringValidator, ph_dim_and_diagrams_from_distance_matrix

# Convert numpy types for JSON serialization
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

class StatisticalValidation:
    """Statistical validation of top performing combinations with larger sample sizes"""
    
    def __init__(self, base_validator, output_dir: str = "entailment_surfaces/statistical_validation"):
        self.base_validator = base_validator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Top performers to test with more samples
        self.top_combinations = [
            ('sbert_concat', 'euclidean'),
            ('sbert_concat', 'chebyshev'),
            ('sbert_concat', 'cosine'),
            ('sbert_concat', 'minkowski_3'),
            ('sbert_concat', 'minkowski_4'),
            ('sbert_concat', 'canberra'),
            ('sbert_concat', 'braycurtis'),
            ('lattice_containment', 'euclidean'),
            ('lattice_containment', 'chebyshev'),
            ('lattice_containment', 'cosine'),
            ('lattice_containment', 'minkowski_3'),
            ('lattice_containment', 'minkowski_4'),
            ('lattice_containment', 'canberra'),
            ('lattice_containment', 'braycurtis'),
            ('order_concat', 'euclidean'),
            ('order_concat', 'chebyshev'),
            ('order_concat', 'cosine'),
            ('order_concat', 'minkowski_3'),
            ('order_concat', 'minkowski_4'),
            ('order_concat', 'canberra'),
            ('order_concat', 'braycurtis'),
            ('hyperbolic_concat', 'euclidean'),
            ('hyperbolic_concat', 'chebyshev'),
            ('hyperbolic_concat', 'cosine'),
            ('hyperbolic_concat', 'minkowski_3'),
            ('hyperbolic_concat', 'minkowski_4'),
            ('hyperbolic_concat', 'canberra'),
            ('hyperbolic_concat', 'braycurtis'),
        ]
        
        # Sample size for statistical validation
        self.sample_size = 10  # 10 independent validation runs
        
        print(f"Statistical Validation initialized for {len(self.top_combinations)} combinations")
        print(f"Testing with {self.sample_size} independent validation runs")
        print(f"Each run uses 30 samples (10 per class) for clustering")
        print(f"Output directory: {self.output_dir}")
    
    def statistical_validation_single_combination(self, space_name: str, metric_name: str) -> Dict:
        """
        Run statistical validation for a single embedding space + metric combination
        """
        print(f"\nStatistical Validation: {space_name} + {metric_name}")
        
        # Generate embeddings - only keep the spaces we need
        print(f"  Generating embedding spaces...")
        all_embeddings = self.base_validator.generate_embedding_spaces_by_class()
        
        # Only keep the embedding spaces we're testing
        needed_spaces = ['sbert_concat', 'lattice_containment']
        filtered_embeddings = {space: all_embeddings[space] for space in needed_spaces if space in all_embeddings}
        
        # Delete the full embeddings dictionary to save memory
        del all_embeddings
        
        # Get the specific space we're testing
        space_embeddings = filtered_embeddings[space_name]
        
        print(f"  Memory optimization: Only keeping {len(filtered_embeddings)} embedding spaces")
        
        print(f"  Testing with {self.sample_size} independent validation runs...")
        
        # Generate samples
        sample_results = self._run_multiple_samples(
            space_name, metric_name, space_embeddings, self.sample_size
        )
        
        # Calculate statistics
        stats_summary = self._calculate_sample_statistics(sample_results)
        
        print(f"    Accuracy: {stats_summary['accuracy_mean']:.3f} ± {stats_summary['accuracy_std']:.3f}")
        print(f"    Silhouette: {stats_summary['silhouette_mean']:.3f} ± {stats_summary['silhouette_std']:.3f}")
        print(f"    Perfect clustering rate: {stats_summary['perfect_clustering_rate']:.1%}")
        
        # Statistical significance testing
        significance_tests = self._run_significance_tests(stats_summary)
        
        return {
            'space': space_name,
            'metric': metric_name,
            'sample_results': sample_results,
            'statistics': stats_summary,
            'significance_tests': significance_tests
        }
    
    def _run_multiple_samples(self, space_name: str, metric_name: str, 
                            space_embeddings: Dict, n_samples: int) -> List[Dict]:
        """Run clustering validation with multiple independent samples"""
        
        sample_results = []
        
        for sample_idx in range(n_samples):
            if sample_idx % 10 == 0 and sample_idx > 0:
                print(f"    Sample {sample_idx}/{n_samples}")
            
            # Generate independent sample indices
            sample_indices = self._generate_independent_sample_indices(sample_idx)
            
            # Run single validation
            result = self._run_single_sample_validation(
                space_name, metric_name, space_embeddings, sample_indices
            )
            
            if result is not None:
                sample_results.append(result)
        
        return sample_results
    
    def _generate_independent_sample_indices(self, sample_idx: int) -> Dict:
        """Generate independent sample indices for each class"""
        # Use different seed for each sample to ensure independence
        np.random.seed(42 + sample_idx)
        
        sample_indices = {}
        
        # Generate multiple samples per class to have enough for clustering
        samples_per_class = 10  # Generate 10 samples per class = 30 total samples
        
        for class_name in ['entailment', 'neutral', 'contradiction']:
            class_samples = []
            
            for i in range(samples_per_class):
                # Generate sample of 1000 points for this class
                indices = np.random.choice(10000, 1000, replace=False)
                
                class_samples.append({
                    'indices': indices,
                    'actual_points': 1000,
                    'sample_idx': f"{sample_idx}_{i}",
                    'class_name': class_name,
                    'class_idx': ['entailment', 'neutral', 'contradiction'].index(class_name)
                })
            
            sample_indices[class_name] = class_samples
        
        return sample_indices
    
    def _run_single_sample_validation(self, space_name: str, metric_name: str,
                                    space_embeddings: Dict, sample_indices: Dict) -> Dict:
        """Run clustering validation on a single independent sample"""
        
        try:
            all_persistence_images = []
            sample_labels = []
            ph_dim_values = {'entailment': [], 'neutral': [], 'contradiction': []}
            
            # Process each class
            for class_name in ['entailment', 'neutral', 'contradiction']:
                class_embeddings = space_embeddings[class_name]
                class_sample_indices = sample_indices[class_name]
                class_idx = ['entailment', 'neutral', 'contradiction'].index(class_name)
                
                for sample_info in class_sample_indices:
                    indices = sample_info['indices']
                    
                    # Extract sample
                    sampled_embeddings = class_embeddings[indices]
                    
                    # Compute distance matrix
                    distance_matrix = self.base_validator.analyzer.compute_distance_matrix_advanced(
                        sampled_embeddings, metric_name
                    )
                    
                    # Get PH-dim and persistence diagrams
                    ph_dim, all_diagrams = ph_dim_and_diagrams_from_distance_matrix(
                        distance_matrix,
                        min_points=200,
                        max_points=1000,
                        point_jump=50,
                        h_dim=0,
                        alpha=1.0
                        #seed=42 <--- REMOVE: this was causing the global seed state to change unintentionally
                    )
                    
                    ph_dim_values[class_name].append(ph_dim)
                    
                    # Convert to persistence image
                    if len(all_diagrams) > 0:
                        persistence_image = self.base_validator.persistence_diagrams_to_images([all_diagrams[0]])
                        if len(persistence_image) > 0:
                            all_persistence_images.append(persistence_image[0])
                            sample_labels.append(class_idx)
            
            # Perform clustering - need at least 3 samples but silhouette needs more
            if len(all_persistence_images) >= 3:
                # Custom clustering analysis that handles silhouette score edge case
                accuracy, sil_score, ari_score = self._safe_clustering_analysis(
                    all_persistence_images, sample_labels
                )
                
                return {
                    'accuracy': accuracy,
                    'silhouette': sil_score,
                    'ari': ari_score,
                    'ph_dim_values': ph_dim_values,
                    'perfect_clustering': accuracy == 1.0
                }
        
        except Exception as e:
            print(f"      Sample validation failed: {e}")
            return None
        
        return None
    
    def _safe_clustering_analysis(self, persistence_images: List[np.ndarray], 
                                true_labels: List[int]) -> Tuple[float, float, float]:
        """Perform clustering analysis with safe silhouette score handling"""
        
        if len(persistence_images) < 3:
            return 0.0, 0.0, 0.0
        
        X = np.vstack(persistence_images)
        
        # K-means with reproducible seed
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        predicted_labels = kmeans.fit_predict(X)
        
        # Calculate accuracy
        accuracy = self.base_validator._calculate_clustering_accuracy(true_labels, predicted_labels)
        
        # Calculate silhouette score safely
        try:
            if len(set(predicted_labels)) > 1 and len(X) > 3:
                sil_score = silhouette_score(X, predicted_labels)
            else:
                sil_score = 0.0  # Default when silhouette can't be calculated
        except:
            sil_score = 0.0
        
        # Calculate ARI
        ari_score = adjusted_rand_score(true_labels, predicted_labels)
        
        return accuracy, sil_score, ari_score
    
    def _calculate_sample_statistics(self, sample_results: List[Dict]) -> Dict:
        """Calculate statistical summary of multiple sample results"""
        
        if not sample_results:
            return {
                'n_samples': 0,
                'accuracy_mean': 0.0,
                'accuracy_std': 0.0,
                'silhouette_mean': 0.0,
                'silhouette_std': 0.0,
                'perfect_clustering_rate': 0.0,
                'ph_dim_stats': {}
            }
        
        # Extract metrics
        accuracies = [r['accuracy'] for r in sample_results]
        silhouettes = [r['silhouette'] for r in sample_results]
        perfect_clustering = [r['perfect_clustering'] for r in sample_results]
        
        # PH-dimension statistics
        ph_dim_stats = {}
        for class_name in ['entailment', 'neutral', 'contradiction']:
            all_ph_dims = []
            for result in sample_results:
                all_ph_dims.extend(result['ph_dim_values'][class_name])
            
            if all_ph_dims:
                ph_dim_stats[class_name] = {
                    'mean': float(np.mean(all_ph_dims)),
                    'std': float(np.std(all_ph_dims)),
                    'min': float(np.min(all_ph_dims)),
                    'max': float(np.max(all_ph_dims)),
                    'n_samples': len(all_ph_dims)
                }
        
        return {
            'n_samples': len(sample_results),
            'accuracy_mean': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
            'accuracy_ci_95': (float(np.percentile(accuracies, 2.5)), float(np.percentile(accuracies, 97.5))),
            'silhouette_mean': float(np.mean(silhouettes)),
            'silhouette_std': float(np.std(silhouettes)),
            'silhouette_ci_95': (float(np.percentile(silhouettes, 2.5)), float(np.percentile(silhouettes, 97.5))),
            'perfect_clustering_rate': float(np.mean(perfect_clustering)),
            'ph_dim_stats': ph_dim_stats
        }
    
    def _run_significance_tests(self, stats_summary: Dict) -> Dict:
        """Run statistical significance tests on sample results"""
        
        significance_tests = {}
        
        # Test if perfect clustering rate is significantly > 0.7 (our success threshold)
        perfect_rate = stats_summary['perfect_clustering_rate']
        n_samples = stats_summary['n_samples']
        
        if n_samples > 0:
            # One-sample proportion test against 0.7
            perfect_count = int(perfect_rate * n_samples)
            
            # Using binomial test
            p_value = stats.binom_test(perfect_count, n_samples, 0.7, alternative='greater')
            
            significance_tests['perfect_clustering_vs_threshold'] = {
                'perfect_rate': perfect_rate,
                'perfect_count': perfect_count,
                'n_samples': n_samples,
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'significantly_better_than_70_percent': p_value < 0.05
            }
        
        return significance_tests
    
    def run_statistical_validation(self) -> Dict:
        """Run complete statistical validation on all top combinations"""
        print("="*60)
        print("PHASE 1 STATISTICAL VALIDATION: ROBUST STATISTICAL TESTING")
        print("="*60)
        print(f"Testing {self.sample_size} independent validation runs per combination")
        print("Each run uses 30 samples (10 per class) for clustering")
        print("Computing confidence intervals and significance tests")
        
        validation_results = {}
        
        for space_name, metric_name in self.top_combinations:
            print(f"\n{'='*60}")
            print(f"STATISTICAL VALIDATION: {space_name} + {metric_name}")
            print(f"{'='*60}")
            
            try:
                result = self.statistical_validation_single_combination(space_name, metric_name)
                validation_results[f"{space_name}_{metric_name}"] = result
                
                # Print summary
                self._print_combination_summary(result)
                
            except Exception as e:
                print(f"  ERROR in statistical validation: {e}")
                validation_results[f"{space_name}_{metric_name}"] = {
                    'error': str(e)
                }
        
        # Save results
        self._save_statistical_results(validation_results)
        
        # Generate summary report
        self._generate_statistical_summary(validation_results)
        
        return validation_results
    
    def _print_combination_summary(self, result: Dict):
        """Print summary for a single combination"""
        print(f"\n  SUMMARY for {result['space']} + {result['metric']}:")
        
        stats = result['statistics']
        print(f"    {stats['n_samples']} samples: Acc={stats['accuracy_mean']:.3f}±{stats['accuracy_std']:.3f}, "
              f"Perfect={stats['perfect_clustering_rate']:.1%}")
        
        # Significance tests
        if 'significance_tests' in result:
            sig_test = result['significance_tests'].get('perfect_clustering_vs_threshold', {})
            if sig_test:
                sig_status = "YES" if sig_test.get('significantly_better_than_70_percent', False) else "NO"
                print(f"    Significantly > 70%: {sig_status} (p={sig_test.get('p_value', 1.0):.4f})")
                print(f"    Perfect clustering: {sig_test.get('perfect_count', 0)}/{sig_test.get('n_samples', 0)} runs")
    
    def _save_statistical_results(self, validation_results: Dict):
        """Save statistical validation results"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"sbert_concat_statistical_validation_{timestamp}.json"
        
        try:
            results_to_save = copy.deepcopy(validation_results)
            
            with open(results_file, 'w') as f:
                json.dump(results_to_save, f, indent=2, default=convert_numpy_types)
            
            print(f"\nStatistical validation results saved to: {results_file}")
            
        except ValueError as e:
            print(f"\nJSON serialization error: {e}")
            print("Attempting to save with simplified data structure...")
            
            # Fallback: save only essential statistics
            simplified_results = {}
            for combo_name, results in validation_results.items():
                if 'error' in results:
                    simplified_results[combo_name] = {'error': results['error']}
                else:
                    simplified_results[combo_name] = {
                        'space': results.get('space', ''),
                        'metric': results.get('metric', ''),
                        'statistics': copy.deepcopy(results.get('statistics', {})),
                        'significance_tests': copy.deepcopy(results.get('significance_tests', {}))
                    }
            
            # Try again with simplified data
            with open(results_file, 'w') as f:
                json.dump(simplified_results, f, indent=2, default=convert_numpy_types)
            
            print(f"Simplified results saved to: {results_file}")
    
    def _generate_statistical_summary(self, validation_results: Dict):
        """Generate statistical summary report"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"mnli_statistical_summary_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("PHASE 1 STATISTICAL VALIDATION SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            f.write("STATISTICAL SIGNIFICANCE RESULTS:\n")
            f.write("-" * 40 + "\n")
            
            for combo_name, results in validation_results.items():
                if 'error' in results:
                    continue
                
                f.write(f"\n{combo_name}:\n")
                
                # Statistics
                if 'statistics' in results:
                    stats = results['statistics']
                    f.write(f"  {stats['n_samples']} samples: {stats['accuracy_mean']:.3f}±{stats['accuracy_std']:.3f} "
                           f"({stats['perfect_clustering_rate']:.1%} perfect)\n")
                
                # Significance tests
                if 'significance_tests' in results:
                    sig_test = results['significance_tests'].get('perfect_clustering_vs_threshold', {})
                    if sig_test:
                        sig_status = "SIGNIFICANT" if sig_test.get('significantly_better_than_70_percent', False) else "not significant"
                        f.write(f"  Statistical significance: {sig_status} (p={sig_test.get('p_value', 1.0):.4f})\n")
                        f.write(f"  Perfect clustering: {sig_test.get('perfect_count', 0)}/{sig_test.get('n_samples', 0)} runs\n")
                
                f.write("\n")
            
            f.write("RECOMMENDATION:\n")
            f.write("-" * 15 + "\n")
            f.write("Combinations with highest statistical confidence for Phase 2\n")
        
        print(f"Statistical summary saved to: {report_file}")


def main():
    """Run statistical validation with increased sample sizes"""
    
    bert_data_path = "data/processed/mnli_full_SBERT_train.pt"
    order_model_path = "models/enhanced_order_embeddings_snli_SBERT_full.pt"
    
    # Initialize base validator
    base_validator = ClusteringValidator(
        bert_data_path=bert_data_path,
        order_model_path=order_model_path,
        output_dir="entailment_surfaces/clustering",
        seed=42
    )
    
    # Run statistical validation
    statistical_validator = StatisticalValidation(base_validator)
    results = statistical_validator.run_statistical_validation()
    
    print("\nStatistical Validation completed!")
    return results

if __name__ == "__main__":
    main()  