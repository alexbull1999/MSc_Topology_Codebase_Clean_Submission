"""
Raw Embedding Statistical Validation: Test clustering on pure embeddings without topology
This tests the same combinations but directly on SBERT/lattice embeddings (no persistence homology)
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from scipy import stats
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import copy
from datetime import datetime
warnings.filterwarnings('ignore')

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

class RawEmbeddingValidator:
    """Statistical validation of clustering on raw embeddings without topology"""
    
    def __init__(self, sbert_data_path: str, output_dir: str = "entailment_surfaces/raw_embedding_validation"):
        self.sbert_data_path = sbert_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Same combinations as the topological version for direct comparison
        self.test_combinations = [
            ('sbert_concat', 'euclidean'),
            ('sbert_concat', 'chebyshev'), 
            ('sbert_concat', 'cosine'),
            ('sbert_concat', 'minkowski_3'),
            ('sbert_concat', 'minkowski_4'),
            ('sbert_concat', 'braycurtis'),
            ('sbert_concat', 'canberra'),
            ('lattice_containment', 'euclidean'),
            ('lattice_containment', 'chebyshev'),
            ('lattice_containment', 'cosine'),
            ('lattice_containment', 'minkowski_3'),
            ('lattice_containment', 'minkowski_4'),
            ('lattice_containment', 'braycurtis'),
            ('lattice_containment', 'canberra')
        ]
        
        self.sample_size = 10
        
        print(f"Raw Embedding Validator initialized for {len(self.test_combinations)} combinations")
        print(f"Testing with {self.sample_size} independent validation runs")
        print(f"Each run uses 30 samples (10 per class) for clustering")
        print(f"Output directory: {self.output_dir}")
    
    def load_data(self) -> Dict:
        """Load BERT data and organize by class"""
        print("Loading BERT data...")
        data = torch.load(self.sbert_data_path)
        
        data_by_class = {
            'entailment': {'premise_bert': [], 'hypothesis_bert': []},
            'neutral': {'premise_bert': [], 'hypothesis_bert': []},
            'contradiction': {'premise_bert': [], 'hypothesis_bert': []}
        }
        
        # Check if labels are already strings or integers
        sample_label = data['labels'][0]
        if isinstance(sample_label, str):
            # Labels are already strings
            for i, label in enumerate(data['labels']):
                if label in data_by_class:
                    data_by_class[label]['premise_bert'].append(data['premise_embeddings'][i])
                    data_by_class[label]['hypothesis_bert'].append(data['hypothesis_embeddings'][i])
        else:
            # Labels are integers, need mapping
            label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
            for i, label_idx in enumerate(data['labels']):
                label = label_map[int(label_idx)]
                data_by_class[label]['premise_bert'].append(data['premise_embeddings'][i])
                data_by_class[label]['hypothesis_bert'].append(data['hypothesis_embeddings'][i])
        
        # Convert to tensors
        for label in data_by_class:
            if len(data_by_class[label]['premise_bert']) > 0:
                data_by_class[label]['premise_bert'] = torch.stack(data_by_class[label]['premise_bert'])
                data_by_class[label]['hypothesis_bert'] = torch.stack(data_by_class[label]['hypothesis_bert'])
            else:
                print(f"Warning: No samples found for class {label}")
        
        print(f"Loaded data: {len(data['labels'])} samples")
        for label, class_data in data_by_class.items():
            if len(class_data['premise_bert']) > 0:
                print(f"  {label}: {len(class_data['premise_bert'])} samples")
            else:
                print(f"  {label}: 0 samples")
        
        return data_by_class
    
    def generate_raw_embeddings(self, data_by_class: Dict) -> Dict:
        """Generate raw embedding spaces without any topological processing"""
        print("Generating raw embedding spaces...")
        
        embeddings_by_space = {}
        
        for space in ['sbert_concat', 'lattice_containment']:
            print(f"  Processing {space}...")
            space_embeddings = {}
            
            for label in data_by_class.keys():
                if space == 'sbert_concat':
                    space_embeddings[label] = torch.cat([
                        data_by_class[label]['premise_bert'],
                        data_by_class[label]['hypothesis_bert']
                    ], dim=1).cpu()
                
                elif space == 'lattice_containment':
                    epsilon = 1e-8
                    premise_bert = data_by_class[label]['premise_bert']
                    hypothesis_bert = data_by_class[label]['hypothesis_bert']
                    space_embeddings[label] = ((premise_bert * hypothesis_bert) / 
                                             (torch.abs(premise_bert) + torch.abs(hypothesis_bert) + epsilon)).cpu()
            
            embeddings_by_space[space] = space_embeddings
        
        print("Raw embedding spaces generated")
        return embeddings_by_space
    
    def compute_distance_matrix(self, embeddings: torch.Tensor, metric: str) -> np.ndarray:
        """
        Compute distance matrix using advanced metrics
        
        Args:
            embeddings: Embedding tensor [n_samples, embed_dim]
            metric: Distance metric name
            
        Returns:
            Distance matrix [n_samples, n_samples]
        """
        # OPTIMIZATION: Only convert to numpy when necessary
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
    
    def generate_independent_sample_indices(self, sample_idx: int) -> Dict:
        """Generate independent sample indices for each class"""
        np.random.seed(42 + sample_idx)
        
        sample_indices = {}
        samples_per_class = 10
        
        for class_name in ['entailment', 'neutral', 'contradiction']:
            class_samples = []
            
            for i in range(samples_per_class):
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
    
    def run_single_sample_validation(self, space_name: str, metric_name: str,
                                   space_embeddings: Dict, sample_indices: Dict) -> Dict:
        """Run clustering validation on a single independent sample using raw embeddings"""
        
        try:
            all_embeddings = []
            sample_labels = []
            
            # Process each class - use 1000 raw embeddings per class (3000 total)
            for class_name in ['entailment', 'neutral', 'contradiction']:
                class_embeddings = space_embeddings[class_name]
                class_idx = ['entailment', 'neutral', 'contradiction'].index(class_name)
                
                # Take 1000 random embeddings from this class
                sample_indices_for_class = np.random.choice(len(class_embeddings), 1000, replace=False)
                sampled_embeddings = class_embeddings[sample_indices_for_class]
                
                # Add each individual embedding (no aggregation)
                for emb in sampled_embeddings:
                    all_embeddings.append(emb.numpy())
                    sample_labels.append(class_idx)
            
            # Perform clustering directly on 3000 raw embeddings
            if len(all_embeddings) >= 3:
                accuracy, sil_score, ari_score = self.safe_clustering_analysis(
                    all_embeddings, sample_labels, metric_name
                )
                
                return {
                    'accuracy': accuracy,
                    'silhouette': sil_score,
                    'ari': ari_score,
                    'perfect_clustering': accuracy == 1.0
                }
        
        except Exception as e:
            print(f"      Sample validation failed: {e}")
            return None
        
        return None
    
    def safe_clustering_analysis(self, embeddings: List[np.ndarray], 
                               true_labels: List[int], metric: str) -> Tuple[float, float, float]:
        """Perform k-means clustering analysis on raw embeddings using specified distance metric"""
        
        if len(embeddings) < 3:
            return 0.0, 0.0, 0.0
        
        X = np.vstack(embeddings)
        
        # Use k-means for all metrics (same as topological version)
        # For non-euclidean metrics, we'll use the distance matrix in the silhouette calculation
        # but still use standard k-means clustering (which uses euclidean internally)
        # This matches what was likely done in the topological version
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        predicted_labels = kmeans.fit_predict(X)
        
        # Calculate accuracy
        accuracy = self.calculate_clustering_accuracy(true_labels, predicted_labels)
        
        # Calculate silhouette score using the specified metric
        try:
            if len(set(predicted_labels)) > 1 and len(X) > 3:
                if metric == 'euclidean':
                    sil_score = silhouette_score(X, predicted_labels)
                else:
                    # Use precomputed distance matrix for silhouette with custom metric
                    distances = self.compute_distance_matrix(torch.tensor(X), metric)
                    sil_score = silhouette_score(distances, predicted_labels, metric='precomputed')
            else:
                sil_score = 0.0
        except:
            sil_score = 0.0
        
        # Calculate ARI
        ari_score = adjusted_rand_score(true_labels, predicted_labels)
        
        return accuracy, sil_score, ari_score
    
    def calculate_clustering_accuracy(self, true_labels: List[int], predicted_labels: List[int]) -> float:
        """Calculate clustering accuracy with optimal label assignment"""
        from itertools import permutations
        
        n_classes = 3
        best_accuracy = 0.0
        
        # Try all possible label mappings
        for perm in permutations(range(n_classes)):
            mapped_predictions = [perm[label] for label in predicted_labels]
            accuracy = np.mean([true == pred for true, pred in zip(true_labels, mapped_predictions)])
            best_accuracy = max(best_accuracy, accuracy)
        
        return best_accuracy
    
    def run_multiple_samples(self, space_name: str, metric_name: str, 
                           space_embeddings: Dict, n_samples: int) -> List[Dict]:
        """Run clustering validation with multiple independent samples"""
        
        sample_results = []
        
        for sample_idx in range(n_samples):
            if sample_idx % 5 == 0 and sample_idx > 0:
                print(f"    Sample {sample_idx}/{n_samples}")
            
            sample_indices = self.generate_independent_sample_indices(sample_idx)
            
            result = self.run_single_sample_validation(
                space_name, metric_name, space_embeddings, sample_indices
            )
            
            if result is not None:
                sample_results.append(result)
        
        return sample_results
    
    def calculate_sample_statistics(self, sample_results: List[Dict]) -> Dict:
        """Calculate statistical summary of multiple sample results"""
        
        if not sample_results:
            return {
                'n_samples': 0,
                'accuracy_mean': 0.0,
                'accuracy_std': 0.0,
                'silhouette_mean': 0.0,
                'silhouette_std': 0.0,
                'perfect_clustering_rate': 0.0
            }
        
        accuracies = [r['accuracy'] for r in sample_results]
        silhouettes = [r['silhouette'] for r in sample_results]
        perfect_clustering = [r['perfect_clustering'] for r in sample_results]
        
        return {
            'n_samples': len(sample_results),
            'accuracy_mean': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
            'accuracy_ci_95': (float(np.percentile(accuracies, 2.5)), float(np.percentile(accuracies, 97.5))),
            'silhouette_mean': float(np.mean(silhouettes)),
            'silhouette_std': float(np.std(silhouettes)),
            'silhouette_ci_95': (float(np.percentile(silhouettes, 2.5)), float(np.percentile(silhouettes, 97.5))),
            'perfect_clustering_rate': float(np.mean(perfect_clustering))
        }
    
    def run_significance_tests(self, stats_summary: Dict) -> Dict:
        """Run statistical significance tests on sample results"""
        
        significance_tests = {}
        
        perfect_rate = stats_summary['perfect_clustering_rate']
        n_samples = stats_summary['n_samples']
        
        if n_samples > 0:
            perfect_count = int(perfect_rate * n_samples)
            
            # Using binomial test against 70% threshold
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
    
    def statistical_validation_single_combination(self, space_name: str, metric_name: str,
                                                 embeddings_by_space: Dict) -> Dict:
        """Run statistical validation for a single embedding space + metric combination"""
        print(f"\nRaw Embedding Validation: {space_name} + {metric_name}")
        
        space_embeddings = embeddings_by_space[space_name]
        
        print(f"  Testing with {self.sample_size} independent validation runs...")
        
        sample_results = self.run_multiple_samples(
            space_name, metric_name, space_embeddings, self.sample_size
        )
        
        stats_summary = self.calculate_sample_statistics(sample_results)
        
        print(f"    Accuracy: {stats_summary['accuracy_mean']:.3f} ± {stats_summary['accuracy_std']:.3f}")
        print(f"    Silhouette: {stats_summary['silhouette_mean']:.3f} ± {stats_summary['silhouette_std']:.3f}")
        print(f"    Perfect clustering rate: {stats_summary['perfect_clustering_rate']:.1%}")
        
        significance_tests = self.run_significance_tests(stats_summary)
        
        return {
            'space': space_name,
            'metric': metric_name,
            'sample_results': sample_results,
            'statistics': stats_summary,
            'significance_tests': significance_tests
        }
    
    def print_combination_summary(self, result: Dict):
        """Print summary for a single combination"""
        print(f"\n  SUMMARY for {result['space']} + {result['metric']}:")
        
        stats = result['statistics']
        print(f"    {stats['n_samples']} samples: Acc={stats['accuracy_mean']:.3f}±{stats['accuracy_std']:.3f}, "
              f"Perfect={stats['perfect_clustering_rate']:.1%}")
        
        if 'significance_tests' in result:
            sig_test = result['significance_tests'].get('perfect_clustering_vs_threshold', {})
            if sig_test:
                sig_status = "YES" if sig_test.get('significantly_better_than_70_percent', False) else "NO"
                print(f"    Significantly > 70%: {sig_status} (p={sig_test.get('p_value', 1.0):.4f})")
                print(f"    Perfect clustering: {sig_test.get('perfect_count', 0)}/{sig_test.get('n_samples', 0)} runs")
    
    def save_results(self, validation_results: Dict):
        """Save validation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"raw_embedding_validation_{timestamp}.json"
        
        try:
            results_to_save = copy.deepcopy(validation_results)
            
            with open(results_file, 'w') as f:
                json.dump(results_to_save, f, indent=2, default=convert_numpy_types)
            
            print(f"\nRaw embedding validation results saved to: {results_file}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def generate_summary_report(self, validation_results: Dict):
        """Generate summary report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"raw_embedding_summary_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("RAW EMBEDDING CLUSTERING VALIDATION SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write("Testing K-means clustering directly on raw embeddings (no topology)\n")
            f.write("Compare with topological results to show topology's contribution\n\n")
            
            f.write("RESULTS:\n")
            f.write("-" * 20 + "\n")
            
            for combo_name, results in validation_results.items():
                if 'error' in results:
                    continue
                
                f.write(f"\n{combo_name}:\n")
                
                if 'statistics' in results:
                    stats = results['statistics']
                    f.write(f"  {stats['n_samples']} samples: {stats['accuracy_mean']:.3f}±{stats['accuracy_std']:.3f} "
                           f"({stats['perfect_clustering_rate']:.1%} perfect)\n")
                
                if 'significance_tests' in results:
                    sig_test = results['significance_tests'].get('perfect_clustering_vs_threshold', {})
                    if sig_test:
                        sig_status = "SIGNIFICANT" if sig_test.get('significantly_better_than_70_percent', False) else "not significant"
                        f.write(f"  Statistical significance: {sig_status} (p={sig_test.get('p_value', 1.0):.4f})\n")
            
            f.write("\n\nCOMPARISON NOTE:\n")
            f.write("If topological processing achieved 100% accuracy while raw embeddings show lower accuracy,\n")
            f.write("this demonstrates that the topological structure was crucial for perfect clustering.\n")
        
        print(f"Summary report saved to: {report_file}")
    
    def run_validation(self) -> Dict:
        """Run complete validation on raw embeddings"""
        print("="*70)
        print("RAW EMBEDDING CLUSTERING VALIDATION")
        print("="*70)
        print("Testing K-means clustering directly on raw embeddings (no topology)")
        print(f"Testing {self.sample_size} independent validation runs per combination")
        
        # Load data
        data_by_class = self.load_data()
        
        # Generate raw embeddings
        embeddings_by_space = self.generate_raw_embeddings(data_by_class)
        
        validation_results = {}
        
        for space_name, metric_name in self.test_combinations:
            print(f"\n{'='*70}")
            print(f"RAW EMBEDDING VALIDATION: {space_name} + {metric_name}")
            print(f"{'='*70}")
            
            try:
                result = self.statistical_validation_single_combination(
                    space_name, metric_name, embeddings_by_space
                )
                validation_results[f"{space_name}_{metric_name}"] = result
                
                self.print_combination_summary(result)
                
            except Exception as e:
                print(f"  ERROR in validation: {e}")
                validation_results[f"{space_name}_{metric_name}"] = {
                    'error': str(e)
                }
        
        # Save results
        self.save_results(validation_results)
        
        # Generate summary report
        self.generate_summary_report(validation_results)
        
        return validation_results


def main():
    """Run raw embedding validation"""
    
    sbert_data_path = "data/processed/mnli_full_SBERT_train.pt"
    
    validator = RawEmbeddingValidator(
        sbert_data_path=sbert_data_path,
        output_dir="entailment_surfaces/raw_embedding_validation"
    )
    
    results = validator.run_validation()
    
    print("\nRaw Embedding Validation completed!")
    print("\nCompare these results with your topological clustering results to demonstrate")
    print("the contribution of persistent homology to achieving 100% clustering accuracy.")
    
    return results

if __name__ == "__main__":
    main()