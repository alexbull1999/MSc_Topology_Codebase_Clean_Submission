"""
Analyze similarity of persistence diagrams within each class from the 100% clustering results.
This helps determine if the diagrams are similar enough to average into prototypes.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional
import sys
import os
from collections import defaultdict
from gph.python import ripser_parallel
from persim import PersistenceImager, bottleneck, wasserstein
import time
import gc

# Add path to existing functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phd_method', 'src_phd'))
from sklearn.metrics.pairwise import pairwise_distances

def ph_dim_and_diagrams_from_distance_matrix(dm: np.ndarray,
                                           min_points=200,
                                           max_points=1000,
                                           point_jump=50,
                                           h_dim=0,
                                           alpha: float = 1.,
                                           seed: int = 42) -> Tuple[float, List[np.ndarray]]:
    """
    Compute both PH dimension and persistence diagrams from distance matrix
    Adapted from your existing function to return both
    """

    assert dm.ndim == 2, dm
    assert dm.shape[0] == dm.shape[1], dm.shape
    
    # np.random.seed(seed)
    
    test_n = range(min_points, max_points, point_jump)
    lengths = []
    all_diagrams = []
    
    for points_number in test_n:
        sample_indices = np.random.choice(dm.shape[0], points_number, replace=False)
        dist_matrix = dm[sample_indices, :][:, sample_indices]
        
        # Compute persistence diagrams (both H0 and H1)
        result = ripser_parallel(dist_matrix, maxdim=1, n_threads=-1, metric="precomputed")
        diagrams = result['dgms']
        
        # Store diagrams for this sample size
        all_diagrams.append({
            'n_points': points_number,
            'H0': diagrams[0],
            'H1': diagrams[1]
        })
        
        # Compute persistence lengths for PH dimension calculation
        d = diagrams[h_dim]
        d = d[d[:, 1] < np.inf]
        lengths.append(np.power((d[:, 1] - d[:, 0]), alpha).sum())
    
    lengths = np.array(lengths)
    
    # Compute PH dimension
    x = np.log(np.array(list(test_n)))
    y = np.log(lengths)
    N = len(x)
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    b = y.mean() - m * x.mean()
    
    phd_score = alpha / (1 - m)
    
    # Return the diagrams from the largest sample size (most stable)
    final_diagrams = all_diagrams[-1]  # Last one has max_points
    
    return phd_score, [final_diagrams['H0'], final_diagrams['H1']]



class PersistenceDiagramCollector:
    """
    Collect persistence diagrams using the same methodology as phdim_clustering_validation_best_metrics.py
    """
    
    def __init__(self, embedding_space='sbert_concat', distance_metric='euclidean', 
                 bert_data_path=None, device='cuda'):
        self.embedding_space = embedding_space
        self.distance_metric = distance_metric
        self.bert_data_path = bert_data_path
        self.device = device
        self.bert_data = None
        
        # Parameters matching your successful clustering
        self.phd_params = {
            'min_points': 200,
            'max_points': 1000,
            'point_jump': 50,
            'h_dim': 0,
            'alpha': 1.0
        }
        
    def load_data(self):
        """Load BERT data same as your existing files"""
        if self.bert_data_path is None:
            # Use default path from your setup
            self.bert_data_path = 'data/processed/snli_full_standard_SBERT.pt'
        
        self.bert_data = torch.load(self.bert_data_path, map_location=self.device, weights_only=False)
        print(f"Loaded BERT data: {self.bert_data['premise_embeddings'].shape}")
        labels_list = self.bert_data['labels']
        unique_labels = sorted(list(set(labels_list))) # e.g., ['contradiction', 'entailment', 'neutral']
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        int_labels = [label_to_int[label] for label in labels_list]
        labels_tensor = torch.tensor(int_labels)
        print(f"Labels distribution: {torch.bincount(labels_tensor)}")
        print(f"Label mapping used for bincount: {label_to_int}") 

        
    def extract_class_embeddings(self, class_idx, n_points=1000):
        """
        Extract embeddings for a specific class
        
        Args:
            class_idx: 0=entailment, 1=neutral, 2=contradiction
            n_points: Number of points to sample
            use_current_random_state: If True, use current numpy random state instead of reseeding
        """
        if self.bert_data is None:
            self.load_data()
            
        # Get indices for this class
        if class_idx == 0:
            class_name = 'entailment'
        elif class_idx == 1:
            class_name = 'neutral'
        elif class_idx == 2:
            class_name = 'contradiction'
        else:
            raise ValueError 

        class_mask = torch.tensor([label == class_name for label in self.bert_data['labels']])
        class_indices = torch.where(class_mask)[0]
        
        # Sample n_points randomly
        if len(class_indices) > n_points:
            sampled_indices = np.random.choice(class_indices.cpu().numpy(), n_points, replace=False)
        else:
            sampled_indices = class_indices.cpu().numpy()
        
        # Extract embeddings based on embedding space
        if self.embedding_space == 'sbert_concat':
            premise_emb = self.bert_data['premise_embeddings'][sampled_indices]
            hypothesis_emb = self.bert_data['hypothesis_embeddings'][sampled_indices]
            embeddings = torch.cat([premise_emb, hypothesis_emb], dim=1)
        elif self.embedding_type == 'lattice':
            # Original lattice containment formula
            embeddings = (premise_batch * hypothesis_batch) / (torch.abs(premise_batch) + torch.abs(hypothesis_batch) + self.epsilon)
        else:
            raise NotImplementedError(f"Embedding space {self.embedding_space} not implemented")
        
        return embeddings.detach().cpu().numpy()

    def collect_persistence_diagrams(self, n_tests=10, n_samples_per_test=10):
        """
        Collect persistence diagrams using the same methodology as your 100% clustering
        10 independent tests, each with 10 samples of 1000 points per class
        
        Following the exact seeding procedure from phdim_clustering_validation_best_metrics.py:
        - Each test run gets a different seed (42, 43, 44, ..., 51)
        - Within each test run, that same seed is used to select ALL samples for that run
        - The seed determines the sequence of random samples for E, N, C across all 10 samples
        """
        print(f"Collecting persistence diagrams for {self.embedding_space} + {self.distance_metric}")
        print(f"Running {n_tests} tests with {n_samples_per_test} samples each")
        print("Following exact seeding procedure from phdim_clustering_validation_best_metrics.py")
        
        class_names = ['entailment', 'neutral', 'contradiction']
        all_diagrams = {
            'entailment': {'H0': [], 'H1': [], 'phd_scores': []},
            'neutral': {'H0': [], 'H1': [], 'phd_scores': []},
            'contradiction': {'H0': [], 'H1': [], 'phd_scores': []}
        }
        
        for test_idx in range(n_tests):
            # Each test run gets its own seed (42, 43, 44, ..., 51)
            test_seed = 42 + test_idx
            print(f"\nTest {test_idx + 1}/{n_tests} (seed={test_seed})")
            
            # Set the seed once for this entire test run
            np.random.seed(test_seed)
            
            for sample_idx in range(n_samples_per_test):
                for class_idx, class_name in enumerate(class_names):
                    # Extract embeddings using the current random state (no re-seeding)
                    embeddings = self.extract_class_embeddings(
                        class_idx, n_points=1000
                    )
                    
                    # Compute distance matrix
                    distance_matrix = pairwise_distances(embeddings, metric=self.distance_metric)
                    
                    # Compute persistence diagrams with fixed seed=42 for reproducible distance matrix calculations
                    phd_score, diagrams = ph_dim_and_diagrams_from_distance_matrix(
                        distance_matrix,
                        min_points=self.phd_params['min_points'],
                        max_points=self.phd_params['max_points'],
                        point_jump=self.phd_params['point_jump'],
                        h_dim=self.phd_params['h_dim'],
                        alpha=self.phd_params['alpha']
                        # seed=42  
                    )
                    
                    # Store results
                    all_diagrams[class_name]['H0'].append(diagrams[0])
                    all_diagrams[class_name]['H1'].append(diagrams[1])
                    all_diagrams[class_name]['phd_scores'].append(phd_score)
                    
                    print(f"  Sample {sample_idx + 1}, {class_name}: H0={len(diagrams[0])}, H1={len(diagrams[1])}, PHD={phd_score:.2f}")
        
        return all_diagrams


class UltraFastPersistenceAnalyzer:
    """
    Ultra-fast persistence analysis using only statistical methods.
    
    Completely avoids expensive distance computations between diagrams.
    Instead uses statistical analysis of persistence features.
    """
    
    def __init__(self, diagrams_data: Dict):
        self.diagrams_data = diagrams_data
        print("Initialized ultra-fast analyzer (statistics-only approach)")
    
    def _preprocess_diagram(self, diagram: np.ndarray) -> np.ndarray:
        """Clean and preprocess diagram"""
        if diagram.size == 0:
            return np.array([]).reshape(0, 2)
        
        # Remove infinite points
        finite_mask = np.isfinite(diagram).all(axis=1)
        finite_diagram = diagram[finite_mask]
        
        return finite_diagram
    
    def _get_comprehensive_statistics(self, diagram: np.ndarray) -> Dict:
        """Extract comprehensive statistics from persistence diagram"""
        if diagram.size == 0:
            return {
                'total_persistence': 0.0,
                'max_persistence': 0.0,
                'mean_persistence': 0.0,
                'std_persistence': 0.0,
                'feature_count': 0,
                'persistence_entropy': 0.0,
                'birth_mean': 0.0,
                'birth_std': 0.0,
                'death_mean': 0.0,
                'death_std': 0.0,
                'midlife_mean': 0.0,
                'midlife_std': 0.0
            }
        
        # Basic persistence statistics
        persistences = diagram[:, 1] - diagram[:, 0]
        persistences = persistences[np.isfinite(persistences) & (persistences > 1e-10)]
        # persistences = persistences[persistences > 1e-10]  # Remove zero persistence
        
        if len(persistences) == 0:
            return {
                'total_persistence': 0.0,
                'max_persistence': 0.0,
                'mean_persistence': 0.0,
                'std_persistence': 0.0,
                'feature_count': 0,
                'persistence_entropy': 0.0,
                'birth_mean': 0.0,
                'birth_std': 0.0,
                'death_mean': 0.0,
                'death_std': 0.0,
                'midlife_mean': 0.0,
                'midlife_std': 0.0
            }
        
        # Birth and death statistics
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        midlife = (births + deaths) / 2
        
        # Persistence entropy
        total_persistence = np.sum(persistences)
        if total_persistence > 1e-10:
            p_normalized = persistences / total_persistence
            entropy = -np.sum(p_normalized * np.log(p_normalized + 1e-10))
        else:
            entropy = 0.0
        
        return {
            'total_persistence': total_persistence,
            'max_persistence': np.max(persistences),
            'mean_persistence': np.mean(persistences),
            'std_persistence': np.std(persistences),
            'feature_count': len(diagram),
            'persistence_entropy': entropy,
            'birth_mean': np.mean(births),
            'birth_std': np.std(births),
            'death_mean': np.mean(deaths),
            'death_std': np.std(deaths),
            'midlife_mean': np.mean(midlife),
            'midlife_std': np.std(midlife)
        }
    
    def _simple_diagram_signature(self, diagram: np.ndarray) -> np.ndarray:
        """
        Create a simple signature vector for the diagram
        This can be used for fast similarity comparisons
        """
        if diagram.size == 0:
            return np.zeros(10)
        
        persistences = diagram[:, 1] - diagram[:, 0]
        persistences = persistences[persistences > 1e-10]
        
        if len(persistences) == 0:
            return np.zeros(10)
        
        # Create signature based on persistence distribution
        signature = np.array([
            np.sum(persistences),                    # Total persistence
            np.max(persistences),                    # Max persistence
            np.mean(persistences),                   # Mean persistence
            np.std(persistences),                    # Std persistence
            len(persistences),                       # Feature count
            np.percentile(persistences, 25),         # 25th percentile
            np.percentile(persistences, 50),         # Median
            np.percentile(persistences, 75),         # 75th percentile
            np.percentile(persistences, 90),         # 90th percentile
            np.sum(persistences > np.mean(persistences))  # Features above mean
        ])
        
        return signature
    
    def _fast_similarity_via_signatures(self, diagrams: list, sample_size: int = 50) -> Dict:
        """
        Fast similarity analysis using signature vectors instead of diagram distances
        """
        n_diagrams = len(diagrams)
        print(f"    Creating signature vectors for {n_diagrams} diagrams...")
        
        # Create signature vectors
        signatures = []
        for i, diagram in enumerate(diagrams):
            processed = self._preprocess_diagram(diagram)
            signature = self._simple_diagram_signature(processed)
            signatures.append(signature)
            
            if i % 25 == 0:
                print(f"      Processed {i}/{n_diagrams} signatures")
        
        signatures = np.array(signatures)
        
        # Compute pairwise distances between signatures (much faster)
        print(f"    Computing signature distances for {sample_size} pairs...")
        
        # Sample pairs
        np.random.seed(42)
        if n_diagrams * (n_diagrams - 1) // 2 <= sample_size:
            pairs = [(i, j) for i in range(n_diagrams) for j in range(i + 1, n_diagrams)]
        else:
            all_pairs = [(i, j) for i in range(n_diagrams) for j in range(i + 1, n_diagrams)]
            selected_indices = np.random.choice(len(all_pairs), sample_size, replace=False)
            pairs = [all_pairs[idx] for idx in selected_indices]
        
        # Compute Euclidean distances between signatures
        signature_distances = []
        for i, j in pairs:
            dist = np.linalg.norm(signatures[i] - signatures[j])
            signature_distances.append(dist)
        
        print(f"    Completed {len(signature_distances)} signature comparisons")
        
        return {
            'signature_distances': signature_distances,
            'n_comparisons': len(signature_distances),
            'signature_mean': np.mean(signature_distances),
            'signature_std': np.std(signature_distances)
        }
    
    def analyze_ultra_fast(self, h_dim: int = 0) -> Dict:
        """
        Ultra-fast analysis using only statistical methods
        """
        dim_label = f'H{h_dim}'
        print(f"\n{'='*50}")
        print(f"ULTRA-FAST ANALYSIS FOR {dim_label}")
        print(f"{'='*50}")
        
        results = {}
        
        for class_name, data in self.diagrams_data.items():
            print(f"\n--- Processing {class_name.upper()} ---")
            diagrams = data[dim_label]
            n_diagrams = len(diagrams)
            
            if n_diagrams < 2:
                print("  Not enough diagrams to compare.")
                continue
            
            print(f"  Total diagrams: {n_diagrams}")
            
            # 1. Statistical analysis
            print("  Computing comprehensive statistics...")
            all_stats = []
            for i, diagram in enumerate(diagrams):
                stats = self._get_comprehensive_statistics(diagram)
                all_stats.append(stats)
                
                if i % 25 == 0:
                    print(f"    Processed {i}/{n_diagrams} statistics")
            
            # Aggregate statistics
            stat_keys = [
                'total_persistence', 'max_persistence', 'mean_persistence', 
                'std_persistence', 'feature_count', 'persistence_entropy',
                'birth_mean', 'birth_std', 'death_mean', 'death_std',
                'midlife_mean', 'midlife_std'
            ]
            
            aggregated_stats = {}
            for key in stat_keys:
                values = [s[key] for s in all_stats]
                values = [v for v in values if not np.isnan(v)]  # Remove NaN values

                if key == 'total_persistence' and h_dim == 0:  # Only for H0 total persistence
                    print(f"DEBUG - {class_name} H0 total persistence values:")
                    print(f"  Raw values (first 10): {values[:10]}")
                    print(f"  All values count: {len(values)}")
                    print(f"  Min: {np.min(values) if values else 'N/A'}")
                    print(f"  Max: {np.max(values) if values else 'N/A'}")
                    print(f"  Mean: {np.mean(values) if values else 'N/A'}")
                    print(f"  Std: {np.std(values) if values else 'N/A'}")
                
                if len(values) > 0:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    cv = (std_val / mean_val) if mean_val > 0 else 0

                # ADD MORE DEBUG HERE
                    if key == 'total_persistence' and h_dim == 0:
                        print(f"  Final mean_val: {mean_val}")
                        print(f"  Final std_val: {std_val}")
                        print(f"  Final CV: {cv}")
                        print()
                    
                    aggregated_stats[key] = {
                        'mean': mean_val,
                        'std': std_val,
                        'cv': cv,
                        'min': np.min(values),
                        'max': np.max(values)
                    }
                else:
                    aggregated_stats[key] = {
                        'mean': 0.0, 'std': 0.0, 'cv': 0.0, 'min': 0.0, 'max': 0.0
                    }
            
            # 2. Fast similarity analysis using signatures
            similarity_results = self._fast_similarity_via_signatures(diagrams, sample_size=50)
            
            # Store results
            results[class_name] = {
                'statistics': aggregated_stats,
                'similarity': similarity_results
            }
            
            # Print key results
            total_pers_cv = aggregated_stats['total_persistence']['cv']
            mean_sig_dist = similarity_results['signature_mean']
            
            print(f"  Key Results:")
            print(f"    Total persistence CV: {total_pers_cv:.3f}")
            print(f"    Mean signature distance: {mean_sig_dist:.3f}")
            print(f"    Feature count CV: {aggregated_stats['feature_count']['cv']:.3f}")
            print(f"    Persistence entropy CV: {aggregated_stats['persistence_entropy']['cv']:.3f}")
            
            # Memory cleanup
            gc.collect()
        
        return results
    
    def generate_ultra_fast_report(self, h0_results: Dict, h1_results: Dict) -> str:
        """Generate comprehensive ultra-fast report"""
        report = []
        report.append("="*80)
        report.append("ULTRA-FAST PERSISTENCE DIAGRAM SIMILARITY ANALYSIS")
        report.append("="*80)
        report.append("Uses statistical analysis and signature vectors instead of expensive distance computations")
        report.append("Focus: Coefficient of Variation (CV) for stability assessment\n")
        
        # Key metrics table
        report.append("KEY STABILITY METRICS")
        report.append("-" * 80)
        header = f"{'CLASS':<15} {'H0_TOTAL_CV':<12} {'H1_TOTAL_CV':<12} {'H0_FEAT_CV':<12} {'H1_FEAT_CV':<12} {'H0_SIG_DIST':<12} {'H1_SIG_DIST':<12}"
        report.append(header)
        report.append("-" * 80)
        
        for class_name in self.diagrams_data.keys():
            h0_total_cv = h0_results.get(class_name, {}).get('statistics', {}).get('total_persistence', {}).get('cv', np.nan)
            h1_total_cv = h1_results.get(class_name, {}).get('statistics', {}).get('total_persistence', {}).get('cv', np.nan)
            h0_feat_cv = h0_results.get(class_name, {}).get('statistics', {}).get('feature_count', {}).get('cv', np.nan)
            h1_feat_cv = h1_results.get(class_name, {}).get('statistics', {}).get('feature_count', {}).get('cv', np.nan)
            h0_sig_dist = h0_results.get(class_name, {}).get('similarity', {}).get('signature_mean', np.nan)
            h1_sig_dist = h1_results.get(class_name, {}).get('similarity', {}).get('signature_mean', np.nan)
            
            row = f"{class_name:<15} {h0_total_cv:<12.3f} {h1_total_cv:<12.3f} {h0_feat_cv:<12.3f} {h1_feat_cv:<12.3f} {h0_sig_dist:<12.3f} {h1_sig_dist:<12.3f}"
            report.append(row)
        
        # Detailed analysis per class
        report.append("\n" + "="*60)
        report.append("DETAILED ANALYSIS PER CLASS")
        report.append("="*60)
        
        for class_name in self.diagrams_data.keys():
            report.append(f"\n{class_name.upper()} CLASS ANALYSIS:")
            report.append("-" * 40)
            
            # H0 analysis
            h0_stats = h0_results.get(class_name, {}).get('statistics', {})
            h0_total_cv = h0_stats.get('total_persistence', {}).get('cv', np.nan)
            h0_entropy_cv = h0_stats.get('persistence_entropy', {}).get('cv', np.nan)
            h0_feat_cv = h0_stats.get('feature_count', {}).get('cv', np.nan)
            
            report.append(f"  H0 Stability:")
            report.append(f"    Total Persistence CV: {h0_total_cv:.3f}")
            report.append(f"    Feature Count CV: {h0_feat_cv:.3f}")
            report.append(f"    Persistence Entropy CV: {h0_entropy_cv:.3f}")
            
            # H1 analysis  
            h1_stats = h1_results.get(class_name, {}).get('statistics', {})
            h1_total_cv = h1_stats.get('total_persistence', {}).get('cv', np.nan)
            h1_entropy_cv = h1_stats.get('persistence_entropy', {}).get('cv', np.nan)
            h1_feat_cv = h1_stats.get('feature_count', {}).get('cv', np.nan)
            
            report.append(f"  H1 Stability:")
            report.append(f"    Total Persistence CV: {h1_total_cv:.3f}")
            report.append(f"    Feature Count CV: {h1_feat_cv:.3f}")
            report.append(f"    Persistence Entropy CV: {h1_entropy_cv:.3f}")
            
            # Overall assessment
            avg_cv = np.nanmean([h0_total_cv, h1_total_cv, h0_feat_cv, h1_feat_cv])
            
            if avg_cv < 0.3:
                assessment = "EXCELLENT - Very stable, ideal for averaging"
            elif avg_cv < 0.5:
                assessment = "GOOD - Stable enough for averaging"
            elif avg_cv < 0.7:
                assessment = "FAIR - Moderate stability, averaging may work"
            else:
                assessment = "POOR - High variability, averaging not recommended"
            
            report.append(f"  Overall Assessment: {assessment}")
            report.append(f"  Average CV: {avg_cv:.3f}")
        
        # Final recommendations
        report.append("\n" + "="*60)
        report.append("FINAL RECOMMENDATIONS")
        report.append("="*60)
        
        all_h1_cvs = []
        for class_name in self.diagrams_data.keys():
            h1_cv = h1_results.get(class_name, {}).get('statistics', {}).get('total_persistence', {}).get('cv', np.nan)
            if not np.isnan(h1_cv):
                all_h1_cvs.append(h1_cv)
        
        if all_h1_cvs:
            overall_avg_cv = np.mean(all_h1_cvs)
            report.append(f"\nOverall H1 Total Persistence CV: {overall_avg_cv:.3f}")
            
            if overall_avg_cv < 0.4:
                final_rec = "PROCEED with averaging - diagrams show excellent stability"
            elif overall_avg_cv < 0.6:
                final_rec = "PROCEED with averaging - diagrams show good stability"
            else:
                final_rec = "CAUTION - High variability, consider alternative approaches"
            
            report.append(f"Final Recommendation: {final_rec}")
        
        return "\n".join(report)


def main():
    """Main function for ultra-fast analysis"""
    import time
    
    print("Starting ULTRA-FAST persistence diagram analysis...")
    print("This version uses only statistical methods - no expensive distance computations")
    
    # Load data
    SAVE_PATH = 'entailment_surfaces/supervised_contrastive_autoencoder/src/persistence_diagrams/collected_diagrams_CONCAT_EUCLIDEAN.pkl'
    
    if not Path(SAVE_PATH).exists():
        print("Running diagram collection...")
        collector = PersistenceDiagramCollector(bert_data_path='data/processed/snli_full_standard_SBERT.pt')
        all_diagrams = collector.collect_persistence_diagrams(n_tests=10, n_samples_per_test=10)
        with open(SAVE_PATH, 'wb') as f:
            pickle.dump(all_diagrams, f)
        print(f"Diagrams collected and saved to {SAVE_PATH}")
    
    with open(SAVE_PATH, 'rb') as f:
        all_diagrams = pickle.load(f)
    
    # Initialize analyzer
    analyzer = UltraFastPersistenceAnalyzer(all_diagrams)
    
    # Run analysis
    print("\n" + "="*60)
    print("STARTING ULTRA-FAST ANALYSIS")
    print("="*60)
    
    start_time = time.time()
    
    h0_results = analyzer.analyze_ultra_fast(h_dim=0)
    h1_results = analyzer.analyze_ultra_fast(h_dim=1)
    
    elapsed = time.time() - start_time
    print(f"\nTotal analysis time: {elapsed:.1f} seconds")
    
    # Generate report
    report = analyzer.generate_ultra_fast_report(h0_results, h1_results)
    print("\n" + report)
    
    # Save report
    REPORT_PATH = 'entailment_surfaces/supervised_contrastive_autoencoder/src/persistence_diagrams/ultra_fast_report.txt'
    with open(REPORT_PATH, 'w') as f:
        f.write(report)
    
    print(f"\nUltra-fast analysis complete! Report saved to {REPORT_PATH}")


if __name__ == '__main__':
    main()