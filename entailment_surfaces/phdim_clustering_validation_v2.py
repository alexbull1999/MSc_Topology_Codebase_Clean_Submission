"""
Efficient Statistical Validation: Subsample First, Then Compute Embeddings
Simple approach: Generate embeddings once, then test each combination completely
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, pairwise_distances
from scipy import stats
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import copy
import time
from datetime import datetime
from persim import PersistenceImager
from gph.python import ripser_parallel
from itertools import permutations
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')

def flush_output():
    sys.stdout.flush()
    sys.stderr.flush()

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def ph_dim_and_diagrams_from_distance_matrix(dm: np.ndarray,
                                            min_points=200,
                                            max_points=1000,
                                            point_jump=50,
                                            h_dim=0,
                                            alpha: float = 1.) -> Tuple[float, List]:
    """Compute PH-dimension and persistence diagrams from distance matrix"""
    assert dm.ndim == 2, dm
    assert dm.shape[0] == dm.shape[1], dm.shape

    test_n = range(min_points, max_points, point_jump)
    lengths = []
    all_diagrams = []

    for points_number in test_n:
        sample_indices = np.random.choice(dm.shape[0], points_number, replace=False)
        dist_matrix = dm[sample_indices, :][:, sample_indices]

        diagrams = ripser_parallel(dist_matrix, maxdim=1, n_threads=-1, metric="precomputed")['dgms']
        all_diagrams.append(diagrams)

        d = diagrams[h_dim]
        d = d[d[:, 1] < np.inf]
        lengths.append(np.power((d[:, 1] - d[:, 0]), alpha).sum())

    lengths = np.array(lengths)

    x = np.log(np.array(list(test_n)))
    y = np.log(lengths)
    N = len(x)
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    
    ph_dimension = alpha / (1 - m)
    
    return ph_dimension, all_diagrams

def compute_distance_matrix_advanced(embeddings: torch.Tensor, metric: str) -> np.ndarray:
    """Compute distance matrix using advanced metrics"""
    embeddings_np = embeddings.detach().cpu().numpy()
    
    sklearn_metrics = ['euclidean', 'manhattan', 'chebyshev', 'cosine', 'correlation', 'braycurtis', 'canberra']
    
    if metric in sklearn_metrics:
        return pairwise_distances(embeddings_np, metric=metric)
    elif metric == 'minkowski_3':
        return pairwise_distances(embeddings_np, metric='minkowski', p=3)
    elif metric == 'minkowski_4':
        return pairwise_distances(embeddings_np, metric='minkowski', p=4)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def persistence_diagrams_to_images(all_diagrams: List) -> List[np.ndarray]:
    """Convert persistence diagrams to standardized images"""
    pimgr = PersistenceImager(
        pixel_size=0.5,
        birth_range=(0, 5),
        pers_range=(0, 5),
        kernel_params={'sigma': 0.3}
    )
    
    persistence_images = []
    
    for diagrams in all_diagrams:
        combined_image = np.zeros((20, 20))
        
        for dim in range(min(2, len(diagrams))):
            diagram = diagrams[dim]
            if len(diagram) > 0:
                finite_diagram = diagram[np.isfinite(diagram).all(axis=1)]
                if len(finite_diagram) > 0:
                    try:
                        img = pimgr.transform([finite_diagram])[0]
                        if img.shape != (20, 20):
                            from scipy.ndimage import zoom
                            zoom_factors = (20 / img.shape[0], 20 / img.shape[1])
                            img = zoom(img, zoom_factors)
                        combined_image += img
                    except:
                        continue
        
        if combined_image.max() > 0:
            combined_image = combined_image / combined_image.max()
        
        persistence_images.append(combined_image.flatten())
    
    return persistence_images

def calculate_clustering_accuracy(true_labels: List[int], predicted_labels: List[int]) -> float:
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

class EfficientStatisticalValidation:
    """Efficient statistical validation - subsample first, then compute embeddings"""
    
    def __init__(self, bert_data_path: str, order_model_path: str, 
                 output_dir: str = "entailment_surfaces/statistical_validation"):
        self.bert_data_path = bert_data_path
        self.order_model_path = order_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 4 embedding spaces as requested
        self.embedding_spaces = ['sbert_concat', 'lattice_containment']
        
        # All distance metrics for all 4 spaces as requested
        self.distance_metrics = [
            'euclidean', 'manhattan', 'chebyshev', 'cosine',
            'minkowski_3', 'minkowski_4', 'canberra', 'braycurtis'
        ]
        
        # Generate all combinations
        self.top_combinations = []
        for space in self.embedding_spaces:
            for metric in self.distance_metrics:
                self.top_combinations.append((space, metric))
        
        self.sample_size = 10
        
        print(f"Efficient Statistical Validation initialized")
        print(f"Testing {len(self.top_combinations)} combinations")
        print(f"Embedding spaces: {self.embedding_spaces}")
        print(f"Distance metrics: {self.distance_metrics}")
        print(f"Device: {self.device}")
        
        self._load_data()
    
    def _load_data(self):
        """Load BERT data and existing order model"""
        print("Loading BERT data...")
        self.bert_data = torch.load(self.bert_data_path, map_location=self.device, weights_only=False)
        
        print(f"BERT data loaded:")
        print(f"  Premise embeddings: {self.bert_data['premise_embeddings'].shape}")
        print(f"  Hypothesis embeddings: {self.bert_data['hypothesis_embeddings'].shape}")
        print(f"  Labels: {len(self.bert_data['labels'])}")
        
        # Organize data by class for efficient sampling
        self._organize_data_by_class()
        
        # Load existing order model
        print("Loading existing order model...")
        from src.order_embeddings_asymmetry import OrderEmbeddingModel
        
        checkpoint = torch.load(self.order_model_path, map_location=self.device, weights_only=False)
        model_config = checkpoint['model_config']
        
        self.order_model = OrderEmbeddingModel(
            bert_dim=model_config['bert_dim'],
            order_dim=model_config['order_dim'],
            asymmetry_weight=model_config.get('asymmetry_weight', 0.2)
        )
        self.order_model.load_state_dict(checkpoint['model_state_dict'])
        self.order_model.to(self.device)
        self.order_model.eval()
        print("Existing order model loaded successfully")
        
        # Load existing hyperbolic pipeline
        print("Loading existing hyperbolic pipeline...")
        try:
            from src.hyperbolic_projection_asymmetry import HyperbolicOrderEmbeddingPipeline
            self.hyperbolic_pipeline = HyperbolicOrderEmbeddingPipeline(self.order_model_path)
            print("Existing hyperbolic pipeline loaded successfully")
        except Exception as e:
            print(f"Could not load hyperbolic pipeline: {e}")
            self.hyperbolic_pipeline = None
    
    def _organize_data_by_class(self):
        """Organize data by class for efficient sampling"""
        premise_embs = self.bert_data['premise_embeddings']
        hypothesis_embs = self.bert_data['hypothesis_embeddings']
        labels = self.bert_data['labels']
        
        self.data_by_class = {}
        
        for label in ['entailment', 'neutral', 'contradiction']:
            mask = torch.tensor([l == label for l in labels], device=self.device, dtype=torch.bool)
            indices = torch.where(mask)[0]
            
            self.data_by_class[label] = {
                'premise_bert': premise_embs[indices],
                'hypothesis_bert': hypothesis_embs[indices],
                'indices': indices,
                'n_samples': len(indices)
            }
            
            print(f"  {label}: {len(indices)} samples available")
    
    def generate_all_sample_indices(self) -> Dict:
        """Generate all sample indices for all runs"""
        print("Generating all sample indices...")
        
        all_sample_indices = {}
        
        for run_idx in range(self.sample_size):
            np.random.seed(42 + run_idx)
            
            run_samples = {}
            for class_name in ['entailment', 'neutral', 'contradiction']:
                class_samples = []
                n_available = self.data_by_class[class_name]['n_samples']
                
                # Generate 10 samples of 1000 points each
                for sample_idx in range(10):
                    indices = np.random.choice(n_available, 1000, replace=False)
                    class_samples.append(indices)
                
                run_samples[class_name] = class_samples
            
            all_sample_indices[run_idx] = run_samples
        
        print(f"Generated sample indices for {self.sample_size} runs")
        return all_sample_indices
    
    def compute_embeddings_for_combination(self, all_sample_indices: Dict, space_name: str) -> Dict:
        """Compute embeddings for one space across all samples from all runs"""
        print(f"  Computing {space_name} embeddings...")
        
        space_embeddings = {}
        
        # Extract sampled BERT data for this space only
        all_sampled_bert_data = {}
        for class_name in ['entailment', 'neutral', 'contradiction']:
            all_premise_samples = []
            all_hypothesis_samples = []
            
            class_data = self.data_by_class[class_name]
            
            # Collect samples from all runs
            for run_idx in range(self.sample_size):
                for sample_indices_list in all_sample_indices[run_idx][class_name]:
                    premise_sample = class_data['premise_bert'][sample_indices_list]
                    hypothesis_sample = class_data['hypothesis_bert'][sample_indices_list]
                    
                    all_premise_samples.append(premise_sample)
                    all_hypothesis_samples.append(hypothesis_sample)
            
            # Stack all samples: [100, 1000, 768]
            all_sampled_bert_data[class_name] = {
                'premise': torch.stack(all_premise_samples),
                'hypothesis': torch.stack(all_hypothesis_samples)
            }
        
        # Compute only the requested embedding space
        for class_name in ['entailment', 'neutral', 'contradiction']:
            premise_samples = all_sampled_bert_data[class_name]['premise']  # [100, 1000, 768]
            hypothesis_samples = all_sampled_bert_data[class_name]['hypothesis']  # [100, 1000, 768]
            
            if space_name == 'sbert_concat':
                class_embeddings = torch.cat([premise_samples, hypothesis_samples], dim=2)  # [100, 1000, 1536]
            
            elif space_name == 'lattice_containment':
                epsilon = 1e-8
                class_embeddings = ((premise_samples * hypothesis_samples) / 
                                  (torch.abs(premise_samples) + torch.abs(hypothesis_samples) + epsilon))
            
            elif space_name == 'order_concat':
                order_premise_list = []
                order_hypothesis_list = []
                
                # Process in batches to avoid memory issues
                batch_size = 10
                for i in range(0, premise_samples.shape[0], batch_size):
                    end_idx = min(i + batch_size, premise_samples.shape[0])
                    
                    with torch.no_grad():
                        batch_premise = premise_samples[i:end_idx].view(-1, 768)  # [batch×1000, 768]
                        batch_hypothesis = hypothesis_samples[i:end_idx].view(-1, 768)  # [batch×1000, 768]
                        
                        order_p = self.order_model(batch_premise)
                        order_h = self.order_model(batch_hypothesis)
                        
                        # Reshape back
                        batch_size_actual = end_idx - i
                        order_p = order_p.view(batch_size_actual, 1000, -1)
                        order_h = order_h.view(batch_size_actual, 1000, -1)
                        
                        order_premise_list.append(order_p.cpu())
                        order_hypothesis_list.append(order_h.cpu())
                
                order_premise_stack = torch.cat(order_premise_list, dim=0)
                order_hypothesis_stack = torch.cat(order_hypothesis_list, dim=0)
                
                class_embeddings = torch.cat([order_premise_stack, order_hypothesis_stack], dim=2)
            
            elif space_name == 'hyperbolic_concat':
                if self.hyperbolic_pipeline:
                    hyperbolic_premise_list = []
                    hyperbolic_hypothesis_list = []
                    
                    # Process in smaller batches
                    batch_size = 5
                    for i in range(0, premise_samples.shape[0], batch_size):
                        end_idx = min(i + batch_size, premise_samples.shape[0])
                        
                        batch_premise = premise_samples[i:end_idx]
                        batch_hypothesis = hypothesis_samples[i:end_idx]
                        
                        try:
                            batch_premise_flat = batch_premise.view(-1, 768)
                            batch_hypothesis_flat = batch_hypothesis.view(-1, 768)
                            
                            with torch.no_grad():
                                results = self.hyperbolic_pipeline.compute_enhanced_hyperbolic_energies(
                                    batch_premise_flat, batch_hypothesis_flat
                                )
                                
                                # Reshape back
                                batch_size_actual = end_idx - i
                                hyp_p = results['premise_hyperbolic'].view(batch_size_actual, 1000, -1)
                                hyp_h = results['hypothesis_hyperbolic'].view(batch_size_actual, 1000, -1)
                                
                                hyperbolic_premise_list.append(hyp_p.cpu())
                                hyperbolic_hypothesis_list.append(hyp_h.cpu())
                        except Exception as e:
                            print(f"    Hyperbolic batch failed, using order embeddings: {e}")
                            # Fallback to order embeddings
                            batch_premise_flat = batch_premise.view(-1, 768)
                            batch_hypothesis_flat = batch_hypothesis.view(-1, 768)
                            
                            with torch.no_grad():
                                order_p = self.order_model(batch_premise_flat)
                                order_h = self.order_model(batch_hypothesis_flat)
                                
                                batch_size_actual = end_idx - i
                                order_p = order_p.view(batch_size_actual, 1000, -1)
                                order_h = order_h.view(batch_size_actual, 1000, -1)
                                
                                hyperbolic_premise_list.append(order_p.cpu())
                                hyperbolic_hypothesis_list.append(order_h.cpu())
                    
                    hyperbolic_premise_stack = torch.cat(hyperbolic_premise_list, dim=0)
                    hyperbolic_hypothesis_stack = torch.cat(hyperbolic_hypothesis_list, dim=0)
                    
                    class_embeddings = torch.cat([hyperbolic_premise_stack, hyperbolic_hypothesis_stack], dim=2)
                else:
                    print(f"    Hyperbolic pipeline not available, using order embeddings")
                    # Same as order_concat
                    order_premise_list = []
                    order_hypothesis_list = []
                    
                    batch_size = 10
                    for i in range(0, premise_samples.shape[0], batch_size):
                        end_idx = min(i + batch_size, premise_samples.shape[0])
                        
                        with torch.no_grad():
                            batch_premise = premise_samples[i:end_idx].view(-1, 768)
                            batch_hypothesis = hypothesis_samples[i:end_idx].view(-1, 768)
                            
                            order_p = self.order_model(batch_premise)
                            order_h = self.order_model(batch_hypothesis)
                            
                            batch_size_actual = end_idx - i
                            order_p = order_p.view(batch_size_actual, 1000, -1)
                            order_h = order_h.view(batch_size_actual, 1000, -1)
                            
                            order_premise_list.append(order_p.cpu())
                            order_hypothesis_list.append(order_h.cpu())
                    
                    order_premise_stack = torch.cat(order_premise_list, dim=0)
                    order_hypothesis_stack = torch.cat(order_hypothesis_list, dim=0)
                    
                    class_embeddings = torch.cat([order_premise_stack, order_hypothesis_stack], dim=2)
            
            space_embeddings[class_name] = class_embeddings.cpu()  # [100, 1000, embedding_dim]
        
        # Clear GPU memory immediately
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return space_embeddings
    
    def run_single_sample_validation(self, space_embeddings: Dict, space_name: str, metric_name: str, run_idx: int) -> Dict:
        """Run clustering validation for a single run"""
        
        all_persistence_images = []
        sample_labels = []
        ph_dim_values = {'entailment': [], 'neutral': [], 'contradiction': []}
        
        # Process each class for this run
        for class_name in ['entailment', 'neutral', 'contradiction']:
            class_idx = ['entailment', 'neutral', 'contradiction'].index(class_name)
            class_embeddings = space_embeddings[class_name]  # [100, 1000, embedding_dim]
            
            # Process each of the 10 samples for this class and run
            start_idx = run_idx * 10
            end_idx = start_idx + 10
            
            for sample_idx in range(start_idx, end_idx):
                sample_embedding = class_embeddings[sample_idx]  # [1000, embedding_dim]
                
                # Compute distance matrix
                distance_matrix = compute_distance_matrix_advanced(sample_embedding, metric_name)
                
                # Get PH-dim and persistence diagrams
                ph_dim, all_diagrams = ph_dim_and_diagrams_from_distance_matrix(
                    distance_matrix,
                    min_points=200,
                    max_points=1000,
                    point_jump=50,
                    h_dim=0,
                    alpha=1.0
                )
                
                ph_dim_values[class_name].append(ph_dim)
                
                # Convert to persistence image
                if len(all_diagrams) > 0:
                    persistence_image = persistence_diagrams_to_images([all_diagrams[0]])
                    if len(persistence_image) > 0:
                        all_persistence_images.append(persistence_image[0])
                        sample_labels.append(class_idx)
        
        # Perform clustering
        if len(all_persistence_images) >= 3:
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
        
        return None
    
    def _safe_clustering_analysis(self, persistence_images: List[np.ndarray], 
                                true_labels: List[int]) -> Tuple[float, float, float]:
        """Perform clustering analysis"""
        
        if len(persistence_images) < 3:
            return 0.0, 0.0, 0.0
        
        X = np.vstack(persistence_images)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        predicted_labels = kmeans.fit_predict(X)
        
        accuracy = calculate_clustering_accuracy(true_labels, predicted_labels)
        
        try:
            if len(set(predicted_labels)) > 1 and len(X) > 3:
                sil_score = silhouette_score(X, predicted_labels)
            else:
                sil_score = 0.0
        except:
            sil_score = 0.0
        
        ari_score = adjusted_rand_score(true_labels, predicted_labels)
        
        return accuracy, sil_score, ari_score
    
    def _calculate_sample_statistics(self, sample_results: List[Dict]) -> Dict:
        """Calculate statistical summary"""
        
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
        
        accuracies = [r['accuracy'] for r in sample_results]
        silhouettes = [r['silhouette'] for r in sample_results]
        perfect_clustering = [r['perfect_clustering'] for r in sample_results]
        
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
            'silhouette_mean': float(np.mean(silhouettes)),
            'silhouette_std': float(np.std(silhouettes)),
            'perfect_clustering_rate': float(np.mean(perfect_clustering)),
            'ph_dim_stats': ph_dim_stats
        }
    
    def _run_ttest_analysis(self, sample_results: List[Dict]) -> Dict:
        """Run t-tests on accuracy results with correct statistical logic"""
        
        if not sample_results:
            return {}
        
        accuracies = [r['accuracy'] for r in sample_results]
        accuracy_std = np.std(accuracies)
        accuracy_mean = np.mean(accuracies)
        
        # Handle zero variance case properly
        if accuracy_std < 1e-10:
            # For zero variance, the test is deterministic based on the mean
            
            # For random chance (33.33%)
            if abs(accuracy_mean - 0.3333) < 1e-6:  # Exactly equal to random chance
                p_val_random = 0.5  # No evidence of being better than random
                sig_random = False
            elif accuracy_mean > 0.3333:
                p_val_random = 0.0  # Definitely better than random
                sig_random = True
            else:
                p_val_random = 1.0  # Definitely worse than random
                sig_random = False
            
            # For strong performance (90%)
            if abs(accuracy_mean - 0.90) < 1e-6:  # Exactly equal to 90%
                p_val_strong = 0.5  # No evidence of being better than 90%
                sig_strong = False
            elif accuracy_mean > 0.90:
                p_val_strong = 0.0  # Definitely better than 90%
                sig_strong = True
            else:
                p_val_strong = 1.0  # Definitely worse than 90%
                sig_strong = False
            
            return {
                'vs_random_chance': {
                    't_statistic': 0.0,
                    'p_value': p_val_random,
                    'significant': sig_random,
                    'threshold': 0.3333,
                    'note': 'zero_variance'
                },
                'vs_strong_performance': {
                    't_statistic': 0.0,
                    'p_value': p_val_strong,
                    'significant': sig_strong,
                    'threshold': 0.90,
                    'note': 'zero_variance'
                }
            }
        
        # Normal case with variance
        t_stat_random, p_val_random = stats.ttest_1samp(
            accuracies, 0.3333, alternative='greater'
        )
        t_stat_strong, p_val_strong = stats.ttest_1samp(
            accuracies, 0.90, alternative='greater'
        )
        
        return {
            'vs_random_chance': {
                't_statistic': float(t_stat_random),
                'p_value': float(p_val_random),
                'significant': p_val_random < 0.05,
                'threshold': 0.3333
            },
            'vs_strong_performance': {
                't_statistic': float(t_stat_strong),
                'p_value': float(p_val_strong),
                'significant': p_val_strong < 0.05,
                'threshold': 0.90
            }
        }
    
    def _run_significance_tests(self, stats_summary: Dict, sample_results: List[Dict]) -> Dict:
        """Run statistical significance tests"""
        
        significance_tests = {}
        
        ttest_results = self._run_ttest_analysis(sample_results)
        significance_tests.update(ttest_results)
        
        perfect_rate = stats_summary['perfect_clustering_rate']
        n_samples = stats_summary['n_samples']
        
        if n_samples > 0:
            perfect_count = int(perfect_rate * n_samples)
            p_value = stats.binom_test(perfect_count, n_samples, 0.7, alternative='greater')
            
            significance_tests['perfect_clustering_vs_threshold'] = {
                'perfect_rate': perfect_rate,
                'perfect_count': perfect_count,
                'n_samples': n_samples,
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'threshold': 0.7
            }
        
        return significance_tests
    
    def run_statistical_validation(self) -> Dict:
        """Run complete efficient statistical validation"""
        print("="*60)
        print("EFFICIENT STATISTICAL VALIDATION")
        print("="*60)
        
        total_start_time = time.time()
        
        # Step 1: Generate all sample indices
        all_sample_indices = self.generate_all_sample_indices()
        
        # Step 2: Test each combination completely (generate embeddings per combination)
        final_results = {}
        
        for combo_idx, (space_name, metric_name) in enumerate(self.top_combinations):
            print(f"\n{'='*60}")
            print(f"TESTING COMBINATION {combo_idx + 1}/{len(self.top_combinations)}: {space_name} + {metric_name}")
            print(f"{'='*60}")
            
            combo_start_time = time.time()
            
            # Generate embeddings for this space only
            space_embeddings = self.compute_embeddings_for_combination(all_sample_indices, space_name)
            
            # Run all 10 validation runs for this combination
            sample_results = []
            for run_idx in range(self.sample_size):
                print(f"  Run {run_idx + 1}/{self.sample_size}...")
                
                result = self.run_single_sample_validation(
                    space_embeddings, space_name, metric_name, run_idx
                )
                
                if result:
                    sample_results.append(result)
                    print(f"    Accuracy: {result['accuracy']:.3f}")
            
            # Calculate statistics for this combination
            if sample_results:
                stats_summary = self._calculate_sample_statistics(sample_results)
                significance_tests = self._run_significance_tests(stats_summary, sample_results)
                
                final_results[f"{space_name}_{metric_name}"] = {
                    'space': space_name,
                    'metric': metric_name,
                    'sample_results': sample_results,
                    'statistics': stats_summary,
                    'significance_tests': significance_tests
                }
                
                # Print results immediately
                self._print_combination_summary(final_results[f"{space_name}_{metric_name}"])
            
            # Clear memory after each combination
            del space_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            combo_time = time.time() - combo_start_time
            print(f"  Combination completed in {combo_time:.1f}s")
        
        # Save results
        self._save_results(final_results)
        
        total_time = time.time() - total_start_time
        print(f"\nTotal time: {total_time:.1f}s")
        
        return final_results
    
    def _print_combination_summary(self, result: Dict):
        """Print summary for a single combination"""
        print(f"\n  SUMMARY for {result['space']} + {result['metric']}:")
        
        stats = result['statistics']
        print(f"    {stats['n_samples']} samples: Acc={stats['accuracy_mean']:.3f}±{stats['accuracy_std']:.3f}")
        print(f"    Silhouette: {stats['silhouette_mean']:.3f}±{stats['silhouette_std']:.3f}")
        print(f"    Perfect clustering rate: {stats['perfect_clustering_rate']:.1%}")
        
        if 'significance_tests' in result:
            sig_tests = result['significance_tests']
            
            if 'vs_random_chance' in sig_tests:
                random_test = sig_tests['vs_random_chance']
                status = "YES" if random_test['significant'] else "NO"
                note = f" ({random_test.get('note', '')})" if 'note' in random_test else ""
                print(f"    Significantly > random (33%): {status} (p={random_test['p_value']:.4f}){note}")
            
            if 'vs_strong_performance' in sig_tests:
                strong_test = sig_tests['vs_strong_performance']
                status = "YES" if strong_test['significant'] else "NO"
                note = f" ({strong_test.get('note', '')})" if 'note' in strong_test else ""
                print(f"    Significantly > 90%: {status} (p={strong_test['p_value']:.4f}){note}")
            
            if 'perfect_clustering_vs_threshold' in sig_tests:
                perfect_test = sig_tests['perfect_clustering_vs_threshold']
                status = "YES" if perfect_test['significant'] else "NO"
                print(f"    Significantly > 70%: {status} (p={perfect_test['p_value']:.4f})")
                print(f"    Perfect clustering: {perfect_test['perfect_count']}/{perfect_test['n_samples']} runs")
    
    def _save_results(self, validation_results: Dict):
        """Save validation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"efficient_validation_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(validation_results, f, indent=2, default=convert_numpy_types)
        
        print(f"\nResults saved to: {results_file}")


def main():
    """Run efficient statistical validation"""
    
    bert_data_path = "data/processed/snli_10k_subset_train_SBERT_STSB_LARGE.pt"
    order_model_path = "models/enhanced_order_embeddings_snli_SBERT_full.pt"
    
    validator = EfficientStatisticalValidation(
        bert_data_path=bert_data_path,
        order_model_path=order_model_path,
        output_dir="entailment_surfaces/statistical_validation"
    )
    
    results = validator.run_statistical_validation()
    
    print("\nEfficient Statistical Validation completed!")
    return results

if __name__ == "__main__":
    main()