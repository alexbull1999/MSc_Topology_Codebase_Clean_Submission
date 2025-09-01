"""
TDA Integration for Hyperbolic Entailment Cones

Implements topological data analysis
on cone violation energies to extract features that distinguish entailment vs
neutral vs contradiction reliably.

Core Hypothesis: Different entailment relationships produce distinct topological
signatures in their cone violation patterns that can be captured via persistent homology.

Expected Patterns:
- Entailment: Tight clusters, minimal topological complexity
- Neutral: Intermediate structure
- Contradiction: Dispersed patterns, complex topological holes

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import ripser
import persim
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance


@dataclass
class TopologicalFeatures:
    """Container for extracted topological features"""
    birth_death_pairs: np.ndarray
    persistence_landscape: np.ndarray
    bottleneck_distance: float
    wasserstein_distance: float
    betti_numbers: List[int]
    total_persistence: float
    max_persistence: float
    n_significant_features: int


class TDAIntegration:
    """
    Apply TDA to validated hyperbolic entailment cone violations, building on cone_validation.py where we showed on
    toy dataset that:
    - Energy hierarchy: entailment (0.015) < neutral (1.113) < contradiction (1.525)
    - Strong correlation with order embeddings (pearson and spearman)
    - Geometric properties validated (asymmetry, transitivity, etc.)
    """
    def __init__(self, results_dir: str = "results/tda_integration"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cone_validation_results = self.load_cone_validation_results()

        self.tda_params = {
            'maxdim': 2, # Compute H0, H1, H2 # Extend to H3 for larger dataset?
            'thresh': 2.0, #Maximum distance for ripser
            'coeff': 2 #coefficient field
        }

        self.topological_features = {}
        self.original_indices = {}
        self.embedding_coordinates = {}

    def load_cone_validation_results(self) -> Dict:
        tda_data_path = Path("validation_results/tda_ready_data_snli_10k.pt")
        if tda_data_path.exists():
            results = torch.load(tda_data_path, weights_only=False)
            print("Loaded TDA-ready data")
            print(f"{len(results['labels'])} samples with cone violations")
            print(f"Validation status: {'PASSED' if results.get('validation_passed', False) else 'FAILED'}")
            return results
        else:
            raise FileNotFoundError(
                f"TDA-ready data not found at {tda_data_path}. "
                "Please run the enhanced cone validation first to generate tda_ready_data_TYPE.pt"
            )

    def extract_cone_violation_patterns(self) -> Dict[str, np.ndarray]:
        """
        Extract cone violation patterns for TDA analysis
        Returns point clouds organised by entailment labels where each point represents the cone violation pattern
        for a premise-hypothesis pair
        """
        cone_violations = self.cone_validation_results['cone_violations'].numpy()
        labels = self.cone_validation_results['labels']

        #Organize by label
        patterns = defaultdict(list)
        original_indices = defaultdict(list)
        for i, label in enumerate(labels):
            patterns[label].append(cone_violations[i])
            original_indices[label].append(i) # preserve original index

        # Convert to arrays
        pattern_arrays = {}
        index_arrays = {}
        for label, violations in patterns.items():
            pattern_arrays[label] = np.array(violations)
            index_arrays[label] = np.array(original_indices[label])
            print(f"{label}: {len(violations)} samples, shape {pattern_arrays[label].shape}")

        self.original_indices = index_arrays

        # Store clean point clouds ready for neural network use
        self.point_clouds = {}
        for label, point_cloud in pattern_arrays.items():
            # Clean the point cloud
            finite_mask = np.isfinite(point_cloud).all(axis=1)
            clean_cloud = point_cloud[finite_mask]
            self.point_clouds[label] = clean_cloud
            print(f"Clean point cloud for {label}: {len(clean_cloud)} points")


        return pattern_arrays

    def compute_persistent_homology(self, point_cloud: np.ndarray, label: str) -> Dict:
        """
        Compute persistent homology for a point cloud of cone violations
        Args:
            point_cloud: Array of shape (n_samples, n_features) representing cone violations
            label: Entailment label for this point cloud
        Returns:
            Dictionary containing persistence diagrams and features
        """
        print(f"Computing persistent homology for {label} ({point_cloud.shape[0]} points)")

        # Ensure point cloud has valid data
        if not np.isfinite(point_cloud).all():
            print(f"Warning: Non-finite values in {label} point cloud, cleaning...")
            finite_mask = np.isfinite(point_cloud).all(axis=1)
            point_cloud = point_cloud[finite_mask]
            print(f"   Cleaned: {point_cloud.shape[0]} valid points remaining")

        if len(point_cloud) < 3:
            print(f"Not enough valid points for {label}, skipping")
            return None

        # Compute persistence diagrams with Ripser
        try:
            result = ripser.ripser(point_cloud, **self.tda_params)
            diagrams = result['dgms']

            # Debug: Print diagram info
            print(f"   TDA for {label}: H0={len(diagrams[0])}, H1={len(diagrams[1])}, H2={len(diagrams[2])}")

            # Check for infinite values
            for dim, diagram in enumerate(diagrams):
                if len(diagram) > 0:
                    n_infinite = np.sum(np.isinf(diagram[:, 1]))  # Count infinite death times
                    print(f"     H{dim}: {len(diagram)} features, {n_infinite} infinite")

            # Extract topological features
            features = self.extract_topological_features(diagrams, label)

            return {
                'diagrams': diagrams,
                'features': features,
                'n_points': point_cloud.shape[0]
            }

        except Exception as e:
            print(f"Error computing persistent homology for {label}: {e}")

    def extract_topological_features(self, diagrams: List[np.ndarray], label: str) -> TopologicalFeatures:
        """Extract meaningful features from persistence diagrams"""

        print(f"   Extracting features for {label}...")

        # Collect finite features only
        finite_births = []
        finite_deaths = []
        finite_lifespans = []
        betti_numbers = []

        for dim, diagram in enumerate(diagrams):
            if len(diagram) > 0:
                # Count all features for Betti numbers (including infinite)
                betti_numbers.append(len(diagram))

                # Extract finite features only
                finite_mask = np.isfinite(diagram[:, 1])  # Death times are finite
                finite_diagram = diagram[finite_mask]

                if len(finite_diagram) > 0:
                    births = finite_diagram[:, 0]
                    deaths = finite_diagram[:, 1]
                    lifespans = deaths - births

                    # Only include positive lifespans
                    positive_mask = lifespans > 0
                    valid_lifespans = lifespans[positive_mask]

                    if len(valid_lifespans) > 0:
                        finite_births.extend(births[positive_mask])
                        finite_deaths.extend(deaths[positive_mask])
                        finite_lifespans.extend(valid_lifespans)

                        print(f"     H{dim}: {len(valid_lifespans)} finite features")
                    else:
                        print(f"     H{dim}: No valid finite features")
                else:
                    print(f"     H{dim}: No finite features")
            else:
                betti_numbers.append(0)
                print(f"     H{dim}: No features")

        # Convert to arrays
        finite_births = np.array(finite_births)
        finite_deaths = np.array(finite_deaths)
        finite_lifespans = np.array(finite_lifespans)

        # Compute meaningful features
        if len(finite_lifespans) > 0:
            total_persistence = np.sum(finite_lifespans)
            max_persistence = np.max(finite_lifespans)
            mean_lifespan = np.mean(finite_lifespans)
            n_significant = np.sum(finite_lifespans > mean_lifespan)

            print(f"   Features: total={total_persistence:.4f}, max={max_persistence:.4f}, significant={n_significant}")
        else:
            total_persistence = 0.0
            max_persistence = 0.0
            n_significant = 0
            print(f"   Features: No finite lifespans - using zeros")

        return TopologicalFeatures(
            birth_death_pairs=np.column_stack([finite_births, finite_deaths]) if len(
                finite_births) > 0 else np.array([]).reshape(0, 2),
            persistence_landscape=finite_lifespans,
            bottleneck_distance=0.0,
            wasserstein_distance=0.0,
            betti_numbers=betti_numbers,
            total_persistence=total_persistence,
            max_persistence=max_persistence,
            n_significant_features=int(n_significant)
        )

    def analyse_topological_signatures(self) -> Dict:
        """
        Main analysis function: Apply TDA to cone violations and extract distinguishing features

        Expected outcomes:
        - Entailment: Low violation energies → tight clusters → simple topology
        - Neutral: Medium violation energies → intermediate structure
        - Contradiction: High violation energies → complex patterns → rich topology
        """
        print("Analysing topological signatures")
        hierarchy = self.cone_validation_results['energy_hierarchy']
        print(f" Entailment: {hierarchy['entailment_mean']:.3f}")
        print(f" Neutral: {hierarchy['neutral_mean']:.3f}")
        print(f" Contradiction: {hierarchy['contradiction_mean']:.3f}")
        print(f"   Validation: {'PASSED' if self.cone_validation_results.get('validation_passed', False) else 'FAILED'}")

        #Extract cone violation patterns
        pattern_arrays = self.extract_cone_violation_patterns()

        #Compute persistent homology for each label
        tda_results = {}
        for label, point_cloud in pattern_arrays.items():
            result = self.compute_persistent_homology(point_cloud, label)
            if result:
                tda_results[label] = result

        # Compare topological signatures between classes
        signature_comparison = self.compare_topological_signatures(tda_results)
        #store results
        self.topological_features = tda_results

        self._compute_per_example_statistics(tda_results)

        del pattern_arrays
        gc.collect()

        analysis_results = {
            'tda_results': tda_results,
            'signature_comparison': signature_comparison,
            'validation_status': self.validate_tda_hypotheses(tda_results),
        }

        return analysis_results

    def _compute_per_example_statistics(self, tda_results: Dict):

        self.class_statistics = {}

        for label, result in tda_results.items():
            features = result['features']
            n_points = result['n_points']

            # Compute per-example averages
            per_example_stats = {
                'total_persistence_per_example': features.total_persistence / n_points if n_points > 0 else 0.0,
                'max_persistence_per_example': features.max_persistence / n_points if n_points > 0 else 0.0,
                'significant_features_per_example': features.n_significant_features / n_points if n_points > 0 else 0.0,
                'betti_sum_per_example': sum(features.betti_numbers) / n_points if n_points > 0 else 0.0
            }

            self.class_statistics[label] = per_example_stats

            print(f"  {label}:")
            print(f"    Persistence per example: {per_example_stats['total_persistence_per_example']:.6f}")
            print(f"    Significant features per example: {per_example_stats['significant_features_per_example']:.6f}")
            print(f"    Betti sum per example: {per_example_stats['betti_sum_per_example']:.6f}")



    def compare_topological_signatures(self, tda_results: Dict) -> Dict:
        """Compare topological signatures between entailment classes"""
        comparisons = {}
        labels = list(tda_results.keys())

        print("Comparing topological signatures between classes")

        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels[i+1:], i+1):
                key = f"{label1}_vs_{label2}"

                #Compare key topological features
                features1 = tda_results[label1]['features']
                features2 = tda_results[label2]['features']

                # Safe Wasserstein distance computation
                wasserstein_dist = self.compute_safe_wasserstein_distance(
                    features1.persistence_landscape,
                    features2.persistence_landscape
                )

                # Safe ratios
                persistence_ratio = self.safe_divide(features1.total_persistence, features2.total_persistence)
                complexity_ratio = self.safe_divide(features1.n_significant_features, features2.n_significant_features)

                # Betti number differences
                betti_diff = np.array(features1.betti_numbers) - np.array(features2.betti_numbers)


                comparison = {
                    'wasserstein_distance': wasserstein_dist,
                    'betti_diff': betti_diff,
                    'persistence_ratio': persistence_ratio,
                    'complexity_ratio': complexity_ratio,
                    'features1_persistence': features1.total_persistence,
                    'features2_persistence': features2.total_persistence,
                    'features1_complexity': features1.n_significant_features,
                    'features2_complexity': features2.n_significant_features
                }

                comparisons[key] = comparison

                print(f"  {key}:")
                print(f"    Wasserstein: {wasserstein_dist:.4f}")
                print(f"    Persistence ratio: {persistence_ratio:.4f}")
                print(f"    Betti diff: {betti_diff}")

        return comparisons

    def compute_safe_wasserstein_distance(self, landscape1: np.ndarray, landscape2: np.ndarray):
        """Safely compute Wasserstein distance between persistence landscapes"""
        # Handle empty landscapes
        if len(landscape1) == 0 and len(landscape2) == 0:
            return 0.0
        elif len(landscape1) == 0:
            return np.mean(landscape2) if len(landscape2) > 0 else 0.0
        elif len(landscape2) == 0:
            return np.mean(landscape1) if len(landscape1) > 0 else 0.0

        # Both non-empty - compute Wasserstein distance
        try:
            # Filter out any remaining infinite values
            landscape1_clean = landscape1[np.isfinite(landscape1)]
            landscape2_clean = landscape2[np.isfinite(landscape2)]

            if len(landscape1_clean) == 0 and len(landscape2_clean) == 0:
                return 0.0
            elif len(landscape1_clean) == 0:
                return np.mean(landscape2_clean)
            elif len(landscape2_clean) == 0:
                return np.mean(landscape1_clean)

            return wasserstein_distance(landscape1_clean, landscape2_clean)
        except Exception as e:
            print(f"    Warning: Wasserstein computation failed: {e}")
            return np.nan

    def safe_divide(self, numerator: float, denominator: float) -> float:
        """Safely divide two numbers"""
        if denominator == 0:
            return 0.0 if numerator == 0 else np.inf
        return numerator / denominator

    def validate_tda_hypotheses(self, tda_results: Dict) -> Dict:
        """Validate core TDA hypotheses"""
        validation = {}

        if not all(label in tda_results for label in ['entailment', 'neutral', 'contradiction']):
            return {'status': 'FAILED', 'reason': 'Missing label data'}

        # Checkpoint 2: Topological Signature Differentiation
        # Expected: Different persistence diagrams for each label type
        ent_features = tda_results['entailment']['features']
        neut_features = tda_results['neutral']['features']
        cont_features = tda_results['contradiction']['features']

        # Test complexity ordering: entailment < neutral < contradiction
        complexity_order = (
                ent_features.n_significant_features <= neut_features.n_significant_features <=
                cont_features.n_significant_features
        )

        persistence_order = (
                ent_features.total_persistence <= neut_features.total_persistence <=
                cont_features.total_persistence
        )

        # Test Betti number ordering (sum of all dimensions)
        ent_betti_sum = sum(ent_features.betti_numbers)
        neut_betti_sum = sum(neut_features.betti_numbers)
        cont_betti_sum = sum(cont_features.betti_numbers)

        betti_order = ent_betti_sum <= neut_betti_sum <= cont_betti_sum

        validation = {
            'status': 'PASS' if (complexity_order and persistence_order) else 'PARTIAL/FAIL',
            'complexity_order_valid': complexity_order,
            'persistence_order_valid': persistence_order,
            'entailment_complexity': ent_features.n_significant_features,
            'neutral_complexity': neut_features.n_significant_features,
            'contradiction_complexity': cont_features.n_significant_features,
            'entailment_persistence': ent_features.total_persistence,
            'neutral_persistence': neut_features.total_persistence,
            'contradiction_persistence': cont_features.total_persistence,
            'entailment_betti_sum': ent_betti_sum,
            'neutral_betti_sum': neut_betti_sum,
            'contradiction_betti_sum': cont_betti_sum
        }

        return validation

    def visualise_topological_analysis(self, analysis_results: Dict):
        """Visualise topological analysis with non-blocking saves"""
        print("\nGenerating visualisations...")

        try:
            tda_results = analysis_results['tda_results']
            if not tda_results:
                print("No TDA results to visualise")
                return

            # Create persistence diagrams
            self._create_persistence_diagrams(tda_results)

            # Create feature comparison plots
            self._create_feature_comparison(tda_results)

            # Create 2D projections
            self._create_2d_projections(tda_results)

            print("All visualizations saved successfully")

        except Exception as e:
            print(f"Error in visualization: {e}")
            import traceback
            traceback.print_exc()

    def _create_persistence_diagrams(self, tda_results):
        """Create persistence diagram plots"""
        fig, axes = plt.subplots(1, len(tda_results), figsize=(6 * len(tda_results), 5))
        if len(tda_results) == 1:
            axes = [axes]

        colours = {'entailment': 'green', 'neutral': 'blue', 'contradiction': 'red'}

        for i, (label, result) in enumerate(tda_results.items()):
            ax = axes[i]
            diagrams = result['diagrams']

            for dim, diagram in enumerate(diagrams):
                if len(diagram) > 0:
                    # Only plot finite points
                    finite_mask = np.isfinite(diagram).all(axis=1)
                    finite_diagram = diagram[finite_mask]
                    if len(finite_diagram) > 0:
                        ax.scatter(finite_diagram[:, 0], finite_diagram[:, 1],
                                   alpha=0.7, label=f'H_{dim}', s=50)

            # Add diagonal line
            if any(len(d) > 0 for d in diagrams):
                all_finite_values = []
                for d in diagrams:
                    if len(d) > 0:
                        finite_d = d[np.isfinite(d).all(axis=1)]
                        if len(finite_d) > 0:
                            all_finite_values.extend(finite_d.flatten())

                if all_finite_values:
                    max_val = max(all_finite_values)
                    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)

            ax.set_title(f'{label.title()} Persistence Diagram')
            ax.set_xlabel('Birth')
            ax.set_ylabel('Death')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'persistence_diagrams_snli_10k.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   Persistence diagrams saved")

    def _create_feature_comparison(self, tda_results):
        """Create feature comparison bar plots"""
        labels_list = list(tda_results.keys())
        colours = {'entailment': 'green', 'neutral': 'blue', 'contradiction': 'red'}

        # Debug: Print feature values first
        print(f"Debug - Feature values:")
        for label in labels_list:
            features = tda_results[label]['features']
            print(f"     {label}:")
            print(f"       Total Persistence: {features.total_persistence}")
            print(f"       Max Persistence: {features.max_persistence}")
            print(f"       Significant Features: {features.n_significant_features}")

        # Extract features (all should be finite now)
        features_comparison = {
            'Total Persistence': [tda_results[l]['features'].total_persistence for l in labels_list],
            'Max Persistence': [tda_results[l]['features'].max_persistence for l in labels_list],
            'Significant Features': [tda_results[l]['features'].n_significant_features for l in labels_list],
            'Total Betti Numbers': [sum(tda_results[l]['features'].betti_numbers) for l in labels_list]
        }

        # Print debug info
        print(f"  Features to plot (should be finite):")
        for name, vals in features_comparison.items():
            print(f"     {name}: {vals}")

            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            for i, (feature_name, values) in enumerate(features_comparison.items()):
                if i < len(axes):
                    ax = axes[i]

                    plot_colors = [colours.get(label, 'gray') for label in labels_list]

                    bars = ax.bar(labels_list, values, color=plot_colors, alpha=0.7)
                    ax.set_title(f'{feature_name} (Fixed)')
                    ax.set_ylabel('Value')

                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01 * max(values),
                                f'{value:.0f}' if isinstance(value, (int, np.integer)) else f'{value:.3f}',
                                ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(self.results_dir / 'feature_comparison_fixed_snli_10k.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Fixed feature comparison saved")

    def _create_2d_projections(self, tda_results):
        """Create 2D projections using UMAP and t-SNE"""
        try:
            # Combine all point clouds
            all_points = []
            all_labels = []
            all_indices = []
            colours = {'entailment': 'green', 'neutral': 'blue', 'contradiction': 'red'}

            current_idx = 0
            for label, result in tda_results.items():
                points = result['point_cloud']
                all_points.append(points)
                all_labels.extend([label] * len(points))

                original_indices_for_label = self.original_indices[label]
                all_indices.extend(original_indices_for_label)
                current_idx += len(points)

            if not all_points:
                print("   No point clouds available for 2D projection")
                return

            all_points = np.vstack(all_points)

            # Clean data
            finite_mask = np.isfinite(all_points).all(axis=1)
            all_points = all_points[finite_mask]
            all_labels = [all_labels[i] for i in range(len(all_labels)) if finite_mask[i]]
            all_indices = [all_indices[i] for i in range(len(all_indices)) if finite_mask[i]]

            if len(all_points) < 5:
                print("   Not enough valid points for 2D projection")
                return

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # UMAP projection
            n_neighbors = min(15, len(all_points) - 1)
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, metric='euclidean', random_state=42)
            umap_embedding = reducer.fit_transform(all_points)

            for label in set(all_labels):
                if label in colours:
                    mask = np.array(all_labels) == label
                    ax1.scatter(umap_embedding[mask, 0], umap_embedding[mask, 1],
                                c=colours[label], label=label, alpha=0.6, s=30)

            ax1.set_title('UMAP: Cone Violation Patterns')
            ax1.set_xlabel('UMAP 1')
            ax1.set_ylabel('UMAP 2')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # t-SNE projection
            perplexity = min(30, len(all_points) // 3)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            tsne_embedding = tsne.fit_transform(all_points)

            for label in set(all_labels):
                if label in colours:
                    mask = np.array(all_labels) == label
                    ax2.scatter(tsne_embedding[mask, 0], tsne_embedding[mask, 1],
                                c=colours[label], label=label, alpha=0.6, s=30)

            ax2.set_title('t-SNE: Cone Violation Patterns')
            ax2.set_xlabel('t-SNE 1')
            ax2.set_ylabel('t-SNE 2')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.results_dir / 'cone_patterns_2d_snli_10k.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("    2D projections saved")

            #Store embedding coordinates for neural network features
            self.embedding_coordinates = {
                'umap_coordinates': umap_embedding,
                'tsne_coordinates': tsne_embedding,
                'sample_labels': all_labels,
                'sample_indices': all_indices,
                'umap_reducer': reducer,
                'tsne_fitted_data': all_points
            }

            del all_points
            gc.collect()

            print(f"    Embedding coordinates saved: {len(all_indices)} samples")
            print(f"    UMAP shape: {umap_embedding.shape}, t-SNE shape: {tsne_embedding.shape}")

        except Exception as e:
            print(f"   Error creating 2D projections: {e}")

    # def extract_tda_features_for_classification(self, analysis_results: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    #     """
    #     Extract TDA-derived features that can be used for entailment classification and preserve original texts
    #
    #     This creates a feature vector for each sample based on its topological properties,
    #     which can later be used to train classifiers or as regularization features.
    #     """
    #     tda_results = analysis_results['tda_results']
    #
    #     feature_vectors = []
    #     labels = []
    #     sample_texts = []
    #     sample_metadata = []
    #
    #     # Get original texts from cone validation results
    #     premise_texts = self.cone_validation_results.get('premise_texts', [])
    #     hypothesis_texts = self.cone_validation_results.get('hypothesis_texts', [])
    #     original_labels = self.cone_validation_results.get('labels', [])
    #
    #     label_index = 0
    #
    #     for label, result in tda_results.items():
    #         original_indices_for_label = self.original_indices[label]
    #
    #         features = result['features']
    #         n_samples = result['n_points']
    #
    #         # Create feature vector for each sample in this class
    #         for i in range(n_samples):
    #             feature_vector = [
    #                 features.total_persistence,
    #                 features.max_persistence,
    #                 features.n_significant_features,
    #                 len(features.betti_numbers),
    #                 np.sum(features.betti_numbers),
    #                 np.std(features.persistence_landscape) if len(features.persistence_landscape) > 0 else 0
    #             ]
    #
    #             feature_vectors.append(feature_vector)
    #             labels.append(label)
    #
    #             # Find corresponding sample in original data by matching label
    #             # This assumes samples are in the same order as the original validation results
    #             original_idx = original_indices_for_label[i]
    #
    #             if original_idx < len(premise_texts):
    #                 sample_texts.append({
    #                     'premise': premise_texts[original_idx],
    #                     'hypothesis': hypothesis_texts[original_idx],
    #                     'label': label,
    #                     'original_index': original_idx
    #                 })
    #
    #                 sample_metadata.append({
    #                     'sample_id': original_idx,
    #                     'tda_label': label,
    #                     'original_label': original_labels[original_idx] if original_idx < len(
    #                         original_labels) else label,
    #                     'topological_features': {
    #                         'total_persistence': features.total_persistence,
    #                         'max_persistence': features.max_persistence,
    #                         'n_significant_features': features.n_significant_features,
    #                         'betti_sum': np.sum(features.betti_numbers),
    #                         'betti_numbers': features.betti_numbers
    #                     }
    #                 })
    #             else:
    #                 # Fallback if index mapping fails
    #                 sample_texts.append({
    #                     'premise': 'TEXT_NOT_FOUND',
    #                     'hypothesis': 'TEXT_NOT_FOUND',
    #                     'label': label,
    #                     'original_index': -1
    #                 })
    #                 sample_metadata.append({
    #                     'sample_id': -1,
    #                     'tda_label': label,
    #                     'original_label': label,
    #                     'topological_features': {
    #                         'total_persistence': features.total_persistence,
    #                         'max_persistence': features.max_persistence,
    #                         'n_significant_features': features.n_significant_features,
    #                         'betti_sum': np.sum(features.betti_numbers),
    #                         'betti_numbers': features.betti_numbers
    #                     }
    #                 })
    #
    #     X = np.array(feature_vectors)
    #     y = np.array(labels)
    #
    #     # Create comprehensive text data dictionary
    #     text_data = {
    #         'sample_texts': sample_texts,
    #         'sample_metadata': sample_metadata,
    #         'feature_names': [
    #             'total_persistence',
    #             'max_persistence',
    #             'n_significant_features',
    #             'n_betti_dimensions',
    #             'betti_sum',
    #             'persistence_std'
    #         ]
    #     }
    #
    #     print(f"\nExtracted TDA features: {X.shape[0]} samples × {X.shape[1]} features")
    #
    #     return X, y, text_data


    def save_neural_network_data(self, analysis_results: Dict):
        """
            NEW: Save all data needed for neural network classification.

            This saves:
            1. Point clouds for each class
            2. Class statistics (per-example averages)
            3. Individual sample data for training
            4. TDA parameters for consistency
        """
        print("\nSaving neural network classification data...")
        # Prepare neural network data package
        nn_data = {
            # Core data for perturbation analysis
            'tda_params': self.tda_params,

            'embedding_coordinates': self.embedding_coordinates,

            # Individual sample data (from validation results)
            'cone_violations': self.cone_validation_results['cone_violations'],
            'labels': self.cone_validation_results['labels'],
            'premise_texts': self.cone_validation_results.get('premise_texts', []),
            'hypothesis_texts': self.cone_validation_results.get('hypothesis_texts', []),


            # Analysis results
            'tda_results': analysis_results['tda_results'],
            'signature_comparison': analysis_results['signature_comparison']
        }

        # Save comprehensive data for neural network
        nn_data_path = self.results_dir / 'neural_network_data_snli_10k.pt'
        torch.save(nn_data, nn_data_path)

        print(f"Neural network data saved to:")
        print(f"  {nn_data_path} (complete PyTorch format)")

        # Print summary
        print(f"\nData summary for neural network:")
        print(f"  Point clouds: {list(self.point_clouds.keys())}")
        for class_name, cloud in self.point_clouds.items():
            print(f"    {class_name}: {cloud.shape}")
        print(f"  Individual samples: {len(self.cone_validation_results['labels'])}")
        print(f"  Per-example statistics computed for: {list(self.class_statistics.keys())}")


def run_tda_analysis():
    """Main execution function for TDA integration"""
    print("=" * 80)
    print("TDA INTEGRATION FOR HYPERBOLIC ENTAILMENT CONES")
    print("=" * 80)

    analyser = TDAIntegration()
    analysis_results = analyser.analyse_topological_signatures()
    analyser.visualise_topological_analysis(analysis_results)

    # # Extract features for downstream tasks
    # X, y, text_data = analyser.extract_tda_features_for_classification(analysis_results)
    #
    # np.savez(analyser.results_dir / 'tda_features_with_texts.npz', X=X, y=y,
    #          sample_texts=text_data['sample_texts'],
    #          sample_metadata=text_data['sample_metadata'],
    #          feature_names=text_data['feature_names'])
    #
    # with open(analyser.results_dir / 'tda_analysis_with_texts.json', 'w') as f:
    #     json.dump({
    #         'analysis_results': {
    #             # Convert numpy arrays to lists for JSON serialization
    #             'feature_matrix_shape': X.shape,
    #             'labels_unique': list(set(y)),
    #             'n_samples_per_label': {label: int(np.sum(y == label)) for label in set(y)}
    #         },
    #         'text_data': text_data,
    #         'methodology': {
    #             'description': 'TDA features extracted from cone violation patterns with preserved texts',
    #             'feature_interpretation': {
    #                 'total_persistence': 'Sum of all finite persistence values - higher values indicate more complex topology',
    #                 'max_persistence': 'Maximum persistence value - indicates most significant topological feature',
    #                 'n_significant_features': 'Number of features above mean persistence',
    #                 'betti_sum': 'Total topological complexity across all dimensions'
    #             }
    #         }
    #     }, f, indent=2, default=str)

    analyser.save_neural_network_data(analysis_results)

    print("\n" + "=" * 80)
    print("TDA ANALYSIS COMPLETED")
    print("=" * 80)
    print("\nFor neural network classification, use:")
    print(f"  {analyser.results_dir}/neural_network_data.pt")

    return analyser, analysis_results

if __name__ == "__main__":
    analyser, results = run_tda_analysis()




















