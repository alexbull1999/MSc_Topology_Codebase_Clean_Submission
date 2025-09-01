
"""
Enhanced Landmark-Based TDA Integration for Hyperbolic Entailment Cones with Asymmetric Features

This script uses a landmark-based approach to generate topological features from the enhanced
10D asymmetric feature space, then combines them with all geometric features for classification.

Enhanced Methodology:
1. Uses the enhanced 10D feature space (cone energies + asymmetric features) 
2. Selects landmarks from this richer geometric space
3. Computes local topological signatures around landmarks in 10D space
4. Propagates TDA features to all points 
5. Combines 10D geometric + 4D topological = 14D final feature space
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import ripser
import logging
import warnings

# --- Setup ---
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EnhancedLandmarkTDAExtractor:
    """
    Extracts per-sample topological features using enhanced 10D geometric features.
    """
    def __init__(self,
                 n_landmarks: int = 400,
                 n_neighbors_for_signature: int = 50,
                 n_neighbors_for_propagation: int = 5,
                 use_enhanced_features: bool = True):
        """
        Initializes the enhanced landmark-based TDA feature extractor.

        Args:
            n_landmarks (int): Number of landmark points (3-5% of dataset size)
            n_neighbors_for_signature (int): Local neighborhood size for TDA computation
            n_neighbors_for_propagation (int): Number of nearest landmarks for propagation
            use_enhanced_features (bool): Whether to use 10D enhanced features or 3D standard
        """
        self.n_landmarks = n_landmarks
        self.k = n_neighbors_for_signature
        self.m = n_neighbors_for_propagation
        self.use_enhanced_features = use_enhanced_features
        
        self.all_points = None
        self.all_labels = None
        self.landmark_indices = None
        self.landmark_signatures = None
        self.propagated_features = None

        feature_type = "Enhanced 10D" if use_enhanced_features else "Standard 3D"
        logging.info(f"Enhanced Landmark TDA Extractor Initialized ({feature_type}):")
        logging.info(f"  - Number of landmarks: {self.n_landmarks}")
        logging.info(f"  - Neighborhood size for landmarks (k): {self.k}")
        logging.info(f"  - Neighbors for propagation (m): {self.m}")
        logging.info(f"  - Using enhanced asymmetric features: {self.use_enhanced_features}")

    def fit_transform(self, enhanced_features: np.ndarray, labels: List[str]) -> np.ndarray:
        """
        Executes the enhanced TDA pipeline using 10D asymmetric features.

        Args:
            enhanced_features (np.ndarray): The enhanced 10D feature vectors [n_samples, 10]
            labels (List[str]): The list of labels for all samples

        Returns:
            np.ndarray: A matrix of shape (n_samples, 4) containing topological features
        """
        self.all_points = enhanced_features
        self.all_labels = np.array(labels)

        logging.info(f"Processing {len(labels)} samples with {enhanced_features.shape[1]}D features")

        # 1. Select landmark points using enhanced features
        self._select_landmarks_stratified()

        # 2. Compute topological signatures for each landmark in enhanced space
        self._compute_enhanced_landmark_signatures()

        # 3. Propagate these features to all points in the dataset
        self._propagate_features()

        return self.propagated_features

    def _select_landmarks_stratified(self):
        """
        Selects a stratified subset of landmarks to ensure representation from all classes.
        """
        logging.info(f"Selecting {self.n_landmarks} landmarks using stratified sampling...")
        
        unique_labels, class_counts = np.unique(self.all_labels, return_counts=True)
        self.landmark_indices = []
        
        for label, count in zip(unique_labels, class_counts):
            # Number of landmarks to pick from this class
            n_landmarks_class = int(np.round(self.n_landmarks * (count / len(self.all_labels))))
            n_landmarks_class = max(1, n_landmarks_class)
            
            # Get indices for the current class
            class_indices = np.where(self.all_labels == label)[0]
            
            if len(class_indices) < n_landmarks_class:
                logging.warning(f"Class '{label}' has fewer samples ({len(class_indices)}) than landmarks to be selected ({n_landmarks_class}). Using all samples as landmarks.")
                selected_indices = class_indices
            else:
                # Use enhanced clustering for better landmark selection
                if len(class_indices) > n_landmarks_class * 3:
                    # Use K-means clustering for better diversity in enhanced space
                    class_points = self.all_points[class_indices]
                    kmeans = MiniBatchKMeans(n_clusters=n_landmarks_class, random_state=42, n_init=3)
                    cluster_labels = kmeans.fit_predict(class_points)
                    
                    # Select one point closest to each cluster center
                    selected_indices = []
                    for cluster_id in range(n_landmarks_class):
                        cluster_mask = cluster_labels == cluster_id
                        if np.any(cluster_mask):
                            cluster_points = class_points[cluster_mask]
                            cluster_center = kmeans.cluster_centers_[cluster_id]
                            
                            # Find closest point to cluster center
                            distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
                            closest_idx = np.argmin(distances)
                            original_idx = class_indices[cluster_mask][closest_idx]
                            selected_indices.append(original_idx)
                    
                    selected_indices = np.array(selected_indices)
                else:
                    # Random selection for smaller classes
                    selected_indices = np.random.choice(class_indices, n_landmarks_class, replace=False)

            self.landmark_indices.extend(selected_indices)
        
        self.landmark_indices = np.array(self.landmark_indices)
        logging.info(f"Selected {len(self.landmark_indices)} total landmarks from enhanced feature space.")

    def _compute_enhanced_landmark_signatures(self):
        """
        Computes local topological signatures using the enhanced 10D feature space.
        """
        logging.info(f"Computing enhanced topological signatures for {len(self.landmark_indices)} landmarks...")

        # Fit nearest neighbors on the enhanced feature space
        nn_finder = NearestNeighbors(n_neighbors=self.k + 1, n_jobs=-1)
        nn_finder.fit(self.all_points)
        
        landmark_points = self.all_points[self.landmark_indices]
        _, neighbor_indices = nn_finder.kneighbors(landmark_points)

        signatures = []
        for i in tqdm(range(len(self.landmark_indices)), desc="Computing Enhanced TDA Signatures"):
            # Get the local neighborhood in enhanced 10D space
            local_neighborhood_indices = neighbor_indices[i]
            local_point_cloud = self.all_points[local_neighborhood_indices]

            # Compute persistence diagram for the enhanced local neighborhood
            try:
                diagrams = ripser.ripser(local_point_cloud, maxdim=1)['dgms']
                
                # Extract enhanced features from the diagram
                h0_persistence = self._calculate_total_persistence(diagrams[0])
                h1_persistence = self._calculate_total_persistence(diagrams[1])
                h1_max_persistence = self._calculate_max_persistence(diagrams[1])
                h1_feature_count = len(diagrams[1]) if diagrams[1] is not None else 0
                
                # Additional enhanced features for richer topological description
                h0_feature_count = len(diagrams[0]) if diagrams[0] is not None else 0
                h1_persistence_variance = self._calculate_persistence_variance(diagrams[1])

            except Exception as e:
                logging.warning(f"Error computing persistence for landmark {i}: {e}")
                # Use zero features if computation fails
                raise

            signatures.append([
                h0_persistence,
                h1_persistence,
                h1_max_persistence,
                h1_feature_count
            ])

        self.landmark_signatures = np.array(signatures)
        logging.info(f"Computed enhanced landmark signatures. Shape: {self.landmark_signatures.shape}")
        
    def _calculate_total_persistence(self, diagram: np.ndarray) -> float:
        """Helper to calculate the sum of (death - birth) for a diagram."""
        if diagram is None or len(diagram) == 0:
            return 0.0
        # Filter out infinite death times for H0
        finite_intervals = diagram[np.isfinite(diagram[:, 1])]
        if len(finite_intervals) == 0:
            return 0.0
        return np.sum(finite_intervals[:, 1] - finite_intervals[:, 0])

    def _calculate_max_persistence(self, diagram: np.ndarray) -> float:
        """Helper to find the max persistence in a diagram."""
        if diagram is None or len(diagram) == 0:
            return 0.0
        finite_intervals = diagram[np.isfinite(diagram[:, 1])]
        if len(finite_intervals) == 0:
            return 0.0
        return np.max(finite_intervals[:, 1] - finite_intervals[:, 0])

    def _calculate_persistence_variance(self, diagram: np.ndarray) -> float:
        """Helper to calculate variance in persistence values."""
        if diagram is None or len(diagram) == 0:
            return 0.0
        finite_intervals = diagram[np.isfinite(diagram[:, 1])]
        if len(finite_intervals) < 2:
            return 0.0
        persistences = finite_intervals[:, 1] - finite_intervals[:, 0]
        return np.var(persistences)

    def _propagate_features(self):
        """
        Propagates the topological signatures from landmarks to all other points
        using a weighted average of the nearest landmarks in enhanced space.
        """
        logging.info("Propagating enhanced topological features to all samples...")
        
        # Fit nearest neighbors on the landmark points in enhanced space
        landmark_points = self.all_points[self.landmark_indices]
        landmark_finder = NearestNeighbors(n_neighbors=self.m, n_jobs=-1)
        landmark_finder.fit(landmark_points)

        # Find the m-nearest landmarks for every point in the enhanced space
        distances, landmark_neighbor_indices = landmark_finder.kneighbors(self.all_points)

        propagated_features = []
        epsilon = 1e-8  # To avoid division by zero

        for i in tqdm(range(len(self.all_points)), desc="Propagating Enhanced Features"):
            # Get the indices and distances of the nearest landmarks
            neighbor_dists = distances[i]
            neighbor_indices = landmark_neighbor_indices[i]

            # Inverse distance weighting in enhanced space
            weights = 1.0 / (neighbor_dists + epsilon)
            weights /= np.sum(weights)  # Normalize weights

            # Get the signatures of the nearest landmarks
            nearest_landmark_signatures = self.landmark_signatures[neighbor_indices]

            # Compute the weighted average of the signatures
            weighted_avg_signature = np.sum(nearest_landmark_signatures * weights[:, np.newaxis], axis=0)
            propagated_features.append(weighted_avg_signature)

        self.propagated_features = np.array(propagated_features)
        logging.info(f"Propagated enhanced features to all points. Final TDA shape: {self.propagated_features.shape}")

    def save_landmark_model(self, output_path: Path):
        """
        Save the trained landmark model for later application to test data

        Args:
            output_path: Path to save the landmark model
        """
        if self.landmark_indices is None or self.landmark_signatures is None:
            raise ValueError("Model must be fitted before saving. Call fit_transform() first.")

        # Extract landmark points (their positions in feature space)
        landmark_points = self.all_points[self.landmark_indices]

        # Prepare model data
        landmark_model_data = {
            'landmark_points': landmark_points,  # [n_landmarks, feature_dim]
            'landmark_signatures': self.landmark_signatures,  # [n_landmarks, 4]
            'landmark_indices': self.landmark_indices,  # Original indices in training data
            'n_landmarks': len(self.landmark_indices),
            'n_neighbors_for_signature': self.k,
            'n_neighbors_for_propagation': self.m,
            'use_enhanced_features': self.use_enhanced_features,
            'feature_names': [
                'cone_energy', 'order_energy', 'hyperbolic_distance',
                'forward_cone', 'backward_cone', 'cone_asymmetry',
                'forward_energy', 'backward_energy', 'asymmetric_energy', 'asymmetry_measure'
            ],
            'tda_feature_names': [
                'tda_h0_total_persistence',
                'tda_h1_total_persistence',
                'tda_h1_max_persistence',
                'tda_h1_feature_count'
            ],
            'training_metadata': {
                'n_training_samples': len(self.all_labels),
                'feature_dim': self.all_points.shape[1],
                'landmark_selection_method': 'stratified_clustering'
            }
        }

        # Save the model
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(landmark_model_data, output_path)


def main():
    """
    Main execution function to run the enhanced landmark-based TDA pipeline.
    """
    # --- Configuration ---
    # Try to load enhanced data first, fall back to standard if not available
    INPUT_PATHS = [
        Path("validation_results/enhanced_tda_ready_data_snli_10k_asymmetry.pt"),  # Enhanced 10D features
    ]
    
    OUTPUT_DIR = Path("results/tda_integration/landmark_tda_features")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- Find and Load Data ---
    data_path = None
    use_enhanced = False
    
    for path in INPUT_PATHS:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError(
            f"No input data found. Please run enhanced cone validation first.\n"
            f"Expected files: {[str(p) for p in INPUT_PATHS]}"
        )
    
    logging.info(f"Loading data from {data_path}...")
    data = torch.load(data_path, map_location='cpu')
    
    # Extract the appropriate feature set
    geometric_features = data['enhanced_features'].numpy()
    feature_type = "Enhanced 10D"
    logging.info("Using enhanced 10D asymmetric features!")
   
    
    labels = data['labels']
    
    logging.info(f"Loaded {len(labels)} samples with {geometric_features.shape[1]}D {feature_type} features")

    # --- Run Enhanced Landmark TDA Feature Extraction ---
    extractor = EnhancedLandmarkTDAExtractor(
        n_landmarks=min(400, len(labels) // 25),  # Adaptive landmark count
        n_neighbors_for_signature=50,
        n_neighbors_for_propagation=5,
        use_enhanced_features=use_enhanced
    )
    
    topological_features = extractor.fit_transform(geometric_features, labels)

    landmark_model_path = OUTPUT_DIR / "asymmetry_landmark_tda_model_blind_tests.pt"
    extractor.save_landmark_model(landmark_model_path)

    # --- Combine Enhanced Features ---
    logging.info("Combining enhanced geometric features with new topological features...")

    # Final enhanced feature set: geometric (10D) + topological (4D) = 14D
    final_features = np.hstack((geometric_features, topological_features))
    
    # Enhanced feature names
    geometric_feature_names = [
        'cone_energy', 'order_energy', 'hyperbolic_distance',          # Standard features (0-2)
        'forward_cone', 'backward_cone', 'cone_asymmetry',            # Cone directional (3-5)
        'forward_energy', 'backward_energy', 'asymmetric_energy', 'asymmetry_measure'  # Order directional (6-9)
    ]
   
    
    topological_feature_names = [
        'tda_h0_total_persistence',
        'tda_h1_total_persistence', 
        'tda_h1_max_persistence',
        'tda_h1_feature_count'
    ]
    
    all_feature_names = geometric_feature_names + topological_feature_names
    
    logging.info(f"Final enhanced feature matrix: {final_features.shape}")
    logging.info(f"Feature breakdown: {len(geometric_feature_names)}D geometric + {len(topological_feature_names)}D topological = {len(all_feature_names)}D total")
    
    # --- Enhanced Analysis ---
    logging.info("\n" + "="*60)
    logging.info("    Enhanced Topological Signatures per Class    ")
    logging.info("="*60)
    
    import pandas as pd
    
    # Create comprehensive feature analysis
    full_df = pd.DataFrame(final_features, columns=all_feature_names)
    full_df['label'] = labels
    
    # Topological features analysis
    tda_df = pd.DataFrame(topological_features, columns=topological_feature_names)
    tda_df['label'] = labels
    
    print("Topological Features by Class:")
    tda_class_averages = tda_df.groupby('label').mean()
    print(tda_class_averages.to_string())
    
    print("\nAsymmetric Features by Class:")
    asymmetric_features = ['forward_cone', 'backward_cone', 'cone_asymmetry', 'forward_energy', 'backward_energy', 'asymmetric_energy', 'asymmetry_measure']
    asymmetric_df = full_df[asymmetric_features + ['label']]
    asymmetric_class_averages = asymmetric_df.groupby('label').mean()
    print(asymmetric_class_averages.to_string())
    
    logging.info("="*60 + "\n")

    # --- Save Enhanced Data for Neural Classifier ---
    classifier_data_path = OUTPUT_DIR / f"enhanced_neural_network_features_snli_10k_2(delete).pt"
    
    enhanced_classifier_data = {
        'features': torch.from_numpy(final_features).float(),
        'labels': data['labels'],
        'feature_names': all_feature_names,
        'geometric_feature_names': geometric_feature_names,
        'topological_feature_names': topological_feature_names,
        'n_geometric_features': len(geometric_feature_names),
        'n_topological_features': len(topological_feature_names), 
        'n_total_features': len(all_feature_names),
        'use_enhanced_features': use_enhanced,
        'premise_texts': data.get('premise_texts', []),
        'hypothesis_texts': data.get('hypothesis_texts', []),
        'sample_metadata': data.get('sample_metadata', []),
        'enhanced_energy_hierarchy': data.get('enhanced_energy_hierarchy', {}),
        'asymmetric_patterns_validated': data.get('asymmetric_patterns_validated', False)
    }
    
    torch.save(enhanced_classifier_data, classifier_data_path)
    
    # Also save a summary for quick reference
    summary_path = OUTPUT_DIR / "feature_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Enhanced TDA Feature Summary\n")
        f.write(f"============================\n\n")
        f.write(f"Dataset: {len(labels)} samples\n")
        f.write(f"Feature Type: {feature_type}\n")
        f.write(f"Total Features: {len(all_feature_names)}\n")
        f.write(f"Geometric Features: {len(geometric_feature_names)}\n")
        f.write(f"Topological Features: {len(topological_feature_names)}\n\n")
        f.write(f"Feature Names:\n")
        for i, name in enumerate(all_feature_names):
            f.write(f"  {i:2d}. {name}\n")

if __name__ == "__main__":
    # Set a seed for reproducibility of landmark selection
    np.random.seed(42)
    main()
