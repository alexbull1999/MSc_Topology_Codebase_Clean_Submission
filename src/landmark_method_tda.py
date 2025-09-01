"""
Landmark-Based TDA Integration for Hyperbolic Entailment Cones

This script replaces the original, computationally intensive TDA pipeline. It uses a 
landmark-based approach to generate unique, per-example topological features in a 
highly efficient and scalable manner. This is the recommended approach for large datasets.

Methodology:
1.  A small, representative subset of points (landmarks) is selected from the data.
2.  A rich "local topological signature" (based on H₀ and H₁ persistence) is 
    computed *only* for these landmark points by analyzing their local neighborhoods.
3.  A unique feature vector is generated for every point in the full dataset by 
    calculating a weighted average of the signatures of its nearest landmarks.
4.  These new topological features are combined with the original geometric features
    (cone energy, etc.) and saved for the neural classifier.
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
warnings.filterwarnings("ignore", category=UserWarning) # Ripser can be noisy
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LandmarkTDAExtractor:
    """
    Extracts per-sample topological features using a landmark-based method.
    """
    def __init__(self,
                 n_landmarks: int = 400,
                 n_neighbors_for_signature: int = 50,
                 n_neighbors_for_propagation: int = 5):
        """
        Initializes the landmark-based TDA feature extractor.

        Args:
            n_landmarks (int): The number of landmark points to select from the dataset.
                               A good value is 3-5% of the total sample size.
            n_neighbors_for_signature (int): The size of the local neighborhood (k) used
                                             to compute the topological signature for each landmark.
            n_neighbors_for_propagation (int): The number of nearest landmarks (m) to use
                                               when propagating features to all other points.
        """
        self.n_landmarks = n_landmarks
        self.k = n_neighbors_for_signature
        self.m = n_neighbors_for_propagation
        
        self.all_points = None
        self.all_labels = None
        self.landmark_indices = None
        self.landmark_signatures = None
        self.propagated_features = None

        logging.info("Landmark TDA Extractor Initialized:")
        logging.info(f"  - Number of landmarks: {self.n_landmarks}")
        logging.info(f"  - Neighborhood size for landmarks (k): {self.k}")
        logging.info(f"  - Neighbors for propagation (m): {self.m}")

    def fit_transform(self, cone_violations: np.ndarray, labels: List[str]) -> np.ndarray:
        """
        Executes the full pipeline: selects landmarks, computes their signatures,
        and propagates the topological features to every sample.

        Args:
            cone_violations (np.ndarray): The geometric feature vectors for all samples.
            labels (List[str]): The list of labels for all samples.

        Returns:
            np.ndarray: A matrix of shape (n_samples, n_topological_features) containing
                        the new feature vector for each sample.
        """
        self.all_points = cone_violations
        self.all_labels = np.array(labels)

        # 1. Select landmark points
        self._select_landmarks_stratified()

        # 2. Compute topological signatures for each landmark
        self._compute_landmark_signatures()

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
            n_landmarks_class = max(1, n_landmarks_class) # Ensure at least one
            
            # Get indices for the current class
            class_indices = np.where(self.all_labels == label)[0]
            
            if len(class_indices) < n_landmarks_class:
                 logging.warning(f"Class '{label}' has fewer samples ({len(class_indices)}) than landmarks to be selected ({n_landmarks_class}). Using all samples as landmarks.")
                 selected_indices = class_indices
            else:
                 # Randomly choose landmarks from this class
                 selected_indices = np.random.choice(class_indices, n_landmarks_class, replace=False)

            self.landmark_indices.extend(selected_indices)
        
        self.landmark_indices = np.array(self.landmark_indices)
        logging.info(f"Selected {len(self.landmark_indices)} total landmarks.")

    def _compute_landmark_signatures(self):
        """
        Computes the local topological signature for each landmark point.
        This is the most computationally intensive step, but is only performed on the landmarks.
        """
        logging.info(f"Computing topological signatures for {len(self.landmark_indices)} landmarks...")

        # Fit nearest neighbors on the *entire* dataset to find local neighborhoods
        nn_finder = NearestNeighbors(n_neighbors=self.k + 1, n_jobs=-1)
        nn_finder.fit(self.all_points)
        
        landmark_points = self.all_points[self.landmark_indices]
        _, neighbor_indices = nn_finder.kneighbors(landmark_points)

        signatures = []
        for i in tqdm(range(len(self.landmark_indices)), desc="Calculating Landmark Signatures"):
            # Get the local neighborhood for the current landmark
            local_neighborhood_indices = neighbor_indices[i]
            local_point_cloud = self.all_points[local_neighborhood_indices]

            # Compute persistence diagram for the local neighborhood
            # We only care about H0 and H1 as H2+ is less informative for small 3D clouds
            diagrams = ripser.ripser(local_point_cloud, maxdim=1)['dgms']
            
            # Extract features from the diagram
            h0_persistence = self._calculate_total_persistence(diagrams[0])
            h1_persistence = self._calculate_total_persistence(diagrams[1])
            h1_max_persistence = self._calculate_max_persistence(diagrams[1])
            h1_feature_count = len(diagrams[1]) if diagrams[1] is not None else 0

            signatures.append([
                h0_persistence,
                h1_persistence,
                h1_max_persistence,
                h1_feature_count
            ])

        self.landmark_signatures = np.array(signatures)
        logging.info(f"Computed landmark signatures. Shape: {self.landmark_signatures.shape}")
        
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

    def _propagate_features(self):
        """
        Propagates the topological signatures from landmarks to all other points
        using a weighted average of the nearest landmarks.
        """
        logging.info("Propagating topological features to all samples...")
        
        # Fit nearest neighbors on the *landmark points* only
        landmark_points = self.all_points[self.landmark_indices]
        landmark_finder = NearestNeighbors(n_neighbors=self.m, n_jobs=-1)
        landmark_finder.fit(landmark_points)

        # Find the m-nearest landmarks for every point in the full dataset
        distances, landmark_neighbor_indices = landmark_finder.kneighbors(self.all_points)

        propagated_features = []
        epsilon = 1e-8  # To avoid division by zero

        for i in tqdm(range(len(self.all_points)), desc="Propagating Features"):
            # Get the indices and distances of the nearest landmarks for the current point
            neighbor_dists = distances[i]
            neighbor_indices = landmark_neighbor_indices[i]

            # Inverse distance weighting
            weights = 1.0 / (neighbor_dists + epsilon)
            weights /= np.sum(weights) # Normalize weights to sum to 1

            # Get the signatures of the nearest landmarks
            nearest_landmark_signatures = self.landmark_signatures[neighbor_indices]

            # Compute the weighted average of the signatures
            weighted_avg_signature = np.sum(nearest_landmark_signatures * weights[:, np.newaxis], axis=0)
            propagated_features.append(weighted_avg_signature)

        self.propagated_features = np.array(propagated_features)
        logging.info(f"Propagated features to all points. Final shape: {self.propagated_features.shape}")

def main():
    """
    Main execution function to run the landmark-based TDA pipeline.
    """
    # --- Configuration ---
    # This assumes the file from your validation script exists
    INPUT_DATA_PATH = Path("validation_results/tda_ready_data_snli_10k.pt") 
    OUTPUT_DIR = Path("results/tda_integration/landmark_tda_features")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- Load Data ---
    if not INPUT_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Input data not found at '{INPUT_DATA_PATH}'. "
            "Please run 'tda_ready_cone_validation_texts_preserved.py' first."
        )
    
    logging.info(f"Loading data from {INPUT_DATA_PATH}...")
    data = torch.load(INPUT_DATA_PATH, map_location='cpu')
    cone_violations = data['cone_violations'].numpy()
    labels = data['labels']
    
    logging.info(f"Loaded {len(labels)} samples.")

    # --- Run Landmark TDA Feature Extraction ---
    # These parameters can be tuned depending on your dataset size and desired resolution
    extractor = LandmarkTDAExtractor(
        n_landmarks=400,
        n_neighbors_for_signature=75,
        n_neighbors_for_propagation=5
    )
    topological_features = extractor.fit_transform(cone_violations, labels)

    # --- Combine Features and Save for Classifier ---
    logging.info("Combining geometric features with new topological features...")

    # The final feature set for the classifier
    final_features = np.hstack((cone_violations, topological_features))
    
    feature_names = [
        'cone_energy',
        'order_energy',
        'hyperbolic_distance',
        'h0_total_persistence',
        'h1_total_persistence',
        'h1_max_persistence',
        'h1_feature_count'
    ]
    
    logging.info(f"Final feature matrix created with shape: {final_features.shape}")
    
    # --- Analyze Results ---
    logging.info("\n" + "="*50)
    logging.info("      Average Topological Signatures per Class      ")
    logging.info("="*50)
    
    # Create a pandas DataFrame for easy analysis
    import pandas as pd
    df = pd.DataFrame(topological_features, columns=feature_names[3:])
    df['label'] = labels
    
    class_averages = df.groupby('label').mean()
    print(class_averages.to_string())
    logging.info("="*50 + "\n")

    # --- Save Data ---
    classifier_data_path = OUTPUT_DIR / "neural_network_features_snli_10k.pt"
    
    classifier_data = {
        'features': torch.from_numpy(final_features).float(),
        'labels': data['labels'],
        'feature_names': feature_names,
        'premise_texts': data.get('premise_texts', []),
        'hypothesis_texts': data.get('hypothesis_texts', []),
        'sample_metadata': data.get('sample_metadata', [])
    }
    
    torch.save(classifier_data, classifier_data_path)
    logging.info(f"Success! Data for the neural classifier saved to:\n{classifier_data_path}")
    logging.info("You can now use this file as input for 'train_classifier.py'.")

if __name__ == "__main__":
    # Set a seed for reproducibility of landmark selection
    np.random.seed(42)
    main()
