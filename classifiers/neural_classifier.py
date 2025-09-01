"""
TDA Neural Network Classifier for Semantic Entailment Detection - CORRECTED VERSION

This module implements a neural network classifier that uses geometric features
combined with UMAP-derived TDA features to classify entailment relationships.

Architecture: 9 input features → 128 → 64 → 32 → 16 → 3 output classes
Features: Per-example geometric + spatial context + UMAP TDA coordinates
"""

import logging

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from Cython.Shadow import returns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import wasserstein_distance
import ripser
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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



class TDAFeatureExtractor:
    """
        Extract features combining per-example geometric data with REAL TDA perturbation analysis.

        Features extracted (15 total):
        1-3: Per-example geometric (cone energy, order energy, hyperbolic distance)
        4-7: Spatial context (local density + distances to 3 class centroids)
        8-9: UMAP coordinates (TDA-derived topological features)
    """

    def __init__(self, k_neighbors: int = 5):
        """
        Initialize the feature extractor.

        Args:
            k_neighbors: Number of neighbors for local density computation
        """
        self.k_neighbors = k_neighbors
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Will be loaded from neural network data
        self.point_clouds = {}
        self.class_statistics = {}
        self.tda_params = {}
        self.class_centroids = {}
        self.training_geometric_features = None
        self.training_labels = None

        # NEW: UMAP data for TDA features
        self.umap_reducer = None
        self.embedding_coordinates = {}

        # Feature names for reference
        self.feature_names = [
            # Core geometric features (3)
            'cone_energy',
            'order_energy',
            'hyperbolic_distance',
            # Spatial features (4)
            'local_density',
            'dist_to_entailment_centroid',
            'dist_to_neutral_centroid',
            'dist_to_contradiction_centroid',
            # UMAP TDA features (2)
            'umap_x',
            'umap_y'
        ]

    def fit(self, training_data: Dict):
        """
        Fit the feature extractor on neural network training data.

        Expected training_data structure (from updated tda_integration):
        {
            'cone_violations': tensor of shape [n_samples, 3] containing
                              [cone_energy, order_energy, hyperbolic_distance] per sample
            'labels': list of string labels ['entailment', 'neutral', 'contradiction']
            'embedding_coordinates': dict with UMAP coordinates and fitted reducer

        }
        """
        logger.info("Fitting TDA feature extractor with REAL perturbation analysis...")

        # Extract geometric features (cone_energy, order_energy, hyperbolic_distance)
        if isinstance(training_data['cone_violations'], torch.Tensor):
            self.training_geometric_features = training_data['cone_violations'].numpy()
        else:
            self.training_geometric_features = np.array(training_data['cone_violations'])

        self.training_labels = training_data['labels']

        # Load UMAP data for TDA features
        self.embedding_coordinates = training_data['embedding_coordinates']
        self.umap_reducer = self.embedding_coordinates['umap_reducer']

        # Compute class centroids in geometric space
        self._compute_class_centroids()

        # Extract features for training data to fit scaler
        logger.info("Extracting features for training samples (this may take a while due to TDA computations)...")
        training_features = []

        for i in range(len(training_data['labels'])):
            if i % 200 == 0:  # Less frequent logging since this is much faster
                logger.info(f"Processing training sample {i}/{len(training_data['labels'])}")

            sample_data = {
                'geometric_features': self.training_geometric_features[i],  # [cone, order, hyperbolic]
                'sample_index': i  # For looking up UMAP coordinates
            }
            features = self._extract_single_sample_features(sample_data)
            training_features.append(features)

        training_features = np.array(training_features)

        # Fit scaler
        self.scaler.fit(training_features)
        self.is_fitted = True

        logger.info(f"Feature extractor fitted. Feature dimension: {training_features.shape[1]}")
        logger.info(f"Class centroids computed for: {list(self.class_centroids.keys())}")
        logger.info(f"UMAP reducer loaded for TDA features")

    def transform(self, sample_data: Dict) -> np.ndarray:
        """
        Transform a single sample into feature vector.

        Args:
            sample_data: Dictionary containing:
                - geometric_features: [cone_energy, order_energy, hyperbolic_distance]

        Returns:
            Normalized feature vector
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transform")

        features = self._extract_single_sample_features(sample_data)
        features = features.reshape(1, -1)
        normalized_features = self.scaler.transform(features)

        return normalized_features.flatten()

    def fit_transform(self, training_data: Dict) -> np.ndarray:
        """
        Fit extractor and transform training data.

        Args:
            training_data: Training data dictionary

        Returns:
            Normalized feature matrix [n_samples, n_features]
        """
        self.fit(training_data)

        # Transform all training samples
        logger.info("Transforming all training samples...")
        features_matrix = []
        for i in range(len(training_data['labels'])):
            if i % 200 == 0:
                logger.info(f"Transforming sample {i}/{len(training_data['labels'])}")

            sample_data = {
                'geometric_features': self.training_geometric_features[i],
                'sample_index': i  # For looking up UMAP coordinates
            }
            features = self.transform(sample_data)
            features_matrix.append(features)

        return np.array(features_matrix)

    def _compute_class_centroids(self):
        """Compute centroid for each class in geometric feature space."""
        labels = self.training_labels

        for class_name in ['entailment', 'neutral', 'contradiction']:
            class_mask = np.array(labels) == class_name
            if np.any(class_mask):
                class_features = self.training_geometric_features[class_mask]
                centroid = np.mean(class_features, axis=0)
                self.class_centroids[class_name] = centroid
                logger.info(f"Computed centroid for {class_name}: {len(class_features)} samples")

    def _extract_single_sample_features(self, sample_data: Dict) -> np.ndarray:
        """Extract all features for a single sample."""
        features = []
        geometric_features = sample_data['geometric_features']

        # Features 1-3: Core geometric features
        features.extend(geometric_features.tolist() if hasattr(geometric_features, 'tolist')
                        else list(geometric_features))

        # Feature 4: Local density (distance-weighted k-NN)
        local_density = self._compute_local_density(geometric_features)
        features.append(local_density)

        # Features 5-7: Distances to class centroids
        for class_name in ['entailment', 'neutral', 'contradiction']:
            if class_name in self.class_centroids:
                distance = np.linalg.norm(geometric_features - self.class_centroids[class_name])
                features.append(distance)
            else:
                features.append(0.0)  # Fallback

        # Features 8-9: UMAP coordinates (TDA-derived features)
        umap_coords = self._compute_umap_coordinates(sample_data)
        features.extend(umap_coords)

        return np.array(features)



    def _compute_local_density(self, geometric_features: np.ndarray) -> float:
        """Compute distance-weighted k-NN density in geometric space."""
        if self.training_geometric_features is None:
            return 0.0

        # Find k nearest neighbors in geometric space
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors, metric='euclidean')
        nbrs.fit(self.training_geometric_features)

        distances, indices = nbrs.kneighbors(geometric_features.reshape(1, -1))
        distances = distances.flatten()

        # Distance-weighted density (avoid division by zero)
        weights = 1.0 / (distances + 1e-8)
        density = np.sum(weights)

        return density

    def _compute_umap_coordinates(self, sample_data: Dict) -> List[float]:
        """
        Compute UMAP coordinates for a sample.

        For training samples: Look up pre-computed coordinates
        For test samples: Transform using fitted UMAP reducer
        """
        # Check if this is a training sample with pre-computed coordinates
        if 'sample_index' in sample_data:
            sample_idx = sample_data['sample_index']

            # Find this sample in the embedding coordinates
            embedding_coords = self.embedding_coordinates
            if sample_idx < len(embedding_coords['umap_coordinates']):
                umap_coords = embedding_coords['umap_coordinates'][sample_idx]
                return umap_coords.tolist()

        # For test samples or if lookup fails: transform using fitted reducer
        geometric_features = sample_data['geometric_features']

        if self.umap_reducer is not None:
            try:
                # Transform single sample using fitted UMAP
                test_sample = geometric_features.reshape(1, -1)
                umap_coords = self.umap_reducer.transform(test_sample)
                return umap_coords.flatten().tolist()
            except Exception as e:
                logger.warning(f"UMAP transform failed: {e}")
                raise
        else:
            logger.warning("UMAP reducer not available")
            raise



class TDANeuralClassifier(nn.Module):
    """
    Neural network classifier using geometric + REAL TDA perturbation features.

    Architecture: 15 → 128 → 64 → 32 → 16 → 3 (raw logits for CrossEntropyLoss)
    """

    def __init__(self, input_dim: int = 15, dropout_rate: float = 0.3):
        """
        Initialize the neural network.

        Args:
            input_dim: Number of input features (15 for full feature set)
            dropout_rate: Dropout probability for regularization
        """
        super(TDANeuralClassifier, self).__init__()

        self.input_dim = input_dim
        self.dropout_rate = dropout_rate

        # Network layers following architecture document
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.layer4 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.dropout4 = nn.Dropout(dropout_rate)

        # Output layer - raw logits (no activation for CrossEntropyLoss)
        self.output = nn.Linear(16, 3)  # 3 classes: entailment, neutral, contradiction

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Raw logits [batch_size, 3] - use with CrossEntropyLoss
        """
        # Layer 1: 128 neurons
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout1(x)

        # Layer 2: 64 neurons
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout2(x)

        # Layer 3: 32 neurons
        x = self.layer3(x)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.dropout3(x)

        # Layer 4: 16 neurons
        x = self.layer4(x)
        x = self.bn4(x)
        x = F.gelu(x)
        x = self.dropout4(x)

        # Output layer - raw logits (no activation)
        x = self.output(x)

        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities (applies softmax to logits)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        probabilities = self.predict_proba(x)
        return torch.argmax(probabilities, dim=1)


def load_neural_network_data(data_path: str) -> Dict:
    """
    Load data from updated TDA integration for neural network training.

    Expected file: 'results/tda_integration/neural_network_data.pt'
    """
    logger.info(f"Loading neural network data from {data_path}")

    try:
        data = torch.load(data_path, map_location='cpu')

        # Validate required fields
        required_fields = ['cone_violations', 'labels', 'embedding_coordinates']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        logger.info(f"Successfully loaded neural network data")
        logger.info(f"Samples: {len(data['labels'])}")

        # Print embedding info
        if 'embedding_coordinates' in data:
            coords = data['embedding_coordinates']
            logger.info(f"UMAP coordinates available: {coords['umap_coordinates'].shape}")
            logger.info(f"UMAP reducer loaded for transforming new samples")

        return data

    except Exception as e:
        logger.error(f"Failed to load neural network data from {data_path}: {e}")
        raise




def create_classifier_from_neural_data(
        data_path: str = "results/tda_integration/neural_network_data_SNLI_1k.pt") -> Tuple[TDANeuralClassifier, TDAFeatureExtractor]:
    """
    Create and initialize classifier from neural network data.

    Args:
        data_path: Path to neural network data file
        device: Device to use ('auto', 'cuda', 'cpu')

    Returns:
        Tuple of (classifier, feature_extractor)
    """
    # Load data
    training_data = load_neural_network_data(data_path)

    # Initialize feature extractor and fit on training data
    logger.info("Initializing feature extractor with REAL TDA perturbation analysis...")
    feature_extractor = TDAFeatureExtractor()
    features_matrix = feature_extractor.fit_transform(training_data)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create classifier with softmax output
    input_dim = features_matrix.shape[1]
    classifier = TDANeuralClassifier(input_dim=input_dim)
    classifier = classifier.to(device)

    logger.info(f"Created classifier with {input_dim} input features on device {device}")
    logger.info("Network outputs raw logits for use with CrossEntropyLoss")

    return classifier, feature_extractor


def prepare_training_data(
        features_matrix: np.ndarray,
        labels: List[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare training data for PyTorch training.

    Args:
        features_matrix: Normalized feature matrix from feature extractor
        labels: String labels

    Returns:
        Tuple of (features_tensor, labels_tensor)
    """
    # Convert features to tensor
    X = torch.FloatTensor(features_matrix)

    # Convert string labels to numeric
    label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    numeric_labels = [label_to_idx[label] for label in labels]
    y = torch.LongTensor(numeric_labels)

    logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Label distribution: {torch.bincount(y)}")

    return X, y


if __name__ == "__main__":
    # Test the implementation
    logger.info("Testing TDA Neural Classifier with REAL perturbation analysis...")

    # Test with neural network data
    neural_data_path = "results/tda_integration/neural_network_data_SNLI_1k.pt"

    try:
        classifier, feature_extractor = create_classifier_from_neural_data(neural_data_path)
        logger.info("✓ Successfully created classifier and feature extractor")

        # Test forward pass with dummy data
        batch_size = 16  # Smaller batch for testing
        input_dim = classifier.input_dim
        dummy_input = torch.randn(batch_size, input_dim)

        # Test forward pass (raw logits)
        output = classifier(dummy_input)
        predictions = classifier.predict(dummy_input)

        logger.info(f"✓ Forward pass successful: {output.shape}")
        logger.info(f"✓ Output is raw logits (no activation): range [{output.min():.4f}, {output.max():.4f}]")
        logger.info(f"✓ Predictions shape: {predictions.shape}")

        # Test with CrossEntropyLoss (correct for raw logits)
        criterion = nn.CrossEntropyLoss()
        dummy_labels = torch.randint(0, 3, (batch_size,))
        loss = criterion(output, dummy_labels)

        logger.info(f"✓ CrossEntropyLoss test successful: {loss.item():.4f}")
        logger.info("✓ Network is ready for CrossEntropyLoss training!")

        logger.info("TDA Neural Classifier implementation test completed successfully!")

        # Test a single sample transformation
        if hasattr(feature_extractor, 'training_geometric_features'):
            test_sample = feature_extractor.training_geometric_features[0]
            test_sample_data = {'geometric_features': test_sample}

            logger.info("Testing single sample transformation with REAL TDA perturbation...")
            features = feature_extractor.transform(test_sample_data)
            logger.info(f"✓ Single sample features extracted: {len(features)} features")
            logger.info(f"Feature range: [{features.min():.4f}, {features.max():.4f}]")

            # Test UMAP transform specifically
            umap_coords = feature_extractor._compute_umap_coordinates(test_sample_data)
            logger.info(f"✓ UMAP coordinates computed: {umap_coords}")

    except FileNotFoundError:
        logger.warning(f"Neural network data file not found at {neural_data_path}")









