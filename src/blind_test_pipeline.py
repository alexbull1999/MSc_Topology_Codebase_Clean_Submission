import argparse
import json
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import os

from text_processing import TextToEmbedding
from entailment_cones_asymmetry import EnhancedHyperbolicConeEmbeddingPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BlindTestProcessor:
    """
    Process test data through the complete pipeline without seeing labels
    """

    def __init__(self,
                 bert_model_name: str = "bert-base-uncased",
                 device: str = None):
        """
        Initialize the blind test processor

        Args:
            bert_model_name: BERT model to use for text embeddings
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing blind test processor on {self.device}")

        # Initialize text processor (label-blind)
        logger.info(f"Loading {bert_model_name} for text processing...")
        self.text_processor = TextToEmbedding(model_name=bert_model_name, device=self.device)

        # Initialize cone pipeline with pre-trained models (label-blind)
        logger.info("Loading pre-trained order embeddings and hyperbolic pipeline...")
        try:
            self.cone_pipeline = EnhancedHyperbolicConeEmbeddingPipeline(model_path="models/enhanced_order_embeddings_snli_10k_asymmetry.pt")
            logger.info("Pre-trained models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load pre-trained models: {e}")
            logger.error("Make sure you have trained order embeddings model available!")
            raise


    def load_test_data(self, test_data_path: str) -> Dict:
        """
        Load raw test data from JSON file

        Args:
            test_data_path: Path to SNLI test JSON file

        Returns:
            Dictionary with premises, hypotheses, and labels
        """
        logger.info(f"Loading test data from {test_data_path}")

        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test data not found at {test_data_path}")

        with open(test_data_path, 'r') as f:
            raw_data = json.load(f)

        # Extract components
        premises = [item[0] for item in raw_data]
        hypotheses = [item[1] for item in raw_data]
        labels = [item[2] for item in raw_data]  # Store but don't use until evaluation

        logger.info(f"Loaded {len(raw_data)} test samples")
        logger.info(f"Labels present but will be ignored until evaluation: {set(labels)}")

        return {
            'premises': premises,
            'hypotheses': hypotheses,
            'labels': labels,  # Keep for final evaluation only
            'n_samples': len(raw_data)
        }

    def extract_bert_embeddings(self, premises: List[str], hypotheses: List[str]) -> Dict:
        """
        Convert text to BERT embeddings (completely label-blind)

        Args:
            premises: List of premise texts
            hypotheses: List of hypothesis texts

        Returns:
            Dictionary with BERT embeddings
        """
        logger.info("Extracting BERT embeddings (label-blind)")

        # Process premises
        logger.info("Processing premise texts...")
        premise_embeddings = self.text_processor.encode_text(premises)

        # Process hypotheses
        logger.info("Processing hypothesis texts...")
        hypothesis_embeddings = self.text_processor.encode_text(hypotheses)

        logger.info(
            f"BERT embeddings extracted: {premise_embeddings.shape} premise, {hypothesis_embeddings.shape} hypothesis")

        return {
            'premise_embeddings': premise_embeddings,
            'hypothesis_embeddings': hypothesis_embeddings,
            'embedding_dim': premise_embeddings.shape[1]
        }

    def extract_geometric_features(self, premise_embeddings: torch.Tensor,
                                   hypothesis_embeddings: torch.Tensor) -> Dict:
        """
        Extract geometric features using pre-trained models (completely label-blind)

        Args:
            premise_embeddings: BERT embeddings for premises [n_samples, 768]
            hypothesis_embeddings: BERT embeddings for hypotheses [n_samples, 768]

        Returns:
            Dictionary with geometric features
        """
        logger.info("Extracting geometric features using pre-trained models (label-blind)")

        # Move to correct device
        premise_embeddings = premise_embeddings.to(self.cone_pipeline.hyperbolic_pipeline.device)
        hypothesis_embeddings = hypothesis_embeddings.to(self.cone_pipeline.hyperbolic_pipeline.device)

        # Extract features using compute_enhanced_cone_energies (completely label-blind)
        logger.info("Computing enhanced cone energies...")
        cone_results = self.cone_pipeline.compute_enhanced_cone_energies(
            premise_embeddings, hypothesis_embeddings
        )

        # Convert to numpy for easier handling
        features_dict = {}
        for key, tensor_val in cone_results.items():
            if isinstance(tensor_val, torch.Tensor):
                features_dict[key] = tensor_val.detach().cpu().numpy()
            else:
                features_dict[key] = tensor_val

        logger.info("Geometric features extracted successfully")

        return features_dict

    def create_geometric_feature_matrix(self, geometric_features: Dict) -> Tuple[np.ndarray, List[str]]:
        """
        Create the 10D geometric feature matrix

        Args:
            geometric_features: Dictionary with all geometric features

        Returns:
            Tuple of (geometric_feature_matrix, geometric_feature_names)
        """
        logger.info("Creating geometric feature matrix...")

        # Define the 10D enhanced feature set (same as training feature labels)
        feature_names = [
            'cone_energy', 'order_energy', 'hyperbolic_distance',  # Standard features (0-2)
            'forward_cone', 'backward_cone', 'cone_asymmetry',  # Cone directional (3-5)
            'forward_energy', 'backward_energy', 'asymmetric_energy', 'asymmetry_measure'  # Order directional (6-9)
        ]

        # Map from cone_results to feature matrix
        feature_mapping = {
            'cone_energy': 'cone_energies',
            'order_energy': 'order_energies',
            'hyperbolic_distance': 'hyperbolic_distances',
            'forward_cone': 'forward_cone_energies',
            'backward_cone': 'backward_cone_energies',
            'cone_asymmetry': 'cone_asymmetries',
            'forward_energy': 'forward_energies',
            'backward_energy': 'backward_energies',
            'asymmetric_energy': 'asymmetric_energies',
            'asymmetry_measure': 'asymmetry_measures'
        }

        # Build feature matrix
        feature_vectors = []
        for feature_name in feature_names:
            key = feature_mapping[feature_name]
            if key in geometric_features:
                feature_vectors.append(geometric_features[key])
            else:
                logger.warning(f"Feature {key} not found in geometric_features, using zeros")
                raise

        # Stack into matrix [n_samples, n_features]
        feature_matrix = np.column_stack(feature_vectors)

        logger.info(f"Geometric feature matrix created: {feature_matrix.shape} ({len(feature_names)} features)")

        return feature_matrix, feature_names


    def apply_landmark_tda(self, geometric_features: np.ndarray, landmark_model_path: str) -> np.ndarray:
        """
        Apply pre-trained landmark TDA model to test geometric features (label-blind)

        Args:
            geometric_features: [n_samples, 10] geometric feature matrix
            landmark_model_path: Path to trained landmark TDA model

        Returns:
            TDA features [n_samples, 4]
        """
        logger.info("Applying pre-trained landmark TDA model (label-blind)...")

        try:
            # Load the saved landmark model
            landmark_data = torch.load(landmark_model_path, map_location='cpu')

            # Extract landmark information
            landmark_points = landmark_data['landmark_points']  # [n_landmarks, 10]
            landmark_signatures = landmark_data['landmark_signatures']  # [n_landmarks, 4]
            n_neighbors = landmark_data['n_neighbors_for_propagation']

            logger.info(f"Loaded landmark model with {len(landmark_points)} landmarks")

            # Set up nearest neighbors finder
            from sklearn.neighbors import NearestNeighbors
            landmark_finder = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
            landmark_finder.fit(landmark_points)

            # Find nearest landmarks for each test sample
            distances, landmark_neighbor_indices = landmark_finder.kneighbors(geometric_features)

            # Compute TDA features by interpolating from nearest landmarks
            tda_features = []
            epsilon = 1e-8  # To avoid division by zero

            for i in range(len(geometric_features)):
                # Get distances and indices of nearest landmarks
                neighbor_dists = distances[i]
                neighbor_indices = landmark_neighbor_indices[i]

                # Inverse distance weighting
                weights = 1.0 / (neighbor_dists + epsilon)
                weights /= np.sum(weights)  # Normalize weights

                # Get signatures of nearest landmarks
                nearest_signatures = landmark_signatures[neighbor_indices]

                # Compute weighted average of signatures
                weighted_signature = np.sum(nearest_signatures * weights[:, np.newaxis], axis=0)
                tda_features.append(weighted_signature)

            tda_features = np.array(tda_features)
            logger.info(f"TDA features computed: {tda_features.shape}")
            return tda_features

        except FileNotFoundError:
            logger.error(f"Landmark TDA model not found at {landmark_model_path}")
            logger.error("Please run landmark_method_tda_asymmetry.py first to create the landmark model")
            raise
        except Exception as e:
            logger.error(f"Failed to apply landmark TDA: {e}")
            raise

    def combine_all_features(self, geometric_features: np.ndarray, tda_features: np.ndarray) -> Tuple[
        np.ndarray, List[str]]:
        """
        Combine geometric and TDA features into final 14D feature matrix

        Args:
            geometric_features: [n_samples, 10] geometric features
            tda_features: [n_samples, 4] TDA features

        Returns:
            Tuple of (final_features, all_feature_names)
        """
        logger.info("Combining geometric and TDA features...")

        # Combine: 10D geometric + 4D topological = 14D total
        final_features = np.hstack([geometric_features, tda_features])

        # All feature names
        geometric_names = [
            'cone_energy', 'order_energy', 'hyperbolic_distance',
            'forward_cone', 'backward_cone', 'cone_asymmetry',
            'forward_energy', 'backward_energy', 'asymmetric_energy', 'asymmetry_measure'
        ]

        tda_names = [
            'tda_h0_total_persistence',
            'tda_h1_total_persistence',
            'tda_h1_max_persistence',
            'tda_h1_feature_count'
        ]

        all_feature_names = geometric_names + tda_names

        logger.info(f"Final feature matrix: {final_features.shape}")
        logger.info(f"  - Geometric: {len(geometric_names)}D")
        logger.info(f"  - Topological: {len(tda_names)}D")
        logger.info(f"  - Total: {len(all_feature_names)}D")

        return final_features, all_feature_names

    def _convert_to_binary_labels(self, labels: List[str]) -> List[int]:
        """
        Convert string labels to binary integers for easy evaluation
        
        Args:
            labels: List of string labels
            
        Returns:
            List of binary integer labels (0=entailment, 1=non-entailment)
        """
        binary_labels = []
        for label in labels:
            if label == 'entailment':
                binary_labels.append(0)  # entailment
            else:  # neutral or contradiction
                binary_labels.append(1)  # non-entailment
        return binary_labels
    

    def process_test_data(self, test_data_path: str, output_path: str,
                          landmark_model_path: Optional[str] = None,
                          include_tda: bool = True) -> Dict:
        """
        Complete blind processing pipeline with optional TDA features

        Args:
            test_data_path: Path to raw SNLI test data
            output_path: Path to save processed features
            landmark_model_path: Path to trained landmark TDA model (required if include_tda=True)
            include_tda: Whether to include TDA features (10D vs 14D output)

        Returns:
            Dictionary with processed features and metadata
        """
        logger.info("=" * 80)
        logger.info("STARTING BLIND TEST DATA PROCESSING")
        if include_tda:
            logger.info("INCLUDING TDA FEATURES (14D output)")
        else:
            logger.info("GEOMETRIC FEATURES ONLY (10D output)")
        logger.info("=" * 80)

        # Step 1: Load raw test data
        test_data = self.load_test_data(test_data_path)

        # Step 2: Extract BERT embeddings (label-blind)
        bert_results = self.extract_bert_embeddings(
            test_data['premises'],
            test_data['hypotheses']
        )

        # Step 3: Extract geometric features (label-blind)
        geometric_features_dict = self.extract_geometric_features(
            bert_results['premise_embeddings'],
            bert_results['hypothesis_embeddings']
        )

        # Step 4: Create geometric feature matrix
        geometric_features, geometric_feature_names = self.create_geometric_feature_matrix(geometric_features_dict)

        # Step 5: Optionally add TDA features
        if include_tda:
            if landmark_model_path is None:
                raise ValueError("landmark_model_path must be provided when include_tda=True")

            # Apply landmark TDA (label-blind)
            tda_features = self.apply_landmark_tda(geometric_features, landmark_model_path)

            # Combine geometric + TDA features
            final_features, final_feature_names = self.combine_all_features(geometric_features, tda_features)
            feature_type = "geometric+tda"
        else:
            # Use geometric features only
            final_features = geometric_features
            final_feature_names = geometric_feature_names
            feature_type = "geometric_only"

        # Prepare final output (labels stored but not used in processing)
        processed_data = {
            'features': torch.from_numpy(final_features).float(),
            'labels': test_data['labels'],  # Available for evaluation but not used in processing
            'binary_labels': self._convert_to_binary_labels(test_data['labels']),
            'feature_names': final_feature_names,
            'n_samples': test_data['n_samples'],
            'n_features': final_features.shape[1],
            'feature_type': feature_type,
            'processing_metadata': {
                'bert_model': self.text_processor.model_name,
                'embedding_dim': bert_results['embedding_dim'],
                'device_used': str(self.device),
                'pipeline_type': 'blind_test',
                'bias_free': True,  # Confirms no labels were used in processing
                'includes_tda': include_tda,
                'landmark_model_path': landmark_model_path if include_tda else None
            },
            # Store raw geometric features for analysis if needed
            'raw_geometric_features': geometric_features_dict
        }

        # Add TDA-specific data if included
        if include_tda:
            processed_data['geometric_features'] = torch.from_numpy(geometric_features).float()
            processed_data['tda_features'] = torch.from_numpy(tda_features).float()
            processed_data['geometric_feature_names'] = geometric_feature_names
            processed_data['tda_feature_names'] = [
                'tda_h0_total_persistence', 'tda_h1_total_persistence',
                'tda_h1_max_persistence', 'tda_h1_feature_count'
            ]

        # Save processed data
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(processed_data, output_path)
        logger.info(f"Blind processed data saved to {output_path}")

        # Summary
        logger.info("=" * 80)
        logger.info("BLIND PROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Samples processed: {processed_data['n_samples']}")
        logger.info(f"Features extracted: {processed_data['n_features']}")
        logger.info(f"Feature type: {feature_type}")
        logger.info(f"Feature names: {', '.join(final_feature_names)}")
        logger.info(f"Labels available for evaluation: {len(set(test_data['labels']))}")
        logger.info("PIPELINE WAS COMPLETELY LABEL-BLIND")
        logger.info("Ready for unbiased classifier evaluation!")

        return processed_data


def main():
    try:
        # Initialize processor
        processor = BlindTestProcessor(
            bert_model_name='bert-base-uncased'
        )

        # Process test data
        processed_data = processor.process_test_data(
            test_data_path="data/raw/snli/test/TEST_snli_10k_subset_balanced.json",
            output_path="blind_tests/snli_10k_test_asymmetry_input.pt",
            landmark_model_path="results/tda_integration/landmark_tda_features/asymmetry_landmark_tda_model_blind_tests.pt",
            include_tda=True
        )

        logger.info("Blind test processing completed successfully!")

    except Exception as e:
        logger.error(f"Blind test processing failed: {e}")
        raise

if __name__ == "__main__":
    main()

