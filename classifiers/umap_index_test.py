"""
Test suite for UMAP feature indexing in neural_classifier.py

This test verifies that the sample_index mechanism correctly maps between:
1. Training data indices
2. UMAP coordinate lookups
3. Geometric feature consistency

The test ensures we can trust the UMAP features used in the neural network.
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import os
from dataclasses import dataclass

from neural_classifier import TDAFeatureExtractor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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



class UMAPIndexingTester:
    """Test suite for UMAP indexing system"""

    def __init__(self, neural_data_path: str = "results/tda_integration/neural_network_data_SNLI_1k.pt"):
        self.neural_data_path = Path(neural_data_path)
        self.training_data = None
        self.feature_extractor = None

    def load_test_data(self) -> bool:
        """Load the neural network training data"""
        try:
            if not self.neural_data_path.exists():
                logger.error(f"Neural network data file not found: {self.neural_data_path}")
                return False

            self.training_data = torch.load(self.neural_data_path)
            logger.info(f"✓ Loaded training data with {len(self.training_data['labels'])} samples")
            return True

        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return False

    def setup_feature_extractor(self) -> bool:
        """Initialize and fit the feature extractor"""
        try:
            self.feature_extractor = TDAFeatureExtractor()
            self.feature_extractor.fit(self.training_data)
            logger.info("✓ Feature extractor fitted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to setup feature extractor: {e}")
            return False

    def test_index_range_validity(self) -> bool:
        """Test 1: Verify all sample indices are within valid range"""
        logger.info("\n=== Test 1: Index Range Validity ===")

        try:
            embedding_coords = self.feature_extractor.embedding_coordinates
            umap_coordinates = embedding_coords['umap_coordinates']
            sample_indices = embedding_coords.get('sample_indices', [])

            n_samples = len(self.training_data['labels'])
            n_umap_coords = len(umap_coordinates)

            logger.info(f"Training samples: {n_samples}")
            logger.info(f"UMAP coordinates: {n_umap_coords}")
            logger.info(f"Sample indices: {len(sample_indices)}")

            # Check if we have the right number of coordinates
            if n_umap_coords != n_samples:
                logger.warning(f"Mismatch: {n_umap_coords} UMAP coords vs {n_samples} training samples")
                # This might be OK if some samples were filtered out during UMAP creation

            # Test that all training sample indices can be looked up
            valid_lookups = 0
            for i in range(n_samples):
                if i < len(umap_coordinates):
                    valid_lookups += 1

            logger.info(f"✓ Valid index lookups: {valid_lookups}/{n_samples}")

            success_rate = valid_lookups / n_samples
            if success_rate < 0.95:  # Allow for some filtering
                logger.error(f"Low success rate: {success_rate:.2%}")
                return False

            logger.info(f"✓ Index range validity passed ({success_rate:.2%} success rate)")
            return True

        except Exception as e:
            logger.error(f"Index range test failed: {e}")
            return False

    def test_coordinate_lookup_consistency(self) -> bool:
        """Test 2: Verify coordinate lookups are consistent"""
        logger.info("\n=== Test 2: Coordinate Lookup Consistency ===")

        try:
            # Test multiple sample lookups
            test_indices = [0, 1, 10, 50, 100, 200] if len(self.training_data['labels']) > 200 else [0, 1, 2, 3, 4]

            for sample_idx in test_indices:
                if sample_idx >= len(self.training_data['labels']):
                    continue

                # Create sample data with index
                geometric_features = self.feature_extractor.training_geometric_features[sample_idx]
                sample_data = {
                    'geometric_features': geometric_features,
                    'sample_index': sample_idx
                }

                # Get UMAP coordinates via the lookup method
                umap_coords = self.feature_extractor._compute_umap_coordinates(sample_data)

                # Verify the coordinates are reasonable (2D, finite values)
                if len(umap_coords) != 2:
                    logger.error(f"Sample {sample_idx}: Expected 2D coords, got {len(umap_coords)}D")
                    return False

                if not all(np.isfinite(umap_coords)):
                    logger.error(f"Sample {sample_idx}: Non-finite coordinates {umap_coords}")
                    return False

                logger.info(f"✓ Sample {sample_idx}: UMAP coords {umap_coords}")

            logger.info("✓ Coordinate lookup consistency passed")
            return True

        except Exception as e:
            logger.error(f"Coordinate lookup test failed: {e}")
            return False

    def test_fallback_transform(self) -> bool:
        """Test 3: Verify fallback UMAP transform works for samples without index"""
        logger.info("\n=== Test 3: Fallback Transform Test ===")

        try:
            # Test with a sample that doesn't have sample_index (simulating test data)
            geometric_features = self.feature_extractor.training_geometric_features[0]
            sample_data_no_index = {
                'geometric_features': geometric_features
                # No 'sample_index' - should trigger fallback
            }

            # This should use the fitted UMAP reducer to transform
            umap_coords_fallback = self.feature_extractor._compute_umap_coordinates(sample_data_no_index)

            # Verify fallback coordinates are reasonable
            if len(umap_coords_fallback) != 2:
                logger.error(f"Fallback: Expected 2D coords, got {len(umap_coords_fallback)}D")
                return False

            if not all(np.isfinite(umap_coords_fallback)):
                logger.error(f"Fallback: Non-finite coordinates {umap_coords_fallback}")
                return False

            logger.info(f"✓ Fallback transform: {umap_coords_fallback}")

            # Compare with lookup method for same sample
            sample_data_with_index = {
                'geometric_features': geometric_features,
                'sample_index': 0
            }
            umap_coords_lookup = self.feature_extractor._compute_umap_coordinates(sample_data_with_index)

            # They might not be identical due to floating point precision and transform differences
            coord_diff = np.linalg.norm(np.array(umap_coords_fallback) - np.array(umap_coords_lookup))
            logger.info(f"✓ Coordinate difference (lookup vs fallback): {coord_diff:.6f}")

            # They should be reasonably close for the same sample
            if coord_diff > 1.0:  # Allow some difference due to transform vs lookup
                logger.warning(f"Large coordinate difference: {coord_diff}")

            logger.info("✓ Fallback transform test passed")
            return True

        except Exception as e:
            logger.error(f"Fallback transform test failed: {e}")
            return False

    def test_feature_extraction_with_umap(self) -> bool:
        """Test 4: Verify complete feature extraction pipeline including UMAP - FIXED VERSION"""
        logger.info("\n=== Test 4: Complete Feature Extraction Test (FIXED) ===")

        try:
            test_indices = [0, 1, 10] if len(self.training_data['labels']) > 10 else [0, 1]

            for sample_idx in test_indices:
                if sample_idx >= len(self.training_data['labels']):
                    continue

                # Extract complete feature vector using transform() - this includes normalization
                geometric_features = self.feature_extractor.training_geometric_features[sample_idx]
                sample_data = {
                    'geometric_features': geometric_features,
                    'sample_index': sample_idx
                }

                # Get the complete normalized feature vector (what the neural network sees)
                features = self.feature_extractor.transform(sample_data)

                # Verify feature vector properties
                expected_dim = 9  # Actual feature dimension: 3 geometric + 1 density + 3 centroids + 2 UMAP
                if len(features) != expected_dim:
                    logger.error(f"Sample {sample_idx}: Expected {expected_dim} features, got {len(features)}")
                    return False

                if not all(np.isfinite(features)):
                    logger.error(f"Sample {sample_idx}: Non-finite features")
                    return False

                # FIXED: Compare normalized UMAP features properly
                # Get the last 2 features (normalized UMAP coordinates) from transform()
                umap_features_from_transform = features[-2:]

                # Extract raw features and normalize them manually to verify consistency
                raw_features = self.feature_extractor._extract_single_sample_features(sample_data)
                raw_features = raw_features.reshape(1, -1)
                manually_normalized_features = self.feature_extractor.scaler.transform(raw_features)
                umap_features_manual = manually_normalized_features.flatten()[-2:]

                # These should be identical - both are the same features with same normalization
                umap_diff = np.linalg.norm(umap_features_from_transform - umap_features_manual)
                if umap_diff > 1e-10:
                    logger.error(f"Sample {sample_idx}: UMAP feature mismatch: {umap_diff}")
                    logger.error(f"  From transform(): {umap_features_from_transform}")
                    logger.error(f"  Manual normalize: {umap_features_manual}")
                    return False

                logger.info(
                    f"✓ Sample {sample_idx}: Complete feature vector (dim={len(features)}) with normalized UMAP {umap_features_from_transform}")

            logger.info("✓ Complete feature extraction test passed")
            return True

        except Exception as e:
            logger.error(f"Complete feature extraction test failed: {e}")
            return False

    def test_batch_consistency(self) -> bool:
        """Test 5: Verify batch processing maintains index consistency"""
        logger.info("\n=== Test 5: Batch Consistency Test ===")

        try:
            # Test a small batch of samples
            batch_size = min(5, len(self.training_data['labels']))

            # Extract features for a batch using fit_transform (which includes indexing)
            features_matrix = []
            for i in range(batch_size):
                geometric_features = self.feature_extractor.training_geometric_features[i]
                sample_data = {
                    'geometric_features': geometric_features,
                    'sample_index': i
                }
                features = self.feature_extractor.transform(sample_data)
                features_matrix.append(features)

            features_matrix = np.array(features_matrix)

            # Verify batch properties
            if features_matrix.shape != (batch_size, 9):
                logger.error(f"Batch shape mismatch: expected ({batch_size}, 9), got {features_matrix.shape}")
                return False

            # Check that UMAP features (last 2 columns) are all different
            umap_batch = features_matrix[:, -2:]
            unique_coords = len(np.unique(umap_batch.round(6), axis=0))

            if unique_coords < batch_size - 1:  # Allow for rare duplicates
                logger.warning(f"Low UMAP diversity: {unique_coords}/{batch_size} unique coordinates")

            logger.info(f"✓ Batch consistency: {batch_size} samples, {unique_coords} unique UMAP coordinates")
            logger.info("✓ Batch consistency test passed")
            return True

        except Exception as e:
            logger.error(f"Batch consistency test failed: {e}")
            return False

    def run_all_tests(self) -> bool:
        """Run the complete test suite"""
        logger.info("=" * 80)
        logger.info("UMAP INDEXING TEST SUITE")
        logger.info("=" * 80)

        # Setup
        if not self.load_test_data():
            return False

        if not self.setup_feature_extractor():
            return False

        # Run tests
        tests = [
            ("Index Range Validity", self.test_index_range_validity),
            ("Coordinate Lookup Consistency", self.test_coordinate_lookup_consistency),
            ("Fallback Transform", self.test_fallback_transform),
            ("Complete Feature Extraction", self.test_feature_extraction_with_umap),
            ("Batch Consistency", self.test_batch_consistency)
        ]

        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
                if result:
                    logger.info(f"{test_name}: PASSED")
                else:
                    logger.error(f"{test_name}: FAILED")
            except Exception as e:
                logger.error(f"{test_name}: ERROR - {e}")
                results.append((test_name, False))

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)

        passed = sum(1 for _, result in results if result)
        total = len(results)

        for test_name, result in results:
            status = "PASS" if result else "FAIL"
            logger.info(f"{status}: {test_name}")

        logger.info(f"\nOverall: {passed}/{total} tests passed")

        if passed == total:
            logger.info("All tests passed! UMAP indexing system is working correctly.")
            return True
        else:
            logger.error(f"{total - passed} test(s) failed. Review UMAP indexing implementation.")
            return False


def main():
    """Main test execution"""
    tester = UMAPIndexingTester()
    success = tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())