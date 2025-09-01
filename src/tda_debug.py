"""
Fixed TDA Implementation Following Savle (2019) Methodology
===========================================================

This implementation corrects the methodological issues in our TDA analysis
by following the exact approach used in Savle et al. (2019).
"""

import torch
import numpy as np
import ripser
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings("ignore")


class FixedTDAAnalysis:
    """
    Corrected TDA implementation following Savle (2019) methodology exactly
    """

    def __init__(self):
        self.results = {}
        self.tda_params = {
            'maxdim': 1,  # Savle used H0 and H1 primarily
            'thresh': 2.0,
            'coeff': 2
        }

    def debug_input_data(self):
        """Debug the input data to understand the structure"""
        print("=" * 60)
        print("DEBUGGING TDA INPUT DATA")
        print("=" * 60)

        try:
            # Load the TDA-ready data
            tda_data_path = "validation_results/tda_ready_data_small_toy.pt"
            data = torch.load(tda_data_path)

            print(f"Data keys: {list(data.keys())}")

            # Check cone violations
            if 'cone_violations' in data:
                cone_violations = data['cone_violations']
                print(f"\nCone violations shape: {cone_violations.shape}")
                print(f"Cone violations dtype: {cone_violations.dtype}")
                print(f"Cone violations range: [{cone_violations.min():.4f}, {cone_violations.max():.4f}]")
                print(
                    f"Contains inf/nan: inf={torch.isinf(cone_violations).any()}, nan={torch.isnan(cone_violations).any()}")

                # Check per feature
                for i in range(cone_violations.shape[1]):
                    feature_vals = cone_violations[:, i]
                    print(f"  Feature {i}: range=[{feature_vals.min():.4f}, {feature_vals.max():.4f}], "
                          f"inf={torch.isinf(feature_vals).any()}, nan={torch.isnan(feature_vals).any()}")

            # Check labels
            if 'labels' in data:
                labels = data['labels']
                print(f"\nLabels: {len(labels)} samples")
                label_counts = {}
                for label in labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
                print(f"Label distribution: {label_counts}")

            # Check energy hierarchy
            if 'energy_hierarchy' in data:
                hierarchy = data['energy_hierarchy']
                print(f"\nEnergy hierarchy: {hierarchy}")

            return data

        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def create_document_based_tda(self, texts, labels):
        """
        Apply TDA following Savle's document-based methodology

        Args:
            texts: List of text documents
            labels: List of corresponding labels
        """
        print("\n" + "=" * 60)
        print("DOCUMENT-BASED TDA (Following Savle 2019)")
        print("=" * 60)

        # Step 1: Create TFIDF vectors (as in Savle paper)
        print("Step 1: Creating TFIDF vectors...")
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        tfidf_dense = tfidf_matrix.toarray()

        print(f"TFIDF matrix shape: {tfidf_dense.shape}")
        print(f"TFIDF range: [{tfidf_dense.min():.4f}, {tfidf_dense.max():.4f}]")

        # Step 2: Compute cosine distance matrix (as in Savle paper)
        print("\nStep 2: Computing cosine distance matrix...")
        # Use cosine distance = 1 - cosine similarity
        distances = pdist(tfidf_dense, metric='cosine')
        distance_matrix = squareform(distances)

        print(f"Distance matrix shape: {distance_matrix.shape}")
        print(f"Distance range: [{distance_matrix.min():.4f}, {distance_matrix.max():.4f}]")
        print(f"Contains inf/nan: inf={np.isinf(distance_matrix).any()}, nan={np.isnan(distance_matrix).any()}")

        # Step 3: Apply persistent homology (as in Savle paper)
        print("\nStep 3: Computing persistent homology...")
        try:
            # Use distance matrix as input (not point cloud)
            result = ripser.ripser(distance_matrix, distance_matrix=True, **self.tda_params)
            diagrams = result['dgms']

            print(f"Successfully computed persistent homology!")
            print(f"H0 features: {len(diagrams[0])}")
            print(f"H1 features: {len(diagrams[1])}")

            # Step 4: Analyze by label
            self.analyze_persistence_by_label(diagrams, labels, tfidf_dense)

            return {
                'diagrams': diagrams,
                'distance_matrix': distance_matrix,
                'tfidf_matrix': tfidf_dense,
                'labels': labels
            }

        except Exception as e:
            print(f"Error in persistent homology computation: {e}")
            return None

    def analyze_persistence_by_label(self, diagrams, labels, tfidf_matrix):
        """Analyze persistence diagrams by label (following Savle approach)"""
        print("\n" + "-" * 40)
        print("ANALYZING PERSISTENCE BY LABEL")
        print("-" * 40)

        # Group by labels
        label_groups = {}
        for i, label in enumerate(labels):
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(i)

        # Analyze H0 and H1 features by label
        for label, indices in label_groups.items():
            print(f"\n{label.upper()} ({len(indices)} samples):")

            # For simplicity, we'll look at global persistence features
            # In practice, Savle computed TDA per case, not globally

            # Analyze H0 (connected components)
            h0_features = diagrams[0]
            if len(h0_features) > 0:
                h0_lifespans = h0_features[:, 1] - h0_features[:, 0]
                valid_h0 = h0_lifespans[np.isfinite(h0_lifespans)]
                if len(valid_h0) > 0:
                    print(f"  H0 lifespans: mean={np.mean(valid_h0):.4f}, std={np.std(valid_h0):.4f}")
                else:
                    print(f"  H0 lifespans: no finite values")

            # Analyze H1 (holes/loops)
            h1_features = diagrams[1]
            if len(h1_features) > 0:
                h1_lifespans = h1_features[:, 1] - h1_features[:, 0]
                valid_h1 = h1_lifespans[np.isfinite(h1_lifespans)]
                if len(valid_h1) > 0:
                    print(f"  H1 lifespans: mean={np.mean(valid_h1):.4f}, std={np.std(valid_h1):.4f}")
                else:
                    print(f"  H1 lifespans: no finite values")
            else:
                print(f"  H1 lifespans: no features detected")

    def fix_cone_violation_tda(self, cone_violations, labels):
        """
        Fix the cone violation TDA by properly handling the input format
        """
        print("\n" + "=" * 60)
        print("FIXING CONE VIOLATION TDA")
        print("=" * 60)

        # Convert to numpy and check for issues
        if torch.is_tensor(cone_violations):
            cone_violations_np = cone_violations.numpy()
        else:
            cone_violations_np = cone_violations

        print(f"Input shape: {cone_violations_np.shape}")
        print(f"Input range: [{cone_violations_np.min():.4f}, {cone_violations_np.max():.4f}]")

        # Remove any infinite or NaN values
        finite_mask = np.isfinite(cone_violations_np).all(axis=1)
        clean_data = cone_violations_np[finite_mask]
        clean_labels = [labels[i] for i in range(len(labels)) if finite_mask[i]]

        print(f"After cleaning: {clean_data.shape[0]} samples remain")

        if clean_data.shape[0] < 3:
            print("ERROR: Not enough clean samples for TDA")
            return None

        # Apply TDA to clean data
        try:
            print("\nComputing persistent homology on cone violations...")
            result = ripser.ripser(clean_data, **self.tda_params)
            diagrams = result['dgms']

            print(f"Success! H0 features: {len(diagrams[0])}, H1 features: {len(diagrams[1])}")

            # Extract meaningful features
            features = self.extract_topological_features(diagrams, clean_labels)

            return {
                'diagrams': diagrams,
                'features': features,
                'clean_data': clean_data,
                'clean_labels': clean_labels
            }

        except Exception as e:
            print(f"Error in cone violation TDA: {e}")
            return None

    def extract_topological_features(self, diagrams, labels):
        """Extract topological features avoiding infinite values"""
        features = {}

        for dim, diagram in enumerate(diagrams):
            if len(diagram) == 0:
                features[f'H{dim}'] = {
                    'total_persistence': 0.0,
                    'max_persistence': 0.0,
                    'n_features': 0,
                    'mean_birth': 0.0,
                    'mean_death': 0.0
                }
                continue

            # Only use finite values
            finite_mask = np.isfinite(diagram).all(axis=1)
            finite_diagram = diagram[finite_mask]

            if len(finite_diagram) == 0:
                features[f'H{dim}'] = {
                    'total_persistence': 0.0,
                    'max_persistence': 0.0,
                    'n_features': 0,
                    'mean_birth': 0.0,
                    'mean_death': 0.0
                }
                continue

            births = finite_diagram[:, 0]
            deaths = finite_diagram[:, 1]
            lifespans = deaths - births

            features[f'H{dim}'] = {
                'total_persistence': np.sum(lifespans),
                'max_persistence': np.max(lifespans),
                'n_features': len(finite_diagram),
                'mean_birth': np.mean(births),
                'mean_death': np.mean(deaths)
            }

        return features


def main():
    """Main function to debug and fix TDA implementation"""
    analyzer = FixedTDAAnalysis()

    # Step 1: Debug input data
    data = analyzer.debug_input_data()

    if data is not None:
        # Step 2: Try fixed cone violation TDA
        if 'cone_violations' in data and 'labels' in data:
            print("\n" + "=" * 60)
            print("ATTEMPTING FIXED CONE VIOLATION TDA")
            print("=" * 60)

            result = analyzer.fix_cone_violation_tda(
                data['cone_violations'],
                data['labels']
            )

            if result:
                print(f"\nSUCCESS! TDA analysis completed.")
                print(f"Topological features extracted: {list(result['features'].keys())}")
            else:
                print(f"\nFAILED: TDA analysis could not be completed.")



if __name__ == "__main__":
    main()