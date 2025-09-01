"""
ChaosNLI TDA Features Pipeline
Precompute TDA features for ChaosNLI dataset with human label distributions
"""

import pickle
import numpy as np
import torch
from datetime import datetime
import os
import time
from tqdm import tqdm
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Import your existing class
from point_cloud_clustering_test import SeparateModelPointCloudGenerator, TopologicalClassifier, train_pytorch_classifier, train_pytorch_classifier_custom

class ChaosNLITDAFeaturePrecomputer:
    """
    Precompute and cache TDA features for ChaosNLI dataset
    Handles human label distributions and uncertainty metrics
    """
    
    def __init__(self, order_model_path, asymmetry_model_path, device='cuda'):
        """Initialize using your existing SeparateModelPointCloudGenerator"""
        self.device = device
        
        print(f"Initializing SeparateModelPointCloudGenerator on {device}...")
        
        # Use your existing class
        self.generator = SeparateModelPointCloudGenerator(
            order_model_path=order_model_path,
            asymmetry_model_path=asymmetry_model_path,
            hyperbolic_model_path=None,  # removed this
            device=device
        )
        
        print("Generator initialized successfully")
    
    def precompute_chaosnli_features(self, data_path, output_path, max_samples=None,
                                   batch_size=32, save_every=1000):
        """
        Precompute TDA features for ChaosNLI dataset
        
        Args:
            data_path: Path to ChaosNLI SBERT tokens (.pkl)
            output_path: Path to save precomputed features (.pkl)
            max_samples: Limit total samples (None = all)
            batch_size: GPU batch size for model inference
            save_every: Save intermediate results every N samples
        """
        print("=" * 80)
        print("PRECOMPUTING CHAOSNLI TDA FEATURES")
        print("=" * 80)
        print(f"Input: {data_path}")
        print(f"Output: {output_path}")
        
        # Load ChaosNLI data
        print("Loading ChaosNLI data...")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        premise_tokens = data['premise_tokens']
        hypothesis_tokens = data['hypothesis_tokens']
        majority_labels = data['labels']  # Majority vote labels
        label_distributions = data['label_distributions']  # [E, N, C] distributions
        label_counts = data['label_counts']  # Raw counts [E, N, C]
        entropies = data['entropies']  # Human annotation entropy
        uids = data['uids']  # Unique identifiers
        old_labels = data['old_labels']  # Original dataset labels
        
        # Apply filtering if requested
        if max_samples and len(premise_tokens) > max_samples:
            print(f"Limiting to first {max_samples} samples")
            premise_tokens = premise_tokens[:max_samples]
            hypothesis_tokens = hypothesis_tokens[:max_samples]
            majority_labels = majority_labels[:max_samples]
            label_distributions = label_distributions[:max_samples]
            label_counts = label_counts[:max_samples]
            entropies = entropies[:max_samples]
            uids = uids[:max_samples]
            old_labels = old_labels[:max_samples]
        
        # Print statistics
        self._print_chaosnli_statistics(majority_labels, entropies, label_distributions, premise_tokens, hypothesis_tokens)
        
        # Precompute features
        all_features = []
        all_majority_labels = []
        all_label_distributions = []
        all_label_counts = []
        all_entropies = []
        all_uids = []
        all_old_labels = []
        all_sample_indices = []
        
        start_time = time.time()
        total_samples = len(premise_tokens)
        processed_count = 0
        
        print(f"\nProcessing {total_samples} samples...")
        
        # Process in batches for GPU efficiency
        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            batch_size_actual = batch_end - i
            
            print(f"Processing batch {i//batch_size + 1}: samples {i+1}-{batch_end}")
            
            # Process each sample in the batch
            for j in range(i, batch_end):
                try:
                    # Extract TDA features using your existing method
                    features = self.generator.extract_interpretable_topological_features(
                        premise_tokens[j], hypothesis_tokens[j]
                    )
                    
                    all_features.append(features)
                    all_majority_labels.append(majority_labels[j])
                    all_label_distributions.append(label_distributions[j])
                    all_label_counts.append(label_counts[j])
                    all_entropies.append(entropies[j])
                    all_uids.append(uids[j])
                    all_old_labels.append(old_labels[j])
                    all_sample_indices.append(j)
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"    Error processing sample {j}: {e}")
                    continue
            
            # Progress update
            if processed_count % 100 == 0:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed
                eta = (total_samples - processed_count) / rate / 3600  # hours
                print(f"  Progress: {processed_count}/{total_samples} "
                      f"({processed_count/total_samples*100:.1f}%) "
                      f"Rate: {rate:.1f} samples/sec, ETA: {eta:.1f}h")
            
            # Periodic saving
            if processed_count % save_every == 0:
                self._save_intermediate_results(
                    output_path, all_features, all_majority_labels, all_label_distributions,
                    all_label_counts, all_entropies, all_uids, all_old_labels, 
                    all_sample_indices, processed_count, total_samples
                )
            
            # Memory cleanup
            torch.cuda.empty_cache()
        
        # Final save
        self._save_final_chaosnli_results(
            output_path, all_features, all_majority_labels, all_label_distributions,
            all_label_counts, all_entropies, all_uids, all_old_labels, all_sample_indices,
            data_path, total_samples, time.time() - start_time
        )
        
        print(f"\n✅ ChaosNLI precomputation completed!")
        print(f"Total time: {(time.time() - start_time)/3600:.1f} hours")
        print(f"Features saved to: {output_path}")
    
    def _print_chaosnli_statistics(self, majority_labels, entropies, label_distributions, premise_tokens, hypothesis_tokens):
        """Print ChaosNLI dataset statistics"""
        print("\nChaosNLI Dataset Statistics:")
        
        # Majority label distribution
        from collections import Counter
        majority_counts = Counter(majority_labels)
        print(f"Majority label distribution: {dict(majority_counts)}")
        
        # Entropy statistics
        print(f"Human annotation entropy:")
        print(f"  Mean: {np.mean(entropies):.3f}")
        print(f"  Min: {np.min(entropies):.3f}")
        print(f"  Max: {np.max(entropies):.3f}")
        print(f"  Std: {np.std(entropies):.3f}")
        
        # Agreement levels
        high_agreement = sum(1 for e in entropies if e < 0.5)
        medium_agreement = sum(1 for e in entropies if 0.5 <= e < 1.0)
        low_agreement = sum(1 for e in entropies if e >= 1.0)
        
        print(f"Agreement levels:")
        print(f"  High agreement (entropy < 0.5): {high_agreement} ({high_agreement/len(entropies)*100:.1f}%)")
        print(f"  Medium agreement (0.5 ≤ entropy < 1.0): {medium_agreement} ({medium_agreement/len(entropies)*100:.1f}%)")
        print(f"  Low agreement (entropy ≥ 1.0): {low_agreement} ({low_agreement/len(entropies)*100:.1f}%)")
        
        # Token statistics
        premise_token_counts = [tokens.shape[0] for tokens in premise_tokens]
        hypothesis_token_counts = [tokens.shape[0] for tokens in hypothesis_tokens]
        total_token_counts = [p + h for p, h in zip(premise_token_counts, hypothesis_token_counts)]
        
        print(f"Token statistics:")
        print(f"  Premise tokens: {np.mean(premise_token_counts):.1f} ± {np.std(premise_token_counts):.1f}")
        print(f"  Hypothesis tokens: {np.mean(hypothesis_token_counts):.1f} ± {np.std(hypothesis_token_counts):.1f}")
        print(f"  Total tokens: {np.mean(total_token_counts):.1f} ± {np.std(total_token_counts):.1f}")
        
        # Average human label distributions by entropy level
        high_ent_mask = np.array(entropies) < 0.5
        med_ent_mask = (np.array(entropies) >= 0.5) & (np.array(entropies) < 1.0)
        low_ent_mask = np.array(entropies) >= 1.0
        
        if np.any(high_ent_mask):
            high_ent_dists = np.array([label_distributions[i] for i in range(len(entropies)) if high_ent_mask[i]])
            print(f"  High agreement avg distribution: {np.mean(high_ent_dists, axis=0)}")
        
        if np.any(med_ent_mask):
            med_ent_dists = np.array([label_distributions[i] for i in range(len(entropies)) if med_ent_mask[i]])
            print(f"  Medium agreement avg distribution: {np.mean(med_ent_dists, axis=0)}")
        
        if np.any(low_ent_mask):
            low_ent_dists = np.array([label_distributions[i] for i in range(len(entropies)) if low_ent_mask[i]])
            print(f"  Low agreement avg distribution: {np.mean(low_ent_dists, axis=0)}")
    
    def _save_intermediate_results(self, output_path, features, majority_labels, label_distributions,
                                 label_counts, entropies, uids, old_labels, indices, processed, total):
        """Save intermediate results for recovery"""
        intermediate_path = output_path.replace('.pkl', f'_intermediate_{processed}.pkl')
        
        data = {
            'features': np.array(features),
            'majority_labels': majority_labels,
            'label_distributions': label_distributions,
            'label_counts': label_counts,
            'entropies': entropies,
            'uids': uids,
            'old_labels': old_labels,
            'sample_indices': indices,
            'processed_count': processed,
            'total_count': total,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(intermediate_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"    Intermediate results saved: {intermediate_path}")
    
    def _save_final_chaosnli_results(self, output_path, features, majority_labels, label_distributions,
                                   label_counts, entropies, uids, old_labels, indices, 
                                   original_data_path, total_samples, total_time):
        """Save final precomputed ChaosNLI features"""
        
        # Create comprehensive metadata
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'original_data_path': original_data_path,
            'total_samples': total_samples,
            'computation_time_hours': total_time / 3600,
            'feature_count': len(features[0]) if features else 0,
            'feature_names': self.get_interpretable_feature_names(),
            'majority_label_distribution': {
                label: majority_labels.count(label) for label in set(majority_labels)
            },
            'entropy_stats': {
                'mean': float(np.mean(entropies)),
                'std': float(np.std(entropies)),
                'min': float(np.min(entropies)),
                'max': float(np.max(entropies))
            },
            'models_used': {
                'order_model': 'separate_models/order_embedding_model_separate_margins.pt',
                'asymmetry_model': 'separate_models/new_independent_asymmetry_transform_model_v2.pt'
            },
            'dataset_type': 'ChaosNLI'
        }
        
        # Convert lists to numpy arrays for consistent format
        final_data = {
            'features': np.array(features),
            'majority_labels': np.array(majority_labels),  # Majority vote labels
            'labels': np.array(majority_labels),  # For compatibility with existing code
            'label_distributions': np.array(label_distributions),  # Human distributions [E, N, C]
            'label_counts': np.array(label_counts),  # Raw counts [E, N, C]
            'entropies': np.array(entropies),  # Human annotation entropy
            'uids': np.array(uids),  # Unique identifiers
            'old_labels': np.array(old_labels),  # Original dataset labels
            'sample_indices': np.array(indices),
            'metadata': metadata
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(final_data, f)
        
        # Save metadata separately as JSON for easy inspection
        metadata_path = output_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nFinal ChaosNLI results saved:")
        print(f"  Features: {output_path}")
        print(f"  Metadata: {metadata_path}")
        print(f"  Feature shape: {final_data['features'].shape}")
        print(f"  Majority labels: {len(set(final_data['majority_labels']))} classes")
    
    def get_interpretable_feature_names(self):
        """Use your existing feature names method"""
        return self.generator.get_interpretable_feature_names()


# Main execution scripts
def precompute_chaosnli_snli_features():
    """Precompute ChaosNLI-SNLI features"""
    
    order_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/order_embedding_model_separate_margins.pt"
    asymmetry_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/new_independent_asymmetry_transform_model_v2.pt"
    
    data_path = "/vol/bitbucket/ahb24/tda_entailment_new/chaosnli_snli_sbert_tokens.pkl"
    output_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_chaosnli_snli_tda_features_NO_TOKEN_THRESHOLD.pkl"
    
    # Create precomputer
    precomputer = ChaosNLITDAFeaturePrecomputer(order_model_path, asymmetry_model_path)
    
    # Precompute ChaosNLI-SNLI features
    precomputer.precompute_chaosnli_features(
        data_path, 
        output_path,
        max_samples=None,  # All samples
        batch_size=32,
        save_every=500
    )

def precompute_chaosnli_mnli_features():
    """Precompute ChaosNLI-MNLI features"""
    
    order_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/mnli_order_embedding_model_separate_margins.pt"
    asymmetry_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/mnli_asymmetry_transform_model_(match_SNLI_v2).pt"
    
    data_path = "/vol/bitbucket/ahb24/tda_entailment_new/chaosnli_mnli_matched_sbert_tokens.pkl"
    output_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_chaosnli_mnli_matched_tda_features_NO_TOKEN_THRESHOLD.pkl"
    
    # Create precomputer
    precomputer = ChaosNLITDAFeaturePrecomputer(order_model_path, asymmetry_model_path)
    
    # Precompute ChaosNLI-MNLI features
    precomputer.precompute_chaosnli_features(
        data_path, 
        output_path,
        max_samples=None,  # All samples
        batch_size=32,
        save_every=500
    )


if __name__ == "__main__":
    precompute_chaosnli_snli_features()