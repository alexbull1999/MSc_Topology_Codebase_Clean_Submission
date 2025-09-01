"""
Precompute TDA Features Pipeline
Run once offline to generate cached features, then use for fast experiments
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
from SNLI_point_cloud_clustering_test import SeparateModelPointCloudGenerator, TopologicalClassifier, train_pytorch_classifier, train_pytorch_classifier_custom

class TDAFeaturePrecomputer:
    """
    Precompute and cache all TDA features for fast experimentation
    """
    
    def __init__(self, order_model_path, asymmetry_model_path, device='cuda'):
        """Initialize using your existing SeparateModelPointCloudGenerator"""
        self.device = device
        
        print(f"Initializing SeparateModelPointCloudGenerator on {device}...")
        
        # Use your existing class - no hyperbolic model needed since you removed it
        self.generator = SeparateModelPointCloudGenerator(
            order_model_path=order_model_path,
            asymmetry_model_path=asymmetry_model_path,
            hyperbolic_model_path=None,  # removed this
            device=device
        )
        
        print("Generator initialized successfully")
    
    def precompute_dataset_features(self, data_path, output_path, max_samples_per_class=None,
                                  batch_size=32, save_every=1000):
        """
        Precompute TDA features for entire dataset
        
        Args:
            data_path: Path to original SBERT tokens (.pkl)
            output_path: Path to save precomputed features (.pkl)
            max_samples_per_class: Limit samples per class (None = all)
            batch_size: GPU batch size for model inference
            save_every: Save intermediate results every N samples
        """
        print("=" * 80)
        print("PRECOMPUTING TDA FEATURES")
        print("=" * 80)
        print(f"Input: {data_path}")
        print(f"Output: {output_path}")
        
        # Load original data
        print("Loading original data...")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        premise_tokens = data['premise_tokens']
        hypothesis_tokens = data['hypothesis_tokens']
        labels = data['labels']
        
        # Organize by class and apply filtering
        samples_by_class = self._organize_and_filter_samples(
            premise_tokens, hypothesis_tokens, labels, max_samples_per_class
        )
        
        # Precompute features for each class
        all_features = []
        all_labels = []
        all_sample_indices = []
        
        start_time = time.time()
        total_samples = sum(len(samples) for samples in samples_by_class.values())
        processed_count = 0
        
        print(f"\nProcessing {total_samples} samples...")
        
        for class_name, class_samples in samples_by_class.items():
            print(f"\nProcessing {class_name}: {len(class_samples)} samples")
            class_start_time = time.time()
            
            # Process in batches for GPU efficiency
            for i in range(0, len(class_samples), batch_size):
                batch_samples = class_samples[i:i+batch_size]
                
                # Extract features for this batch
                batch_features, batch_labels, batch_indices = self._process_batch(
                    batch_samples, class_name, i
                )
                
                all_features.extend(batch_features)
                all_labels.extend(batch_labels)
                all_sample_indices.extend(batch_indices)
                
                processed_count += len(batch_samples)
                
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
                        output_path, all_features, all_labels, all_sample_indices,
                        processed_count, total_samples
                    )
                
                # Memory cleanup
                torch.cuda.empty_cache()
            
            class_time = time.time() - class_start_time
            print(f"  {class_name} completed in {class_time/60:.1f} minutes")
        
        # Final save
        self._save_final_results(output_path, all_features, all_labels, all_sample_indices,
                               data_path, total_samples, time.time() - start_time)
        
        print(f"\n✅ Precomputation completed!")
        print(f"Total time: {(time.time() - start_time)/3600:.1f} hours")
        print(f"Features saved to: {output_path}")
    
    def _organize_and_filter_samples(self, premise_tokens, hypothesis_tokens, labels, max_samples_per_class):
        """Organize samples by class and apply filtering"""
        
        samples_by_class = {'entailment': [], 'neutral': [], 'contradiction': []}
        
        for i, label in enumerate(labels):
            if label in samples_by_class:
                # Apply token filtering
                total_tokens = premise_tokens[i].shape[0] + hypothesis_tokens[i].shape[0]
                #COULD TRY REMOVING THIS CONDITION BELOW?
                if total_tokens > 0: #WAS AT >= 40 for original 
                    sample_data = {
                        'premise_tokens': premise_tokens[i],
                        'hypothesis_tokens': hypothesis_tokens[i],
                        'original_index': i
                    }
                    samples_by_class[label].append(sample_data)
                    
                    # Stop if we have enough samples for this class
                    if max_samples_per_class and len(samples_by_class[label]) >= max_samples_per_class:
                        continue
        
        # Print statistics
        print("\nDataset statistics after filtering:")
        for class_name, samples in samples_by_class.items():
            print(f"  {class_name}: {len(samples)} samples")
            if samples:
                token_counts = [
                    s['premise_tokens'].shape[0] + s['hypothesis_tokens'].shape[0] 
                    for s in samples
                ]
                print(f"    Token stats: {np.mean(token_counts):.0f} ± {np.std(token_counts):.0f}")
        
        return samples_by_class
    
    def _process_batch(self, batch_samples, class_name, batch_start_idx):
        """Process a batch of samples and extract TDA features"""
        
        batch_features = []
        batch_labels = []
        batch_indices = []
        
        for j, sample in enumerate(batch_samples):
            try:
                # Use your existing method directly
                features = self.generator.extract_interpretable_topological_features(
                    sample['premise_tokens'], sample['hypothesis_tokens']
                )
                
                batch_features.append(features)
                batch_labels.append(class_name)
                batch_indices.append(sample['original_index'])
                
            except Exception as e:
                print(f"    Error processing {class_name} sample {batch_start_idx + j}: {e}")
                continue
        
        return batch_features, batch_labels, batch_indices
    
    # Remove the duplicate methods since we're using the generator
    # extract_interpretable_topological_features and generate_premise_hypothesis_point_cloud
    # are now called via self.generator
    
    def _save_intermediate_results(self, output_path, features, labels, indices, processed, total):
        """Save intermediate results for recovery"""
        intermediate_path = output_path.replace('.pkl', f'_intermediate_{processed}.pkl')
        
        data = {
            'features': np.array(features),
            'labels': labels,
            'sample_indices': indices,
            'processed_count': processed,
            'total_count': total,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(intermediate_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"    Intermediate results saved: {intermediate_path}")
    
    def _save_final_results(self, output_path, features, labels, indices, 
                          original_data_path, total_samples, total_time):
        """Save final precomputed features"""
        
        # Create comprehensive metadata
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'original_data_path': original_data_path,
            'total_samples': total_samples,
            'computation_time_hours': total_time / 3600,
            'feature_count': len(features[0]) if features else 0,
            'feature_names': self.get_interpretable_feature_names(),
            'class_distribution': {
                label: labels.count(label) for label in set(labels)
            },
            'models_used': {
                'order_model': 'separate_models/order_embedding_model_separate_margins.pt',
                'asymmetry_model': 'separate_models/new_independent_asymmetry_transform_model_v2.pt'
            }
        }
        
        # Save main data
        final_data = {
            'features': np.array(features),
            'labels': np.array(labels),
            'sample_indices': np.array(indices),
            'metadata': metadata
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(final_data, f)
        
        # Save metadata separately as JSON for easy inspection
        metadata_path = output_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nFinal results saved:")
        print(f"  Features: {output_path}")
        print(f"  Metadata: {metadata_path}")
        print(f"  Shape: {final_data['features'].shape}")
    
    def get_interpretable_feature_names(self):
        """Use your existing feature names method"""
        return self.generator.get_interpretable_feature_names()


# Main execution scripts
def precompute_snli_features():
    """Precompute SNLI features (run once overnight)"""
    
    order_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/order_embedding_model_separate_margins.pt"
    asymmetry_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/new_independent_asymmetry_transform_model_v2.pt"
    
    train_data_path = "/vol/bitbucket/ahb24/tda_entailment_new/snli_train_sbert_tokens.pkl"
    val_data_path = "/vol/bitbucket/ahb24/tda_entailment_new/snli_val_sbert_tokens.pkl"
    
    # Create precomputer
    precomputer = TDAFeaturePrecomputer(order_model_path, asymmetry_model_path)
    
    # Precompute train features
    precomputer.precompute_dataset_features(
        train_data_path, 
        "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_snli_train_tda_features_NO_TOKEN_THRESHOLD.pkl",
        max_samples_per_class=None,  # 150k total samples
        batch_size=32,
        save_every=1000
    )
    
    # Precompute validation features
    precomputer.precompute_dataset_features(
        val_data_path,
        "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_snli_val_tda_features_NO_TOKEN_THRESHOLD.pkl",
        max_samples_per_class=None,  # All validation samples
        batch_size=32,
        save_every=500
    )


if __name__ == "__main__":
    # Step 1: Precompute features (run once overnight)
    precompute_snli_features()
    