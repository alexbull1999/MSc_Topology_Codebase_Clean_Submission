"""
ChaosNLI Persistence Image Precomputer
Precomputes persistence images for ChaosNLI dataset while preserving human label distributions
"""

import pickle
import numpy as np
import torch
from datetime import datetime
import time
import json
from sklearn.metrics import pairwise_distances
from gph.python import ripser_parallel
from persim import PersistenceImager
from scipy.ndimage import zoom

# Import your existing clustering classes to use the same functions
from point_cloud_clustering_test import SeparateModelPointCloudGenerator


class ChaosNLIPersistenceImagePrecomputer:
    """
    Precompute persistence images for ChaosNLI while preserving human annotation data
    """
    
    def __init__(self, order_model_path, asymmetry_model_path, device='cuda'):
        """Initialize using your existing generator"""
        self.device = device
        
        print(f"Initializing SeparateModelPointCloudGenerator on {device}...")
        
        self.generator = SeparateModelPointCloudGenerator(
            order_model_path=order_model_path,
            asymmetry_model_path=asymmetry_model_path,
            hyperbolic_model_path=None,  # Not needed
            device=device
        )
        
        print("Generator initialized successfully")
    
    def compute_distance_matrix(self, point_cloud: torch.Tensor, metric: str = 'braycurtis') -> np.ndarray:
        """Compute distance matrix for point cloud - SAME AS CLUSTERING CODE"""
        point_cloud_np = point_cloud.numpy()
        distance_matrix = pairwise_distances(point_cloud_np, metric=metric)
        return distance_matrix
    
    def ph_dim_and_diagrams_from_distance_matrix(self, dm: np.ndarray,
                                 min_points: int = 50,
                                 max_points: int = 1000,
                                 point_jump: int = 25,
                                 h_dim: int = 0,
                                 alpha: float = 1.0):
        """EXACT COPY from clustering code - compute persistence on FULL point cloud"""        
        assert dm.ndim == 2 and dm.shape[0] == dm.shape[1]
        
        print(f"Computing persistence on full {dm.shape[0]} point cloud...")
    
        # Compute persistence diagrams on FULL point cloud with H1 dimension
        full_diagrams = ripser_parallel(dm, maxdim=1, n_threads=-1, metric="precomputed")['dgms']
        
        print(f"  H0 features: {len(full_diagrams[0])}")
        print(f"  H1 features: {len(full_diagrams[1])}")
        
        # For PH-dimension, still need to subsample (that's what PH-dim measures)
        test_n = range(min_points, min(max_points, dm.shape[0]), point_jump)
        lengths = []
        
        for points_number in test_n:
            if points_number >= dm.shape[0]:
                break
                
            sample_indices = np.random.choice(dm.shape[0], points_number, replace=False)
            dist_matrix = dm[sample_indices, :][:, sample_indices]
            
            # Compute persistence diagrams - this is for PH-DIM calculation
            sub_diagrams = ripser_parallel(dist_matrix, maxdim=0, n_threads=-1, metric="precomputed")['dgms']
            
            # Extract specific dimension for PH-dim calculation
            d = sub_diagrams[h_dim]
            d = d[d[:, 1] < np.inf]
            lengths.append(np.power((d[:, 1] - d[:, 0]), alpha).sum())
        
        if len(lengths) < 2:
            ph_dimension = 0.0
        else:
            lengths = np.array(lengths)
            
            # Compute PH dimension
            x = np.log(np.array(list(test_n[:len(lengths)])))
            y = np.log(lengths)
            N = len(x)
            
            if N < 2:
                ph_dimension = 0.0
            else:
                m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
                ph_dimension = alpha / (1 - m) if m != 1 else 0.0
        
        # Return FULL topology diagrams (not subsampled ones!)
        return ph_dimension, full_diagrams
    
    def persistence_diagrams_to_images(self, all_diagrams) -> list:
        """EXACT COPY from clustering code with best configs"""
        
        # First, analyze the actual range of data that exists
        all_birth_times = []
        all_death_times = []
        all_lifespans = []
        valid_diagrams_count = 0
        
        print("Analyzing persistence diagram ranges...")
        
        for diagram_idx, diagrams in enumerate(all_diagrams):
            if diagrams is None:
                continue
            if isinstance(diagrams, (list, tuple)) and len(diagrams) == 0:
                continue
            if isinstance(diagrams, np.ndarray) and diagrams.size == 0:
                continue

            for dim in range(min(2, len(diagrams))):
                diagram = diagrams[dim]
                
                # Handle different possible diagram formats
                if isinstance(diagram, np.ndarray) and diagram.size > 0:
                    # Check if it's 1D (empty) or 2D (has features)
                    if diagram.ndim == 1:
                        continue
                    elif diagram.ndim == 2 and diagram.shape[1] >= 2:
                        # Valid 2D diagram with birth/death columns
                        finite_mask = np.isfinite(diagram).all(axis=1)
                        finite_diagram = diagram[finite_mask]
                        
                        if len(finite_diagram) > 0:
                            all_birth_times.extend(finite_diagram[:, 0])
                            all_death_times.extend(finite_diagram[:, 1])
                            lifespans = finite_diagram[:, 1] - finite_diagram[:, 0]
                            all_lifespans.extend(lifespans)
                            valid_diagrams_count += 1
        
        print(f"Found {valid_diagrams_count} valid diagrams with {len(all_lifespans)} total finite features")
        
        if len(all_lifespans) == 0:
            print("❌ No finite features found across all diagrams!")
            return []
        
        # Calculate actual data ranges
        min_birth = np.min(all_birth_times)
        max_birth = np.max(all_birth_times)
        min_life = np.min(all_lifespans)
        max_life = np.max(all_lifespans)
        
        print(f"Actual persistence ranges:")
        print(f"  Birth times: {min_birth:.4f} - {max_birth:.4f}")
        print(f"  Lifespans: {min_life:.4f} - {max_life:.4f}")
        
        # Use data-driven ranges with padding
        birth_padding = max(0.01, (max_birth - min_birth) * 0.1)
        life_padding = max(0.001, (max_life - min_life) * 0.1)
        
        birth_range = (max(0, min_birth - birth_padding), max_birth + birth_padding)
        pers_range = (max(0.001, min_life - life_padding), max_life + life_padding)
        
        print(f"Adjusted persistence image parameters:")
        print(f"  birth_range: ({birth_range[0]:.4f}, {birth_range[1]:.4f})")
        print(f"  pers_range: ({pers_range[0]:.4f}, {pers_range[1]:.4f})")
        
        # USING BEST CONFIGS FROM PARAMETER SEARCH (from your clustering code)
        pixel_size = max(0.001, (pers_range[1] - pers_range[0]) / 138.9)
        sigma = max(0.001, (pers_range[1] - pers_range[0]) / 82.6)
        target_resolution = 30
        
        pimgr = PersistenceImager(
            pixel_size=pixel_size,
            birth_range=birth_range,
            pers_range=pers_range,
            kernel_params={'sigma': sigma}
        )
        
        print(f"PersistenceImager config: pixel_size={pixel_size:.4f}, sigma={sigma:.4f}")
        
        persistence_images = []
        successful_conversions = 0
        
        for diagram_idx, diagrams in enumerate(all_diagrams):
            if diagrams is None:
                continue
            if isinstance(diagrams, (list, tuple)) and len(diagrams) == 0:
                continue
            if isinstance(diagrams, np.ndarray) and diagrams.size == 0:
                continue
                
            combined_image = np.zeros((target_resolution, target_resolution))
            has_content = False
            
            # Process H0 and H1 diagrams with robust handling
            for dim in range(min(2, len(diagrams))):
                diagram = diagrams[dim]
                
                # Robust diagram handling
                if isinstance(diagram, np.ndarray) and diagram.size > 0:
                    if diagram.ndim == 1:
                        # 1D array means no features
                        continue
                    elif diagram.ndim == 2 and diagram.shape[1] >= 2:
                        # Valid 2D diagram
                        finite_mask = np.isfinite(diagram).all(axis=1)
                        finite_diagram = diagram[finite_mask]
                        
                        if len(finite_diagram) > 0:
                            try:
                                img = pimgr.transform([finite_diagram])[0]
                                
                                # Resize if needed
                                if img.shape != (target_resolution, target_resolution):
                                    zoom_factors = (target_resolution / img.shape[0], target_resolution / img.shape[1])
                                    img = zoom(img, zoom_factors)
                                
                                combined_image += img
                                has_content = True
                                
                            except Exception as e:
                                print(f"    Failed to convert diagram {diagram_idx}, dim {dim}: {e}")
                                continue
            
            # Only add if image has content
            if has_content and combined_image.max() > 0:
                combined_image = combined_image / combined_image.max()
                persistence_images.append(combined_image.flatten())  # Flatten to 900D vector
                successful_conversions += 1
        
        print(f"\nPersistence image conversion results:")
        print(f"  Successful: {successful_conversions}/{len(all_diagrams)}")
        print(f"  Success rate: {successful_conversions/len(all_diagrams)*100:.1f}%" if all_diagrams else "N/A")
        
        return persistence_images
    
    def precompute_chaosnli_persistence_images(self, data_path, output_path, max_samples=None,
                                             batch_size=32, save_every=500):
        """
        Precompute persistence images for ChaosNLI dataset
        """
        print("=" * 80)
        print("PRECOMPUTING CHAOSNLI PERSISTENCE IMAGES")
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
        total_samples = len(premise_tokens)
        if max_samples and total_samples > max_samples:
            print(f"Limiting to first {max_samples} samples")
            premise_tokens = premise_tokens[:max_samples]
            hypothesis_tokens = hypothesis_tokens[:max_samples]
            majority_labels = majority_labels[:max_samples]
            label_distributions = label_distributions[:max_samples]
            label_counts = label_counts[:max_samples]
            entropies = entropies[:max_samples]
            uids = uids[:max_samples]
            old_labels = old_labels[:max_samples]
            total_samples = max_samples
        
        # Print ChaosNLI statistics
        self._print_chaosnli_statistics(majority_labels, entropies, label_distributions, premise_tokens, hypothesis_tokens)
        
        # Process samples
        all_persistence_images = []
        all_majority_labels = []
        all_label_distributions = []
        all_label_counts = []
        all_entropies = []
        all_uids = []
        all_old_labels = []
        all_sample_indices = []
        
        start_time = time.time()
        processed_count = 0
        
        print(f"\nProcessing {total_samples} samples...")
        
        # Process in batches
        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            batch_samples = list(range(i, batch_end))
            
            print(f"Processing batch {i//batch_size + 1}: samples {i+1}-{batch_end}")
            
            # Extract persistence images for this batch
            batch_results = self._process_chaosnli_batch_persistence_images(
                batch_samples, premise_tokens, hypothesis_tokens,
                majority_labels, label_distributions, label_counts, 
                entropies, uids, old_labels
            )
            
            for result in batch_results:
                if result is not None:
                    all_persistence_images.append(result['persistence_image'])
                    all_majority_labels.append(result['majority_label'])
                    all_label_distributions.append(result['label_distribution'])
                    all_label_counts.append(result['label_counts'])
                    all_entropies.append(result['entropy'])
                    all_uids.append(result['uid'])
                    all_old_labels.append(result['old_label'])
                    all_sample_indices.append(result['sample_index'])
            
            processed_count += len(batch_samples)
            
            # Progress update
            if processed_count % 100 == 0:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed
                eta = (total_samples - processed_count) / rate / 3600
                print(f"  Progress: {processed_count}/{total_samples} "
                      f"({processed_count/total_samples*100:.1f}%) "
                      f"Rate: {rate:.1f} samples/sec, ETA: {eta:.1f}h")
            
            # Periodic saving
            if processed_count % save_every == 0:
                self._save_intermediate_results(
                    output_path, all_persistence_images, all_majority_labels, 
                    all_label_distributions, all_label_counts, all_entropies,
                    all_uids, all_old_labels, all_sample_indices, 
                    processed_count, total_samples
                )
            
            # Memory cleanup
            torch.cuda.empty_cache()
        
        # Final save
        self._save_final_chaosnli_results(
            output_path, all_persistence_images, all_majority_labels,
            all_label_distributions, all_label_counts, all_entropies,
            all_uids, all_old_labels, all_sample_indices,
            data_path, total_samples, time.time() - start_time
        )
        
        print(f"\n✅ ChaosNLI persistence image precomputation completed!")
        print(f"Total time: {(time.time() - start_time)/3600:.1f} hours")
        print(f"Persistence images saved to: {output_path}")
    
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
    
    def _process_chaosnli_batch_persistence_images(self, batch_indices, premise_tokens, hypothesis_tokens,
                                                  majority_labels, label_distributions, label_counts,
                                                  entropies, uids, old_labels):
        """Process a batch and extract persistence images for ChaosNLI"""
        
        # First collect all diagrams for this batch
        batch_diagrams = []
        batch_metadata = []
        
        for idx in batch_indices:
            try:
                # Generate point cloud
                point_cloud, stats = self.generator.generate_premise_hypothesis_point_cloud(
                    premise_tokens[idx], hypothesis_tokens[idx]
                )
                
                # Compute persistence diagrams if sufficient points
                if stats['sufficient_for_phd']:
                    distance_matrix = self.compute_distance_matrix(point_cloud)
                    ph_dim, diagrams = self.ph_dim_and_diagrams_from_distance_matrix(
                        distance_matrix, min_points=50, max_points=min(1000, point_cloud.shape[0]), point_jump=25
                    )
                    
                    batch_diagrams.append(diagrams)
                    batch_metadata.append({
                        'majority_label': majority_labels[idx],
                        'label_distribution': label_distributions[idx],
                        'label_counts': label_counts[idx],
                        'entropy': entropies[idx],
                        'uid': uids[idx],
                        'old_label': old_labels[idx],
                        'sample_index': idx,
                        'ph_dim': ph_dim,
                        'point_cloud_stats': stats
                    })
                else:
                    #INSUFFICIENT POINTS
                    raise
                    
            except Exception as e:
                print(f"    Error processing ChaosNLI sample {idx}: {e}")
                raise
        
        # Convert all diagrams to persistence images at once
        persistence_images = self.persistence_diagrams_to_images(batch_diagrams)
        
        # Match images with metadata
        batch_results = []
        for i, metadata in enumerate(batch_metadata):
            if i < len(persistence_images):
                persistence_image = persistence_images[i]
            else:
                # Create empty image if conversion failed
                persistence_image = np.zeros(900)  # 30x30 flattened
            
            batch_results.append({
                'persistence_image': persistence_image,
                'majority_label': metadata['majority_label'],
                'label_distribution': metadata['label_distribution'],
                'label_counts': metadata['label_counts'],
                'entropy': metadata['entropy'],
                'uid': metadata['uid'],
                'old_label': metadata['old_label'],
                'sample_index': metadata['sample_index']
            })
        
        return batch_results
    
    def _save_intermediate_results(self, output_path, images, majority_labels, label_distributions,
                                 label_counts, entropies, uids, old_labels, indices, processed, total):
        """Save intermediate results for ChaosNLI"""
        intermediate_path = output_path.replace('.pkl', f'_intermediate_{processed}.pkl')
        
        data = {
            'persistence_images': np.array(images) if images else np.array([]),
            'majority_labels': majority_labels,
            'labels': majority_labels,  # For compatibility
            'label_distributions': label_distributions,
            'label_counts': label_counts,
            'entropies': entropies,
            'uids': uids,
            'old_labels': old_labels,
            'sample_indices': indices,
            'processed_count': processed,
            'total_count': total,
            'timestamp': datetime.now().isoformat(),
            'image_shape': (30, 30),
            'feature_count': 900
        }
        
        with open(intermediate_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"    Intermediate ChaosNLI results saved: {intermediate_path}")
    
    def _save_final_chaosnli_results(self, output_path, images, majority_labels, label_distributions,
                                   label_counts, entropies, uids, old_labels, indices,
                                   original_data_path, total_samples, total_time):
        """Save final precomputed ChaosNLI persistence images"""
        
        # Create comprehensive metadata
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'original_data_path': original_data_path,
            'total_samples': total_samples,
            'computation_time_hours': total_time / 3600,
            'image_shape': (30, 30),
            'feature_count': 900,
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
            'persistence_config': {
                'target_resolution': 30,
                'method': 'same_as_clustering_code'
            },
            'dataset_type': 'ChaosNLI'
        }
        
        # Save main data
        final_data = {
            'persistence_images': np.array(images) if images else np.array([]),
            'majority_labels': np.array(majority_labels),
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
        
        # Save metadata separately as JSON
        metadata_path = output_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nFinal ChaosNLI results saved:")
        print(f"  Persistence images: {output_path}")
        print(f"  Metadata: {metadata_path}")
        print(f"  Shape: {final_data['persistence_images'].shape}")
        print(f"  Image resolution: 30x30 (900 features per sample)")
        print(f"  Human label distributions preserved: {len(final_data['label_distributions'])}")


# Main execution functions
def precompute_chaosnli_snli_persistence_images():
    """Precompute ChaosNLI-SNLI persistence images"""
    
    order_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/order_embedding_model_separate_margins.pt"
    asymmetry_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/new_independent_asymmetry_transform_model_v2.pt"
    
    data_path = "/vol/bitbucket/ahb24/tda_entailment_new/chaosnli_snli_sbert_tokens.pkl"
    output_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_chaosnli_snli_persistence_images.pkl"
    
    # Create precomputer
    precomputer = ChaosNLIPersistenceImagePrecomputer(order_model_path, asymmetry_model_path)
    
    # Precompute ChaosNLI-SNLI persistence images
    precomputer.precompute_chaosnli_persistence_images(
        data_path, 
        output_path,
        max_samples=None,  # All samples
        batch_size=16,  # Smaller batches due to persistence computation
        save_every=1000
    )

def precompute_chaosnli_mnli_persistence_images():
    """Precompute ChaosNLI-MNLI persistence images"""
    
    order_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/order_embedding_model_separate_margins.pt"
    asymmetry_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/new_independent_asymmetry_transform_model_v2.pt"
    
    data_path = "/vol/bitbucket/ahb24/tda_entailment_new/chaosnli_mnli_matched_sbert_tokens.pkl"
    output_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_chaosnli_mnli_matched_persistence_images.pkl"
    
    # Create precomputer
    precomputer = ChaosNLIPersistenceImagePrecomputer(order_model_path, asymmetry_model_path)
    
    # Precompute ChaosNLI-MNLI persistence images
    precomputer.precompute_chaosnli_persistence_images(
        data_path, 
        output_path,
        max_samples=None,  # All samples
        batch_size=16,
        save_every=250
    )


if __name__ == "__main__":
    # Precompute ChaosNLI-SNLI persistence images
    precompute_chaosnli_mnli_persistence_images()