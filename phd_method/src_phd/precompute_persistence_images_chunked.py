"""
Chunked Persistence Image Precomputer for Classification
Processes each chunk file separately to avoid memory issues
"""

import pickle
import numpy as np
import torch
from datetime import datetime
import time
import json
import gc
from pathlib import Path
from sklearn.metrics import pairwise_distances
from gph.python import ripser_parallel
from persim import PersistenceImager
from scipy.ndimage import zoom

# Import your existing clustering classes to use the same functions
from point_cloud_clustering_test import SeparateModelPointCloudGenerator


class ChunkedPersistenceImagePrecomputer:
    """
    Precompute persistence images from chunked SBERT token files
    """
    
    def __init__(self, order_model_path, asymmetry_model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
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
    
    def process_single_chunk(self, chunk_path, output_path, batch_size=16, save_every=1000):
        """
        Process a single chunk file to extract persistence images
        """
        print("=" * 80)
        print(f"PROCESSING SINGLE CHUNK: {chunk_path}")
        print("=" * 80)
        
        # Load chunk data
        print("Loading chunk data...")
        with open(chunk_path, 'rb') as f:
            data = pickle.load(f)
        
        premise_tokens = data['premise_tokens']
        hypothesis_tokens = data['hypothesis_tokens'] 
        labels = data['labels']
        chunk_metadata = data['metadata']
        
        print(f"Chunk info: {chunk_metadata.get('chunk_info', 'No chunk info')}")
        print(f"Loaded {len(labels)} samples from chunk")
        
        # Organize by class
        samples_by_class = self._organize_samples_by_class(premise_tokens, hypothesis_tokens, labels)
        
        # Process each class
        all_persistence_images = []
        all_labels = []
        all_sample_indices = []
        
        start_time = time.time()
        total_samples = sum(len(samples) for samples in samples_by_class.values())
        processed_count = 0
        
        print(f"\nProcessing {total_samples} samples from chunk...")
        
        for class_name, class_samples in samples_by_class.items():
            print(f"\nProcessing {class_name}: {len(class_samples)} samples")
            class_start_time = time.time()
            
            # Process in batches
            for i in range(0, len(class_samples), batch_size):
                batch_samples = class_samples[i:i+batch_size]
                
                # Extract persistence images for this batch
                batch_results = self._process_batch_persistence_images(
                    batch_samples, class_name, i
                )
                
                for result in batch_results:
                    if result is not None:
                        all_persistence_images.append(result['persistence_image'])
                        all_labels.append(result['label'])
                        all_sample_indices.append(result['sample_index'])
                
                processed_count += len(batch_samples)
                
                # Progress update
                if processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed
                    eta = (total_samples - processed_count) / rate / 60 if rate > 0 else 0
                    print(f"  Progress: {processed_count}/{total_samples} "
                          f"({processed_count/total_samples*100:.1f}%) "
                          f"Rate: {rate:.1f} samples/sec, ETA: {eta:.1f}min")
                
                # Memory cleanup
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                
                gc.collect()
            
            class_time = time.time() - class_start_time
            print(f"  {class_name} completed in {class_time/60:.1f} minutes")
        
        # Save chunk results
        self._save_chunk_results(
            output_path, all_persistence_images, all_labels, all_sample_indices,
            chunk_path, chunk_metadata, total_samples, time.time() - start_time
        )
        
        print(f"\n✅ Chunk processing completed!")
        print(f"Chunk time: {(time.time() - start_time)/60:.1f} minutes")
        print(f"Persistence images saved to: {output_path}")
        
        # Clean up memory before next chunk
        del data, premise_tokens, hypothesis_tokens, labels
        del all_persistence_images, all_labels, all_sample_indices
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
    
    def process_all_chunks(self, chunk_base_path, output_base_path, num_chunks=5, batch_size=16):
        """
        Process all chunk files and save persistence images for each
        
        Args:
            chunk_base_path: Base path for chunk files (e.g., "snli_train_sbert_tokens.pkl")
            output_base_path: Base path for output files
            num_chunks: Number of chunk files to process
            batch_size: Batch size for processing
        """
        print("=" * 80)
        print("PROCESSING ALL CHUNKS FOR PERSISTENCE IMAGES")
        print("=" * 80)
        print(f"Input base: {chunk_base_path}")
        print(f"Output base: {output_base_path}")
        print(f"Number of chunks: {num_chunks}")
        
        total_start_time = time.time()
        
        for chunk_idx in range(1, num_chunks + 1):
            print(f"\n{'='*60}")
            print(f"PROCESSING CHUNK {chunk_idx}/{num_chunks}")
            print(f"{'='*60}")
            
            # Construct paths
            base_path = Path(chunk_base_path)
            chunk_path = base_path.parent / f"{base_path.stem}_chunk_{chunk_idx}_of_{num_chunks}{base_path.suffix}"
            
            output_base = Path(output_base_path)
            output_path = output_base.parent / f"{output_base.stem}_chunk_{chunk_idx}_of_{num_chunks}{output_base.suffix}"
            
            # Check if chunk file exists
            if not chunk_path.exists():
                print(f"❌ Chunk file not found: {chunk_path}")
                continue
            
            # Check if output already exists
            if output_path.exists():
                print(f"⚠️  Output already exists: {output_path}")
                print("Skipping this chunk...")
                continue
            
            try:
                # Process this chunk
                chunk_start_time = time.time()
                self.process_single_chunk(chunk_path, output_path, batch_size)
                
                chunk_time = time.time() - chunk_start_time
                print(f"✅ Chunk {chunk_idx} completed in {chunk_time/60:.1f} minutes")
                
            except Exception as e:
                print(f"❌ Error processing chunk {chunk_idx}: {e}")
                print("Continuing to next chunk...")
                continue
        
        total_time = time.time() - total_start_time
        print(f"\n{'='*80}")
        print("ALL CHUNKS PROCESSING COMPLETED!")
        print(f"Total time: {total_time/3600:.1f} hours")
        print(f"{'='*80}")
    
    def _organize_samples_by_class(self, premise_tokens, hypothesis_tokens, labels):
        """Organize samples by class"""
        
        samples_by_class = {'entailment': [], 'neutral': [], 'contradiction': []}
        
        for i, label in enumerate(labels):
            if label in samples_by_class:
                # Apply token filtering (same as clustering code)
                total_tokens = premise_tokens[i].shape[0] + hypothesis_tokens[i].shape[0]
                if total_tokens > 0:  # No minimum token threshold
                    sample_data = {
                        'premise_tokens': premise_tokens[i],
                        'hypothesis_tokens': hypothesis_tokens[i],
                        'original_index': i
                    }
                    samples_by_class[label].append(sample_data)
        
        # Print statistics
        print("\nChunk statistics after filtering:")
        for class_name, samples in samples_by_class.items():
            print(f"  {class_name}: {len(samples)} samples")
            if samples:
                token_counts = [
                    s['premise_tokens'].shape[0] + s['hypothesis_tokens'].shape[0] 
                    for s in samples
                ]
                print(f"    Token stats: {np.mean(token_counts):.0f} ± {np.std(token_counts):.0f}")
        
        return samples_by_class
    
    def _process_batch_persistence_images(self, batch_samples, class_name, batch_start_idx):
        """Process a batch and extract persistence images"""
        
        # First collect all diagrams for this batch
        batch_diagrams = []
        batch_metadata = []
        
        for j, sample in enumerate(batch_samples):
            try:
                # Generate point cloud
                point_cloud, stats = self.generator.generate_premise_hypothesis_point_cloud(
                    sample['premise_tokens'], sample['hypothesis_tokens']
                )
                
                # Compute persistence diagrams if sufficient points
                if stats['sufficient_for_phd']:
                    distance_matrix = self.compute_distance_matrix(point_cloud)
                    ph_dim, diagrams = self.ph_dim_and_diagrams_from_distance_matrix(
                        distance_matrix, min_points=50, max_points=min(1000, point_cloud.shape[0]), point_jump=25
                    )
                    
                    batch_diagrams.append(diagrams)
                    batch_metadata.append({
                        'label': class_name,
                        'sample_index': sample['original_index'],
                        'ph_dim': ph_dim,
                        'point_cloud_stats': stats
                    })
                else:
                    # Skip samples with insufficient points
                    continue
                    
            except Exception as e:
                print(f"    Error processing {class_name} sample {batch_start_idx + j}: {e}")
                continue
        
        # Convert all diagrams to persistence images at once
        persistence_images = self.persistence_diagrams_to_images(batch_diagrams)
        
        # Match images with metadata
        batch_results = []
        for i, metadata in enumerate(batch_metadata):
            if i < len(persistence_images):
                persistence_image = persistence_images[i]
                batch_results.append({
                    'persistence_image': persistence_image,
                    'label': metadata['label'],
                    'sample_index': metadata['sample_index']
                })
        
        return batch_results
    
    def _save_chunk_results(self, output_path, images, labels, indices, 
                           chunk_path, chunk_metadata, total_samples, processing_time):
        """Save persistence images for a single chunk"""
        
        # Create comprehensive metadata
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'chunk_source_path': str(chunk_path),
            'original_chunk_metadata': chunk_metadata,
            'total_samples_processed': total_samples,
            'successful_samples': len(images),
            'processing_time_minutes': processing_time / 60,
            'image_shape': (30, 30),
            'feature_count': 900,  # 30x30 flattened
            'class_distribution': {
                label: labels.count(label) for label in set(labels)
            },
            'models_used': {
                'order_model': 'separate_models/order_embedding_model_separate_margins.pt',
                'asymmetry_model': 'separate_models/new_independent_asymmetry_transform_model_v2.pt'
            },
            'persistence_config': {
                'target_resolution': 30,
                'method': 'same_as_clustering_code'
            }
        }
        
        # Save chunk data
        chunk_data = {
            'persistence_images': np.array(images) if images else np.array([]),
            'labels': np.array(labels),
            'sample_indices': np.array(indices),
            'metadata': metadata
        }
        
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(chunk_data, f)
        
        # Save metadata separately as JSON
        metadata_path = str(output_path).replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            # Make chunk metadata JSON serializable
            json_metadata = metadata.copy()
            if 'original_chunk_metadata' in json_metadata:
                # Convert any non-serializable items
                json_metadata['original_chunk_metadata'] = str(json_metadata['original_chunk_metadata'])
            json.dump(json_metadata, f, indent=2)
        
        print(f"\nChunk results saved:")
        print(f"  Persistence images: {output_path}")
        print(f"  Metadata: {metadata_path}")
        print(f"  Shape: {chunk_data['persistence_images'].shape}")
        print(f"  Success rate: {len(images)}/{total_samples} ({len(images)/total_samples*100:.1f}%)")


# Main execution functions
def process_snli_chunks():
    """Process SNLI chunk files"""
    
    order_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/order_embedding_model_separate_margins.pt"
    asymmetry_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/new_independent_asymmetry_transform_model_v2.pt"
    
    # Create precomputer
    precomputer = ChunkedPersistenceImagePrecomputer(order_model_path, asymmetry_model_path)
    
    # Process SNLI chunks
    snli_chunk_base = "/vol/bitbucket/ahb24/tda_entailment_new/chunked_snli_train_sbert_tokens.pkl"
    snli_output_base = "/vol/bitbucket/ahb24/tda_entailment_new/chunked_snli_train_persistence_images.pkl"
    
    precomputer.process_all_chunks(
        chunk_base_path=snli_chunk_base,
        output_base_path=snli_output_base,
        num_chunks=5,
        batch_size=16
    )


def process_mnli_chunks():
    """Process MNLI chunk files"""
    
    order_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/mnli_order_embedding_model_separate_margins.pt"
    asymmetry_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/mnli_asymmetry_transform_model_(match_SNLI_v2).pt"
    
    # Create precomputer  
    precomputer = ChunkedPersistenceImagePrecomputer(order_model_path, asymmetry_model_path)
    
    # Process MNLI chunks
    mnli_chunk_base = "/vol/bitbucket/ahb24/tda_entailment_new/chunked_mnli_train_sbert_tokens.pkl"
    mnli_output_base = "/vol/bitbucket/ahb24/tda_entailment_new/MNLI_ORDER_ASYMM_MODELS_chunked_mnli_train_persistence_images.pkl"
    
    precomputer.process_all_chunks(
        chunk_base_path=mnli_chunk_base,
        output_base_path=mnli_output_base,
        num_chunks=5,
        batch_size=16
    )


if __name__ == "__main__":
    # Process SNLI chunks
    print("Starting SNLI chunk processing...")
    process_snli_chunks()
    
    print("\n" + "="*80)
    print("Starting MNLI chunk processing...")
    process_mnli_chunks()