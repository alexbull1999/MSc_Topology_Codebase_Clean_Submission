import torch
from transformers import AutoTokenizer, AutoModel
import json
import os
from typing import List, Tuple, Dict, Optional
import numpy as np

class EnhancedSubtokenTextToEmbedding:
    """
    Enhanced subtoken processing to dramatically increase points per sample while preserving semantic context.
    Uses longer sequences, multi-layer embeddings, and sliding windows.
    """

    def __init__(self, model_name="roberta-base", device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize enhanced subtoken-level text processing pipeline"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        print(f"Loading {model_name} on {self.device}...")

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['TMPDIR'] = '/vol/bitbucket/ahb24/temp'
        os.environ['TEMP'] = '/vol/bitbucket/ahb24/temp'
        os.environ['TMP'] = '/vol/bitbucket/ahb24/temp'
        os.makedirs('/vol/bitbucket/ahb24/temp', exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Enable gradient checkpointing to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        print("Enhanced subtoken-level text processing pipeline ready")

    def encode_text_multilayer(self, texts: List[str], 
                              batch_size: int = 8,  # Reduced due to memory requirements
                              max_length: int = 256,  # Increased from 128
                              layers_to_use: List[int] = None,
                              include_special_tokens: bool = True) -> List[torch.Tensor]:
        """
        Convert list of texts to multi-layer token embeddings for maximum point density
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size (reduced due to longer sequences + multiple layers)
            max_length: Maximum sequence length (increased to 256)
            layers_to_use: Which transformer layers to include (default: last 4 layers)
            include_special_tokens: Whether to include [CLS], [SEP] tokens
            
        Returns:
            List of tensors, each containing multi-layer token embeddings for one text
            Each tensor has shape [n_tokens * n_layers, hidden_size]
        """
        
        if layers_to_use is None:
            # Use last 4 layers by default (layers 9, 10, 11, 12 for RoBERTa-base)
            layers_to_use = [9, 10, 11, 12]
        
        all_multilayer_embeddings = []

        # Clear GPU cache before starting
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]

            # Tokenize batch with longer sequences
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                return_attention_mask=True
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embeddings from multiple layers
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # outputs.hidden_states contains embeddings from all layers
                # Shape: (num_layers, batch_size, seq_len, hidden_size)
                hidden_states = outputs.hidden_states
                attention_mask = inputs['attention_mask']

                # Process each sample in the batch
                for j in range(attention_mask.shape[0]):
                    sample_mask = attention_mask[j]
                    
                    # Collect embeddings from selected layers for this sample
                    sample_multilayer_embeddings = []
                    
                    for layer_idx in layers_to_use:
                        # Get embeddings from this layer for this sample
                        layer_embeddings = hidden_states[layer_idx][j]  # [seq_len, hidden_size]
                        
                        # Filter out padding tokens
                        valid_embeddings = layer_embeddings[sample_mask.bool()]
                        
                        # Optionally remove special tokens
                        if not include_special_tokens and valid_embeddings.shape[0] > 2:
                            valid_embeddings = valid_embeddings[1:-1]  # Remove [CLS] and [SEP]
                        
                        # Move to CPU immediately to save GPU memory
                        valid_embeddings = valid_embeddings.cpu().clone()
                        sample_multilayer_embeddings.append(valid_embeddings)

                    # Stack all layers for this sample
                    # Shape: [n_tokens * n_layers, hidden_size]
                    combined_embeddings = torch.cat(sample_multilayer_embeddings, dim=0)
                    all_multilayer_embeddings.append(combined_embeddings)

                 # Clear intermediate results from GPU
                del outputs
                del hidden_states
                del inputs

            # Clear cache every batch for memory efficiency
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        # Final cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return all_multilayer_embeddings

    def encode_text_sliding_window(self, texts: List[str],
                                  window_size: int = 32,
                                  stride: int = 16,
                                  batch_size: int = 8) -> List[torch.Tensor]:
        """
        Create sliding window embeddings to capture different contextual views
        
        Args:
            texts: List of texts to encode
            window_size: Size of each sliding window
            stride: Step size between windows
            batch_size: Batch size for processing
            
        Returns:
            List of tensors, each containing sliding window embeddings
            Each tensor has shape [n_windows * window_tokens, hidden_size]
        """
        
        all_sliding_embeddings = []
        
        for text in texts:
            # Tokenize the full text first
            tokens = self.tokenizer.tokenize(text)
            
            if len(tokens) <= window_size:
                # If text is shorter than window, just process normally
                window_embeddings = self.encode_text_multilayer([text], batch_size=1, 
                                                               max_length=window_size)[0]
                all_sliding_embeddings.append(window_embeddings)
                continue
            
            # Create sliding windows
            text_windows = []
            for start in range(0, len(tokens) - window_size + 1, stride):
                window_tokens = tokens[start:start + window_size]
                window_text = self.tokenizer.convert_tokens_to_string(window_tokens)
                text_windows.append(window_text)
            
            # Process all windows for this text
            window_embeddings_list = []
            for window_batch_start in range(0, len(text_windows), batch_size):
                window_batch = text_windows[window_batch_start:window_batch_start + batch_size]
                
                # Get embeddings for this batch of windows (single layer for efficiency)
                inputs = self.tokenizer(
                    window_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=window_size,
                    return_attention_mask=True
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask']
                    
                    # Extract valid tokens for each window
                    for k in range(embeddings.shape[0]):
                        valid_tokens = embeddings[k][attention_mask[k].bool()]
                        # Remove special tokens
                        if valid_tokens.shape[0] > 2:
                            valid_tokens = valid_tokens[1:-1]

                        # Move to CPU immediately
                        valid_tokens = valid_tokens.cpu().clone()
                        window_embeddings_list.append(valid_tokens)
                        
                    # Clear GPU memory
                    del outputs
                    del inputs

                # Clear cache periodically
                if window_batch_start % (batch_size * 3) == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Combine all window embeddings for this text
            if window_embeddings_list:
                combined_sliding = torch.cat(window_embeddings_list, dim=0)
                all_sliding_embeddings.append(combined_sliding)
            else:
                # Fallback to regular processing
                fallback = self.encode_text_multilayer([text], batch_size=1)[0]
                all_sliding_embeddings.append(fallback)
        
        return all_sliding_embeddings

    def create_enhanced_premise_hypothesis_pointcloud(self, premise: str, hypothesis: str,
                                                    method: str = "multilayer",
                                                    **kwargs) -> torch.Tensor:
        """
        Create enhanced point cloud with dramatically more points while preserving semantics
        
        Args:
            premise: Premise text
            hypothesis: Hypothesis text
            method: Enhancement method ("multilayer", "sliding", "hybrid")
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Enhanced point cloud tensor with many more points
        """
        
        if method == "multilayer":
            # Use multi-layer embeddings for maximum semantic density
            premise_embeddings = self.encode_text_multilayer([premise], **kwargs)[0]
            hypothesis_embeddings = self.encode_text_multilayer([hypothesis], **kwargs)[0]
            
        elif method == "sliding":
            # Use sliding window embeddings for contextual variety
            premise_embeddings = self.encode_text_sliding_window([premise], **kwargs)[0]
            hypothesis_embeddings = self.encode_text_sliding_window([hypothesis], **kwargs)[0]
            
        elif method == "hybrid":
            # Combine multilayer and sliding approaches
            premise_multi = self.encode_text_multilayer([premise], 
                                                       layers_to_use=kwargs.get('layers_to_use', [11, 12]))[0]
            hypothesis_multi = self.encode_text_multilayer([hypothesis],
                                                          layers_to_use=kwargs.get('layers_to_use', [11, 12]))[0]
            
            # Add some sliding window embeddings for variety
            premise_sliding = self.encode_text_sliding_window([premise], 
                                                            window_size=kwargs.get('window_size', 64))[0]
            hypothesis_sliding = self.encode_text_sliding_window([hypothesis],
                                                               window_size=kwargs.get('window_size', 64))[0]
            
            # Combine all embeddings
            premise_embeddings = torch.cat([premise_multi, premise_sliding], dim=0)
            hypothesis_embeddings = torch.cat([hypothesis_multi, hypothesis_sliding], dim=0)
            
        else:
            raise ValueError(f"Unknown enhancement method: {method}")
        
        # Combine premise and hypothesis embeddings
        enhanced_pointcloud = torch.cat([premise_embeddings, hypothesis_embeddings], dim=0)
        
        return enhanced_pointcloud

    def process_enhanced_dataset(self, dataset_path: str,
                               method: str = "multilayer",
                               include_class_separation: bool = True,
                               **method_kwargs) -> Dict:
        """
        Process dataset with enhanced point generation methods
        
        Args:
            dataset_path: Path to JSON dataset
            method: Enhancement method to use
            include_class_separation: Whether to organize by class
            **method_kwargs: Parameters for the enhancement method
            
        Returns:
            Dict with enhanced pointclouds and metadata
        """
        
        print(f"Processing dataset with enhanced method: {method}")
        print(f"Method parameters: {method_kwargs}")
        
        # Load dataset
        with open(dataset_path, "r") as file:
            data = json.load(file)

        # For large datasets, automatically use chunked processing
        if len(data) > 1000:  # Lower threshold for enhanced processing
            print(f"Large dataset detected ({len(data)} samples). Using chunked processing...")
            return self.process_enhanced_dataset_chunked(
                dataset_path, chunk_size=200, method=method,
                include_class_separation=include_class_separation, **method_kwargs
            )

        premises = [item[0] for item in data]
        hypotheses = [item[1] for item in data]
        labels = [item[2] for item in data]

        print(f"Dataset contains {len(data)} premise-hypothesis pairs")

        # Create enhanced point clouds
        enhanced_pointclouds = []
        pointcloud_sizes = []
        failed_samples = []
        
        for i in range(len(premises)):
            if i % 100 == 0:
                print(f"Processing sample {i}/{len(premises)}")
            
            try:
                current_premise = premises[i]
                current_hypothesis = hypotheses[i]
                
                enhanced_pointcloud = self.create_enhanced_premise_hypothesis_pointcloud(
                    current_premise, current_hypothesis, method=method, **method_kwargs
                )
                
                enhanced_pointclouds.append(enhanced_pointcloud)
                pointcloud_sizes.append(enhanced_pointcloud.shape[0])
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                failed_samples.append(i)
                raise

        print(f"Enhanced point cloud statistics:")
        print(f"  Average points per sample: {np.mean(pointcloud_sizes):.1f}")
        print(f"  Min points: {np.min(pointcloud_sizes)}")
        print(f"  Max points: {np.max(pointcloud_sizes)}")
        print(f"  Embedding dimension: {enhanced_pointclouds[0].shape[1]}")

        # Prepare result
        result = {
            "pointclouds": enhanced_pointclouds,
            "pointcloud_sizes": pointcloud_sizes,
            "labels": labels,
            "texts": {
                "premises": premises,
                "hypotheses": hypotheses
            },
            "metadata": {
                "model_name": self.model_name,
                "enhancement_method": method,
                "method_parameters": method_kwargs,
                "embedding_dim": enhanced_pointclouds[0].shape[1],
                "n_samples": len(data),
                "avg_points_per_sample": float(np.mean(pointcloud_sizes)),
                "min_points": int(np.min(pointcloud_sizes)),
                "max_points": int(np.max(pointcloud_sizes))
            }
        }

        if include_class_separation:
            class_pointclouds = self.organize_pointclouds_by_class(enhanced_pointclouds, labels)
            result["class_pointclouds"] = class_pointclouds

        print("Enhanced dataset processing complete")
        return result

    def process_enhanced_dataset_chunked(self, dataset_path: str, chunk_size: int = 200,
                                       method: str = "multilayer",
                                       include_class_separation: bool = True,
                                       **method_kwargs) -> Dict:
        """
        Process large datasets in chunks for enhanced subtoken processing
        """
        print(f"Processing dataset with enhanced method in chunks: {method}")
        print(f"Method parameters: {method_kwargs}")
        
        # Load dataset
        with open(dataset_path, "r") as file:
            data = json.load(file)
        
        total_samples = len(data)
        print(f"Dataset contains {total_samples} premise-hypothesis pairs")
        print(f"Processing in chunks of {chunk_size} samples...")

        # Process in chunks
        all_enhanced_pointclouds = []
        all_pointcloud_sizes = []
        all_labels = []
        all_premises = []
        all_hypotheses = []

        for chunk_idx in range(0, total_samples, chunk_size):
            end_idx = min(chunk_idx + chunk_size, total_samples)
            chunk_data = data[chunk_idx:end_idx]
                        
            # Extract chunk data
            premises = [item[0] for item in chunk_data]
            hypotheses = [item[1] for item in chunk_data]
            labels = [item[2] for item in chunk_data]

            # Process chunk - create enhanced point clouds
            chunk_pointclouds = []
            chunk_pointcloud_sizes = []
            
            for i in range(len(premises)):
                if i % 25 == 0:
                    print(f"  Processing sample {i}/{len(chunk_data)} in current chunk")

                try:
                    current_premise = premises[i]
                    current_hypothesis = hypotheses[i]
                    
                    enhanced_pointcloud = self.create_enhanced_premise_hypothesis_pointcloud(
                        current_premise, current_hypothesis, method=method, **method_kwargs
                    )
                    
                    chunk_pointclouds.append(enhanced_pointcloud)
                    chunk_pointcloud_sizes.append(enhanced_pointcloud.shape[0])
                    
                except Exception as e:
                    print(f"  Error processing sample {i} in chunk: {e}")
                    raise
                
                # Clear cache every 10 samples
                if i % 10 == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            # Store results
            all_enhanced_pointclouds.extend(chunk_pointclouds)
            all_pointcloud_sizes.extend(chunk_pointcloud_sizes)
            all_labels.extend(labels)
            all_premises.extend(premises)
            all_hypotheses.extend(hypotheses)

        print(f"\nEnhanced point cloud statistics:")
        print(f"  Average points per sample: {np.mean(all_pointcloud_sizes):.1f}")
        print(f"  Min points: {np.min(all_pointcloud_sizes)}")
        print(f"  Max points: {np.max(all_pointcloud_sizes)}")
        print(f"  Embedding dimension: {all_enhanced_pointclouds[0].shape[1]}")

        # Prepare final result
        result = {
            "pointclouds": all_enhanced_pointclouds,
            "pointcloud_sizes": all_pointcloud_sizes,
            "labels": all_labels,
            "texts": {
                "premises": all_premises,
                "hypotheses": all_hypotheses
            },
            "metadata": {
                "model_name": self.model_name,
                "enhancement_method": method,
                "method_parameters": method_kwargs,
                "embedding_dim": all_enhanced_pointclouds[0].shape[1],
                "n_samples": len(all_labels),
                "avg_points_per_sample": float(np.mean(all_pointcloud_sizes)),
                "min_points": int(np.min(all_pointcloud_sizes)),
                "max_points": int(np.max(all_pointcloud_sizes)),
                "processed_in_chunks": True,
                "chunk_size": chunk_size
            }
        }

        if include_class_separation:
            class_pointclouds = self.organize_pointclouds_by_class(all_enhanced_pointclouds, all_labels)
            result["class_pointclouds"] = class_pointclouds

        print("Enhanced dataset processing complete")
        return result

    def organize_pointclouds_by_class(self, pointclouds: List[torch.Tensor], 
                                    labels: List[str]) -> Dict[str, List[torch.Tensor]]:
        """Organize point clouds by entailment class"""
        class_pointclouds = {}
        
        unique_labels = list(set(labels))
        for label in unique_labels:
            class_clouds = [pointclouds[i] for i in range(len(labels)) if labels[i] == label]
            class_pointclouds[label] = class_clouds
            
            total_points = sum(cloud.shape[0] for cloud in class_clouds)
            avg_points = total_points / len(class_clouds) if class_clouds else 0
            
            print(f"Class '{label}': {len(class_clouds)} samples, "
                  f"avg {avg_points:.1f} points per sample")

        return class_pointclouds

    def save_processed_data(self, processed_data: Dict, output_path: str):
        """Save enhanced processed data"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(processed_data, output_path)
        print(f"Saved enhanced processed data to {output_path}")


def main():
    """Process dataset with best enhancement method"""
    
    # Set working directory to avoid home directory issues
    original_cwd = os.getcwd()
    work_dir = "/vol/bitbucket/ahb24/temp_processing"
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    
    # Process full dataset with the most promising method
    processor = EnhancedSubtokenTextToEmbedding()
    
    dataset_path = "/homes/ahb24/MSc_Topology_Codebase/data/raw/snli/train/snli_full_train.json"
    output_path = "/vol/bitbucket/ahb24/phd_processed_data"
    if not os.path.exists(output_path):
        print(f"ERROR: Output directory not found at {output_path}")
        return None
    else:
        print(f"Output directory found: {output_path}")
    
    try:
        # Use multi-layer method as it should give most semantic density
        enhanced_data = processor.process_enhanced_dataset(
            dataset_path=dataset_path,
            method="multilayer",
            layers_to_use=[9, 10, 11, 12],  # Last 4 layers
            max_length=256,  # Longer sequences
            include_class_separation=True
        )
        
        # Save the enhanced data
        full_output_path = "/vol/bitbucket/ahb24/phd_processed_data/snli_full_phd_roberta_enhanced_multilayer.pt"
        processor.save_processed_data(enhanced_data, full_output_path)
        
        print(f"\SUCCESS: Enhanced processing complete!")
        print(f"Average points per sample: {enhanced_data['metadata']['avg_points_per_sample']:.1f}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        raise
    
    # Return to original directory
    os.chdir(original_cwd)


if __name__ == "__main__":
    main()