"""
SBERT Token Extractor - Chunked Processing
Extracts token-level embeddings from SBERT's hidden layers for premise-hypothesis pairs
Processes data in chunks to avoid memory issues
"""

import torch
from transformers import AutoTokenizer, AutoModel
import json
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import pickle
import gc
import math


class SBERTTokenExtractor:
    """Extract token-level embeddings from SBERT hidden layers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        
        print(f"Loading SBERT model: {model_name}")
        print(f"Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
  
        self.hidden_size = self.model.config.hidden_size
        print(f"Hidden size: {self.hidden_size}")
        print(f"SBERT token extractor ready")
    
    def extract_token_embeddings(self, text: str, max_length: int = 256) -> torch.Tensor:
        """
        Extract token-level embeddings from SBERT's last hidden layer
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Token embeddings [num_tokens, hidden_size]
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state[0]
            attention_mask = inputs['attention_mask'][0]
            
            valid_tokens = token_embeddings[attention_mask.bool()]
            
        return valid_tokens.cpu()
    
    def process_premise_hypothesis_pair(self, premise: str, hypothesis: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a premise-hypothesis pair to extract token embeddings
        
        Args:
            premise: Premise text
            hypothesis: Hypothesis text
            
        Returns:
            Tuple of (premise_tokens, hypothesis_tokens)
        """
        premise_tokens = self.extract_token_embeddings(premise)
        hypothesis_tokens = self.extract_token_embeddings(hypothesis)
        
        return premise_tokens, hypothesis_tokens
    
    def process_dataset_chunked(self, data_path: str, output_base_path: str, num_chunks: int = 5):
        """
        Process entire dataset in chunks and save token embeddings
        
        Args:
            data_path: Path to JSON dataset file
            output_base_path: Base path for output files (will add chunk numbers)
            num_chunks: Number of chunks to divide the data into
        """
        print(f"Processing dataset: {data_path}")
        
        # Load dataset
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_samples = len(data)
        chunk_size = math.ceil(total_samples / num_chunks)
        
        print(f"Total samples: {total_samples}")
        print(f"Number of chunks: {num_chunks}")
        print(f"Samples per chunk: {chunk_size}")
        
        # Process each chunk
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_samples)
            chunk_data = data[start_idx:end_idx]
            
            print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks}")
            print(f"Samples {start_idx} to {end_idx - 1} ({len(chunk_data)} samples)")
            
            # Create output path for this chunk
            base_path = Path(output_base_path)
            chunk_output_path = base_path.parent / f"{base_path.stem}_chunk_{chunk_idx + 1}_of_{num_chunks}{base_path.suffix}"
            
            # Process chunk
            processed_data = {
                'premise_tokens': [],
                'hypothesis_tokens': [],
                'labels': [],
                'token_counts': {'premise': [], 'hypothesis': []},
                'metadata': {
                    'model_name': self.model_name,
                    'hidden_size': self.hidden_size,
                    'chunk_info': {
                        'chunk_idx': chunk_idx + 1,
                        'total_chunks': num_chunks,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'chunk_size': len(chunk_data)
                    },
                    'total_dataset_size': total_samples
                }
            }
            
            # Process samples in this chunk
            for i, (premise, hypothesis, label) in enumerate(chunk_data):
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(chunk_data)} samples in chunk {chunk_idx + 1}")
                
                try:
                    premise_tokens, hypothesis_tokens = self.process_premise_hypothesis_pair(premise, hypothesis)
                    
                    processed_data['premise_tokens'].append(premise_tokens)
                    processed_data['hypothesis_tokens'].append(hypothesis_tokens)  
                    processed_data['labels'].append(label)
                    processed_data['token_counts']['premise'].append(premise_tokens.shape[0])
                    processed_data['token_counts']['hypothesis'].append(hypothesis_tokens.shape[0])
                    
                except Exception as e:
                    print(f"Error processing sample {start_idx + i}: {e}")
                    continue
            
            # Save chunk
            print(f"Saving chunk {chunk_idx + 1} to: {chunk_output_path}")
            
            chunk_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(chunk_output_path, 'wb') as f:
                pickle.dump(processed_data, f)
            
            # Print chunk statistics
            premise_counts = processed_data['token_counts']['premise']
            hypothesis_counts = processed_data['token_counts']['hypothesis']
            
            print(f"Chunk {chunk_idx + 1} Statistics:")
            print(f"Successfully processed: {len(processed_data['labels'])} samples")
            print(f"Premise token counts - Mean: {np.mean(premise_counts):.1f}")
            print(f"Hypothesis token counts - Mean: {np.mean(hypothesis_counts):.1f}")
            
            # Label distribution for this chunk
            label_counts = {}
            for label in processed_data['labels']:
                label_counts[label] = label_counts.get(label, 0) + 1
            print(f"Label distribution: {label_counts}")
            
            # Clean up memory
            del processed_data
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            print(f"Chunk {chunk_idx + 1} completed and saved")


def main():
    """Process SNLI datasets in chunks"""
    
    extractor = SBERTTokenExtractor()
    
    # Process SNLI training data in chunks
    snli_data_path = "MSc_Topology_Codebase/data/raw/snli/train/snli_full_train.json"
    snli_output_path = "/vol/bitbucket/ahb24/tda_entailment_new/chunked_snli_train_sbert_tokens.pkl"
    
    if Path(snli_data_path).exists():
        print("Processing SNLI training data in chunks...")
        extractor.process_dataset_chunked(snli_data_path, snli_output_path, num_chunks=5)
    else:
        print(f"SNLI data not found at: {snli_data_path}")
    
    # Process MNLI training data in chunks  
    mnli_data_path = "MSc_Topology_Codebase/data/raw/mnli/train/mnli_full_train.json"
    mnli_output_path = "/vol/bitbucket/ahb24/tda_entailment_new/chunked_mnli_train_sbert_tokens.pkl"
    
    if Path(mnli_data_path).exists():
        print("\nProcessing MNLI training data in chunks...")
        extractor.process_dataset_chunked(mnli_data_path, mnli_output_path, num_chunks=5)
    else:
        print(f"MNLI data not found at: {mnli_data_path}")
    
    print("\nSBERT token extraction completed!")


if __name__ == "__main__":
    main()