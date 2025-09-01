"""
SBERT Token Extractor
Extracts token-level embeddings from SBERT's hidden layers for premise-hypothesis pairs
"""

import torch
from transformers import AutoTokenizer, AutoModel
import json
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import pickle
import gc


class SBERTTokenExtractor:
    """Extract token-level embeddings from SBERT hidden layers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        
        print(f"Loading SBERT model: {model_name}")
        print(f"Device: {self.device}")
        

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval() # Set to eval mode
  
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
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get token embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden_size]
            attention_mask = inputs['attention_mask'][0]
            
            # Remove padding tokens
            valid_tokens = token_embeddings[attention_mask.bool()]  # [num_valid_tokens, hidden_size]
            
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
    
    def process_dataset(self, data_path: str, output_path: str, max_samples: int = None):
        """
        Process entire dataset and save token embeddings
        
        Args:
            data_path: Path to JSON dataset file
            output_path: Path to save processed embeddings
            max_samples: Maximum number of samples to process (None for all)
        """
        print(f"Processing dataset: {data_path}")
        
        # Load dataset
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
            print(f"Processing first {max_samples} samples")
        
        print(f"Total samples to process: {len(data)}")
        
        # Process data
        processed_data = {
            'premise_tokens': [],
            'hypothesis_tokens': [],
            'labels': [],
            'token_counts': {'premise': [], 'hypothesis': []},
            'metadata': {
                'model_name': self.model_name,
                'hidden_size': self.hidden_size,
                'total_samples': len(data)
            }
        }
        
        for i, (premise, hypothesis, label) in enumerate(data):
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(data)} samples")
            
            try:
                premise_tokens, hypothesis_tokens = self.process_premise_hypothesis_pair(premise, hypothesis)
                
                processed_data['premise_tokens'].append(premise_tokens)
                processed_data['hypothesis_tokens'].append(hypothesis_tokens)  
                processed_data['labels'].append(label)
                processed_data['token_counts']['premise'].append(premise_tokens.shape[0])
                processed_data['token_counts']['hypothesis'].append(hypothesis_tokens.shape[0])
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # Save processed data
        print(f"Saving processed data to: {output_path}")
        
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle for efficient loading
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        # Print statistics
        premise_counts = processed_data['token_counts']['premise']
        hypothesis_counts = processed_data['token_counts']['hypothesis']
        
        print(f"\nProcessing Statistics:")
        print(f"Successfully processed: {len(processed_data['labels'])} samples")
        print(f"Premise token counts - Mean: {np.mean(premise_counts):.1f}, "
              f"Min: {np.min(premise_counts)}, Max: {np.max(premise_counts)}")
        print(f"Hypothesis token counts - Mean: {np.mean(hypothesis_counts):.1f}, "
              f"Min: {np.min(hypothesis_counts)}, Max: {np.max(hypothesis_counts)}")
        
        # Label distribution
        label_counts = {}
        for label in processed_data['labels']:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"Label distribution: {label_counts}")
        
        return processed_data


def main():
    """Process SNLI datasets"""
    
    extractor = SBERTTokenExtractor()
    
    # # Process training data (for order embedding training)
    # train_data_path = "MSc_Topology_Codebase/data/raw/mnli/train/mnli_full_train.json"  # Adjust path as needed
    # train_output_path = "/vol/bitbucket/ahb24/tda_entailment_new/mnli_train_sbert_tokens_ALL_SAMPLES.pkl"
    
    # if Path(train_data_path).exists():
    #     print("Processing SNLI training data...")
    #     processed_data = extractor.process_dataset(train_data_path, train_output_path, max_samples=None)  # Limit for memory
    #     del processed_data
    # else:
    #     print(f"Training data not found at: {train_data_path}")
    
    # Process validation data (for clustering tests)
    val_data_path = "MSc_Topology_Codebase/data/raw/snli/test/snli_full_test.json"  # Adjust path as needed  
    val_output_path = "/vol/bitbucket/ahb24/tda_entailment_new/snli_test_sbert_tokens.pkl"
    
    if Path(val_data_path).exists():
        print("\nProcessing SNLI validation data...")
        extractor.process_dataset(val_data_path, val_output_path, max_samples=None)  # Smaller for clustering tests
    else:
        print(f"Validation data not found at: {val_data_path}")

    
    print("\nSBERT token extraction completed!")


if __name__ == "__main__":
    main()