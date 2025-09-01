"""
ChaosNLI SBERT Token Extractor
Adapts existing SBERT token extractor for ChaosNLI JSONL format
"""

import torch
from transformers import AutoTokenizer, AutoModel
import json
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import pickle


class ChaosNLISBERTTokenExtractor:
    """Extract token-level embeddings from SBERT hidden layers for ChaosNLI data"""
    
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
        print(f"ChaosNLI SBERT token extractor ready")
    
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
        """Process a premise-hypothesis pair to extract token embeddings"""
        premise_tokens = self.extract_token_embeddings(premise)
        hypothesis_tokens = self.extract_token_embeddings(hypothesis)
        
        return premise_tokens, hypothesis_tokens
    
    def load_chaosnli_data(self, data_path: str) -> List[Dict]:
        """
        Load ChaosNLI JSONL data
        
        Args:
            data_path: Path to ChaosNLI JSONL file
            
        Returns:
            List of parsed JSON objects
        """
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def process_chaosnli_dataset(self, data_path: str, output_path: str, max_samples: int = None):
        """
        Process ChaosNLI dataset and save token embeddings
        
        Args:
            data_path: Path to ChaosNLI JSONL file
            output_path: Path to save processed embeddings
            max_samples: Maximum number of samples to process (None for all)
        """
        print(f"Processing ChaosNLI dataset: {data_path}")
        
        # Load ChaosNLI JSONL data
        data = self.load_chaosnli_data(data_path)
        
        if max_samples:
            data = data[:max_samples]
            print(f"Processing first {max_samples} samples")
        
        print(f"Total samples to process: {len(data)}")
        
        # Process data
        processed_data = {
            'premise_tokens': [],
            'hypothesis_tokens': [],
            'labels': [],  # majority labels
            'label_distributions': [],  # Human label distributions [E, N, C]
            'label_counts': [],  # Raw counts [E, N, C]
            'entropies': [],  # Human annotation entropy
            'uids': [],  # Unique identifiers
            'old_labels': [],  # Original dataset labels
            'token_counts': {'premise': [], 'hypothesis': []},
            'metadata': {
                'model_name': self.model_name,
                'hidden_size': self.hidden_size,
                'total_samples': len(data)
            }
        }
        
        for i, sample in enumerate(data):
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(data)} samples")
            
            try:
                # Extract text from ChaosNLI format
                premise = sample['example']['premise']
                hypothesis = sample['example']['hypothesis']
                
                # Extract token embeddings
                premise_tokens, hypothesis_tokens = self.process_premise_hypothesis_pair(premise, hypothesis)
                
                # Store embeddings
                processed_data['premise_tokens'].append(premise_tokens)
                processed_data['hypothesis_tokens'].append(hypothesis_tokens)
                
                # Store labels and distributions
                processed_data['labels'].append(sample['majority_label'])
                processed_data['label_distributions'].append(sample['label_dist'])
                processed_data['label_counts'].append(sample['label_count'])
                processed_data['entropies'].append(sample['entropy'])
                processed_data['uids'].append(sample['uid'])
                processed_data['old_labels'].append(sample['old_label'])
                
                # Store token counts
                processed_data['token_counts']['premise'].append(premise_tokens.shape[0])
                processed_data['token_counts']['hypothesis'].append(hypothesis_tokens.shape[0])
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # Save processed data
        print(f"Saving processed data to: {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        # Print statistics
        self._print_processing_statistics(processed_data)
        
        return processed_data
    
    def _print_processing_statistics(self, processed_data: Dict):
        """Print processing statistics"""
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
        print(f"Majority label distribution: {label_counts}")
        
        # Entropy statistics
        entropies = processed_data['entropies']
        print(f"Human annotation entropy - Mean: {np.mean(entropies):.3f}, "
              f"Min: {np.min(entropies):.3f}, Max: {np.max(entropies):.3f}")
        
        # Agreement analysis
        high_agreement = sum(1 for e in entropies if e < 0.5)
        medium_agreement = sum(1 for e in entropies if 0.5 <= e < 1.0)
        low_agreement = sum(1 for e in entropies if e >= 1.0)
        
        print(f"Agreement levels:")
        print(f"  High agreement (entropy < 0.5): {high_agreement} ({high_agreement/len(entropies)*100:.1f}%)")
        print(f"  Medium agreement (0.5 ≤ entropy < 1.0): {medium_agreement} ({medium_agreement/len(entropies)*100:.1f}%)")
        print(f"  Low agreement (entropy ≥ 1.0): {low_agreement} ({low_agreement/len(entropies)*100:.1f}%)")


def main():
    """Process ChaosNLI datasets"""
    
    extractor = ChaosNLISBERTTokenExtractor()
    
    # ChaosNLI-SNLI
    snli_data_path = "MSc_Topology_Codebase/data/chaosNLI/chaosNLI_v1.0/chaosNLI_snli.jsonl"
    snli_output_path = "/vol/bitbucket/ahb24/tda_entailment_new/chaosnli_snli_sbert_tokens.pkl"
    
    if Path(snli_data_path).exists():
        print("Processing ChaosNLI-SNLI data...")
        extractor.process_chaosnli_dataset(snli_data_path, snli_output_path)
    else:
        print(f"ChaosNLI-SNLI data not found at: {snli_data_path}")
    
    # ChaosNLI-MNLI
    mnli_data_path = "MSc_Topology_Codebase/data/chaosNLI/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl"
    mnli_output_path = "/vol/bitbucket/ahb24/tda_entailment_new/chaosnli_mnli_matched_sbert_tokens.pkl"
    
    if Path(mnli_data_path).exists():
        print("\nProcessing ChaosNLI-MNLI data...")
        extractor.process_chaosnli_dataset(mnli_data_path, mnli_output_path)
    else:
        print(f"ChaosNLI-MNLI data not found at: {mnli_data_path}")
    
    print("\nChaosNLI SBERT token extraction completed!")


if __name__ == "__main__":
    main()