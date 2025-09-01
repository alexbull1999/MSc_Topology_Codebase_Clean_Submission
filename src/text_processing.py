import torch
from transformers import AutoTokenizer, AutoModel
import json
import os
from typing import List, Dict, Tuple
import numpy as np

class TextToEmbedding:
    """Text to embedding pipeline using BERT. Converts premise-hypothesis pairs to contextualised embeddings"""

    def __init__(self, model_name="bert-base-uncased", device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize text processing pipeline"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        print(f"Loading {model_name} on {device}...")

        # Set environment variable for memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  #Added for mem_mgmt

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval() # Set to eval mode

        # Enable gradient checkpointing to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable() #added for mem_mgmt

        print("Text processing pipeline ready")

    def encode_text(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Convert list of texts to BERT embeddings using [CLS] token
        Args:
            texts (List[str]): list of texts to encode
            batch_size (int, optional): batch size. Defaults to 32.
        Returns:
            torch.Tensor: BERT embeddings (of shape (n_texts, hidden_size))
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]

            #Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )

            # Move each tensor in the inputs dictionary to GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}


            #Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Use [CLS] token embeddings (first token) (added for mem_mgmt)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].clone()
                
                # Move to CPU immediately to free GPU memory
                batch_embeddings = batch_embeddings.cpu()
                embeddings.append(batch_embeddings)
                
                # Clear intermediate results from GPU
                del outputs
                del inputs

             # Clear cache every few batches
            if (i // batch_size + 1) % 5 == 0:
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # batch_embeddings = outputs.last_hidden_state[:, 0, :] (PREVIOUS)
                # embeddings.append(batch_embeddings)
            
        # Final cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return torch.cat(embeddings, dim=0)


    def process_entailment_dataset(self, dataset_path:str) -> Dict:
        """Process entailment dataset to embeddings and return structured data
        Args:
            dataset_path: Path to JSON file with entailment pairs
        Returns:
            Dict containing embeddings and metadata
        """
        print(f"Processing dataset: {dataset_path}...")

        #Load dataset
        with open(dataset_path, "r") as file:
            data = json.load(file)

        #Extract premises, hypotheses, labels
        premises = [item[0] for item in data]
        hypotheses = [item[1] for item in data]
        labels = [item[2] for item in data]

        print(f"Dataset contains {len(data)} premise-hypothesis pairs")
        print("Generating premise embeddings...")
        premise_embeddings = self.encode_text(premises)
        print("Generating hypothesis embeddings...")
        hypothesis_embeddings = self.encode_text(hypotheses)

        #Prepare output
        result = {
            "premise_embeddings": premise_embeddings,
            "hypothesis_embeddings": hypothesis_embeddings,
            "labels": labels,
            "texts": {
                "premises": premises,
                "hypotheses": hypotheses
            },
            "metadata": {
                "model_name": self.model_name,
                "embedding_dim": premise_embeddings.shape[1],
                "n_samples": len(data),
                "label_counts": self._analyze_labels(labels)
            }
        }

        print("Dataset processing complete")
        return result

    def _analyze_labels(self, labels: List[str]) -> Dict:
        """Analyze label distribution in dataset"""
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts


    def process_single_pair(self, premise: str, hypothesis: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a single premise-hypothesis pair.
        Args:
            premise: premise text
            hypothesis: hypothesis text
        Returns:
            Tuple of (premise-embeddings, hypothesis-embeddings)
        """
        premise_embeddings = self.encode_text([premise])
        hypothesis_embeddings = self.encode_text([hypothesis])
        return premise_embeddings[0], hypothesis_embeddings[0]

    def save_processed_data(self, processed_data: Dict, output_path: str):
        """Save processed embeddings and metadata
        Args:
            processed_data (Dict): Output from process_entailment_dataset
            output_path (str): Path to save the processed data
        """
        torch.save(processed_data, output_path)
        print(f"Saved processed data to {output_path}")

    def load_processed_data(self, data_path: str) -> Dict:
        """Load previously processed data"""
        data = torch.load(data_path)
        print(f"Loaded processed data from {data_path}")
        print(f"Contains {data['metadata']['n_samples']} samples")
        return data

    def validate_embeddings(self, processed_data: Dict):
        """Validate that embeddings are reasonable"""

        premise_embs = processed_data["premise_embeddings"]
        hypothesis_embs = processed_data["hypothesis_embeddings"]
        print(f"Premise embeddings shape: {premise_embs.shape}")
        print(f"Hypothesis embeddings shape: {hypothesis_embs.shape}")
        print(f"Embedding dimension: {premise_embs.shape[1]}")

        #Check for reasonable ranges
        print(f"Premise embedding range: [{premise_embs.min():.3f}, {premise_embs.max():.3f}]")
        print(f"Hypothesis embedding range: [{hypothesis_embs.min():.3f}, {hypothesis_embs.max():.3f}]")

        #Check for NaN values
        assert not torch.isnan(premise_embs).any(), "NaN values in premise embeddings"
        assert not torch.isnan(hypothesis_embs).any(), "NaN values in hypothesis embeddings"

def test_text_processing():
    """Test text processing pipeline on toy data"""

    processor = TextToEmbedding()
    premise = "All cars are fast"
    hypothesis = "Some cars are fast"

    premise_emb, hypothesis_emb = processor.process_single_pair(premise, hypothesis)
    print(f"Single pair test - Premise shape: {premise_emb.shape}, Hypothesis shape: {hypothesis_emb.shape}")

    # Test dataset
    data_path = "data/raw/snli/train/snli_10k_subset_balanced.json"
    if os.path.exists(data_path):
        processed_data = processor.process_entailment_dataset(data_path)
        processor.validate_embeddings(processed_data)

        # Save processed data
        output_path = "data/processed/snli_10k_subset_train_BERT_BASE.pt"
        processor.save_processed_data(processed_data, output_path)

        print("Text processing pipeline test completed successfully")
        return processed_data
    else:
        print("Text processing pipeline test failed")

if __name__ == "__main__":
    test_text_processing()





