import torch
from transformers import AutoTokenizer, AutoModel
import json
import os
from typing import List, Dict, Tuple
import numpy as np

class TextToEmbedding:
    """Text to embedding pipeline using roBERTa. Converts premise-hypothesis pairs to contextualised embeddings"""

    def __init__(self, model_name="roberta-base", device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize text processing pipeline"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        print(f"Loading {model_name} on {device}...")

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval() # Set to eval mode

        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable

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

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

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
                # Use [CLS] token embeddings (first token)
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(batch_embeddings)

                del outputs
                del inputs

            if (i // batch_size + 1) % 5 == 0:
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return torch.cat(embeddings, dim=0)

    def concatenate_premise_hypothesis_embeddings(self, premise_embeddings: torch.Tensor, hypothesis_embeddings: torch.Tensor) -> torch.Tensor:
        """Concatenate premise and hypothesis embeddings for PHD analysis
        Args:
            premise_embeddings (torch.Tensor): premise embeddings tensor [n_samples, hidden_size]
            hypothesis_embeddings (torch.Tensor): hypothesis embeddings tensor [n_samples, hidden_size]
        Returns:
            torch.Tensor: concatenated premise and hypothesis embeddings [n_samples, 2*hidden_size]
        """
        assert premise_embeddings.shape == hypothesis_embeddings.shape

        concatenated = torch.cat([premise_embeddings, hypothesis_embeddings], dim=1)
        print(f"Concatenated embeddings shape: {concatenated.shape}")
        return concatenated

    def organize_embeddings_by_class(self, concatenated_embeddings: torch.Tensor, labels: List[str]) -> Dict[str, torch.Tensor]:
        """Organize concatenated embeddings by entailment class for PHD computation
        Args:
            concatenated_embeddings: Concatenated premise-hypothesis embeddings
            labels: List of entailment labels
        Returns:
            Dict mapping class names to their corresponding embeddings
        """
        class_embeddings = {}

        unique_labels = list(set(labels))
        for label in unique_labels:
            mask = torch.tensor([labels[i] == label for i in range(len(labels))], dtype=torch.bool)

            #Extract embeddings for this class
            class_embs = concatenated_embeddings[mask]
            class_embeddings[label] = class_embs

            print(f"Class '{label}': {class_embs.shape[0]} samples, embedding dim: {class_embs.shape[1]}")

        return class_embeddings


    def process_entailment_dataset(self, dataset_path:str, include_class_separation: bool = True) -> Dict:
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

        # For large datasets, automatically use chunked processing
        if len(data) > 5000:
            print(f"Large dataset detected ({len(data)} samples). Using chunked processing...")
            return self.process_entailment_dataset_chunked(
                dataset_path, chunk_size=2000, include_class_separation=include_class_separation
            )

        #Extract premises, hypotheses, labels
        premises = [item[0] for item in data]
        hypotheses = [item[1] for item in data]
        labels = [item[2] for item in data]

        print(f"Dataset contains {len(data)} premise-hypothesis pairs")
        print("Generating premise embeddings...")
        premise_embeddings = self.encode_text(premises)
        print("Generating hypothesis embeddings...")
        hypothesis_embeddings = self.encode_text(hypotheses)

        print("Concatenating premise and hypothesis embeddings...")
        concatenated_embeddings = self.concatenate_premise_hypothesis_embeddings(premise_embeddings, hypothesis_embeddings)

        #Prepare output
        result = {
            "premise_embeddings": premise_embeddings,
            "hypothesis_embeddings": hypothesis_embeddings,
            "concatenated_embeddings": concatenated_embeddings,
            "labels": labels,
            "texts": {
                "premises": premises,
                "hypotheses": hypotheses
            },
            "metadata": {
                "model_name": self.model_name,
                "concatenated_embedding_dim": concatenated_embeddings.shape[1],
                "n_samples": len(data),
                "label_counts": self._analyze_labels(labels)
            }
        }

        if include_class_separation:
            class_embeddings = self.organize_embeddings_by_class(concatenated_embeddings, labels)
            result["class_embeddings"] = class_embeddings

            result["metadata"]["class_embedding_shapes"] = {
                label: embs.shape for label, embs in class_embeddings.items()
            }

        print("Dataset processing complete")
        return result

    def process_entailment_dataset_chunked(self, dataset_path: str, chunk_size: int = 1000, 
                                         include_class_separation: bool = True) -> Dict:
        """Process large datasets in chunks to handle memory constraints
        
        Args:
            dataset_path: Path to JSON file with entailment pairs
            chunk_size: Number of samples to process at once
            include_class_separation: Whether to organize by class
        """
        print(f"Processing dataset in chunks: {dataset_path}...")

        # Load dataset
        with open(dataset_path, "r") as file:
            data = json.load(file)
        
        total_samples = len(data)
        print(f"Dataset contains {total_samples} premise-hypothesis pairs")
        print(f"Processing in chunks of {chunk_size} samples...")

        # Process in chunks
        all_premise_embeddings = []
        all_hypothesis_embeddings = []
        all_labels = []
        all_premises = []
        all_hypotheses = []

        for chunk_idx in range(0, total_samples, chunk_size):
            end_idx = min(chunk_idx + chunk_size, total_samples)
            chunk_data = data[chunk_idx:end_idx]
            
            print(f"\nProcessing chunk {chunk_idx//chunk_size + 1}/{(total_samples-1)//chunk_size + 1}")
            
            # Extract chunk data
            premises = [item[0] for item in chunk_data]
            hypotheses = [item[1] for item in chunk_data]
            labels = [item[2] for item in chunk_data]

            # Process chunk
            print("  Generating premise embeddings...")
            premise_embeddings = self.encode_text(premises, batch_size=8)  # Even smaller batches
            
            print("  Generating hypothesis embeddings...")
            hypothesis_embeddings = self.encode_text(hypotheses, batch_size=8)

            # Store results
            all_premise_embeddings.append(premise_embeddings)
            all_hypothesis_embeddings.append(hypothesis_embeddings)
            all_labels.extend(labels)
            all_premises.extend(premises)
            all_hypotheses.extend(hypotheses)

        # Combine all chunks
        print("\nCombining all chunks...")
        final_premise_embeddings = torch.cat(all_premise_embeddings, dim=0)
        final_hypothesis_embeddings = torch.cat(all_hypothesis_embeddings, dim=0)

        print("Concatenating premise and hypothesis embeddings...")
        concatenated_embeddings = self.concatenate_premise_hypothesis_embeddings(
            final_premise_embeddings, final_hypothesis_embeddings
        )

        # Prepare final result
        result = {
            "premise_embeddings": final_premise_embeddings,
            "hypothesis_embeddings": final_hypothesis_embeddings,
            "concatenated_embeddings": concatenated_embeddings,
            "labels": all_labels,
            "texts": {
                "premises": all_premises,
                "hypotheses": all_hypotheses
            },
            "metadata": {
                "model_name": self.model_name,
                "concatenated_embedding_dim": concatenated_embeddings.shape[1],
                "n_samples": len(all_labels),
                "label_counts": self._analyze_labels(all_labels),
                "processed_in_chunks": True,
                "chunk_size": chunk_size
            }
        }

        if include_class_separation:
            class_embeddings = self.organize_embeddings_by_class(concatenated_embeddings, all_labels)
            result["class_embeddings"] = class_embeddings
            result["metadata"]["class_embedding_shapes"] = {
                label: embs.shape for label, embs in class_embeddings.items()
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
        concatenated_embs = processed_data["concatenated_embeddings"]
        print(f"Premise embeddings shape: {premise_embs.shape}")
        print(f"Hypothesis embeddings shape: {hypothesis_embs.shape}")
        print(f"Concatenated embeddings shape: {concatenated_embs.shape}")
        print(f"Concatenated embedding dimension: {concatenated_embs.shape[1]}")

        expected_dim = premise_embs.shape[1] + hypothesis_embs.shape[1]
        assert concatenated_embs.shape[1] == expected_dim, \
            f"Concatenated dim {concatenated_embs.shape[1]} != expected {expected_dim}"

        #Check for reasonable ranges
        print(f"Premise embedding range: [{premise_embs.min():.3f}, {premise_embs.max():.3f}]")
        print(f"Hypothesis embedding range: [{hypothesis_embs.min():.3f}, {hypothesis_embs.max():.3f}]")

        #Check for NaN values
        assert not torch.isnan(premise_embs).any(), "NaN values in premise embeddings"
        assert not torch.isnan(hypothesis_embs).any(), "NaN values in hypothesis embeddings"

        print(f"Concatenated embedding range: [{concatenated_embs.min():.3f}, {concatenated_embs.max():.3f}]")
        assert not torch.isnan(concatenated_embs).any(), "NaN values in concatenated embeddings"

        if "class_embeddings" in processed_data:
            print("\nClass embedding validation:")
            for label, class_embs in processed_data["class_embeddings"].items():
                print(f"  {label}: {class_embs.shape[0]} samples, dim {class_embs.shape[1]}")
                assert not torch.isnan(class_embs).any(), f"NaN values in {label} embeddings"

def test_text_processing():
    """Test text processing pipeline on toy data"""

    processor = TextToEmbedding()
    premise = "All cars are fast"
    hypothesis = "Some cars are fast"

    premise_emb, hypothesis_emb = processor.process_single_pair(premise, hypothesis)
    print(f"Single pair test - Premise shape: {premise_emb.shape}, Hypothesis shape: {hypothesis_emb.shape}")

    # Test dataset
    data_path = "data/raw/snli/train/snli_full_train.json"
    output_path = "/vol/bitbucket/ahb24/phd_processed_data"
    if not os.path.exists(output_path):
        print(f"ERROR: Output directory not found at {output_path}")
        return None
    else:
        print(f"Output directory found: {output_path}")

    if os.path.exists(data_path):
        processed_data = processor.process_entailment_dataset(data_path, include_class_separation=True)
        processor.validate_embeddings(processed_data)

        # Display class organization results
        if "class_embeddings" in processed_data:
            print(f"\nClass organization summary:")
            for label, embeddings in processed_data["class_embeddings"].items():
                print(f"  {label}: {embeddings.shape[0]} samples with {embeddings.shape[1]}D embeddings")

        # Save processed data
        full_output_path = "/vol/bitbucket/ahb24/phd_processed_data/snli_full_phd_roberta.pt"
        processor.save_processed_data(processed_data, full_output_path)

        print("Text processing pipeline test completed successfully")
        return processed_data
    else:
        print("Text processing pipeline test failed")

if __name__ == "__main__":
    test_text_processing()





