"""
Data Loading Module for Global Dataset Contrastive Training
Clean implementation with flexible embedding types
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Iterator


class FlexibleEmbedder:
    """
    Generates various types of embeddings from premise-hypothesis pairs
    """
    
    def __init__(self, embedding_type='lattice', epsilon=1e-8):
        """
        Initialize embedder
        
        Args:
            embedding_type: 'lattice', 'concat', 'difference', or 'cosine_concat'
            epsilon: Small value for numerical stability
        """
        self.embedding_type = embedding_type
        self.epsilon = epsilon
        
        print(f"FlexibleEmbedder initialized with type: '{embedding_type}'")
        print(f"Output dimension will be: {self.get_output_dim()}")
    
    def get_output_dim(self):
        """Get the output dimension for the current embedding type"""
        if self.embedding_type == 'lattice':
            return 768
        elif self.embedding_type == 'concat':
            return 1536  # 768 * 2
        elif self.embedding_type == 'difference':
            return 768
        elif self.embedding_type == 'cosine_concat':
            return 1537  # 768 * 2 + 1
        else:
            raise ValueError(f"Unknown embedding_type: {self.embedding_type}")
    
    def generate_embeddings(self, premise_embeddings, hypothesis_embeddings, batch_size=1000):
        """
        Generate embeddings from premise-hypothesis pairs
        
        Args:
            premise_embeddings: Tensor of premise embeddings [N, 768]
            hypothesis_embeddings: Tensor of hypothesis embeddings [N, 768]
            batch_size: Batch size for processing to avoid memory issues
            
        Returns:
            embeddings: Tensor of embeddings [N, output_dim]
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Generating {self.embedding_type} embeddings on {device}")
        print(f"Processing {len(premise_embeddings)} samples in batches of {batch_size}")
        
        total_samples = len(premise_embeddings)
        all_embeddings = []
        
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            batch_num = i // batch_size + 1
            total_batches = (total_samples - 1) // batch_size + 1
            
            if batch_num % 5 == 1:  # Print every 5th batch
                print(f"  Processing batch {batch_num}/{total_batches}")
            
            # Get batch
            premise_batch = premise_embeddings[i:end_idx].to(device)
            hypothesis_batch = hypothesis_embeddings[i:end_idx].to(device)
            
            # Compute embeddings based on type
            with torch.no_grad():
                if self.embedding_type == 'lattice':
                    # Original lattice containment formula
                    batch_embeddings = (premise_batch * hypothesis_batch) / (
                        torch.abs(premise_batch) + torch.abs(hypothesis_batch) + self.epsilon
                    )
                
                elif self.embedding_type == 'concat':
                    # Simple concatenation: [premise, hypothesis]
                    batch_embeddings = torch.cat([premise_batch, hypothesis_batch], dim=1)
                
                elif self.embedding_type == 'difference':
                    # Element-wise difference: premise - hypothesis
                    batch_embeddings = premise_batch - hypothesis_batch
                
                elif self.embedding_type == 'cosine_concat':
                    # Concatenation + cosine similarity
                    cos_sim = torch.cosine_similarity(premise_batch, hypothesis_batch, dim=1)
                    batch_embeddings = torch.cat([premise_batch, hypothesis_batch, cos_sim.unsqueeze(1)], dim=1)
                
                else:
                    raise ValueError(f"Unknown embedding_type: {self.embedding_type}")
            
            # Move back to CPU and store
            all_embeddings.append(batch_embeddings.cpu())
            
            # Clear GPU memory
            del premise_batch, hypothesis_batch, batch_embeddings
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Concatenate all batches
        final_embeddings = torch.cat(all_embeddings, dim=0)
        
        print(f"Generated {self.embedding_type} embeddings: {final_embeddings.shape}")
        return final_embeddings


class EntailmentDataset(Dataset):
    """
    Dataset for entailment classification
    """
    
    def __init__(self, embeddings, labels):
        """
        Initialize dataset
        
        Args:
            embeddings: Tensor of embeddings [N, embedding_dim]
            labels: List or tensor of labels (strings or integers)
        """
        self.embeddings = embeddings
        self.labels = self._process_labels(labels)
        
        # Verify data consistency
        assert len(self.embeddings) == len(self.labels), \
            f"Embedding count ({len(self.embeddings)}) != label count ({len(self.labels)})"
        
        print(f"EntailmentDataset created: {len(self)} samples")
        print(f"  Embedding shape: {self.embeddings.shape}")
        print(f"  Class distribution: {self.get_class_distribution()}")
    
    def _process_labels(self, labels):
        """Convert labels to integer format"""
        if isinstance(labels, torch.Tensor):
            return labels.long()
        
        # If labels are strings, convert to integers
        if isinstance(labels[0], str):
            label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
            return torch.tensor([label_map[label] for label in labels], dtype=torch.long)
        
        # If already integers, convert to tensor
        return torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return {
            'embeddings': self.embeddings[idx],
            'labels': self.labels[idx]
        }
    
    def get_class_distribution(self):
        """Get distribution of classes in the dataset"""
        unique_labels, counts = torch.unique(self.labels, return_counts=True)
        
        class_names = ['entailment', 'neutral', 'contradiction']
        distribution = {}
        
        for label, count in zip(unique_labels.tolist(), counts.tolist()):
            distribution[class_names[label]] = count
        
        return distribution


class BalancedBatchSampler(Sampler):
    """
    Sampler that yields batches with equal number of samples from each class
    """
    
    def __init__(self, dataset, batch_size: int, samples_per_class: int = None):
        """
        Args:
            dataset: Dataset with labels accessible via dataset.labels
            batch_size: Total batch size
            samples_per_class: Number of samples per class in each batch.
                              If None, will be batch_size // num_classes
        """
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Get labels
        if hasattr(dataset, 'labels'):
            self.labels = dataset.labels.tolist() if torch.is_tensor(dataset.labels) else dataset.labels
        else:
            # Extract labels by iterating through dataset
            self.labels = [dataset[i]['labels'].item() for i in range(len(dataset))]
        
        # Group indices by class
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_indices[label].append(idx)
        
        self.num_classes = len(self.class_indices)
        
        # Set samples per class
        if samples_per_class is None:
            self.samples_per_class = batch_size // self.num_classes
        else:
            self.samples_per_class = samples_per_class
            
        # Ensure batch size is compatible
        assert self.samples_per_class * self.num_classes <= batch_size, \
            f"samples_per_class ({self.samples_per_class}) * num_classes ({self.num_classes}) " \
            f"must be <= batch_size ({batch_size})"
        
        # Calculate number of batches
        min_class_size = min(len(indices) for indices in self.class_indices.values())
        self.num_batches = min_class_size // self.samples_per_class
        
        print(f"BalancedBatchSampler initialized:")
        print(f"  Classes: {list(self.class_indices.keys())}")
        print(f"  Samples per class per batch: {self.samples_per_class}")
        print(f"  Effective batch size: {self.samples_per_class * self.num_classes}")
        print(f"  Number of batches: {self.num_batches}")
        for class_label, indices in self.class_indices.items():
            print(f"  Class {class_label}: {len(indices)} samples")
    
    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle indices for each class
        shuffled_indices = {}
        for class_label, indices in self.class_indices.items():
            shuffled = indices.copy()
            np.random.shuffle(shuffled)
            shuffled_indices[class_label] = shuffled
        
        # Generate batches
        for batch_idx in range(self.num_batches):
            batch = []
            
            # Add samples from each class
            for class_label in sorted(self.class_indices.keys()):
                start_idx = batch_idx * self.samples_per_class
                end_idx = start_idx + self.samples_per_class
                class_samples = shuffled_indices[class_label][start_idx:end_idx]
                batch.extend(class_samples)
            
            # Shuffle the batch to mix classes
            np.random.shuffle(batch)
            yield batch
    
    def __len__(self) -> int:
        return self.num_batches


class GlobalDataLoader:
    """
    Main data loader class for global dataset contrastive training
    """
    
    def __init__(self, train_path, val_path, test_path, embedding_type='lattice',
                 sample_size=None, random_state=42, batch_size=1000):
        """
        Initialize data loader
        
        Args:
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Path to test data
            embedding_type: Type of embeddings to generate
            sample_size: Number of samples to use from training (None for all)
            random_state: Random seed for reproducibility
            batch_size: Batch size for embedding generation
        """
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.embedding_type = embedding_type
        self.sample_size = sample_size
        self.random_state = random_state
        self.batch_size = batch_size
        
        self.embedder = FlexibleEmbedder(embedding_type=embedding_type)
        
        # Will be populated by load_data()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        print(f"GlobalDataLoader initialized:")
        print(f"  Embedding type: {embedding_type}")
        print(f"  Output dimension: {self.embedder.get_output_dim()}")
        print(f"  Sample size: {sample_size if sample_size else 'All'}")
    
    def load_snli_data(self, data_path, split_name, apply_sampling=False):
        """Load SNLI data from preprocessed torch file"""
        print(f"Loading {split_name} data from {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        data = torch.load(data_path, weights_only=False)
        print(f"  Loaded keys: {list(data.keys())}")
        
        # Only sample from training data if requested
        if apply_sampling and self.sample_size:
            np.random.seed(self.random_state)
            total_samples = len(data['labels'])
            
            if self.sample_size > total_samples:
                print(f"  Warning: Requested sample size {self.sample_size} > total samples {total_samples}")
                self.sample_size = total_samples
            
            indices = np.random.choice(total_samples, self.sample_size, replace=False)
            
            data = {
                'premise_embeddings': torch.stack([data['premise_embeddings'][i] for i in indices]),
                'hypothesis_embeddings': torch.stack([data['hypothesis_embeddings'][i] for i in indices]),
                'labels': [data['labels'][i] for i in indices]
            }
            
            print(f"  Subsampled to {len(data['labels'])} samples")
        
        print(f"  Final dataset: {len(data['labels'])} {split_name} samples")
        return data
    
    def generate_embeddings(self, data, split_name):
        """Generate embeddings from raw data"""
        print(f"Generating embeddings for {split_name}...")
        
        embeddings = self.embedder.generate_embeddings(
            data['premise_embeddings'], 
            data['hypothesis_embeddings'],
            batch_size=self.batch_size
        )
        
        return embeddings, data['labels']
    
    def load_data(self):
        """Main method to load and prepare all data"""
        print("Starting data loading pipeline...")
        print("=" * 60)
        
        # Load each split separately
        train_data = self.load_snli_data(self.train_path, "training", apply_sampling=True)
        val_data = self.load_snli_data(self.val_path, "validation", apply_sampling=False)
        test_data = self.load_snli_data(self.test_path, "test", apply_sampling=False)
        
        # Generate embeddings for each split
        train_embeddings, train_labels = self.generate_embeddings(train_data, "training")
        val_embeddings, val_labels = self.generate_embeddings(val_data, "validation")
        test_embeddings, test_labels = self.generate_embeddings(test_data, "test")
        
        # Create dataset objects
        self.train_dataset = EntailmentDataset(train_embeddings, train_labels)
        self.val_dataset = EntailmentDataset(val_embeddings, val_labels)
        self.test_dataset = EntailmentDataset(test_embeddings, test_labels)
        
        print(f"\nData loading pipeline completed!")
        print(f"Output embedding dimension: {self.embedder.get_output_dim()}")
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def get_dataloaders(self, batch_size=1020, balanced_sampling=True, num_workers=0):
        """
        Create PyTorch DataLoaders for training
        
        Args:
            batch_size: Batch size for training
            balanced_sampling: Whether to use balanced batch sampling
            num_workers: Number of workers for data loading
            
        Returns:
            train_loader, val_loader, test_loader
        """
        if self.train_dataset is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if balanced_sampling:
            # Use balanced sampling for training
            train_sampler = BalancedBatchSampler(
                self.train_dataset, 
                batch_size=batch_size
            )
            
            train_loader = DataLoader(
                self.train_dataset,
                batch_sampler=train_sampler,
                num_workers=num_workers
            )
            
            # Use balanced sampling for validation too (for consistency)
            val_sampler = BalancedBatchSampler(
                self.val_dataset,
                batch_size=batch_size
            )
            
            val_loader = DataLoader(
                self.val_dataset,
                batch_sampler=val_sampler,
                num_workers=num_workers
            )
        else:
            # Regular random sampling
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
            
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
        
        # Test loader is always regular
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        print(f"DataLoaders created:")
        print(f"  Batch size: {batch_size}")
        print(f"  Balanced sampling: {balanced_sampling}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader


def test_data_loader():
    """Test data loading functionality with synthetic data"""
    print("Testing GlobalDataLoader...")
    
    # Create synthetic data
    n_samples = 1000
    premise_embeddings = torch.randn(n_samples, 768)
    hypothesis_embeddings = torch.randn(n_samples, 768)
    labels = torch.randint(0, 3, (n_samples,)).tolist()
    
    # Save synthetic data
    synthetic_data = {
        'premise_embeddings': premise_embeddings,
        'hypothesis_embeddings': hypothesis_embeddings,
        'labels': ['entailment' if l == 0 else 'neutral' if l == 1 else 'contradiction' for l in labels]
    }
    
    test_path = 'test_synthetic_data.pt'
    torch.save(synthetic_data, test_path)
    
    try:
        # Test different embedding types
        for embedding_type in ['lattice', 'concat', 'difference']:
            print(f"\nTesting embedding type: {embedding_type}")
            
            # Create data loader
            data_loader = GlobalDataLoader(
                train_path=test_path,
                val_path=test_path,
                test_path=test_path,
                embedding_type=embedding_type,
                sample_size=500
            )
            
            # Load data
            train_dataset, val_dataset, test_dataset = data_loader.load_data()
            
            # Create dataloaders
            train_loader, val_loader, test_loader = data_loader.get_dataloaders(
                batch_size=64, balanced_sampling=True
            )
            
            # Test one batch
            batch = next(iter(train_loader))
            print(f"  Batch shape: {batch['embeddings'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")
    
    finally:
        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)
    
    print("✅ Data loader test completed!")


if __name__ == "__main__":
    test_data_loader()