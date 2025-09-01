"""
Data Loading Module for Supervised Contrastive Autoencoder
Handles SNLI data loading, lattice containment embedding generation, and batch processing
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Iterator
import os
from collections import defaultdict

class LatticeContainmentEmbedder:
    """
    Generates lattice containment embeddings from premise-hypothesis pairs
    """
    
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
    
    # def generate_embeddings(self, premise_embeddings, hypothesis_embeddings, batch_size=1000):
    #     """
    #     Generate lattice containment embeddings from premise-hypothesis pairs
        
    #     Args:
    #         premise_embeddings: Tensor of premise embeddings [N, 768]
    #         hypothesis_embeddings: Tensor of hypothesis embeddings [N, 768]
    #         batch_size: Batch size for processing to avoid memory issues
            
    #     Returns:
    #         lattice_embeddings: Tensor of lattice containment embeddings [N, 768]
    #     """
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    #     print(f"Generating lattice containment embeddings on {device}")
    #     print(f"Processing {len(premise_embeddings)} samples in batches of {batch_size}")
        
    #     total_samples = len(premise_embeddings)
    #     all_lattice_embeddings = []
        
    #     for i in range(0, total_samples, batch_size):
    #         end_idx = min(i + batch_size, total_samples)
    #         batch_num = i // batch_size + 1
    #         total_batches = (total_samples - 1) // batch_size + 1
            
    #         print(f"Processing batch {batch_num}/{total_batches}")
            
    #         # Get batch
    #         premise_batch = premise_embeddings[i:end_idx]
    #         hypothesis_batch = hypothesis_embeddings[i:end_idx]
            
    #         # Move to device
    #         premise_batch = premise_batch.to(device)
    #         hypothesis_batch = hypothesis_batch.to(device)
            
    #         # Compute lattice embeddings
    #         with torch.no_grad():
    #             lattice_batch = (premise_batch * hypothesis_batch) / (
    #                 torch.abs(premise_batch) + torch.abs(hypothesis_batch) + self.epsilon
    #             )
            
    #         # Move back to CPU and store
    #         all_lattice_embeddings.append(lattice_batch.cpu())
            
    #         # Clear GPU memory
    #         del premise_batch, hypothesis_batch, lattice_batch
    #         if device.type == 'cuda':
    #             torch.cuda.empty_cache()
        
    #     # Concatenate all batches
    #     lattice_embeddings = torch.cat(all_lattice_embeddings, dim=0)
        
    #     print(f"Generated lattice embeddings shape: {lattice_embeddings.shape}")
    #     return lattice_embeddings

    def generate_embeddings(self, premise_embeddings, hypothesis_embeddings, batch_size=1000):
        """
        Generate SBERT concatenation embeddings from premise-hypothesis pairs
    
        Args:
            premise_embeddings: Tensor of premise embeddings [N, 768]
            hypothesis_embeddings: Tensor of hypothesis embeddings [N, 768]
            batch_size: Batch size for processing to avoid memory issues
        
        Returns:
            concat_embeddings: Tensor of concatenated embeddings [N, 1536]
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        print(f"Generating SBERT concatenation embeddings on {device}")
        print(f"Processing {len(premise_embeddings)} samples in batches of {batch_size}")
    
        total_samples = len(premise_embeddings)
        all_concat_embeddings = []
    
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            batch_num = i // batch_size + 1
            total_batches = (total_samples - 1) // batch_size + 1
        
            print(f"Processing batch {batch_num}/{total_batches}")
        
            # Get batch
            premise_batch = premise_embeddings[i:end_idx]
            hypothesis_batch = hypothesis_embeddings[i:end_idx]
        
            # Move to device
            premise_batch = premise_batch.to(device)
            hypothesis_batch = hypothesis_batch.to(device)
        
            # Compute concatenation embeddings
            with torch.no_grad():
                # Simple concatenation: [premise, hypothesis]
                concat_batch = torch.cat([premise_batch, hypothesis_batch], dim=1)
        
            # Move back to CPU and store
            all_concat_embeddings.append(concat_batch.cpu())
        
            # Clear GPU memory
            del premise_batch, hypothesis_batch, concat_batch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
        # Concatenate all batches
        concat_embeddings = torch.cat(all_concat_embeddings, dim=0)
    
        print(f"Generated concatenation embeddings shape: {concat_embeddings.shape}")
        return concat_embeddings


class EntailmentDataset(Dataset):
    """
    Dataset for entailment classification with lattice containment embeddings
    """
    
    def __init__(self, lattice_embeddings, labels):
        """
        Initialize dataset
        
        Args:
            lattice_embeddings: Tensor of lattice containment embeddings [N, 768]
            labels: List or tensor of labels (strings or integers)
        """
        self.lattice_embeddings = lattice_embeddings
        self.labels = self._process_labels(labels)
        
        # Verify data consistency
        assert len(self.lattice_embeddings) == len(self.labels), \
            f"Embedding count ({len(self.lattice_embeddings)}) != label count ({len(self.labels)})"
    
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
        return len(self.lattice_embeddings)
    
    def __getitem__(self, idx):
        return {
            'embeddings': self.lattice_embeddings[idx],
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


class EntailmentDataLoader:
    """
    Main data loader class for the supervised contrastive autoencoder
    """
    
    def __init__(self, train_path, val_path, test_path, sample_size=None, 
                 random_state=42, batch_size=1000):
        """
        Initialize data loader
        
        Args:
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Path to test data
            sample_size: Number of samples to use from training (None for all)
            random_state: Random seed for reproducibility
            batch_size: Batch size for lattice embedding generation
        """
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.sample_size = sample_size
        self.random_state = random_state
        self.batch_size = batch_size
        
        self.embedder = LatticeContainmentEmbedder()
        
        # Will be populated by load_data()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def load_snli_data(self, data_path, split_name, apply_sampling=False):
        """Load SNLI data from preprocessed torch file"""
        print(f"Loading {split_name} data from {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        data = torch.load(data_path, weights_only=False)
        
        # Only sample from training data if requested
        if apply_sampling and self.sample_size:
            np.random.seed(self.random_state)
            total_samples = len(data['labels'])
            
            if self.sample_size > total_samples:
                print(f"Warning: Requested sample size {self.sample_size} > total samples {total_samples}")
                self.sample_size = total_samples
            
            indices = np.random.choice(total_samples, self.sample_size, replace=False)
            
            data = {
                'premise_embeddings': torch.stack([data['premise_embeddings'][i] for i in indices]),
                'hypothesis_embeddings': torch.stack([data['hypothesis_embeddings'][i] for i in indices]),
                'labels': [data['labels'][i] for i in indices]
            }
        
        print(f"Loaded {len(data['labels'])} {split_name} samples")
        return data
    
    def generate_lattice_embeddings(self, data, split_name):
        """Generate lattice containment embeddings from raw data"""
        print(f"Generating lattice containment embeddings for {split_name}...")
        
        lattice_embeddings = self.embedder.generate_embeddings(
            data['premise_embeddings'], 
            data['hypothesis_embeddings'],
            batch_size=self.batch_size
        )
        
        return lattice_embeddings, data['labels']
    
    def load_data(self):
        """Main method to load and prepare all data"""
        print("Starting data loading pipeline...")
        print("=" * 50)
        
        # Load each split separately
        train_data = self.load_snli_data(self.train_path, "training", apply_sampling=True)
        val_data = self.load_snli_data(self.val_path, "validation", apply_sampling=False)
        test_data = self.load_snli_data(self.test_path, "test", apply_sampling=False)
        
        # Generate lattice embeddings for each split
        train_embeddings, train_labels = self.generate_lattice_embeddings(train_data, "training")
        val_embeddings, val_labels = self.generate_lattice_embeddings(val_data, "validation")
        test_embeddings, test_labels = self.generate_lattice_embeddings(test_data, "test")
        
        # Create dataset objects
        self.train_dataset = EntailmentDataset(train_embeddings, train_labels)
        self.val_dataset = EntailmentDataset(val_embeddings, val_labels)
        self.test_dataset = EntailmentDataset(test_embeddings, test_labels)
        
        # Print dataset statistics
        print("\nDataset Statistics:")
        print("-" * 30)
        
        for split_name, dataset in [('Train', self.train_dataset), 
                                   ('Validation', self.val_dataset), 
                                   ('Test', self.test_dataset)]:
            distribution = dataset.get_class_distribution()
            print(f"{split_name}: {distribution}")
        
        print("\nData loading pipeline completed!")
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def get_dataloaders(self, batch_size=32, shuffle=True, num_workers=0):
        """
        Create PyTorch DataLoaders for training
        
        Args:
            batch_size: Batch size for training
            shuffle: Whether to shuffle training data
            num_workers: Number of workers for data loading
            
        Returns:
            train_loader, val_loader, test_loader
        """
        if self.train_dataset is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_loader, val_loader, test_loader

    def get_balanced_dataloaders(self, batch_size=32, samples_per_class=None, num_workers=0):
        """
        Create balanced PyTorch DataLoaders for contrastive training
    
        Args:
            batch_size: Total batch size
            samples_per_class: Number of samples per class per batch
            num_workers: Number of workers for data loading
        
        Returns:
            train_loader, val_loader, test_loader with balanced batching
        """
        if self.train_dataset is None:
            raise ValueError("Data not loaded. Call load_data() first.")
    
        # Use balanced sampler for BOTH training AND validation
        train_sampler = BalancedBatchSampler(
            self.train_dataset, 
            batch_size=batch_size, 
            samples_per_class=samples_per_class
        )
    
        val_sampler = BalancedBatchSampler(
            self.val_dataset, 
            batch_size=batch_size, 
            samples_per_class=samples_per_class
        )
    
        train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers
        )
    
        val_loader = DataLoader(
            self.val_dataset,
            batch_sampler=val_sampler,
            num_workers=num_workers
        )
    
        # Test loader can be regular (evaluation doesn't compute contrastive loss)
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    
        return train_loader, val_loader, test_loader


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
        print(f"  Samples per class: {self.samples_per_class}")
        print(f"  Batch size: {self.samples_per_class * self.num_classes}")
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


def test_data_loader():
    """Test data loading functionality with synthetic data"""
    print("Testing Data Loading Module")
    print("=" * 40)
    
    # Create synthetic data for testing
    print("Creating synthetic test data...")
    
    n_samples = 1000
    embedding_dim = 768
    
    # Generate random embeddings
    premise_embeddings = torch.randn(n_samples, embedding_dim)
    hypothesis_embeddings = torch.randn(n_samples, embedding_dim)
    labels = torch.randint(0, 3, (n_samples,))
    
    # Convert to expected format
    synthetic_data = {
        'premise_embeddings': premise_embeddings,
        'hypothesis_embeddings': hypothesis_embeddings,
        'labels': labels.tolist()
    }
    
    # Test lattice embedding generation
    print("\nTesting lattice embedding generation...")
    embedder = LatticeContainmentEmbedder()
    lattice_embeddings = embedder.generate_embeddings(
        premise_embeddings, hypothesis_embeddings, batch_size=200
    )
    
    print(f"Input shape: {premise_embeddings.shape}")
    print(f"Output shape: {lattice_embeddings.shape}")
    
    # Test dataset creation
    print("\nTesting dataset creation...")
    dataset = EntailmentDataset(lattice_embeddings, labels)
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    
    # Test data loader
    print("\nTesting data loader...")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    batch = next(iter(dataloader))
    print(f"Batch embeddings shape: {batch['embeddings'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    print(f"Batch labels: {batch['labels'][:10]}")

if __name__ == "__main__":
    test_data_loader()