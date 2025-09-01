# FILE: data_loader.py (FIXED VERSION - Compatible with your existing format)
"""
Data loader for InfoNCE + Order Embeddings training
Compatible with your existing SNLI data format
"""

import torch
from torch.utils.data import DataLoader, Dataset
import os


class SNLIDataset(Dataset):
    """
    SNLI dataset for premise-hypothesis pairs
    Compatible with your existing data format:
    {
        'premise_embeddings': torch.Tensor [N, 768],
        'hypothesis_embeddings': torch.Tensor [N, 768],
        'labels': list [N]
    }
    """
    
    def __init__(self, data_path, embedding_type='concat'):
        print(f"Loading SNLI dataset from {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data using your existing format
        self.data = torch.load(data_path, weights_only=False)
        print(f"  Loaded data with keys: {list(self.data.keys())}")
        
        # Extract components using your exact format
        self.premise_embeddings = self.data['premise_embeddings']
        self.hypothesis_embeddings = self.data['hypothesis_embeddings']
        
        # Handle labels - convert strings to integers if needed
        labels = self.data['labels']
        if isinstance(labels, list) and len(labels) > 0 and isinstance(labels[0], str):
            # Convert string labels to integers
            label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
            self.labels = torch.tensor([label_map[label] for label in labels], dtype=torch.long)
            print(f"  Converted string labels to integers using: {label_map}")
        elif isinstance(labels, list):
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = labels.long()
        
        self.embedding_type = embedding_type
        
        print(f"  Dataset loaded successfully:")
        print(f"    Total samples: {len(self)}")
        print(f"    Premise embeddings shape: {self.premise_embeddings.shape}")
        print(f"    Hypothesis embeddings shape: {self.hypothesis_embeddings.shape}")
        print(f"    Labels shape: {self.labels.shape}")
        
        # Check label distribution
        unique_labels, counts = torch.unique(self.labels, return_counts=True)
        label_names = ['entailment', 'neutral', 'contradiction']
        for label, count in zip(unique_labels, counts):
            if label < len(label_names):
                print(f"    {label_names[label]}: {count} samples")
            else:
                print(f"    Unknown label {label}: {count} samples")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'premise_embedding': self.premise_embeddings[idx],
            'hypothesis_embedding': self.hypothesis_embeddings[idx],
            'label': self.labels[idx]
        }


def create_data_loaders(config):
    """
    Create train/val/test data loaders
    """
    try:
        print("Creating data loaders...")
        
        train_dataset = SNLIDataset(
            config['data']['train_path'],
            config['data']['embedding_type']
        )
        
        val_dataset = SNLIDataset(
            config['data']['val_path'],
            config['data']['embedding_type']
        )
        
        test_dataset = SNLIDataset(
            config['data']['test_path'],
            config['data']['embedding_type']
        )
        
        print("Creating DataLoader objects...")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        print("✅ Data loaders created successfully!")
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        print("Debugging data structure...")
        
        # Debug first available file
        for path_key in ['train_path', 'val_path', 'test_path']:
            path = config['data'][path_key]
            if os.path.exists(path):
                print(f"Inspecting {path}:")
                try:
                    data = torch.load(path, weights_only=False)
                    print(f"  Type: {type(data)}")
                    if isinstance(data, dict):
                        print(f"  Keys: {list(data.keys())}")
                        for key, value in data.items():
                            if hasattr(value, 'shape'):
                                print(f"  {key} shape: {value.shape}")
                            elif isinstance(value, (list, tuple)):
                                print(f"  {key} length: {len(value)} (type: {type(value[0]) if len(value) > 0 else 'empty'})")
                            else:
                                print(f"  {key} type: {type(value)}")
                        
                        # Check a few label samples
                        if 'labels' in data:
                            labels = data['labels']
                            if isinstance(labels, list) and len(labels) > 0:
                                print(f"  First 5 labels: {labels[:5]}")
                                print(f"  Label types: {[type(l) for l in labels[:3]]}")
                    break
                except Exception as inspect_e:
                    print(f"  Error inspecting: {inspect_e}")
            else:
                print(f"File not found: {path}")
        
        raise


# Test function to debug data loading
def test_data_loading():
    """
    Test data loading with a simple config
    """
    test_config = {
        'data': {
            'train_path': 'data/processed/snli_full_standard_SBERT.pt',
            'val_path': 'data/processed/snli_full_standard_SBERT_validation.pt',
            'test_path': 'data/processed/snli_full_standard_SBERT_test.pt',
            'embedding_type': 'concat',
            'batch_size': 32
        }
    }
    
    try:
        train_loader, val_loader, test_loader = create_data_loaders(test_config)
        
        # Test a batch
        print("\nTesting batch loading...")
        for batch in train_loader:
            print(f"Batch premise shape: {batch['premise_embedding'].shape}")
            print(f"Batch hypothesis shape: {batch['hypothesis_embedding'].shape}")
            print(f"Batch labels shape: {batch['label'].shape}")
            print(f"Label distribution in batch: {torch.unique(batch['label'], return_counts=True)}")
            break
        
        print("✅ Data loading test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False


if __name__ == "__main__":
    test_data_loading()