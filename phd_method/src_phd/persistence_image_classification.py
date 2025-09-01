"""
Persistence Image Classification Pipeline
Pure topological (CNN) + SBERT baseline + Hybrid classification
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import argparse
from sklearn.metrics import accuracy_score, f1_score
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import gc
from torch.utils.data import Dataset, DataLoader
import time

class PersistenceImageDataset(Dataset):
    """Dataset for persistence images"""
    
    def __init__(self, persistence_images, labels, transform_to_image=True):
        self.persistence_images = persistence_images
        self.labels = labels
        self.transform_to_image = transform_to_image
        
    def __len__(self):
        return len(self.persistence_images)
    
    def __getitem__(self, idx):
        image = self.persistence_images[idx]
        label = self.labels[idx]
        
        if self.transform_to_image:
            # Reshape flattened 900D vector back to 30x30 image
            image = image.reshape(30, 30)
            # Add channel dimension for CNN: (1, 30, 30)
            image = torch.FloatTensor(image).unsqueeze(0)
        else:
            # Keep as flattened vector for standard NN
            image = torch.FloatTensor(image)
            
        return image, label

class HybridDataset(Dataset):
    """Dataset for hybrid persistence images + SBERT features"""
    
    def __init__(self, persistence_images, sbert_features, labels):
        self.persistence_images = persistence_images
        self.sbert_features = sbert_features
        self.labels = labels
        
    def __len__(self):
        return len(self.persistence_images)
    
    def __getitem__(self, idx):
        # Persistence image: reshape to 30x30 for CNN
        p_image = self.persistence_images[idx].reshape(30, 30)
        p_image = torch.FloatTensor(p_image).unsqueeze(0)  # (1, 30, 30)
        
        # SBERT features: keep as vector
        sbert_feat = torch.FloatTensor(self.sbert_features[idx])
        
        label = self.labels[idx]
        
        return p_image, sbert_feat, label

class PersistenceImageCNN(nn.Module):
    """CNN for persistence images (30x30)"""
    
    def __init__(self, num_classes=3):
        super().__init__()
        
        # CNN layers for 30x30 images
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 30x30 -> 30x30
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 30x30 -> 30x30
        self.pool = nn.MaxPool2d(2, 2)  # Halves dimensions
        
        # After conv1 + pool: 30 -> 15
        # After conv2 + pool: 15 -> 7 (with some rounding)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: (batch_size, 1, 30, 30)
        x = self.pool(F.relu(self.conv1(x)))  # -> (batch, 32, 15, 15)
        x = self.pool(F.relu(self.conv2(x)))  # -> (batch, 64, 7, 7)
        
        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class SBERTClassifier(nn.Module):
    """Standard NN for SBERT features"""
    
    def __init__(self, input_dim=3072, num_classes=3):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        
        return x

class HybridClassifier(nn.Module):
    """Hybrid CNN (persistence) + Standard NN (SBERT)"""
    
    def __init__(self, sbert_dim=3072, num_classes=3):
        super().__init__()
        
        # CNN branch for persistence images
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 30 -> 15
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 15 -> 7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Standard NN branch for SBERT features
        self.sbert_branch = nn.Sequential(
            nn.Linear(sbert_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256, 128),  # Concatenate both branches
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, persistence_images, sbert_features):
        # Process persistence images through CNN
        cnn_features = self.cnn_branch(persistence_images)
        
        # Process SBERT features through standard NN
        sbert_features = self.sbert_branch(sbert_features)
        
        # Concatenate and classify
        combined = torch.cat([cnn_features, sbert_features], dim=1)
        output = self.fusion(combined)
        
        return output

def memory_efficient_standardize(X):
    """Standardize features without creating copies"""
    print(f"Standardizing {X.shape} in-place...")
    
    # Compute mean and std in chunks to avoid memory issues
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    
    # Standardize in-place
    X -= mean
    X /= std
    
    return X, mean, std

def apply_standardization(X, mean, std):
    """Apply existing standardization parameters"""
    X_scaled = X.copy()
    X_scaled -= mean
    X_scaled /= std
    return X_scaled


def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, patience=25, device='cuda'):
    """Train any of the models with early stopping"""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"Training model on {device}...")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_data in train_loader:
            if len(batch_data) == 2:  # Pure topological or SBERT
                inputs, labels = batch_data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
            else:  # Hybrid
                p_images, sbert_feats, labels = batch_data
                p_images = p_images.to(device)
                sbert_feats = sbert_feats.to(device)
                labels = labels.to(device)
                outputs = model(p_images, sbert_feats)
            
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 2:  # Pure topological or SBERT
                    inputs, labels = batch_data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                else:  # Hybrid
                    p_images, sbert_feats, labels = batch_data
                    p_images = p_images.to(device)
                    sbert_feats = sbert_feats.to(device)
                    labels = labels.to(device)
                    outputs = model(p_images, sbert_feats)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = val_correct / val_total
        scheduler.step(val_loss)
        
        # Progress logging
        if epoch % 20 == 0 or epoch < 10:
            print(f"Epoch {epoch:3d}: Train Loss {train_loss/len(train_loader):.4f} "
                  f"(Acc {train_acc:.3f}), Val Loss {val_loss/len(val_loader):.4f} "
                  f"(Acc {val_acc:.3f}), Best Val {best_val_acc:.3f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.3f}")
    return model, best_val_acc


def load_precomputed_persistence_images(train_path, val_path):
    """Load precomputed persistence images"""
    print("Loading precomputed persistence images...")
    
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)
    
    X_train = train_data['persistence_images']
    y_train = train_data['labels']
    X_val = val_data['persistence_images']
    y_val = val_data['labels']
    
    # Convert string labels to indices
    label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    if isinstance(y_train[0], str):
        y_train = np.array([label_to_idx[label] for label in y_train])
        y_val = np.array([label_to_idx[label] for label in y_val])
    
    print(f"Persistence images: Train {X_train.shape}, Val {X_val.shape}")
    print(f"Image shape: 30x30 = {X_train.shape[1]} features per sample")
    
    return X_train, y_train, X_val, y_val

def compute_sbert_baseline_features(data_path):
    """Compute SBERT baseline features (same as before)"""
    print(f"Computing SBERT baseline features from {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    premise_tokens = data['premise_tokens']
    hypothesis_tokens = data['hypothesis_tokens']
    labels = data['labels']
    
    # Filter by token count and organize by class
    samples_by_class = {'entailment': [], 'neutral': [], 'contradiction': []}
    
    for i, label in enumerate(labels):
        if label in samples_by_class:
            total_tokens = premise_tokens[i].shape[0] + hypothesis_tokens[i].shape[0]
            if total_tokens >= 0:  # NO FILTERING
                samples_by_class[label].append({
                    'premise': premise_tokens[i],
                    'hypothesis': hypothesis_tokens[i]
                })
    
    # Extract SBERT features
    all_features = []
    all_labels = []
    label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    
    batch_size = 1000
    
    for class_name, samples in samples_by_class.items():
        print(f"Processing {class_name}: {len(samples)} samples")
        
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i+batch_size]
            batch_features = []
            
            for sample in batch_samples:
                premise_mean = torch.mean(sample['premise'], dim=0)
                hypothesis_mean = torch.mean(sample['hypothesis'], dim=0)
                
                # Create SBERT baseline features [premise, hypothesis, diff, product]
                features = torch.cat([
                    premise_mean,
                    hypothesis_mean,
                    premise_mean - hypothesis_mean,
                    premise_mean * hypothesis_mean
                ]).numpy()
                
                batch_features.append(features)
                all_labels.append(label_to_idx[class_name])
            
            all_features.extend(batch_features)
            
            if i + batch_size < len(samples):
                print(f"  Processed {i + batch_size}/{len(samples)} {class_name} samples")
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"SBERT features: {X.shape}, class distribution: {dict(zip(unique_classes, counts))}")
    
    return X, y

def align_datasets(X_persistence, y_persistence, X_sbert, y_sbert):
    """Align persistence and SBERT datasets by taking minimum per class"""
    print("Aligning persistence and SBERT datasets...")
    
    # Find minimum samples per class
    min_samples = []
    for class_idx in [0, 1, 2]:
        persistence_count = np.sum(y_persistence == class_idx)
        sbert_count = np.sum(y_sbert == class_idx)
        min_count = min(persistence_count, sbert_count)
        min_samples.append(min_count)
        print(f"Class {class_idx}: persistence={persistence_count}, sbert={sbert_count}, using={min_count}")
    
    # Sample aligned data
    persistence_indices = []
    sbert_indices = []
    
    for class_idx in [0, 1, 2]:
        p_class_indices = np.where(y_persistence == class_idx)[0]
        s_class_indices = np.where(y_sbert == class_idx)[0]
        
        sample_count = min_samples[class_idx]
        p_selected = np.random.choice(p_class_indices, sample_count, replace=False)
        s_selected = np.random.choice(s_class_indices, sample_count, replace=False)
        
        persistence_indices.extend(p_selected)
        sbert_indices.extend(s_selected)
    
    # Shuffle to mix classes
    combined_indices = list(zip(persistence_indices, sbert_indices))
    np.random.shuffle(combined_indices)
    persistence_indices, sbert_indices = zip(*combined_indices)
    
    X_persistence_aligned = X_persistence[list(persistence_indices)]
    X_sbert_aligned = X_sbert[list(sbert_indices)]
    y_aligned = y_persistence[list(persistence_indices)]
    
    print(f"Aligned datasets: {X_persistence_aligned.shape} persistence, {X_sbert_aligned.shape} SBERT")
    
    return X_persistence_aligned, X_sbert_aligned, y_aligned

def run_persistence_image_experiments(persistence_train_path, persistence_val_path, 
                                    sbert_train_path, sbert_val_path, device='cuda'):
    """Run all three experiments: Pure Topological, SBERT Baseline, Hybrid"""
    
    print("=" * 80)
    print("PERSISTENCE IMAGE CLASSIFICATION EXPERIMENTS")
    print("=" * 80)
    
    # 1. Load persistence images
    X_train_persistence, y_train_persistence, X_val_persistence, y_val_persistence = load_precomputed_persistence_images(
        persistence_train_path, persistence_val_path
    )
    
    # 2. Compute SBERT features
    X_train_sbert, y_train_sbert = compute_sbert_baseline_features(sbert_train_path)
    X_val_sbert, y_val_sbert = compute_sbert_baseline_features(sbert_val_path)
    
    # 3. Align datasets for fair comparison
    X_train_persistence_aligned, X_train_sbert_aligned, y_train_aligned = align_datasets(
        X_train_persistence, y_train_persistence, X_train_sbert, y_train_sbert
    )
    
    # For validation, just take minimum
    min_val = min(len(X_val_persistence), len(X_val_sbert))
    X_val_persistence_aligned = X_val_persistence[:min_val]
    X_val_sbert_aligned = X_val_sbert[:min_val]
    y_val_aligned = y_val_persistence[:min_val]
    
    results = {}
    
    # Experiment 1: Pure Topological (Persistence Images CNN)
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: PURE TOPOLOGICAL (PERSISTENCE IMAGES)")
    print("=" * 60)
    
    # Create datasets and loaders
    train_dataset_topo = PersistenceImageDataset(X_train_persistence_aligned, y_train_aligned, transform_to_image=True)
    val_dataset_topo = PersistenceImageDataset(X_val_persistence_aligned, y_val_aligned, transform_to_image=True)
    
    train_loader_topo = DataLoader(train_dataset_topo, batch_size=32, shuffle=True)
    val_loader_topo = DataLoader(val_dataset_topo, batch_size=64, shuffle=False)
    
    # Train CNN model
    model_topo = PersistenceImageCNN()
    model_topo, topo_acc = train_model(model_topo, train_loader_topo, val_loader_topo, device=device)
    
    results['topological'] = {
        'model': model_topo,
        'accuracy': topo_acc,
        'approach': 'Persistence Images CNN'
    }
    
    # Cleanup
    del train_dataset_topo, val_dataset_topo, train_loader_topo, val_loader_topo
    gc.collect()
    
    # Experiment 2: SBERT Baseline
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: SBERT BASELINE")
    print("=" * 60)
    
    # Memory-efficient standardization instead of sklearn
    print("Standardizing SBERT features...")
    X_train_sbert_scaled, mean, std = memory_efficient_standardize(X_train_sbert_aligned.astype(np.float32))
    X_val_sbert_scaled = apply_standardization(X_val_sbert_aligned.astype(np.float32), mean, std)
    
    # Force garbage collection after scaling
    gc.collect()
    
    # Create datasets and loaders
    train_dataset_sbert = PersistenceImageDataset(X_train_sbert_scaled, y_train_aligned, transform_to_image=False)
    val_dataset_sbert = PersistenceImageDataset(X_val_sbert_scaled, y_val_aligned, transform_to_image=False)
    
    train_loader_sbert = DataLoader(train_dataset_sbert, batch_size=32, shuffle=True)
    val_loader_sbert = DataLoader(val_dataset_sbert, batch_size=64, shuffle=False)
    
    # Train SBERT model
    model_sbert = SBERTClassifier()
    model_sbert, sbert_acc = train_model(model_sbert, train_loader_sbert, val_loader_sbert, device=device)
    
    results['sbert'] = {
        'model': model_sbert,
        'accuracy': sbert_acc,
        'approach': 'SBERT Standard NN',
        'mean': mean,
        'std': std
    }
    
    # Cleanup
    del train_dataset_sbert, val_dataset_sbert, train_loader_sbert, val_loader_sbert
    gc.collect()
    
    # Experiment 3: Hybrid (Persistence Images + SBERT)
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: HYBRID (PERSISTENCE + SBERT)")
    print("=" * 60)
    
    # Create hybrid datasets
    train_dataset_hybrid = HybridDataset(X_train_persistence_aligned, X_train_sbert_scaled, y_train_aligned)
    val_dataset_hybrid = HybridDataset(X_val_persistence_aligned, X_val_sbert_scaled, y_val_aligned)
    
    train_loader_hybrid = DataLoader(train_dataset_hybrid, batch_size=32, shuffle=True)
    val_loader_hybrid = DataLoader(val_dataset_hybrid, batch_size=64, shuffle=False)
    
    # Train hybrid model
    model_hybrid = HybridClassifier()
    model_hybrid, hybrid_acc = train_model(model_hybrid, train_loader_hybrid, val_loader_hybrid, device=device)
    
    results['hybrid'] = {
        'model': model_hybrid,
        'accuracy': hybrid_acc,
        'approach': 'Persistence CNN + SBERT NN',
        'mean': mean,
        'std': std
    }
    
    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL RESULTS COMPARISON")
    print("=" * 80)
    print(f"Pure Topological (Persistence CNN):  {topo_acc:.3f}")
    print(f"SBERT Baseline (Standard NN):        {sbert_acc:.3f}")
    print(f"Hybrid (Persistence + SBERT):        {hybrid_acc:.3f}")
    
    improvement_vs_sbert = hybrid_acc - sbert_acc
    improvement_vs_topo = hybrid_acc - topo_acc
    topo_vs_sbert = topo_acc - sbert_acc
    
    print(f"\nPerformance differences:")
    print(f"  Topological vs SBERT: {topo_vs_sbert:+.3f}")
    print(f"  Hybrid vs SBERT: {improvement_vs_sbert:+.3f}")
    print(f"  Hybrid vs Topological: {improvement_vs_topo:+.3f}")
    
    if topo_acc > sbert_acc:
        print(f"\n🎉 Topological approach beats SBERT baseline by {topo_vs_sbert:.3f}!")
    else:
        print(f"\n📊 SBERT still ahead by {-topo_vs_sbert:.3f}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Persistence Image Classification Pipeline')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Configure paths
    persistence_train_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_snli_train_persistence_images.pkl"
    persistence_val_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_snli_val_persistence_images.pkl"
    sbert_train_path = "/vol/bitbucket/ahb24/tda_entailment_new/snli_train_sbert_tokens.pkl"
    sbert_val_path = "/vol/bitbucket/ahb24/tda_entailment_new/snli_val_sbert_tokens.pkl"
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Run experiments
    results = run_persistence_image_experiments(
        persistence_train_path, persistence_val_path,
        sbert_train_path, sbert_val_path,
        device=device
    )
    
    print("\nAll experiments completed!")