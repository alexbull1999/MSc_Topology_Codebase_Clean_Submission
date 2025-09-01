
"""
Train on SNLI, Evaluate on ChaosNLI
Trains persistence image models on SNLI, then evaluates uncertainty quantification on ChaosNLI
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

class EnhancedPersistenceImageCNN(nn.Module):
    """Enhanced CNN with deeper architecture and batch normalization"""
    
    def __init__(self, num_classes=3):
        super().__init__()
        
        # Deeper CNN with batch normalization
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Fixed output size
        
        # More sophisticated fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # 30x30 -> 15x15
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)  # 15x15 -> 7x7
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.adaptive_pool(x)  # -> 4x4
        
        # Fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        
        return x

class AttentionPersistenceImageCNN(nn.Module):
    """CNN with spatial attention for persistence images"""
    
    def __init__(self, num_classes=3):
        super().__init__()
        
        # CNN backbone
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Spatial attention mechanism
        self.attention_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.attention_sigmoid = nn.Sigmoid()
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 30x30 -> 15x15
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 15x15 -> 7x7
        x = F.relu(self.bn3(self.conv3(x)))  # 7x7
        
        # Spatial attention
        attention_weights = self.attention_sigmoid(self.attention_conv(x))
        x = x * attention_weights  # Apply attention
        
        # Global pooling and classification
        x = self.global_pool(x)  # -> (batch, 128, 1, 1)
        x = x.view(-1, 128)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class MultiScalePersistenceImageCNN(nn.Module):
    """Multi-scale CNN that processes different scales of persistence images"""
    
    def __init__(self, num_classes=3):
        super().__init__()
        
        # Branch 1: Fine-grained features (3x3 kernels)
        self.fine_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.fine_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Branch 2: Medium-scale features (5x5 kernels)
        self.medium_conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.medium_conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        
        # Branch 3: Large-scale features (7x7 kernels)
        self.large_conv1 = nn.Conv2d(1, 32, kernel_size=7, padding=3)
        self.large_conv2 = nn.Conv2d(32, 64, kernel_size=7, padding=3)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(64)
        
        # Combine all scales (64 features from each of 3 branches)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 3 * 4 * 4, 256)  # 3 branches
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        # Process at different scales
        fine = self.pool(F.relu(self.fine_conv1(x)))
        fine = self.pool(F.relu(self.bn(self.fine_conv2(fine))))
        fine = self.adaptive_pool(fine)
        
        medium = self.pool(F.relu(self.medium_conv1(x)))
        medium = self.pool(F.relu(self.bn(self.medium_conv2(medium))))
        medium = self.adaptive_pool(medium)
        
        large = self.pool(F.relu(self.large_conv1(x)))
        large = self.pool(F.relu(self.bn(self.large_conv2(large))))
        large = self.adaptive_pool(large)
        
        # Concatenate all scales
        x = torch.cat([fine, medium, large], dim=1)  # Combine along channel dimension
        x = x.view(-1, 64 * 3 * 4 * 4)
        
        # Classification
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

def train_model(model, train_loader, val_loader, epochs=200, lr=1e-3, patience=25, device='cuda'):
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

def load_snli_mnli_persistence_images(snli_train_path, snli_val_path, 
                                     mnli_train_path, mnli_val_path):
    """Load SNLI + MNLI persistence images for training"""
    print("Loading SNLI + MNLI persistence images...")
    
    # Load SNLI data
    with open(snli_train_path, 'rb') as f:
        snli_train_data = pickle.load(f)
    with open(snli_val_path, 'rb') as f:
        snli_val_data = pickle.load(f)
    
    # Load MNLI data
    with open(mnli_train_path, 'rb') as f:
        mnli_train_data = pickle.load(f)
    with open(mnli_val_path, 'rb') as f:
        mnli_val_data = pickle.load(f)
    
    # Combine training data
    X_train = np.concatenate([
        snli_train_data['persistence_images'],
        mnli_train_data['persistence_images']
    ], axis=0)
    
    y_train = np.concatenate([
        snli_train_data['labels'],
        mnli_train_data['labels']
    ])
    
    # Combine validation data
    X_val = np.concatenate([
        snli_val_data['persistence_images'],
        mnli_val_data['persistence_images']
    ], axis=0)
    
    y_val = np.concatenate([
        snli_val_data['labels'],
        mnli_val_data['labels']
    ])
    
    # Convert string labels to indices
    label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    if isinstance(y_train[0], str):
        y_train = np.array([label_to_idx[label] for label in y_train])
        y_val = np.array([label_to_idx[label] for label in y_val])
    
    print(f"Combined SNLI+MNLI persistence images:")
    print(f"  Train: {X_train.shape} (SNLI: {snli_train_data['persistence_images'].shape}, MNLI: {mnli_train_data['persistence_images'].shape})")
    print(f"  Val: {X_val.shape} (SNLI: {snli_val_data['persistence_images'].shape}, MNLI: {mnli_val_data['persistence_images'].shape})")
    
    return X_train, y_train, X_val, y_val


def load_chaosnli_persistence_images(data_path):
    """Load ChaosNLI persistence images for evaluation"""
    print("Loading ChaosNLI persistence images...")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X_chaos = data['persistence_images']
    majority_labels = data['majority_labels']
    label_distributions = data['label_distributions']
    entropies = data['entropies']
    
    # Convert string labels to indices
    label_to_idx = {'e': 0, 'n': 1, 'c': 2}
    if isinstance(majority_labels[0], str):
        y_chaos = np.array([label_to_idx[label] for label in majority_labels])
    else:
        y_chaos = majority_labels
    
    print(f"ChaosNLI persistence images: {X_chaos.shape}")
    print(f"Human label distributions: {label_distributions.shape}")
    print(f"Entropy range: {np.min(entropies):.3f} - {np.max(entropies):.3f}")
    
    return X_chaos, y_chaos, label_distributions, entropies

def compute_snli_mnli_sbert_baseline_features(snli_data_path, mnli_data_path):
    """Compute SBERT baseline features from SNLI + MNLI"""
    print(f"Computing SBERT baseline features from SNLI + MNLI...")
    
    # Load SNLI data
    with open(snli_data_path, 'rb') as f:
        snli_data = pickle.load(f)
    
    # Load MNLI data  
    with open(mnli_data_path, 'rb') as f:
        mnli_data = pickle.load(f)
    
    # Combine data
    premise_tokens = snli_data['premise_tokens'] + mnli_data['premise_tokens']
    hypothesis_tokens = snli_data['hypothesis_tokens'] + mnli_data['hypothesis_tokens']
    labels = snli_data['labels'] + mnli_data['labels']
    
    print(f"Combined data: SNLI {len(snli_data['labels'])} + MNLI {len(mnli_data['labels'])} = {len(labels)} total samples")
    
    # Rest of the function remains the same...
    samples_by_class = {'entailment': [], 'neutral': [], 'contradiction': []}
    
    for i, label in enumerate(labels):
        if label in samples_by_class:
            total_tokens = premise_tokens[i].shape[0] + hypothesis_tokens[i].shape[0]
            if total_tokens >= 0:  # NO FILTERING
                samples_by_class[label].append({
                    'premise': premise_tokens[i],
                    'hypothesis': hypothesis_tokens[i]
                })
    
    # Extract SBERT features (same as before)
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
    print(f"Combined SBERT features: {X.shape}, class distribution: {dict(zip(unique_classes, counts))}")
    
    return X, y

def compute_chaosnli_sbert_features(data_path):
    """Compute SBERT features for ChaosNLI dataset"""
    print(f"Computing ChaosNLI SBERT baseline features from {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    premise_tokens = data['premise_tokens']
    hypothesis_tokens = data['hypothesis_tokens']
    
    all_features = []
    
    print(f"Processing {len(premise_tokens)} ChaosNLI samples...")
    
    batch_size = 1000
    for i in range(0, len(premise_tokens), batch_size):
        batch_end = min(i + batch_size, len(premise_tokens))
        batch_features = []
        
        for j in range(i, batch_end):
            premise_mean = torch.mean(premise_tokens[j], dim=0)
            hypothesis_mean = torch.mean(hypothesis_tokens[j], dim=0)
            
            # Create SBERT baseline features [premise, hypothesis, diff, product]
            features = torch.cat([
                premise_mean,
                hypothesis_mean,
                premise_mean - hypothesis_mean,
                premise_mean * hypothesis_mean
            ]).numpy()
            
            batch_features.append(features)
        
        all_features.extend(batch_features)
        
        if batch_end < len(premise_tokens):
            print(f"  Processed {batch_end}/{len(premise_tokens)} samples")
    
    X_chaos_sbert = np.array(all_features)
    print(f"ChaosNLI SBERT features: {X_chaos_sbert.shape}")
    
    return X_chaos_sbert

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

def evaluate_chaosnli_uncertainty(model, X_chaos, label_distributions, entropies, 
                                mean=None, std=None, model_type='persistence', device='cuda'):
    """Evaluate model on ChaosNLI uncertainty quantification"""
    print(f"\nEvaluating {model_type} model on ChaosNLI uncertainty quantification...")
    
    model = model.to(device)
    model.eval()
    
    # Prepare features for model
    if model_type == 'sbert' and mean is not None and std is not None:
        X_chaos_scaled = apply_standardization(X_chaos.astype(np.float32), mean, std)
        X_chaos_tensor = torch.FloatTensor(X_chaos_scaled)
    elif model_type == 'persistence':
        # Reshape for CNN: (batch, 1, 30, 30)
        X_chaos_images = X_chaos.reshape(-1, 1, 30, 30)
        X_chaos_tensor = torch.FloatTensor(X_chaos_images)
    else:
        X_chaos_tensor = torch.FloatTensor(X_chaos)
    
    # Get model probabilities
    all_probs = []
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(X_chaos_tensor), batch_size):
            batch = X_chaos_tensor[i:i+batch_size].to(device)
            if model_type == 'hybrid':
                # For hybrid model, need both persistence and SBERT features
                # This would need to be handled differently
                pass
            else:
                logits = model(batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
    
    model_probs = np.vstack(all_probs)
    
    # Compute uncertainty metrics
    jsd_scores = []
    kl_scores = []
    
    for i in range(len(model_probs)):
        model_dist = model_probs[i] / model_probs[i].sum()
        human_dist = label_distributions[i] / label_distributions[i].sum()
        
        # Add small epsilon for numerical stability
        eps = 1e-10
        model_dist = np.clip(model_dist + eps, eps, 1.0)
        human_dist = np.clip(human_dist + eps, eps, 1.0)
        model_dist = model_dist / model_dist.sum()
        human_dist = human_dist / human_dist.sum()
        
        # Jensen-Shannon Distance
        jsd = jensenshannon(model_dist, human_dist)
        jsd_scores.append(jsd)
        
        # KL Divergence
        kl = entropy(human_dist, model_dist)
        kl_scores.append(kl)
    
    avg_jsd = np.mean(jsd_scores)
    avg_kl = np.mean(kl_scores)
    
    # Traditional accuracy
    model_preds = np.argmax(model_probs, axis=1)
    human_preds = np.argmax(label_distributions, axis=1)
    traditional_acc = np.mean(model_preds == human_preds)
    
    # Analyze by entropy levels
    high_ent_mask = entropies < 0.5
    med_ent_mask = (entropies >= 0.5) & (entropies < 1.0)
    low_ent_mask = entropies >= 1.0
    
    print(f"ChaosNLI Results for {model_type}:")
    print(f"  Overall JSD: {avg_jsd:.4f}")
    print(f"  Overall KL:  {avg_kl:.4f}")
    print(f"  Traditional Accuracy: {traditional_acc:.4f}")
    
    if np.any(high_ent_mask):
        high_jsd = np.mean([jsd_scores[i] for i in range(len(jsd_scores)) if high_ent_mask[i]])
        print(f"  High agreement (entropy < 0.5): JSD={high_jsd:.4f} ({np.sum(high_ent_mask)} samples)")
    
    if np.any(med_ent_mask):
        med_jsd = np.mean([jsd_scores[i] for i in range(len(jsd_scores)) if med_ent_mask[i]])
        print(f"  Medium agreement (0.5 ≤ entropy < 1.0): JSD={med_jsd:.4f} ({np.sum(med_ent_mask)} samples)")
    
    if np.any(low_ent_mask):
        low_jsd = np.mean([jsd_scores[i] for i in range(len(jsd_scores)) if low_ent_mask[i]])
        print(f"  Low agreement (entropy ≥ 1.0): JSD={low_jsd:.4f} ({np.sum(low_ent_mask)} samples)")
    
    # Compare with baselines
    random_jsd = 0.32
    random_kl = 0.54
    roberta_jsd = 0.22
    bart_kl = 0.47
    
    print(f"  Beats random: JSD={'YES' if avg_jsd < random_jsd else 'NO'}, KL={'YES' if avg_kl < random_kl else 'NO'}")
    print(f"  vs RoBERTa JSD: {roberta_jsd:.4f}, vs BART KL: {bart_kl:.4f}")
    
    return {
        'jsd': avg_jsd,
        'kl': avg_kl,
        'accuracy': traditional_acc,
        'beats_random_jsd': avg_jsd < random_jsd,
        'beats_random_kl': avg_kl < random_kl,
        'beats_roberta_jsd': avg_jsd < roberta_jsd,
        'beats_bart_kl': avg_kl < bart_kl
    }


def run_complete_architecture_comparison(snli_persistence_train_path, snli_persistence_val_path,
                                       mnli_persistence_train_path, mnli_persistence_val_path,
                                       snli_sbert_train_path, snli_sbert_val_path,
                                       mnli_sbert_train_path, mnli_sbert_val_path,
                                       chaosnli_snli_persistence_path, chaosnli_snli_sbert_path,
                                       chaosnli_mnli_persistence_path, chaosnli_mnli_sbert_path, device='cuda'):
    """Compare all architectures: CNN variants + SBERT + Hybrid"""
    
    print("=" * 80)
    print("COMPLETE ARCHITECTURE COMPARISON")
    print("=" * 80)
    
    # Load SNLI+MNLI training data
    X_train_persistence, y_train_persistence, X_val_persistence, y_val_persistence = load_snli_mnli_persistence_images(
        snli_persistence_train_path, snli_persistence_val_path,
        mnli_persistence_train_path, mnli_persistence_val_path
    )
    
    # Load both ChaosNLI evaluation datasets
    X_chaos_snli_persistence, y_chaos_snli_persistence, chaos_snli_label_distributions, chaos_snli_entropies = load_chaosnli_persistence_images(
        chaosnli_snli_persistence_path
    )
    
    X_chaos_mnli_persistence, y_chaos_mnli_persistence, chaos_mnli_label_distributions, chaos_mnli_entropies = load_chaosnli_persistence_images(
        chaosnli_mnli_persistence_path
    )

    # Create datasets for persistence models
    train_dataset_topo = PersistenceImageDataset(X_train_persistence, y_train_persistence, transform_to_image=True)
    val_dataset_topo = PersistenceImageDataset(X_val_persistence, y_val_persistence, transform_to_image=True)
    
    train_loader_topo = DataLoader(train_dataset_topo, batch_size=128, shuffle=True)
    val_loader_topo = DataLoader(val_dataset_topo, batch_size=256, shuffle=False)
    
    # Define all models to compare
    models_to_test = {
        'Original CNN': PersistenceImageCNN(),
        'Enhanced CNN': EnhancedPersistenceImageCNN(),
        'Attention CNN': AttentionPersistenceImageCNN(),
        'Multi-Scale CNN': MultiScalePersistenceImageCNN(),
    }
    
    results = {}
    
    # Train and evaluate each persistence CNN architecture
    for model_name, model in models_to_test.items():
        print(f"\n" + "=" * 60)
        print(f"TRAINING {model_name.upper()}")
        print("=" * 60)
        
        trained_model, val_acc = train_model(model, train_loader_topo, val_loader_topo, device=device)
        
        print(f"{model_name} SNLI+MNLI validation accuracy: {val_acc:.3f}")
        
        print("Evaluating on SNLI set of ChaosNLI")
        # Evaluate on ChaosNLI-SNLI uncertainty quantification
        chaos_snli_results = evaluate_chaosnli_uncertainty(
            trained_model, X_chaos_snli_persistence, chaos_snli_label_distributions, chaos_snli_entropies,
            model_type='persistence', device=device
        )


        print("Evaluating on MNLI set of ChaosNLI")
        # Evaluate on ChaosNLI-MNLI uncertainty quantification
        chaos_mnli_results = evaluate_chaosnli_uncertainty(
            trained_model, X_chaos_mnli_persistence, chaos_mnli_label_distributions, chaos_mnli_entropies,
            model_type='persistence', device=device
        )
        
        results[model_name] = {
            'model': trained_model,
            'snli_mnli_accuracy': val_acc,
            'chaosnli_snli_jsd': chaos_snli_results['jsd'],
            'chaosnli_snli_kl': chaos_snli_results['kl'],
            'chaosnli_mnli_jsd': chaos_mnli_results['jsd'],
            'chaosnli_mnli_kl': chaos_mnli_results['kl'],
            'beats_roberta_snli': chaos_snli_results['beats_roberta_jsd'],
            'beats_roberta_mnli': chaos_mnli_results['beats_roberta_jsd'],
            'beats_bart_snli': chaos_snli_results['beats_bart_kl'],
            'beats_bart_mnli': chaos_mnli_results['beats_bart_kl'],
            'model_type': 'persistence'
        }
        
        # Clear GPU memory
        del trained_model
        gc.collect()
        torch.cuda.empty_cache()
    
    # Print comprehensive comparison
    print("\n" + "=" * 80)
    print("COMPLETE ARCHITECTURE COMPARISON RESULTS")
    print("=" * 80)
    
    print("📊 SNLI+MNLI Validation Accuracies:")
    snli_sorted = sorted(results.items(), key=lambda x: x[1]['snli_mnli_accuracy'], reverse=True)
    for i, (model_name, result) in enumerate(snli_sorted):
        print(f"  {i+1}. {model_name:<20}: {result['snli_mnli_accuracy']:.3f}")
    
    print("\n🎯 ChaosNLI-SNLI Uncertainty Quantification (JSD):")
    snli_jsd_sorted = sorted(results.items(), key=lambda x: x[1]['chaosnli_snli_jsd'])
    for i, (model_name, result) in enumerate(snli_jsd_sorted):
        jsd_status = "✅" if result['beats_roberta_snli'] else "❌" 
        print(f"  {i+1}. {model_name:<20}: {result['chaosnli_snli_jsd']:.4f} {jsd_status}")
    
    print("\n🎯 ChaosNLI-MNLI Uncertainty Quantification (JSD):")
    mnli_jsd_sorted = sorted(results.items(), key=lambda x: x[1]['chaosnli_mnli_jsd'])
    for i, (model_name, result) in enumerate(mnli_jsd_sorted):
        jsd_status = "✅" if result['beats_roberta_mnli'] else "❌" 
        print(f"  {i+1}. {model_name:<20}: {result['chaosnli_mnli_jsd']:.4f} {jsd_status}")
    
    # Determine overall winners
    best_accuracy_model = snli_sorted[0][0]
    best_snli_jsd_model = snli_jsd_sorted[0][0]
    best_mnli_jsd_model = mnli_jsd_sorted[0][0]
    
    print(f"\n🏆 SUMMARY:")
    print(f"  Best SNLI+MNLI accuracy: {best_accuracy_model}")
    print(f"  Best ChaosNLI-SNLI JSD: {best_snli_jsd_model}")
    print(f"  Best ChaosNLI-MNLI JSD: {best_mnli_jsd_model}")
    
    # Check if any model beats published baselines
    beating_roberta_snli = [name for name, result in results.items() if result['beats_roberta_snli']]
    beating_roberta_mnli = [name for name, result in results.items() if result['beats_roberta_mnli']]
    
    if beating_roberta_snli:
        print(f"\n🎉 Models beating RoBERTa JSD on SNLI: {', '.join(beating_roberta_snli)}")
    if beating_roberta_mnli:
        print(f"🎉 Models beating RoBERTa JSD on MNLI: {', '.join(beating_roberta_mnli)}")
    
    if not beating_roberta_snli and not beating_roberta_mnli:
        print(f"\n📈 None beat published baselines yet - but larger training data should help!")
    
    return results

def run_snli_train_chaosnli_eval(snli_persistence_train_path, snli_persistence_val_path,
                                       mnli_persistence_train_path, mnli_persistence_val_path,
                                       snli_sbert_train_path, snli_sbert_val_path,
                                       mnli_sbert_train_path, mnli_sbert_val_path,
                                       chaosnli_snli_persistence_path, chaosnli_snli_sbert_path,
                                       chaosnli_mnli_persistence_path, chaosnli_mnli_sbert_path, device='cuda'):
    """Train on SNLI, evaluate on ChaosNLI"""
    
    print("=" * 80)
    print("TRAIN ON SNLI, EVALUATE ON CHAOSNLI")
    print("=" * 80)
    
    # Load SNLI+MNLI training data
    X_train_persistence, y_train_persistence, X_val_persistence, y_val_persistence = load_snli_mnli_persistence_images(
        snli_persistence_train_path, snli_persistence_val_path,
        mnli_persistence_train_path, mnli_persistence_val_path
    )
    
    X_train_sbert, y_train_sbert = compute_snli_mnli_sbert_baseline_features(snli_sbert_train_path, mnli_sbert_train_path)
    X_val_sbert, y_val_sbert = compute_snli_mnli_sbert_baseline_features(snli_sbert_val_path, mnli_sbert_val_path)
    
    # 2. Align SNLI datasets
    X_train_persistence_aligned, X_train_sbert_aligned, y_train_aligned = align_datasets(
        X_train_persistence, y_train_persistence, X_train_sbert, y_train_sbert
    )
    
    min_val = min(len(X_val_persistence), len(X_val_sbert))
    X_val_persistence_aligned = X_val_persistence[:min_val]
    X_val_sbert_aligned = X_val_sbert[:min_val]
    y_val_aligned = y_val_persistence[:min_val]
    
    # 3. Load ChaosNLI evaluation data
    X_chaos_snli_persistence, y_chaos_snli_persistence, chaos_snli_label_distributions, chaos_snli_entropies = load_chaosnli_persistence_images(
        chaosnli_snli_persistence_path
    )
    
    X_chaos_mnli_persistence, y_chaos_mnli_persistence, chaos_mnli_label_distributions, chaos_mnli_entropies = load_chaosnli_persistence_images(
        chaosnli_mnli_persistence_path
    )

    X_chaos_snli_sbert = compute_chaosnli_sbert_features(chaosnli_snli_sbert_path)

    X_chaos_mnli_sbert = compute_chaosnli_sbert_features(chaosnli_mnli_sbert_path)
    
    # 4. Train Persistence Image CNN on SNLI
    print("\n" + "=" * 60)
    print("TRAINING PERSISTENCE IMAGE CNN ON SNLI")
    print("=" * 60)
    
    train_dataset_topo = PersistenceImageDataset(X_train_persistence_aligned, y_train_aligned, transform_to_image=True)
    val_dataset_topo = PersistenceImageDataset(X_val_persistence_aligned, y_val_aligned, transform_to_image=True)
    #was 32/64 for original run
    train_loader_topo = DataLoader(train_dataset_topo, batch_size=128, shuffle=True)
    val_loader_topo = DataLoader(val_dataset_topo, batch_size=256, shuffle=False)
    
    model_topo = PersistenceImageCNN()
    model_topo, topo_acc = train_model(model_topo, train_loader_topo, val_loader_topo, device=device)
    
    print(f"Persistence CNN SNLI validation accuracy: {topo_acc:.3f}")
    
    # 5. Train SBERT Classifier on SNLI
    print("\n" + "=" * 60)
    print("TRAINING SBERT CLASSIFIER ON SNLI")
    print("=" * 60)
    
    # Memory-efficient standardization
    X_train_sbert_scaled, mean, std = memory_efficient_standardize(X_train_sbert_aligned.astype(np.float32))
    X_val_sbert_scaled = apply_standardization(X_val_sbert_aligned.astype(np.float32), mean, std)
    gc.collect()
    
    train_dataset_sbert = PersistenceImageDataset(X_train_sbert_scaled, y_train_aligned, transform_to_image=False)
    val_dataset_sbert = PersistenceImageDataset(X_val_sbert_scaled, y_val_aligned, transform_to_image=False)
    
    train_loader_sbert = DataLoader(train_dataset_sbert, batch_size=128, shuffle=True)
    val_loader_sbert = DataLoader(val_dataset_sbert, batch_size=256, shuffle=False)
    
    model_sbert = SBERTClassifier()
    model_sbert, sbert_acc = train_model(model_sbert, train_loader_sbert, val_loader_sbert, device=device)
    
    print(f"SBERT SNLI validation accuracy: {sbert_acc:.3f}")
    
    # Cleanup training data
    del train_dataset_topo, val_dataset_topo, train_loader_topo, val_loader_topo
    del train_dataset_sbert, val_dataset_sbert, train_loader_sbert, val_loader_sbert
    gc.collect()
    
    # 6. Evaluate both models on ChaosNLI
    print("\n" + "=" * 80)
    print("EVALUATING ON CHAOSNLI SNLI DATASET UNCERTAINTY QUANTIFICATION")
    print("=" * 80)
    
    # Evaluate Persistence CNN on ChaosNLI
    persistence_results = evaluate_chaosnli_uncertainty(
        model_topo, X_chaos_snli_persistence, chaos_snli_label_distributions, chaos_snli_entropies,
        model_type='persistence', device=device
    )
    
    # Evaluate SBERT on ChaosNLI
    X_chaos_snli_sbert_scaled = apply_standardization(X_chaos_snli_sbert.astype(np.float32), mean, std)
    sbert_results = evaluate_chaosnli_uncertainty(
        model_sbert, X_chaos_snli_sbert_scaled, chaos_snli_label_distributions, chaos_snli_entropies,
        mean=mean, std=std, model_type='sbert', device=device
    )
    
    print("\n" + "=" * 80)
    print("EVALUATING ON CHAOSNLI MNLI DATASET UNCERTAINTY QUANTIFICATION")
    print("=" * 80)

    # Evaluate Persistence CNN on ChaosNLI
    persistence_results = evaluate_chaosnli_uncertainty(
        model_topo, X_chaos_mnli_persistence, chaos_mnli_label_distributions, chaos_mnli_entropies,
        model_type='persistence', device=device
    )
    
    # Evaluate SBERT on ChaosNLI
    X_chaos_mnli_sbert_scaled = apply_standardization(X_chaos_mnli_sbert.astype(np.float32), mean, std)
    sbert_results = evaluate_chaosnli_uncertainty(
        model_sbert, X_chaos_mnli_sbert_scaled, chaos_mnli_label_distributions, chaos_mnli_entropies,
        mean=mean, std=std, model_type='sbert', device=device
    )
    

def run_hybrid_snli_train_chaosnli_eval(snli_persistence_train_path, snli_persistence_val_path,
                                       mnli_persistence_train_path, mnli_persistence_val_path,
                                       snli_sbert_train_path, snli_sbert_val_path,
                                       mnli_sbert_train_path, mnli_sbert_val_path,
                                       chaosnli_snli_persistence_path, chaosnli_snli_sbert_path,
                                       chaosnli_mnli_persistence_path, chaosnli_mnli_sbert_path, device='cuda'):
    """Train hybrid model on SNLI, evaluate on ChaosNLI"""
    
    print("\n" + "=" * 80)
    print("HYBRID MODEL: TRAIN ON SNLI, EVALUATE ON CHAOSNLI")
    print("=" * 80)
    
    # Load SNLI data
    X_train_persistence, y_train_persistence, X_val_persistence, y_val_persistence = load_snli_mnli_persistence_images(
        snli_persistence_train_path, snli_persistence_val_path,
        mnli_persistence_train_path, mnli_persistence_val_path
    )
    
    X_train_sbert, y_train_sbert = compute_snli_mnli_sbert_baseline_features(snli_sbert_train_path, mnli_sbert_train_path)
    X_val_sbert, y_val_sbert = compute_snli_mnli_sbert_baseline_features(snli_sbert_val_path, mnli_sbert_val_path)
    
    # Align datasets
    X_train_persistence_aligned, X_train_sbert_aligned, y_train_aligned = align_datasets(
        X_train_persistence, y_train_persistence, X_train_sbert, y_train_sbert
    )
    
    min_val = min(len(X_val_persistence), len(X_val_sbert))
    X_val_persistence_aligned = X_val_persistence[:min_val]
    X_val_sbert_aligned = X_val_sbert[:min_val]
    y_val_aligned = y_val_persistence[:min_val]
    
    # Standardize SBERT features
    X_train_sbert_scaled, mean, std = memory_efficient_standardize(X_train_sbert_aligned.astype(np.float32))
    X_val_sbert_scaled = apply_standardization(X_val_sbert_aligned.astype(np.float32), mean, std)
    gc.collect()
    
    # Train hybrid model
    print("Training Hybrid CNN + SBERT model on SNLI...")
    
    train_dataset_hybrid = HybridDataset(X_train_persistence_aligned, X_train_sbert_scaled, y_train_aligned)
    val_dataset_hybrid = HybridDataset(X_val_persistence_aligned, X_val_sbert_scaled, y_val_aligned)
    
    train_loader_hybrid = DataLoader(train_dataset_hybrid, batch_size=128, shuffle=True)
    val_loader_hybrid = DataLoader(val_dataset_hybrid, batch_size=236, shuffle=False)
    
    model_hybrid = HybridClassifier()
    model_hybrid, hybrid_acc = train_model(model_hybrid, train_loader_hybrid, val_loader_hybrid, device=device)
    
    print(f"Hybrid SNLI validation accuracy: {hybrid_acc:.3f}")
    
    # Load ChaosNLI data
    X_chaos_snli_persistence, y_chaos_snli_persistence, chaos_snli_label_distributions, chaos_snli_entropies = load_chaosnli_persistence_images(
        chaosnli_snli_persistence_path
    )
    
    X_chaos_mnli_persistence, y_chaos_mnli_persistence, chaos_mnli_label_distributions, chaos_mnli_entropies = load_chaosnli_persistence_images(
        chaosnli_mnli_persistence_path
    )

    X_chaos_snli_sbert = compute_chaosnli_sbert_features(chaosnli_snli_sbert_path)
    X_chaos_snli_sbert_scaled = apply_standardization(X_chaos_snli_sbert.astype(np.float32), mean, std)
    X_chaos_mnli_sbert = compute_chaosnli_sbert_features(chaosnli_mnli_sbert_path)    
    X_chaos_mnli_sbert_scaled = apply_standardization(X_chaos_snli_sbert.astype(np.float32), mean, std)
    
    # Evaluate hybrid model on ChaosNLI
    print("\nEvaluating Hybrid model on ChaosNLI...")
    print("Evaluating on SNLI CHAOSNLI")
    
    model_hybrid.eval()
    model_hybrid = model_hybrid.to(device)
    
    # Get hybrid model probabilities
    all_probs = []
    batch_size = 128
    
    with torch.no_grad():
        for i in range(0, len(X_chaos_snli_persistence), batch_size):
            batch_end = min(i + batch_size, len(X_chaos_snli_persistence))
            
            # Persistence images: reshape for CNN
            p_batch = X_chaos_snli_persistence[i:batch_end].reshape(-1, 1, 30, 30)
            p_batch = torch.FloatTensor(p_batch).to(device)
            
            # SBERT features
            s_batch = torch.FloatTensor(X_chaos_snli_sbert_scaled[i:batch_end]).to(device)
            
            logits = model_hybrid(p_batch, s_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    
    model_probs = np.vstack(all_probs)
    
    # Compute uncertainty metrics for hybrid
    jsd_scores = []
    kl_scores = []
    
    for i in range(len(model_probs)):
        model_dist = model_probs[i] / model_probs[i].sum()
        human_dist = chaos_snli_label_distributions[i] / chaos_snli_label_distributions[i].sum()
        
        # Add small epsilon for numerical stability
        eps = 1e-10
        model_dist = np.clip(model_dist + eps, eps, 1.0)
        human_dist = np.clip(human_dist + eps, eps, 1.0)
        model_dist = model_dist / model_dist.sum()
        human_dist = human_dist / human_dist.sum()
        
        # Jensen-Shannon Distance
        jsd = jensenshannon(model_dist, human_dist)
        jsd_scores.append(jsd)
        
        # KL Divergence
        kl = entropy(human_dist, model_dist)
        kl_scores.append(kl)
    
    avg_jsd = np.mean(jsd_scores)
    avg_kl = np.mean(kl_scores)
    
    # Traditional accuracy
    model_preds = np.argmax(model_probs, axis=1)
    human_preds = np.argmax(chaos_snli_label_distributions, axis=1)
    traditional_acc = np.mean(model_preds == human_preds)
    
    hybrid_results = {
        'jsd': avg_jsd,
        'kl': avg_kl,
        'accuracy': traditional_acc,
        'beats_random_jsd': avg_jsd < 0.32,
        'beats_random_kl': avg_kl < 0.54
    }
    
    print(f"Hybrid ChaosNLI Results:")
    print(f"  JSD: {avg_jsd:.4f}")
    print(f"  KL:  {avg_kl:.4f}")
    print(f"  Accuracy: {traditional_acc:.4f}")

    print("\n\nEvaluating on MNLI CHAOSNLI")
    
    model_hybrid.eval()
    model_hybrid = model_hybrid.to(device)
    
    # Get hybrid model probabilities
    all_probs = []
    batch_size = 128
    
    with torch.no_grad():
        for i in range(0, len(X_chaos_mnli_persistence), batch_size):
            batch_end = min(i + batch_size, len(X_chaos_mnli_persistence))
            
            # Persistence images: reshape for CNN
            p_batch = X_chaos_mnli_persistence[i:batch_end].reshape(-1, 1, 30, 30)
            p_batch = torch.FloatTensor(p_batch).to(device)
            
            # SBERT features
            s_batch = torch.FloatTensor(X_chaos_mnli_sbert_scaled[i:batch_end]).to(device)
            
            logits = model_hybrid(p_batch, s_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    
    model_probs = np.vstack(all_probs)
    
    # Compute uncertainty metrics for hybrid
    jsd_scores = []
    kl_scores = []
    
    for i in range(len(model_probs)):
        model_dist = model_probs[i] / model_probs[i].sum()
        human_dist = chaos_mnli_label_distributions[i] / chaos_mnli_label_distributions[i].sum()
        
        # Add small epsilon for numerical stability
        eps = 1e-10
        model_dist = np.clip(model_dist + eps, eps, 1.0)
        human_dist = np.clip(human_dist + eps, eps, 1.0)
        model_dist = model_dist / model_dist.sum()
        human_dist = human_dist / human_dist.sum()
        
        # Jensen-Shannon Distance
        jsd = jensenshannon(model_dist, human_dist)
        jsd_scores.append(jsd)
        
        # KL Divergence
        kl = entropy(human_dist, model_dist)
        kl_scores.append(kl)
    
    avg_jsd = np.mean(jsd_scores)
    avg_kl = np.mean(kl_scores)
    
    # Traditional accuracy
    model_preds = np.argmax(model_probs, axis=1)
    human_preds = np.argmax(chaos_mnli_label_distributions, axis=1)
    traditional_acc = np.mean(model_preds == human_preds)
    
    hybrid_results = {
        'jsd': avg_jsd,
        'kl': avg_kl,
        'accuracy': traditional_acc,
        'beats_random_jsd': avg_jsd < 0.32,
        'beats_random_kl': avg_kl < 0.54
    }
    
    print(f"Hybrid ChaosNLI Results:")
    print(f"  JSD: {avg_jsd:.4f}")
    print(f"  KL:  {avg_kl:.4f}")
    print(f"  Accuracy: {traditional_acc:.4f}") 
    
    return {
        'snli_training_accuracy': hybrid_acc,
        'chaosnli_uncertainty': hybrid_results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train on SNLI, Evaluate on ChaosNLI')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--include-hybrid', action='store_true', help='Also train and evaluate hybrid model')
    
    args = parser.parse_args()
    
    # Configure paths
    snli_persistence_train_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_snli_train_persistence_images.pkl"
    snli_persistence_val_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_snli_val_persistence_images.pkl"
    snli_sbert_train_path = "/vol/bitbucket/ahb24/tda_entailment_new/snli_train_sbert_tokens.pkl"
    snli_sbert_val_path = "/vol/bitbucket/ahb24/tda_entailment_new/snli_val_sbert_tokens.pkl"
    mnli_persistence_train_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_mnli_train_persistence_images_intermediate_48000.pkl"
    mnli_persistence_val_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_mnli_val_matched_persistence_images.pkl"
    mnli_sbert_train_path = "/vol/bitbucket/ahb24/tda_entailment_new/mnli_train_sbert_tokens.pkl"
    mnli_sbert_val_path = "/vol/bitbucket/ahb24/tda_entailment_new/mnli_val_matched_sbert_tokens.pkl"
    chaosnli_snli_persistence_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_chaosnli_snli_persistence_images.pkl"
    chaosnli_snli_sbert_path = "/vol/bitbucket/ahb24/tda_entailment_new/chaosnli_snli_sbert_tokens.pkl"
    chaosnli_mnli_persistence_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_chaosnli_mnli_matched_persistence_images.pkl"
    chaosnli_mnli_sbert_path = "/vol/bitbucket/ahb24/tda_entailment_new/chaosnli_mnli_matched_sbert_tokens.pkl"
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Run main experiment
    # results = run_snli_train_chaosnli_eval(
    #     snli_persistence_train_path, snli_persistence_val_path,
    #         mnli_persistence_train_path, mnli_persistence_val_path,
    #         snli_sbert_train_path, snli_sbert_val_path,
    #         mnli_sbert_train_path, mnli_sbert_val_path,
    #         chaosnli_snli_persistence_path, chaosnli_snli_sbert_path,
    #         chaosnli_mnli_persistence_path, chaosnli_mnli_sbert_path,
    #     device=device
    # )

    results = run_complete_architecture_comparison(
        snli_persistence_train_path, snli_persistence_val_path,
        mnli_persistence_train_path, mnli_persistence_val_path,
        snli_sbert_train_path, snli_sbert_val_path,
        mnli_sbert_train_path, mnli_sbert_val_path,
        chaosnli_snli_persistence_path, chaosnli_snli_sbert_path,
        chaosnli_mnli_persistence_path, chaosnli_mnli_sbert_path,
        device=device
    )

    # Run hybrid experiment if requested
    if args.include_hybrid:
        hybrid_results = run_hybrid_snli_train_chaosnli_eval(
            snli_persistence_train_path, snli_persistence_val_path,
            mnli_persistence_train_path, mnli_persistence_val_path,
            snli_sbert_train_path, snli_sbert_val_path,
            mnli_sbert_train_path, mnli_sbert_val_path,
            chaosnli_snli_persistence_path, chaosnli_snli_sbert_path,
            chaosnli_mnli_persistence_path, chaosnli_mnli_sbert_path,
            device=device
        )
        
        print("\n" + "=" * 80)
    
    print("\nExperiment completed!")

