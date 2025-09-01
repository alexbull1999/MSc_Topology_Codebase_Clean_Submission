
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

class ConfidenceAwarePersistenceCNN(nn.Module):
    """Persistence CNN with confidence calibration for high-agreement samples"""
    
    def __init__(self, num_classes=3):
        super().__init__()
        
        # Standard CNN backbone
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Separate pathways for classification and confidence
        self.classification_head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output confidence score [0,1]
        )
        
    def forward(self, x):
        # Feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        features = x.view(-1, 64 * 7 * 7)
        
        # Classification logits
        logits = self.classification_head(features)
        
        # Confidence score
        confidence = self.confidence_head(features)
        
        # Apply confidence to logits (higher confidence = more extreme probabilities)
        # Low confidence flattens distribution, high confidence sharpens it
        temperature = 1.0 / (confidence + 0.1)  # Avoid division by zero
        calibrated_logits = logits / temperature
        
        return calibrated_logits

class EntropyRegularizedLoss(nn.Module):
    """Loss that penalizes high entropy on clear cases"""
    
    def __init__(self, entropy_weight=0.1):
        super().__init__()
        self.entropy_weight = entropy_weight
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, logits, targets, confidence_scores=None):
        # Standard classification loss
        ce_loss = self.ce_loss(logits, targets)
        
        # Entropy regularization - encourage low entropy (high confidence)
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        entropy_loss = torch.mean(entropy)
        
        # If we have confidence scores, use them to weight entropy loss
        if confidence_scores is not None:
            # High confidence samples should have low entropy
            confidence_loss = torch.mean((1 - confidence_scores) * entropy)
            total_loss = ce_loss + self.entropy_weight * confidence_loss
        else:
            total_loss = ce_loss + self.entropy_weight * entropy_loss
            
        return total_loss


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

def train_with_confidence_calibration(model, train_loader, val_loader, epochs=200, device='cuda'):
    """Training with confidence-aware loss"""
    
    model = model.to(device)
    criterion = EntropyRegularizedLoss(entropy_weight=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Since we now only return logits:
            logits = model(inputs)
            
            # For confidence loss, manually get confidence if needed:
            if hasattr(model, 'confidence_head'):
                features = model.pool(F.relu(model.conv1(inputs)))
                features = model.pool(F.relu(model.conv2(features)))
                features = features.view(-1, 64 * 7 * 7)
                confidence = model.confidence_head(features)
                loss = criterion(logits, labels, confidence)
            else:
                loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation - FIXED VERSION
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Same logic as training - get logits and confidence separately
                logits = model(inputs)
                
                if hasattr(model, 'confidence_head'):
                    features = model.pool(F.relu(model.conv1(inputs)))
                    features = model.pool(F.relu(model.conv2(features)))
                    features = features.view(-1, 64 * 7 * 7)
                    confidence = model.confidence_head(features)
                    loss = criterion(logits, labels, confidence)
                else:
                    loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = val_correct / val_total
        scheduler.step(val_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Train Loss {train_loss/len(train_loader):.4f} "
                  f"(Acc {train_correct/train_total:.3f}), Val Loss {val_loss/len(val_loader):.4f} "
                  f"(Acc {val_acc:.3f})")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= 25:
            break
    
    return model, best_val_acc


def load_snli_persistence_images(train_path, val_path):
    """Load SNLI persistence images for training"""
    print("Loading SNLI persistence images...")
    
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
    
    print(f"SNLI persistence images: Train {X_train.shape}, Val {X_val.shape}")
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

def compute_sbert_baseline_features(data_path):
    """Compute SBERT baseline features"""
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


def run_snli_train_chaosnli_eval(snli_persistence_train_path, snli_persistence_val_path,
                                snli_sbert_train_path, snli_sbert_val_path,
                                chaosnli_persistence_path, chaosnli_sbert_path, device='cuda'):
    """Train on SNLI, evaluate on ChaosNLI"""
    
    print("=" * 80)
    print("TRAIN ON SNLI, EVALUATE ON CHAOSNLI")
    print("=" * 80)
    
    # 1. Load SNLI training data
    X_train_persistence, y_train_persistence, X_val_persistence, y_val_persistence = load_snli_persistence_images(
        snli_persistence_train_path, snli_persistence_val_path
    )
    
    X_train_sbert, y_train_sbert = compute_sbert_baseline_features(snli_sbert_train_path)
    X_val_sbert, y_val_sbert = compute_sbert_baseline_features(snli_sbert_val_path)
    
    # 2. Align SNLI datasets
    X_train_persistence_aligned, X_train_sbert_aligned, y_train_aligned = align_datasets(
        X_train_persistence, y_train_persistence, X_train_sbert, y_train_sbert
    )
    
    min_val = min(len(X_val_persistence), len(X_val_sbert))
    X_val_persistence_aligned = X_val_persistence[:min_val]
    X_val_sbert_aligned = X_val_sbert[:min_val]
    y_val_aligned = y_val_persistence[:min_val]
    
    # 3. Load ChaosNLI evaluation data
    X_chaos_persistence, y_chaos_persistence, chaos_label_distributions, chaos_entropies = load_chaosnli_persistence_images(
        chaosnli_persistence_path
    )
    X_chaos_sbert = compute_chaosnli_sbert_features(chaosnli_sbert_path)
    
    # 4. Train Persistence Image CNN on SNLI
    print("\n" + "=" * 60)
    print("TRAINING PERSISTENCE IMAGE CNN ON SNLI")
    print("=" * 60)
    
    train_dataset_topo = PersistenceImageDataset(X_train_persistence_aligned, y_train_aligned, transform_to_image=True)
    val_dataset_topo = PersistenceImageDataset(X_val_persistence_aligned, y_val_aligned, transform_to_image=True)
    #was 32/64 for original run
    train_loader_topo = DataLoader(train_dataset_topo, batch_size=128, shuffle=True)
    val_loader_topo = DataLoader(val_dataset_topo, batch_size=256, shuffle=False)
    
    model_topo = ConfidenceAwarePersistenceCNN()
    model_topo, topo_acc = train_with_confidence_calibration(model_topo, train_loader_topo, val_loader_topo, device=device)
    
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
    model_sbert, sbert_acc = train_with_confidence_calibration(model_sbert, train_loader_sbert, val_loader_sbert, device=device)
    
    print(f"SBERT SNLI validation accuracy: {sbert_acc:.3f}")
    
    # Cleanup training data
    del train_dataset_topo, val_dataset_topo, train_loader_topo, val_loader_topo
    del train_dataset_sbert, val_dataset_sbert, train_loader_sbert, val_loader_sbert
    gc.collect()
    
    # 6. Evaluate both models on ChaosNLI
    print("\n" + "=" * 80)
    print("EVALUATING ON CHAOSNLI UNCERTAINTY QUANTIFICATION")
    print("=" * 80)
    
    # Evaluate Persistence CNN on ChaosNLI
    persistence_results = evaluate_chaosnli_uncertainty(
        model_topo, X_chaos_persistence, chaos_label_distributions, chaos_entropies,
        model_type='persistence', device=device
    )
    
    # Evaluate SBERT on ChaosNLI
    sbert_results = evaluate_chaosnli_uncertainty(
        model_sbert, X_chaos_sbert, chaos_label_distributions, chaos_entropies,
        mean=mean, std=std, model_type='sbert', device=device
    )
    
    # 7. Final comparison
    print("\n" + "=" * 80)
    print("FINAL CHAOSNLI EVALUATION RESULTS")
    print("=" * 80)
    print(f"SNLI Training Results:")
    print(f"  Persistence CNN:  {topo_acc:.3f}")
    print(f"  SBERT Baseline:   {sbert_acc:.3f}")
    
    print(f"\nChaosNLI Uncertainty Quantification:")
    print(f"  Persistence CNN:")
    print(f"    JSD: {persistence_results['jsd']:.4f}")
    print(f"    KL:  {persistence_results['kl']:.4f}")
    print(f"    Accuracy: {persistence_results['accuracy']:.4f}")
    
    print(f"  SBERT Baseline:")
    print(f"    JSD: {sbert_results['jsd']:.4f}")
    print(f"    KL:  {sbert_results['kl']:.4f}")
    print(f"    Accuracy: {sbert_results['accuracy']:.4f}")
    
    # Determine which is better at uncertainty quantification
    persistence_better_jsd = persistence_results['jsd'] < sbert_results['jsd']
    persistence_better_kl = persistence_results['kl'] < sbert_results['kl']
    
    print(f"\nUncertainty Quantification Comparison:")
    print(f"  JSD: {'Persistence CNN' if persistence_better_jsd else 'SBERT'} is better")
    print(f"  KL:  {'Persistence CNN' if persistence_better_kl else 'SBERT'} is better")
    
    if persistence_better_jsd and persistence_better_kl:
        print(f"\n🎉 Persistence CNN wins on both uncertainty metrics!")
    elif persistence_better_jsd or persistence_better_kl:
        print(f"\n📊 Mixed results - each approach has strengths")
    else:
        print(f"\n📈 SBERT baseline still leads on uncertainty quantification")
    
    return {
        'snli_training': {
            'persistence_cnn': topo_acc,
            'sbert_baseline': sbert_acc
        },
        'chaosnli_evaluation': {
            'persistence_cnn': persistence_results,
            'sbert_baseline': sbert_results
        }
    }

def run_hybrid_snli_train_chaosnli_eval(snli_persistence_train_path, snli_persistence_val_path,
                                       snli_sbert_train_path, snli_sbert_val_path,
                                       chaosnli_persistence_path, chaosnli_sbert_path, device='cuda'):
    """Train hybrid model on SNLI, evaluate on ChaosNLI"""
    
    print("\n" + "=" * 80)
    print("HYBRID MODEL: TRAIN ON SNLI, EVALUATE ON CHAOSNLI")
    print("=" * 80)
    
    # Load SNLI data
    X_train_persistence, y_train_persistence, X_val_persistence, y_val_persistence = load_snli_persistence_images(
        snli_persistence_train_path, snli_persistence_val_path
    )
    
    X_train_sbert, y_train_sbert = compute_sbert_baseline_features(snli_sbert_train_path)
    X_val_sbert, y_val_sbert = compute_sbert_baseline_features(snli_sbert_val_path)
    
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
    X_chaos_persistence, y_chaos_persistence, chaos_label_distributions, chaos_entropies = load_chaosnli_persistence_images(
        chaosnli_persistence_path
    )
    X_chaos_sbert = compute_chaosnli_sbert_features(chaosnli_sbert_path)
    X_chaos_sbert_scaled = apply_standardization(X_chaos_sbert.astype(np.float32), mean, std)
    
    # Evaluate hybrid model on ChaosNLI
    print("\nEvaluating Hybrid model on ChaosNLI...")
    
    model_hybrid.eval()
    model_hybrid = model_hybrid.to(device)
    
    # Get hybrid model probabilities
    all_probs = []
    batch_size = 128
    
    with torch.no_grad():
        for i in range(0, len(X_chaos_persistence), batch_size):
            batch_end = min(i + batch_size, len(X_chaos_persistence))
            
            # Persistence images: reshape for CNN
            p_batch = X_chaos_persistence[i:batch_end].reshape(-1, 1, 30, 30)
            p_batch = torch.FloatTensor(p_batch).to(device)
            
            # SBERT features
            s_batch = torch.FloatTensor(X_chaos_sbert_scaled[i:batch_end]).to(device)
            
            logits = model_hybrid(p_batch, s_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    
    model_probs = np.vstack(all_probs)
    
    # Compute uncertainty metrics for hybrid
    jsd_scores = []
    kl_scores = []
    
    for i in range(len(model_probs)):
        model_dist = model_probs[i] / model_probs[i].sum()
        human_dist = chaos_label_distributions[i] / chaos_label_distributions[i].sum()
        
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
    human_preds = np.argmax(chaos_label_distributions, axis=1)
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
    chaosnli_persistence_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_chaosnli_snli_persistence_images.pkl"
    chaosnli_sbert_path = "/vol/bitbucket/ahb24/tda_entailment_new/chaosnli_snli_sbert_tokens.pkl"
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Run main experiment
    results = run_snli_train_chaosnli_eval(
        snli_persistence_train_path, snli_persistence_val_path,
        snli_sbert_train_path, snli_sbert_val_path,
        chaosnli_persistence_path, chaosnli_sbert_path,
        device=device
    )

    gc.collect()

    # Run hybrid experiment if requested
    if args.include_hybrid:
        hybrid_results = run_hybrid_snli_train_chaosnli_eval(
            snli_persistence_train_path, snli_persistence_val_path,
            snli_sbert_train_path, snli_sbert_val_path,
            chaosnli_persistence_path, chaosnli_sbert_path,
            device=device
        )
        
        print("\n" + "=" * 80)
        print("COMPLETE RESULTS INCLUDING HYBRID")
        print("=" * 80)
        print(f"SNLI Validation Accuracies:")
        print(f"  Persistence CNN:  {results['snli_training']['persistence_cnn']:.3f}")
        print(f"  SBERT Baseline:   {results['snli_training']['sbert_baseline']:.3f}")
        print(f"  Hybrid Model:     {hybrid_results['snli_training_accuracy']:.3f}")
        
        print(f"\nChaosNLI Uncertainty Quantification (JSD):")
        print(f"  Persistence CNN:  {results['chaosnli_evaluation']['persistence_cnn']['jsd']:.4f}")
        print(f"  SBERT Baseline:   {results['chaosnli_evaluation']['sbert_baseline']['jsd']:.4f}")
        print(f"  Hybrid Model:     {hybrid_results['chaosnli_uncertainty']['jsd']:.4f}")
    
    print("\nExperiment completed!")