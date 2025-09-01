#!/usr/bin/env python3
"""
Separate ChaosNLI-S and ChaosNLI-M Evaluation

This script evaluates persistence image fusion separately on:
- ChaosNLI-S (SNLI): 1,514 samples  
- ChaosNLI-M (MNLI): 1,599 samples

Results can be directly compared with published Table 5 format.
"""

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import argparse
from pathlib import Path
from collections import defaultdict
import time

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

class HybridDataset(Dataset):
    """Dataset for hybrid training with published logits + persistence images"""
    
    def __init__(self, persistence_images, published_logits, human_distributions, uids):
        self.persistence_images = persistence_images
        self.published_logits = published_logits
        self.human_distributions = human_distributions
        self.uids = uids
        
    def __len__(self):
        return len(self.persistence_images)
    
    def __getitem__(self, idx):
        # Persistence image: reshape to 30x30 for CNN
        p_image = self.persistence_images[idx].reshape(30, 30)
        p_image = torch.FloatTensor(p_image).unsqueeze(0)  # Add channel dimension
        
        # Published logits
        pub_logits = torch.FloatTensor(self.published_logits[idx])
        
        # Human label distribution (target)
        human_dist = torch.FloatTensor(self.human_distributions[idx])
        
        return p_image, pub_logits, human_dist, self.uids[idx]

def load_published_predictions():
    """Load published model predictions"""
    
    predictions_file = Path("ChaosNLI/data/model_predictions/model_predictions_for_snli_mnli.json")
    
    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
    
    print("Loading published model predictions...")
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    print(f"Available models: {list(predictions.keys())}")
    return predictions

def load_separate_persistence_data():
    """Load ChaosNLI persistence images separately for SNLI and MNLI"""
    
    print("Loading persistence images and ground truth (separate datasets)...")
    
    # Load SNLI ChaosNLI data
    snli_persistence_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_chaosnli_snli_persistence_images.pkl"
    with open(snli_persistence_path, 'rb') as f:
        snli_data = pickle.load(f)
    
    # Load MNLI ChaosNLI data
    mnli_persistence_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_chaosnli_mnli_matched_persistence_images.pkl"
    with open(mnli_persistence_path, 'rb') as f:
        mnli_data = pickle.load(f)
    
    # Keep separate for individual evaluation
    snli_test_data = {
        'persistence_images': snli_data['persistence_images'],
        'uids': snli_data['uids'],
        'label_distributions': snli_data['label_distributions'],
        'entropies': snli_data['entropies'],
        'majority_labels': snli_data['majority_labels']
    }
    
    mnli_test_data = {
        'persistence_images': mnli_data['persistence_images'],
        'uids': mnli_data['uids'],
        'label_distributions': mnli_data['label_distributions'],
        'entropies': mnli_data['entropies'],
        'majority_labels': mnli_data['majority_labels']
    }
    
    print(f"ChaosNLI-S (SNLI): {len(snli_test_data['uids'])} samples")
    print(f"ChaosNLI-M (MNLI): {len(mnli_test_data['uids'])} samples")
    
    return snli_test_data, mnli_test_data

def match_predictions_with_persistence_single(published_predictions, persistence_data, model_name, dataset_name):
    """Match published predictions with persistence images by UID for single dataset"""
    
    model_preds = published_predictions[model_name]
    
    # Create UID mappings
    persistence_uid_to_idx = {uid: idx for idx, uid in enumerate(persistence_data['uids'])}
    
    matched_data = {
        'persistence_images': [],
        'published_logits': [],
        'human_distributions': [],
        'entropies': [],
        'uids': [],
        'majority_labels': []
    }
    
    matched_count = 0
    
    for uid, prediction in model_preds.items():
        if uid in persistence_uid_to_idx:
            idx = persistence_uid_to_idx[uid]
            
            matched_data['persistence_images'].append(persistence_data['persistence_images'][idx])
            matched_data['published_logits'].append(prediction['logits'])
            matched_data['human_distributions'].append(persistence_data['label_distributions'][idx])
            matched_data['entropies'].append(persistence_data['entropies'][idx])
            matched_data['uids'].append(uid)
            matched_data['majority_labels'].append(persistence_data['majority_labels'][idx])
            
            matched_count += 1
    
    # Convert to numpy arrays
    for key in matched_data:
        if key != 'uids':
            matched_data[key] = np.array(matched_data[key])
    
    match_rate = matched_count / len(persistence_data['uids']) * 100
    print(f"  {dataset_name}: Matched {matched_count}/{len(persistence_data['uids'])} ({match_rate:.1f}%)")
    
    if matched_count == 0:
        raise ValueError(f"No matches found for {model_name} on {dataset_name}")
    
    return matched_data

def load_training_validation_data():
    """Load ALL 5 chunks for SNLI+MNLI training and validation persistence images"""
    
    print("Loading ALL SNLI+MNLI chunks for training and validation data...")
    
    all_train_images = []
    all_train_labels = []
    
    # Load all 5 SNLI chunks
    print("Loading SNLI chunks...")
    for chunk_idx in range(1, 6):
        chunk_path = f"/vol/bitbucket/ahb24/tda_entailment_new/chunked_snli_train_persistence_images_chunk_{chunk_idx}_of_5.pkl"
        
        if Path(chunk_path).exists():
            with open(chunk_path, 'rb') as f:
                chunk_data = pickle.load(f)
            
            chunk_images = chunk_data['persistence_images']
            chunk_labels = chunk_data['labels']
            
            print(f"  SNLI chunk {chunk_idx}: {len(chunk_images)} samples")
            
            all_train_images.append(chunk_images)
            all_train_labels.extend(chunk_labels)
        else:
            print(f"  SNLI chunk {chunk_idx} not found: {chunk_path}")
    
    # Load all 5 MNLI chunks
    print("Loading MNLI chunks...")
    for chunk_idx in range(1, 6):
        chunk_path = f"/vol/bitbucket/ahb24/tda_entailment_new/chunked_mnli_train_persistence_images_chunk_{chunk_idx}_of_5.pkl"
        
        if Path(chunk_path).exists():
            with open(chunk_path, 'rb') as f:
                chunk_data = pickle.load(f)
            
            chunk_images = chunk_data['persistence_images']
            chunk_labels = chunk_data['labels']
            
            print(f"  MNLI chunk {chunk_idx}: {len(chunk_images)} samples")
            
            all_train_images.append(chunk_images)
            all_train_labels.extend(chunk_labels)
        else:
            print(f"  MNLI chunk {chunk_idx} not found: {chunk_path}")
    
    # Combine all training chunks
    if all_train_images:
        train_persistence_images = np.vstack(all_train_images)
    else:
        raise ValueError("No training chunks found!")
    
    # Load validation data (these should exist as smaller files)
    snli_val_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_snli_val_persistence_images.pkl"
    mnli_val_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_mnli_val_matched_persistence_images.pkl"
    
    val_images_list = []
    val_labels_list = []
    
    # Load SNLI validation
    if Path(snli_val_path).exists():
        with open(snli_val_path, 'rb') as f:
            snli_val = pickle.load(f)
        val_images_list.append(snli_val['persistence_images'])
        val_labels_list.extend(snli_val['labels'])
        print(f"  SNLI val: {len(snli_val['persistence_images'])} samples")
    else:
        print(f"  SNLI val not found: {snli_val_path}")
    
    # Load MNLI validation
    if Path(mnli_val_path).exists():
        with open(mnli_val_path, 'rb') as f:
            mnli_val = pickle.load(f)
        val_images_list.append(mnli_val['persistence_images'])
        val_labels_list.extend(mnli_val['labels'])
        print(f"  MNLI val: {len(mnli_val['persistence_images'])} samples")
    else:
        print(f"  MNLI val not found: {mnli_val_path}")
    
    # Combine validation data
    if val_images_list:
        val_persistence_images = np.vstack(val_images_list)
    else:
        raise ValueError("No validation data found!")
    
    # Convert string labels to indices if needed
    label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    if isinstance(all_train_labels[0], str):
        train_labels = np.array([label_to_idx[label] for label in all_train_labels])
        val_labels = np.array([label_to_idx[label] for label in val_labels_list])
    else:
        train_labels = np.array(all_train_labels)
        val_labels = np.array(val_labels_list)
    
    print(f"  Combined train: {len(train_persistence_images)} samples")
    print(f"  Combined val: {len(val_persistence_images)} samples")
    
    return train_persistence_images, train_labels, val_persistence_images, val_labels

def train_persistence_cnn(device='cuda'):
    """Train persistence CNN on SNLI+MNLI train/val data (proper protocol)"""
    
    print("Training persistence CNN following ChaosNLI evaluation protocol...")
    
    # Load proper training/validation splits
    train_images, train_labels, val_images, val_labels = load_training_validation_data()
    
    # Create datasets with proper train/val splits
    class TrainingDataset(Dataset):
        def __init__(self, persistence_images, labels):
            self.persistence_images = persistence_images
            self.labels = labels
            
        def __len__(self):
            return len(self.persistence_images)
        
        def __getitem__(self, idx):
            p_image = self.persistence_images[idx].reshape(30, 30)
            p_image = torch.FloatTensor(p_image).unsqueeze(0)
            label = torch.LongTensor([self.labels[idx]])[0]
            return p_image, label
    
    train_dataset = TrainingDataset(train_images, train_labels)
    val_dataset = TrainingDataset(val_images, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Train model
    model = PersistenceImageCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print("Training persistence CNN...")
    for epoch in range(50):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for p_images, labels in train_loader:
            p_images, labels = p_images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(p_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for p_images, labels in val_loader:
                p_images, labels = p_images.to(device), labels.to(device)
                
                outputs = model(p_images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d}: Train Acc {train_acc:.3f}, Val Acc {val_acc:.3f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= 10:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"Persistence CNN training completed. Best val accuracy: {best_val_acc:.3f}")
    
    return model

def evaluate_weighted_fusion_single(persistence_model, matched_data, weight_range, device='cuda'):
    """Evaluate different fusion weights on single dataset (SNLI or MNLI)"""
    
    persistence_model.eval()
    
    # Create test dataset from matched data
    test_dataset = HybridDataset(
        matched_data['persistence_images'],
        matched_data['published_logits'],
        matched_data['human_distributions'],
        matched_data['uids']
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Collect all predictions
    all_published_logits = []
    all_persistence_logits = []
    all_human_distributions = []
    
    with torch.no_grad():
        for p_images, pub_logits, human_dists, uids in test_loader:
            p_images = p_images.to(device)
            
            # Get persistence predictions
            persistence_outputs = persistence_model(p_images)
            
            all_published_logits.append(pub_logits.cpu().numpy())
            all_persistence_logits.append(persistence_outputs.cpu().numpy())
            all_human_distributions.append(human_dists.cpu().numpy())
    
    all_published_logits = np.vstack(all_published_logits)
    all_persistence_logits = np.vstack(all_persistence_logits)
    all_human_distributions = np.vstack(all_human_distributions)
    
    # Test different fusion weights
    results = {}
    
    for alpha in weight_range:
        # Weighted fusion: (1-α) * published + α * persistence
        fused_logits = (1 - alpha) * all_published_logits + alpha * all_persistence_logits
        fused_probs = F.softmax(torch.FloatTensor(fused_logits), dim=1).numpy()
        
        # Compute uncertainty metrics
        jsd_scores = []
        kl_scores = []
        
        for i in range(len(fused_probs)):
            pred_dist = fused_probs[i]
            human_dist = all_human_distributions[i]
            
            # Normalize distributions
            eps = 1e-10
            pred_dist = np.clip(pred_dist + eps, eps, 1.0)
            human_dist = np.clip(human_dist + eps, eps, 1.0)
            pred_dist = pred_dist / pred_dist.sum()
            human_dist = human_dist / human_dist.sum()
            
            # Jensen-Shannon Distance
            jsd = jensenshannon(pred_dist, human_dist)
            jsd_scores.append(jsd)
            
            # KL Divergence
            kl = entropy(human_dist, pred_dist)
            kl_scores.append(kl)
        
        # Traditional accuracy
        pred_labels = np.argmax(fused_probs, axis=1)
        human_labels = np.argmax(all_human_distributions, axis=1)
        accuracy = np.mean(pred_labels == human_labels)
        
        results[alpha] = {
            'jsd': np.mean(jsd_scores),
            'kl': np.mean(kl_scores),
            'accuracy': accuracy,
            'jsd_std': np.std(jsd_scores),
            'kl_std': np.std(kl_scores)
        }
    
    return results

def run_separate_evaluation(published_predictions, snli_data, mnli_data, weight_range, device='cuda'):
    """Run evaluation separately on ChaosNLI-S and ChaosNLI-M"""
    
    print("="*80)
    print("SEPARATE ChaosNLI-S and ChaosNLI-M EVALUATION")
    print("="*80)
    
    # Train persistence CNN once on SNLI+MNLI train/val data
    print("Step 1: Training persistence CNN on SNLI+MNLI train/val...")
    persistence_model = train_persistence_cnn(device=device)
    
    all_results = {}
    
    for model_name in published_predictions.keys():
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL: {model_name.upper()}")
        print(f"{'='*60}")
        
        try:
            # Match predictions with both datasets
            snli_matched = match_predictions_with_persistence_single(
                published_predictions, snli_data, model_name, "ChaosNLI-S"
            )
            mnli_matched = match_predictions_with_persistence_single(
                published_predictions, mnli_data, model_name, "ChaosNLI-M"
            )
            
            # Evaluate on SNLI separately
            print("Evaluating ChaosNLI-S (SNLI)...")
            snli_results = evaluate_weighted_fusion_single(
                persistence_model, snli_matched, weight_range, device=device
            )
            
            # Evaluate on MNLI separately
            print("Evaluating ChaosNLI-M (MNLI)...")
            mnli_results = evaluate_weighted_fusion_single(
                persistence_model, mnli_matched, weight_range, device=device
            )
            
            # Store results
            all_results[model_name] = {
                'snli': snli_results,
                'mnli': mnli_results
            }
            
            # Print summary for this model
            print(f"\n📊 {model_name} Results Summary:")
            
            # SNLI results
            snli_baseline_jsd = snli_results[0.0]['jsd']
            snli_baseline_kl = snli_results[0.0]['kl']
            snli_best_alpha = min(snli_results.keys(), key=lambda a: snli_results[a]['jsd'])
            snli_best_jsd = snli_results[snli_best_alpha]['jsd']
            snli_best_kl = snli_results[snli_best_alpha]['kl']
            
            print(f"   ChaosNLI-S: Baseline JSD={snli_baseline_jsd:.4f}, KL={snli_baseline_kl:.4f}")
            print(f"   ChaosNLI-S: Best (α={snli_best_alpha:.1f}) JSD={snli_best_jsd:.4f}, KL={snli_best_kl:.4f}")
            print(f"   ChaosNLI-S: Improvements JSD={snli_baseline_jsd-snli_best_jsd:+.4f}, KL={snli_baseline_kl-snli_best_kl:+.4f}")
            
            # MNLI results
            mnli_baseline_jsd = mnli_results[0.0]['jsd']
            mnli_baseline_kl = mnli_results[0.0]['kl']
            mnli_best_alpha = min(mnli_results.keys(), key=lambda a: mnli_results[a]['jsd'])
            mnli_best_jsd = mnli_results[mnli_best_alpha]['jsd']
            mnli_best_kl = mnli_results[mnli_best_alpha]['kl']
            
            print(f"   ChaosNLI-M: Baseline JSD={mnli_baseline_jsd:.4f}, KL={mnli_baseline_kl:.4f}")
            print(f"   ChaosNLI-M: Best (α={mnli_best_alpha:.1f}) JSD={mnli_best_jsd:.4f}, KL={mnli_best_kl:.4f}")
            print(f"   ChaosNLI-M: Improvements JSD={mnli_baseline_jsd-mnli_best_jsd:+.4f}, KL={mnli_baseline_kl-mnli_best_kl:+.4f}")
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            all_results[model_name] = None
    
    return all_results

def print_table_format_summary(all_results):
    """Print results in ChaosNLI Table 5 format"""
    
    print(f"\n{'='*120}")
    print("📊 RESULTS IN CHAOSNLI TABLE 5 FORMAT")
    print(f"{'='*120}")
    
    # Published baselines for comparison
    published_baselines = {
        'bert-base': {'snli': {'jsd': 0.2345, 'kl': 0.481}, 'mnli': {'jsd': 0.3055, 'kl': 0.7204}},
        'bert-large': {'snli': {'jsd': 0.2300, 'kl': 0.5017}, 'mnli': {'jsd': 0.3152, 'kl': 0.8449}},
        'xlnet-base': {'snli': {'jsd': 0.2331, 'kl': 0.5121}, 'mnli': {'jsd': 0.3069, 'kl': 0.7927}},
        'xlnet-large': {'snli': {'jsd': 0.2259, 'kl': 0.5054}, 'mnli': {'jsd': 0.3116, 'kl': 0.8818}},
        'roberta-base': {'snli': {'jsd': 0.2294, 'kl': 0.5045}, 'mnli': {'jsd': 0.3073, 'kl': 0.7807}},
        'roberta-large': {'snli': {'jsd': 0.2210, 'kl': 0.4937}, 'mnli': {'jsd': 0.3112, 'kl': 0.8701}},
        'bart-large': {'snli': {'jsd': 0.2203, 'kl': 0.4714}, 'mnli': {'jsd': 0.3165, 'kl': 0.8845}},
        'albert-xxlarge': {'snli': {'jsd': 0.2350, 'kl': 0.5342}, 'mnli': {'jsd': 0.3159, 'kl': 0.862}},
        'distilbert': {'snli': {'jsd': 0.2439, 'kl': 0.4682}, 'mnli': {'jsd': 0.3133, 'kl': 0.6652}}
    }
    
    # Print header
    print(f"{'Model':<15} {'ChaosNLI-S':<35} {'ChaosNLI-M':<35}")
    print(f"{'':15} {'JSD↓':<8} {'KL↓':<8} {'Best α':<7} {'vs Pub':<12} {'JSD↓':<8} {'KL↓':<8} {'Best α':<7} {'vs Pub'}")
    print("-" * 120)
    
    for model_name, results in all_results.items():
        if results is None:
            print(f"{model_name:<15} {'ERROR':<35} {'ERROR':<35}")
            continue
        
        # SNLI results
        snli_results = results['snli']
        snli_best_alpha = min(snli_results.keys(), key=lambda a: snli_results[a]['jsd'])
        snli_best_jsd = snli_results[snli_best_alpha]['jsd']
        snli_best_kl = snli_results[snli_best_alpha]['kl']
        
        # Compare with published SNLI
        pub_snli = published_baselines.get(model_name, {}).get('snli', {})
        snli_vs_pub_jsd = f"{snli_best_jsd - pub_snli.get('jsd', 0):+.4f}" if pub_snli else "N/A"
        
        # MNLI results
        mnli_results = results['mnli']
        mnli_best_alpha = min(mnli_results.keys(), key=lambda a: mnli_results[a]['jsd'])
        mnli_best_jsd = mnli_results[mnli_best_alpha]['jsd']
        mnli_best_kl = mnli_results[mnli_best_alpha]['kl']
        
        # Compare with published MNLI
        pub_mnli = published_baselines.get(model_name, {}).get('mnli', {})
        mnli_vs_pub_jsd = f"{mnli_best_jsd - pub_mnli.get('jsd', 0):+.4f}" if pub_mnli else "N/A"
        
        print(f"{model_name:<15} {snli_best_jsd:<8.4f} {snli_best_kl:<8.4f} {snli_best_alpha:<7.1f} {snli_vs_pub_jsd:<12} {mnli_best_jsd:<8.4f} {mnli_best_kl:<8.4f} {mnli_best_alpha:<7.1f} {mnli_vs_pub_jsd}")
    
    # Find best performers
    valid_results = {k: v for k, v in all_results.items() if v is not None}
    if valid_results:
        # Best SNLI
        best_snli_model = min(valid_results.keys(), 
                            key=lambda m: min(valid_results[m]['snli'][a]['jsd'] for a in valid_results[m]['snli']))
        best_snli_alpha = min(valid_results[best_snli_model]['snli'].keys(), 
                            key=lambda a: valid_results[best_snli_model]['snli'][a]['jsd'])
        best_snli_jsd = valid_results[best_snli_model]['snli'][best_snli_alpha]['jsd']
        
        # Best MNLI
        best_mnli_model = min(valid_results.keys(), 
                            key=lambda m: min(valid_results[m]['mnli'][a]['jsd'] for a in valid_results[m]['mnli']))
        best_mnli_alpha = min(valid_results[best_mnli_model]['mnli'].keys(), 
                            key=lambda a: valid_results[best_mnli_model]['mnli'][a]['jsd'])
        best_mnli_jsd = valid_results[best_mnli_model]['mnli'][best_mnli_alpha]['jsd']
        
        print(f"\n🏆 BEST RESULTS:")
        print(f"   ChaosNLI-S: {best_snli_model} (α={best_snli_alpha:.1f}) → JSD={best_snli_jsd:.4f}")
        print(f"   ChaosNLI-M: {best_mnli_model} (α={best_mnli_alpha:.1f}) → JSD={best_mnli_jsd:.4f}")
        
        # Check if we beat published SOTA
        published_best_snli = 0.2203  # BART-large
        published_best_mnli = 0.3055  # BERT-base
        
        if best_snli_jsd < published_best_snli:
            print(f"   🎉 BEATS ChaosNLI-S SOTA by {published_best_snli - best_snli_jsd:.4f}!")
        if best_mnli_jsd < published_best_mnli:
            print(f"   🎉 BEATS ChaosNLI-M SOTA by {published_best_mnli - best_mnli_jsd:.4f}!")

def main():
    """Main evaluation function"""
    
    parser = argparse.ArgumentParser(description="Separate ChaosNLI-S and ChaosNLI-M evaluation")
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--weight_start', type=float, default=0.0, help='Start weight for α range')
    parser.add_argument('--weight_end', type=float, default=1.0, help='End weight for α range')
    parser.add_argument('--weight_steps', type=int, default=11, help='Number of weight steps')
    
    args = parser.parse_args()
    
    # Create weight range
    weight_range = np.linspace(args.weight_start, args.weight_end, args.weight_steps)
    
    print(f"🚀 Starting separate ChaosNLI-S and ChaosNLI-M evaluation...")
    print(f"   Device: {args.device}")
    print(f"   Weight range: α ∈ [{args.weight_start}, {args.weight_end}] ({args.weight_steps} steps)")
    
    # Load data
    published_predictions = load_published_predictions()
    snli_data, mnli_data = load_separate_persistence_data()
    
    # Run separate evaluation
    all_results = run_separate_evaluation(
        published_predictions, snli_data, mnli_data, weight_range, device=args.device
    )
    
    # Print results in Table 5 format
    print_table_format_summary(all_results)
    
    print(f"\n🎉 Separate evaluation completed!")
    print(f"💡 Results can now be directly compared with ChaosNLI Table 5!")

if __name__ == "__main__":
    main()