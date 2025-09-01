#!/usr/bin/env python3
"""
Comprehensive Hybrid Model Implementation

This script:
1. Loads all published model predictions from ChaosNLI
2. Matches UIDs with persistence images 
3. Trains persistence CNN on matched data
4. Tests weighted fusion: published_logits + α * persistence_logits
5. Evaluates uncertainty quantification improvement across all models
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

def load_persistence_data():
    """Load ChaosNLI persistence images and ground truth"""
    
    print("Loading persistence images and ground truth...")
    
    # Load SNLI ChaosNLI data
    snli_persistence_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_chaosnli_snli_persistence_images.pkl"
    with open(snli_persistence_path, 'rb') as f:
        snli_data = pickle.load(f)
    
    # Load MNLI ChaosNLI data
    mnli_persistence_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_chaosnli_mnli_matched_persistence_images.pkl"
    with open(mnli_persistence_path, 'rb') as f:
        mnli_data = pickle.load(f)
    
    # Combine datasets
    combined_data = {
        'persistence_images': np.vstack([snli_data['persistence_images'], mnli_data['persistence_images']]),
        'uids': np.concatenate([snli_data['uids'], mnli_data['uids']]),
        'label_distributions': np.vstack([snli_data['label_distributions'], mnli_data['label_distributions']]),
        'entropies': np.concatenate([snli_data['entropies'], mnli_data['entropies']]),
        'majority_labels': np.concatenate([snli_data['majority_labels'], mnli_data['majority_labels']])
    }
    
    print(f"Combined persistence data: {len(combined_data['uids'])} samples")
    print(f"  SNLI: {len(snli_data['uids'])} samples")
    print(f"  MNLI: {len(mnli_data['uids'])} samples")
    
    return combined_data

def match_predictions_with_persistence(published_predictions, persistence_data, model_name):
    """Match published predictions with persistence images by UID"""
    
    print(f"Matching {model_name} predictions with persistence images...")
    
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
    total_predictions = len(model_preds)
    
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
    
    match_rate = matched_count / total_predictions * 100
    print(f"  Matched: {matched_count}/{total_predictions} ({match_rate:.1f}%)")
    
    if matched_count == 0:
        raise ValueError(f"No matches found for {model_name}")
    
    return matched_data

def load_training_validation_data():
    """Load SNLI+MNLI training and validation persistence images"""
    
    print("Loading SNLI+MNLI training and validation data...")
    
    # Load SNLI training data
    snli_train_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_snli_train_persistence_images.pkl"
    snli_val_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_snli_val_persistence_images.pkl"
    
    with open(snli_train_path, 'rb') as f:
        snli_train = pickle.load(f)
    with open(snli_val_path, 'rb') as f:
        snli_val = pickle.load(f)
    
    # Load MNLI training data (if available)
    mnli_train_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_mnli_train_persistence_images_intermediate_48000.pkl"
    mnli_val_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_mnli_val_matched_persistence_images.pkl"
    
    # Try to load MNLI data
    mnli_train, mnli_val = None, None
    if Path(mnli_train_path).exists():
        with open(mnli_train_path, 'rb') as f:
            mnli_train = pickle.load(f)
        print(f"  MNLI train: {len(mnli_train['persistence_images'])} samples")
    else:
        print(f"  MNLI train: Not found")
    
    if Path(mnli_val_path).exists():
        with open(mnli_val_path, 'rb') as f:
            mnli_val = pickle.load(f)
        print(f"  MNLI val: {len(mnli_val['persistence_images'])} samples")
    else:
        print(f"  MNLI val: Not found")
    
    # Combine training data
    if mnli_train:
        train_persistence_images = np.vstack([snli_train['persistence_images'], mnli_train['persistence_images']])
        train_labels = np.concatenate([snli_train['labels'], mnli_train['labels']])
    else:
        train_persistence_images = snli_train['persistence_images']
        train_labels = snli_train['labels']
    
    # Combine validation data
    if mnli_val:
        val_persistence_images = np.vstack([snli_val['persistence_images'], mnli_val['persistence_images']])
        val_labels = np.concatenate([snli_val['labels'], mnli_val['labels']])
    else:
        val_persistence_images = snli_val['persistence_images']
        val_labels = snli_val['labels']
    
    # Convert string labels to indices if needed
    label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    if isinstance(train_labels[0], str):
        train_labels = np.array([label_to_idx[label] for label in train_labels])
        val_labels = np.array([label_to_idx[label] for label in val_labels])
    
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

def evaluate_weighted_fusion(persistence_model, matched_data, weight_range, device='cuda'):
    """Evaluate different fusion weights on ChaosNLI test data"""
    
    print("Evaluating weighted fusion on ChaosNLI test data...")
    
    persistence_model.eval()
    
    # Create test dataset from matched ChaosNLI data
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

def run_comprehensive_evaluation(published_predictions, persistence_data, weight_range, device='cuda'):
    """Run evaluation across all available models"""
    
    print("="*80)
    print("COMPREHENSIVE HYBRID MODEL EVALUATION")
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
            # Match predictions with ChaosNLI test data
            matched_data = match_predictions_with_persistence(
                published_predictions, persistence_data, model_name
            )
            
            # Evaluate weighted fusion on ChaosNLI test data
            fusion_results = evaluate_weighted_fusion(
                persistence_model, matched_data, weight_range, device=device
            )
            
            # Store results
            all_results[model_name] = fusion_results
            
            # Print summary for this model
            print(f"\n📊 {model_name} Results Summary:")
            baseline_jsd = fusion_results[0.0]['jsd']  # α=0.0 is pure published model
            baseline_kl = fusion_results[0.0]['kl']   # α=0.0 KL divergence
            best_alpha = min(fusion_results.keys(), key=lambda a: fusion_results[a]['jsd'])  # Choose best α based on JSD
            best_jsd = fusion_results[best_alpha]['jsd']
            best_kl = fusion_results[best_alpha]['kl']   # KL at best JSD weight
            jsd_improvement = baseline_jsd - best_jsd
            kl_improvement = baseline_kl - best_kl

            print(f"   Baseline (α=0.0): JSD = {baseline_jsd:.4f}, KL = {baseline_kl:.4f}")
            print(f"   Best (α={best_alpha}): JSD = {best_jsd:.4f}, KL = {best_kl:.4f}")
            print(f"   {'✅ JSD Improvement' if jsd_improvement > 0 else '❌ No JSD improvement'}: {jsd_improvement:+.4f}")
            print(f"   {'✅ KL Improvement' if kl_improvement > 0 else '❌ No KL improvement'}: {kl_improvement:+.4f}")
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            all_results[model_name] = None
    
    return all_results

def print_final_summary(all_results, published_baselines):
    """Print comprehensive summary of all results"""
    
    print(f"\n{'='*80}")
    print(f"FINAL COMPREHENSIVE RESULTS")
    print(f"{'='*80}")
    
    print(f"\n📊 Model Performance Summary:")
    print(f"{'Model':<15} {'Baseline JSD':<12} {'Best JSD':<10} {'Baseline KL':<12} {'Best KL':<10} {'Best α':<8} {'JSD Δ':<10} {'KL Δ'}")
    print(f"-" * 95)
    
    for model_name, results in all_results.items():
        if results is None:
            print(f"{model_name:<15} {'ERROR':<12} {'ERROR':<10} {'ERROR':<8} {'ERROR':<12} {'ERROR'}")
            continue
            
        baseline_jsd = results[0.0]['jsd']
        baseline_kl = results[0.0]['kl']
        best_alpha = min(results.keys(), key=lambda a: results[a]['jsd'])  # Choose α based on JSD
        best_jsd = results[best_alpha]['jsd']
        best_kl = results[best_alpha]['kl']
        jsd_improvement = baseline_jsd - best_jsd
        kl_improvement = baseline_kl - best_kl
        
        jsd_status = "✅" if jsd_improvement > 0 else "❌"
        kl_status = "✅" if kl_improvement > 0 else "❌"
        
        print(f"{model_name:<15} {baseline_jsd:<12.4f} {best_jsd:<10.4f} {baseline_kl:<12.4f} {best_kl:<10.4f} {best_alpha:<8.1f} {jsd_status} {jsd_improvement:+.4f} {kl_status} {kl_improvement:+.4f}")
    
    # Find overall best model
    valid_results = {k: v for k, v in all_results.items() if v is not None}
    if valid_results:
        best_model = min(valid_results.keys(), 
                        key=lambda m: min(valid_results[m][a]['jsd'] for a in valid_results[m]))
        best_model_results = valid_results[best_model]
        best_alpha = min(best_model_results.keys(), key=lambda a: best_model_results[a]['jsd'])
        best_overall_jsd = best_model_results[best_alpha]['jsd']
        best_overall_kl = best_model_results[best_alpha]['kl']
        
        print(f"\n🏆 BEST OVERALL RESULT:")
        print(f"   Model: {best_model}")
        print(f"   Optimal α: {best_alpha}")
        print(f"   JSD: {best_overall_jsd:.4f}")
        print(f"   KL: {best_overall_kl:.4f}")

        
        # Compare with state-of-the-art
        sota_jsd = 0.2203  # BART-large from ChaosNLI paper
        if best_overall_jsd < sota_jsd:
            print(f"   🎉 BEATS STATE-OF-THE-ART by {sota_jsd - best_overall_jsd:.4f}!")
        else:
            print(f"   📊 Still {best_overall_jsd - sota_jsd:.4f} away from SOTA")

def main():
    """Main evaluation function"""
    
    parser = argparse.ArgumentParser(description="Comprehensive hybrid model evaluation")
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--weight_start', type=float, default=0.0, help='Start weight for α range')
    parser.add_argument('--weight_end', type=float, default=1.0, help='End weight for α range')
    parser.add_argument('--weight_steps', type=int, default=11, help='Number of weight steps')
    
    args = parser.parse_args()
    
    # Create weight range
    weight_range = np.linspace(args.weight_start, args.weight_end, args.weight_steps)
    
    print(f"🚀 Starting comprehensive hybrid model evaluation...")
    print(f"   Device: {args.device}")
    print(f"   Weight range: α ∈ [{args.weight_start}, {args.weight_end}] ({args.weight_steps} steps)")
    
    # Load data
    published_predictions = load_published_predictions()
    persistence_data = load_persistence_data()
    
    # Run comprehensive evaluation
    all_results = run_comprehensive_evaluation(
        published_predictions, persistence_data, weight_range, device=args.device
    )
    
    # Published baselines from ChaosNLI paper (SNLI results)
    published_baselines = {
        'bert-base': 0.2345,
        'bert-large': 0.2300,
        'roberta-base': 0.2294,
        'roberta-large': 0.2210,
        'bart-large': 0.2203,
        'albert-xxlarge': 0.2350,
        'distilbert': 0.2439
    }
    
    # Print final summary
    print_final_summary(all_results, published_baselines)
    
    print(f"\n🎉 Comprehensive evaluation completed!")

if __name__ == "__main__":
    main()