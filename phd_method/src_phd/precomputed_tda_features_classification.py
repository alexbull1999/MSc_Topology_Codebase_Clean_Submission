"""
Simple Enhanced Classification Pipeline
Loads precomputed topological features + does SBERT baseline + hybrid + ChaosNLI
"""

import numpy as np
import torch
import pickle
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import gc

# Import your existing functions
from point_cloud_clustering_test import (
    TopologicalClassifier, 
    train_pytorch_classifier,
    SeparateModelPointCloudGenerator
)

def load_precomputed_topological_features(train_path, val_path, sample_train=None):
    """Load precomputed topological features"""
    print("Loading precomputed topological features...")
    
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)
    
    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = val_data['features']
    y_val = val_data['labels']
    
    # Convert string labels to indices
    label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    if isinstance(y_train[0], str):
        y_train = np.array([label_to_idx[label] for label in y_train])
        y_val = np.array([label_to_idx[label] for label in y_val])
    
    # Sample training data if requested - BALANCED sampling
    if sample_train:
        indices = []
        for class_idx in [0, 1, 2]:
            class_indices = np.where(y_train == class_idx)[0]
            selected = np.random.choice(class_indices, min(sample_train, len(class_indices)), replace=False)
            indices.extend(selected)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        # Verify we have all classes
        unique_classes = np.unique(y_train)
        print(f"Classes in sampled training data: {unique_classes}")
        if len(unique_classes) != 3:
            print(f"WARNING: Only {len(unique_classes)} classes found after sampling!")
    
    print(f"Topological features: Train {X_train.shape}, Val {X_val.shape}")
    return X_train, y_train, X_val, y_val


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

def run_classifiers(X_train, y_train, X_val, y_val, feature_type=""):
    """Run all classifiers and return results"""
    print(f"\nRunning {feature_type} classification...")
    
    # Check class distribution
    unique_classes, counts = np.unique(y_train, return_counts=True)
    print(f"Training set class distribution: {dict(zip(unique_classes, counts))}")
    
    if len(unique_classes) < 2:
        raise ValueError(f"Training data only contains {len(unique_classes)} class(es): {unique_classes}")
    
    # Memory-efficient standardization
    X_train_scaled, mean, std = memory_efficient_standardize(X_train.astype(np.float32))
    X_val_scaled = apply_standardization(X_val.astype(np.float32), mean, std)
    
    # Force garbage collection
    gc.collect()
    
    results = {}

    # Sklearn classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_val_scaled)
        y_proba = clf.predict_proba(X_val_scaled)
        
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        
        results[name] = {
            'accuracy': accuracy,
            'f1_macro': f1,
            'model': clf,
            'mean': mean,
            'std': std,
            'probabilities': y_proba
        }
        print(f"{name}: {accuracy:.3f}")
    
    # PyTorch model (using your existing function)
    pytorch_model, pytorch_acc = train_pytorch_classifier(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Get PyTorch probabilities
    pytorch_model.eval()
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        logits = pytorch_model(X_val_tensor)
        pytorch_proba = torch.softmax(logits, dim=1).numpy()
    
    results['PyTorch NN'] = {
        'accuracy': pytorch_acc,
        'f1_macro': f1_score(y_val, np.argmax(pytorch_proba, axis=1), average='macro'),
        'model': pytorch_model,
        'mean': mean,
        'std': std,
        'probabilities': pytorch_proba
    }
    print(f"PyTorch NN: {pytorch_acc:.3f}")
    
    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
    results['best_model'] = best_model
    results['best_accuracy'] = results[best_model]['accuracy']
    
    return results

def create_hybrid_features(topo_features, sbert_features):
    """Simply concatenate topological and SBERT features"""
    print(f"Creating hybrid features: {topo_features.shape} + {sbert_features.shape}")
    hybrid = np.concatenate([topo_features, sbert_features], axis=1)
    print(f"Hybrid features: {hybrid.shape}")
    return hybrid

def evaluate_chaosnli(model, mean, std, chaosnli_path, model_name):
    """Evaluate model on ChaosNLI uncertainty quantification"""
    print(f"\nEvaluating {model_name} on ChaosNLI...")
    
    # Load ChaosNLI data
    with open(chaosnli_path, 'rb') as f:
        data = pickle.load(f)
    
    X_chaos = data['features'].astype(np.float32)
    
    # Check for human distributions
    if 'label_distributions' in data:
        human_distributions = data['label_distributions']
    else:
        print("ERROR: human distributions not found in ChaosNLI data...")
        raise ValueError("Missing label_distributions in ChaosNLI data")
    
    # Scale features using saved parameters
    X_chaos_scaled = apply_standardization(X_chaos, mean, std)
    
    # Get model probabilities - handle both sklearn and PyTorch models
    if hasattr(model, 'predict_proba'):  # Sklearn model
        model_probs = model.predict_proba(X_chaos_scaled)
    else:  # PyTorch model
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_chaos_scaled)
            logits = model(X_tensor)
            model_probs = torch.softmax(logits, dim=1).numpy()
    
    # Compute JSD and KL divergence
    jsd_scores = []
    kl_scores = []
    
    for i in range(len(model_probs)):
        model_dist = model_probs[i] / model_probs[i].sum()
        human_dist = human_distributions[i] / human_distributions[i].sum()
        
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
    human_preds = np.argmax(human_distributions, axis=1)
    traditional_acc = np.mean(model_preds == human_preds)
    
    # Compare with baselines
    random_jsd = 0.32
    random_kl = 0.54
    roberta_jsd = 0.22
    bart_kl = 0.47
    
    print(f"ChaosNLI Results for {model_name}:")
    print(f"  JSD: {avg_jsd:.4f} (vs random: {random_jsd:.4f}, vs RoBERTa: {roberta_jsd:.4f})")
    print(f"  KL:  {avg_kl:.4f} (vs random: {random_kl:.4f}, vs BART: {bart_kl:.4f})")
    print(f"  Accuracy: {traditional_acc:.4f}")
    print(f"  Beats random: JSD={'YES' if avg_jsd < random_jsd else 'NO'}, KL={'YES' if avg_kl < random_kl else 'NO'}")
    
    return {
        'jsd': avg_jsd,
        'kl': avg_kl,
        'accuracy': traditional_acc,
        'beats_random_jsd': avg_jsd < random_jsd,
        'beats_random_kl': avg_kl < random_kl
    }

def compute_sbert_baseline_features(data_path, sample_size=None):
    """Compute SBERT baseline features directly without loading heavy models"""
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
    
    # Sample if requested - BALANCED sampling per class
    if sample_size:
        for class_name in samples_by_class:
            if len(samples_by_class[class_name]) > sample_size:
                np.random.shuffle(samples_by_class[class_name])
                samples_by_class[class_name] = samples_by_class[class_name][:sample_size]
    
    # Extract SBERT features in batches to avoid OOM
    all_features = []
    all_labels = []
    label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    
    batch_size = 1000  # Process in smaller batches
    
    for class_name, samples in samples_by_class.items():
        print(f"Processing {class_name}: {len(samples)} samples")
        
        # Process in batches
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
            
            # Print progress
            if i + batch_size < len(samples):
                print(f"  Processed {i + batch_size}/{len(samples)} {class_name} samples")
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    # Verify class distribution
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"SBERT features: {X.shape}, class distribution: {dict(zip(unique_classes, counts))}")
    
    return X, y

def run_snli_experiment(topo_train_path, topo_val_path, sbert_train_path, sbert_val_path, sample_train=None):
    """Run SNLI experiment: topological + SBERT + hybrid classification"""
    
    print("=" * 80)
    print("SNLI CLASSIFICATION EXPERIMENT")
    print("=" * 80)
    
    # 1. Load precomputed topological features
    X_train_topo, y_train_topo, X_val_topo, y_val_topo = load_precomputed_topological_features(
        topo_train_path, topo_val_path, sample_train
    )
    
    # 2. Compute SBERT baseline features
    X_train_sbert, y_train_sbert = compute_sbert_baseline_features(
        sbert_train_path, sample_size=sample_train
    )
    X_val_sbert, y_val_sbert = compute_sbert_baseline_features(
        sbert_val_path, sample_size=None  # Use all validation data
    )

    # 3. Align datasets by taking samples that exist in both
    # Create sets of indices that have the same labels
    print("Aligning topological and SBERT datasets...")
    
    # For training data - match by taking minimum per class
    min_train_samples = []
    for class_idx in [0, 1, 2]:
        topo_class_count = np.sum(y_train_topo == class_idx)
        sbert_class_count = np.sum(y_train_sbert == class_idx)
        min_count = min(topo_class_count, sbert_class_count)
        min_train_samples.append(min_count)
        print(f"Class {class_idx}: topo={topo_class_count}, sbert={sbert_class_count}, using={min_count}")
    
    # Sample aligned training data
    topo_indices = []
    sbert_indices = []
    
    for class_idx in [0, 1, 2]:
        topo_class_indices = np.where(y_train_topo == class_idx)[0]
        sbert_class_indices = np.where(y_train_sbert == class_idx)[0]
        
        sample_count = min_train_samples[class_idx]
        topo_selected = np.random.choice(topo_class_indices, sample_count, replace=False)
        sbert_selected = np.random.choice(sbert_class_indices, sample_count, replace=False)
        
        topo_indices.extend(topo_selected)
        sbert_indices.extend(sbert_selected)
    
    # Shuffle to mix classes
    combined_indices = list(zip(topo_indices, sbert_indices))
    np.random.shuffle(combined_indices)
    topo_indices, sbert_indices = zip(*combined_indices)
    
    X_train_topo_aligned = X_train_topo[list(topo_indices)]
    y_train_aligned = y_train_topo[list(topo_indices)]
    X_train_sbert_aligned = X_train_sbert[list(sbert_indices)]
    
    # For validation - just take minimum
    min_val = min(len(X_val_topo), len(X_val_sbert))
    X_val_topo_aligned = X_val_topo[:min_val]
    y_val_aligned = y_val_topo[:min_val]
    X_val_sbert_aligned = X_val_sbert[:min_val]
    
    print(f"Final aligned datasets:")
    print(f"  Train: topo={X_train_topo_aligned.shape}, sbert={X_train_sbert_aligned.shape}")
    print(f"  Val: topo={X_val_topo_aligned.shape}, sbert={X_val_sbert_aligned.shape}")
    
    # 4. Create hybrid features
    X_train_hybrid = create_hybrid_features(X_train_topo_aligned, X_train_sbert_aligned)
    X_val_hybrid = create_hybrid_features(X_val_topo_aligned, X_val_sbert_aligned)
    
    # 5. Run classification experiments
    topo_results = run_classifiers(X_train_topo_aligned, y_train_aligned, X_val_topo_aligned, y_val_aligned, "Topological")
    del X_train_topo_aligned, X_val_topo_aligned
    gc.collect()
    sbert_results = run_classifiers(X_train_sbert_aligned, y_train_aligned, X_val_sbert_aligned, y_val_aligned, "SBERT Baseline")
    # Force cleanup before hybrid
    del X_train_sbert_aligned, X_val_sbert_aligned
    gc.collect()
    hybrid_results = run_classifiers(X_train_hybrid, y_train_aligned, X_val_hybrid, y_val_aligned, "Hybrid")
    
    
    # 6. Print comparison
    print("\n" + "=" * 60)
    print("SNLI CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"Topological (21 features):     {topo_results['best_accuracy']:.3f} ({topo_results['best_model']})")
    print(f"SBERT Baseline (3072 features): {sbert_results['best_accuracy']:.3f} ({sbert_results['best_model']})")
    print(f"Hybrid (3093 features):        {hybrid_results['best_accuracy']:.3f} ({hybrid_results['best_model']})")
    
    improvement_vs_sbert = hybrid_results['best_accuracy'] - sbert_results['best_accuracy']
    improvement_vs_topo = hybrid_results['best_accuracy'] - topo_results['best_accuracy']
    
    print(f"\nHybrid improvements:")
    print(f"  vs SBERT: {improvement_vs_sbert:+.3f}")
    print(f"  vs Topological: {improvement_vs_topo:+.3f}")
    
    return {
        'topological': topo_results,
        'sbert': sbert_results,
        'hybrid': hybrid_results
    }

def run_chaosnli_experiment(snli_results, chaosnli_path):
    """Run ChaosNLI uncertainty quantification experiment"""
    
    print("\n" + "=" * 80)
    print("CHAOSNLI UNCERTAINTY QUANTIFICATION")
    print("=" * 80)
    
    # Only evaluate topological model as requested
    topo_results = snli_results['topological']
    best_model = topo_results['PyTorch NN']['model']
    mean = topo_results['PyTorch NN']['mean']
    std = topo_results['PyTorch NN']['std']
    
    chaos_result = evaluate_chaosnli(
        best_model, mean, std, chaosnli_path, "Topological PyTorch NN"
    )
    
    print("\n" + "=" * 60)
    print("CHAOSNLI SUMMARY")
    print("=" * 60)
    
    jsd_status = "YES" if chaos_result['beats_random_jsd'] else "NO"
    kl_status = "YES" if chaos_result['beats_random_kl'] else "NO"
    print(f"Topological: JSD={chaos_result['jsd']:.3f} {jsd_status}, KL={chaos_result['kl']:.3f} {kl_status}")
    
    return {'topological': chaos_result}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Classification Pipeline')
    parser.add_argument('--experiment', choices=['snli', 'chaosnli', 'both'], 
                       default='snli', help='Which experiment to run')
    
    args = parser.parse_args()
    
    # Configure your paths
    topo_train_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_snli_train_tda_features_NO_TOKEN_THRESHOLD.pkl"
    topo_val_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_snli_val_tda_features_NO_TOKEN_THRESHOLD.pkl"
    sbert_train_path = "/vol/bitbucket/ahb24/tda_entailment_new/snli_train_sbert_tokens.pkl"
    sbert_val_path = "/vol/bitbucket/ahb24/tda_entailment_new/snli_val_sbert_tokens.pkl"
    chaosnli_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_chaosnli_snli_tda_features_NO_TOKEN_THRESHOLD.pkl"
    
    # Run SNLI experiment
    if args.experiment in ['snli', 'both']:
        snli_results = run_snli_experiment(
            topo_train_path=topo_train_path,
            topo_val_path=topo_val_path,
            sbert_train_path=sbert_train_path,
            sbert_val_path=sbert_val_path,
            sample_train=None
        )
    
    # Run ChaosNLI experiment
    if args.experiment in ['chaosnli', 'both']:
        if args.experiment == 'chaosnli':
            # Need to run SNLI first to get models
            snli_results = run_snli_experiment(
                topo_train_path, topo_val_path, sbert_train_path, sbert_val_path, None
            )
        
        chaosnli_results = run_chaosnli_experiment(snli_results, chaosnli_path)
    
    print("\nAll experiments completed!")