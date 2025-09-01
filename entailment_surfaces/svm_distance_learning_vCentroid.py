import numpy as np
import pandas as pd
import torch
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances
import sys
import os
import json
from datetime import datetime

# Add project paths
sys.path.append('MSc_Topology_Codebase/entailment_surfaces')

def load_snli_data(data_path, sample_size=None, random_state=42):
    """Load SNLI data from preprocessed torch file"""
    print(f"Loading SNLI data from {data_path}")
    data = torch.load(data_path, weights_only=False)
    
    if sample_size:
        np.random.seed(random_state)
        total_samples = len(data['labels'])
        indices = np.random.choice(total_samples, sample_size, replace=False)
        
        # Use list comprehension for indexing
        data = {
            'premise_embeddings': torch.stack([data['premise_embeddings'][i] for i in indices]),
            'hypothesis_embeddings': torch.stack([data['hypothesis_embeddings'][i] for i in indices]),
            'labels': [data['labels'][i] for i in indices]  # Keep labels as list since they're strings
        }
    
    print(f"Loaded {len(data['labels'])} samples")
    return data

def generate_lattice_containment_embeddings(premise_embeddings, hypothesis_embeddings):
    """Generate lattice containment embedding space for each premise-hypothesis pair"""
    print("Generating lattice containment embedding space")
    embeddings = []
    epsilon = 1e-8
    
    for i, (p_emb, h_emb) in enumerate(zip(premise_embeddings, hypothesis_embeddings)):
        if i % 10000 == 0:
            print(f"Processing sample {i}")
        
        # Convert to tensors if needed
        if not torch.is_tensor(p_emb):
            p_emb = torch.tensor(p_emb)
        if not torch.is_tensor(h_emb):
            h_emb = torch.tensor(h_emb)
        
        # Apply lattice containment formula: (P * H) / (|P| + |H| + epsilon)
        lattice_embedding = (p_emb * h_emb) / (torch.abs(p_emb) + torch.abs(h_emb) + epsilon)
        embeddings.append(lattice_embedding.cpu().numpy())
    
    return np.array(embeddings)

def prepare_labels(labels):
    """Convert labels to numerical format"""
    label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    
    if torch.is_tensor(labels):
        # Assuming labels are already numerical 0,1,2
        return labels.numpy()
    else:
        # If labels are strings, map them
        return np.array([label_map.get(label, label) for label in labels])

def compute_class_centroids(embeddings, labels):
    """Compute centroids for each class in lattice containment space"""
    print("Computing class centroids")
    
    class_names = ['Entailment', 'Neutral', 'Contradiction']
    centroids = {}
    
    for class_idx in range(3):
        class_embeddings = embeddings[labels == class_idx]
        centroid = np.mean(class_embeddings, axis=0)
        centroids[class_idx] = centroid
        
        print(f"{class_names[class_idx]} Centroid:")
        print(f"  Shape: {centroid.shape}")
        print(f"  Norm: {np.linalg.norm(centroid):.4f}")
        print(f"  Sample count: {len(class_embeddings)}")
    
    return centroids

def compute_centroid_distances(embeddings, centroids, distance_metric='cosine'):
    """Compute distances from each sample to all class centroids"""
    print(f"Computing {distance_metric} distances to centroids")
    
    n_samples = len(embeddings)
    centroid_features = np.zeros((n_samples, 3))  # 3 distances per sample
    
    for i, embedding in enumerate(embeddings):
        if i % 10000 == 0:
            print(f"Processing sample {i}")
        
        for class_idx in range(3):
            if distance_metric == 'cosine':
                # Cosine distance between embedding and centroid
                distance = cosine_distances([embedding], [centroids[class_idx]])[0, 0]
            elif distance_metric == 'euclidean':
                # Euclidean distance
                distance = np.linalg.norm(embedding - centroids[class_idx])
            else:
                raise ValueError(f"Unknown distance metric: {distance_metric}")
            
            centroid_features[i, class_idx] = distance
    
    return centroid_features

def analyze_centroid_features(features, labels, distance_metric):
    """Analyze distribution of centroid distance features by class"""
    print("\n" + "="*60)
    print(f"CENTROID DISTANCE FEATURES ANALYSIS ({distance_metric.upper()})")
    print("="*60)
    
    class_names = ['Entailment', 'Neutral', 'Contradiction']
    feature_names = ['Dist_to_E', 'Dist_to_N', 'Dist_to_C']
    
    print(f"Feature shape: {features.shape}")
    print(f"Total samples: {len(features)}")
    
    for class_idx in range(3):
        class_features = features[labels == class_idx]
        print(f"\n{class_names[class_idx]} Class (n={len(class_features)}):")
        
        for feat_idx, feat_name in enumerate(feature_names):
            values = class_features[:, feat_idx]
            print(f"  {feat_name}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

def train_svm_centroid(X, y, test_size=0.2, random_state=42):
    """Train SVM on centroid distance features"""
    print("Splitting data into train/test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print("Scaling features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try both RBF and linear kernels
    kernels = ['rbf', 'linear']
    results = {}
    
    for kernel in kernels:
        print(f"\nTraining SVM with {kernel} kernel")
        svm = SVC(kernel=kernel, probability=True, random_state=random_state)
        svm.fit(X_train_scaled, y_train)
        
        train_score = svm.score(X_train_scaled, y_train)
        test_score = svm.score(X_test_scaled, y_test)
        
        print(f"SVM {kernel.upper()} Training Accuracy: {train_score:.4f}")
        print(f"SVM {kernel.upper()} Test Accuracy: {test_score:.4f}")
        
        results[kernel] = {
            'model': svm,
            'scaler': scaler,
            'train_score': train_score,
            'test_score': test_score,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test
        }
    
    # Return the best performing model
    best_kernel = max(results.keys(), key=lambda k: results[k]['test_score'])
    print(f"\nBest kernel: {best_kernel} (test accuracy: {results[best_kernel]['test_score']:.4f})")
    
    return results[best_kernel], results

def validate_svm_centroid(svm_result, distance_metric, sample_size):
    """Comprehensive validation of SVM centroid approach"""
    svm = svm_result['model']
    scaler = svm_result['scaler']
    X_test_scaled = svm_result['X_test_scaled']
    y_test = svm_result['y_test']
    
    print("\n" + "="*60)
    print("SVM CENTROID VALIDATION RESULTS")
    print("="*60)
    
    y_pred = svm.predict(X_test_scaled)
    
    # Calculate test accuracy manually to verify
    manual_accuracy = np.mean(y_pred == y_test)
    svm_accuracy = svm.score(X_test_scaled, y_test)
    
    print(f"Manual Test Accuracy: {manual_accuracy:.4f}")
    print(f"SVM Test Accuracy: {svm_accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Entailment', 'Neutral', 'Contradiction']))
    
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("                Predicted")
    print("               E   N   C")
    class_names = ['E', 'N', 'C']
    for i, (actual_class, row) in enumerate(zip(class_names, conf_matrix)):
        print(f"Actual {actual_class} {row[0]:4d} {row[1]:3d} {row[2]:3d}")
    
    # Extract decision function values
    decision_values = svm.decision_function(X_test_scaled)
    
    print(f"\nDecision Function Shape: {decision_values.shape}")
    print(f"Decision Function Stats:")
    print(f"  Mean: {np.mean(decision_values, axis=0)}")
    print(f"  Std: {np.std(decision_values, axis=0)}")
    print(f"  Min: {np.min(decision_values, axis=0)}")
    print(f"  Max: {np.max(decision_values, axis=0)}")
    
    # Save results
    save_centroid_results(svm_result, decision_values, y_pred, distance_metric, sample_size)
    
    return decision_values, y_pred

def save_centroid_results(svm_result, decision_values, y_pred, distance_metric, sample_size):
    """Save centroid SVM validation results to file"""
    os.makedirs('entailment_surfaces/svm_validation_results', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    svm = svm_result['model']
    X_test_scaled = svm_result['X_test_scaled']
    y_test = svm_result['y_test']
    
    test_score = svm.score(X_test_scaled, y_test)
    class_names = ['Entailment', 'Neutral', 'Contradiction']
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Prepare results dictionary
    results = {
        'timestamp': timestamp,
        'experiment_config': {
            'approach': 'centroid_distances',
            'sample_size': sample_size,
            'embedding_space': 'lattice_containment',
            'distance_metric': distance_metric,
            'svm_kernel': svm.kernel,
            'feature_dimensions': 3,
            'test_size': 0.2,
            'random_state': 42
        },
        'performance_metrics': {
            'test_accuracy': float(test_score),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        },
        'decision_function_stats': {
            'shape': list(decision_values.shape),
            'mean_per_class': np.mean(decision_values, axis=0).tolist(),
            'std_per_class': np.std(decision_values, axis=0).tolist(),
            'min_per_class': np.min(decision_values, axis=0).tolist(),
            'max_per_class': np.max(decision_values, axis=0).tolist()
        }
    }
    
    # Save as JSON
    json_filename = f'entailment_surfaces/svm_validation_results/svm_centroid_{distance_metric}_{timestamp}_n{sample_size}.json'
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as readable text file
    txt_filename = f'entailment_surfaces/svm_validation_results/svm_centroid_{distance_metric}_{timestamp}_n{sample_size}.txt'
    with open(txt_filename, 'w') as f:
        f.write("SVM CENTROID VALIDATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Approach: Centroid Distances\n")
        f.write(f"Sample Size: {sample_size}\n")
        f.write(f"Embedding Space: lattice_containment\n")
        f.write(f"Distance Metric: {distance_metric}\n")
        f.write(f"SVM Kernel: {svm.kernel}\n")
        f.write(f"Feature Dimensions: 3\n\n")
        
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Test Accuracy: {test_score:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write("                Predicted\n")
        f.write("               E   N   C\n")
        for i, (actual_class, row) in enumerate(zip(['E', 'N', 'C'], conf_matrix)):
            f.write(f"Actual {actual_class} {row[0]:4d} {row[1]:3d} {row[2]:3d}\n")
        f.write("\n")
        
        f.write("Classification Report:\n")
        for class_name in class_names:
            metrics = class_report[class_name]
            f.write(f"{class_name:12s}: precision={metrics['precision']:.3f}, "
                   f"recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}\n")
    
    print(f"\nResults saved to:")
    print(f"  JSON: {json_filename}")
    print(f"  TXT:  {txt_filename}")

def main():
    # Configuration
    data_path = 'data/processed/snli_full_standard_SBERT.pt'
    sample_size = 50000  # Start with same size for comparison
    distance_metric = 'euclidean'  # Try cosine first based on Phase 1 success
    
    print("Starting SVM Centroid Distance Training Pipeline")
    print("="*60)
    
    # Load data
    data = load_snli_data(data_path, sample_size)
    
    # Generate lattice containment embedding space
    embeddings = generate_lattice_containment_embeddings(
        data['premise_embeddings'], 
        data['hypothesis_embeddings']
    )
    
    # Prepare labels
    labels = prepare_labels(data['labels'])
    
    # Compute class centroids
    centroids = compute_class_centroids(embeddings, labels)
    
    # Compute centroid distance features
    X = compute_centroid_distances(embeddings, centroids, distance_metric)
    
    # Analyze centroid features
    analyze_centroid_features(X, labels, distance_metric)
    
    # Train SVM on centroid features
    best_result, all_results = train_svm_centroid(X, labels)
    
    # Validate best SVM
    decision_values, y_pred = validate_svm_centroid(best_result, distance_metric, sample_size)
    
    print("\n" + "="*60)
    print("SVM CENTROID TRAINING COMPLETE")
    print("="*60)
    
    return best_result, centroids, decision_values

if __name__ == "__main__":
    svm_result, centroids, decision_values = main()