import numpy as np
import pandas as pd
import torch
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import sys
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.append('MSc_Topology_Codebase/entailment_surfaces')

def flush_output():
    """Force output to appear immediately in SLURM"""
    sys.stdout.flush()
    sys.stderr.flush()

def load_snli_data(data_path, sample_size=None, random_state=42):
    """Load SNLI data from preprocessed torch file"""
    print(f"Loading SNLI data from {data_path}")
    data = torch.load(data_path, weights_only=False)
    
    if sample_size:
        np.random.seed(random_state)
        indices = np.random.choice(len(data['labels']), sample_size, replace=False)
        
         # Use list comprehension for indexing instead of tensor indexing
        data = {
            'premise_embeddings': torch.stack([data['premise_embeddings'][i] for i in indices]),
            'hypothesis_embeddings': torch.stack([data['hypothesis_embeddings'][i] for i in indices]),
            'labels': [data['labels'][i] for i in indices]
        }
    
    
    print(f"Loaded {len(data['labels'])} samples")
    return data


def generate_lattice_containment_embeddings(premise_embeddings, hypothesis_embeddings, batch_size=1000):
    """Generate lattice containment embedding space for each premise-hypothesis pair"""
    print("Generating lattice containment embedding space")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    epsilon = 1e-8
    total_samples = len(premise_embeddings)
    print(f"Processing {total_samples} samples in batches of {batch_size}")
    
    # Process in batches to avoid memory issues
    all_lattice_embeddings = []
    
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        print(f"Processing batch {i//batch_size + 1}/{(total_samples-1)//batch_size + 1}")
        
        # Get batch
        premise_batch = premise_embeddings[i:end_idx]
        hypothesis_batch = hypothesis_embeddings[i:end_idx]
        
        # Convert to tensors if needed
        if not torch.is_tensor(premise_batch):
            premise_batch = torch.tensor(premise_batch)
        if not torch.is_tensor(hypothesis_batch):
            hypothesis_batch = torch.tensor(hypothesis_batch)
        
        # Move to device
        premise_batch = premise_batch.to(device)
        hypothesis_batch = hypothesis_batch.to(device)
        
        # Compute lattice embeddings for this batch
        with torch.no_grad():
            lattice_batch = (premise_batch * hypothesis_batch) / (
                torch.abs(premise_batch) + torch.abs(hypothesis_batch) + epsilon
            )
            
            # Move back to CPU and store
            all_lattice_embeddings.append(lattice_batch.cpu().numpy())
        
        # Clear GPU memory
        del premise_batch, hypothesis_batch, lattice_batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Concatenate all batches
    return np.concatenate(all_lattice_embeddings, axis=0)

def prepare_labels(labels):
    """Convert labels to numerical format"""
    label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    
    if torch.is_tensor(labels):
        # Assuming labels are already numerical 0,1,2
        return labels.numpy()
    else:
        # If labels are strings, map them
        return np.array([label_map.get(label, label) for label in labels])


def create_binary_ec_dataset(X, y):
    """Create binary dataset with only Entailment (0) and Contradiction (2) samples"""
    print("Creating binary Entailment vs Contradiction dataset")
    
    # Filter out neutral samples (label 1)
    binary_mask = (y == 0) | (y == 2)
    X_binary = X[binary_mask]
    y_binary = y[binary_mask]
    
    # Convert contradiction labels from 2 to 1 for binary classification
    y_binary = np.where(y_binary == 2, 1, y_binary)
    
    print(f"Binary dataset: {len(X_binary)} samples")
    print(f"  Entailment (0): {np.sum(y_binary == 0)} samples")
    print(f"  Contradiction (1): {np.sum(y_binary == 1)} samples")
    
    return X_binary, y_binary


def train_binary_svm(X_train_binary, y_train_binary, random_state=42):
    """Train binary SVM to separate Entailment from Contradiction"""
    print("\n" + "="*60)
    print("STEP 1: TRAINING BINARY ENTAILMENT vs CONTRADICTION SVM")
    print("="*60)
    
    print("Scaling features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_binary)
    
    print("Setting up GridSearchCV for binary SVM optimization")
    
    # Define parameter grid
    param_grid = {
        'C': [1], # original tests: 'C': [0.1, 1, 10, 100] (1 was best)
        'gamma': ['scale'] #original tests: 'gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }
    
    print(f"Parameter grid: {param_grid}")
    print(f"Total combinations to test: {len(param_grid['C']) * len(param_grid['gamma'])}")
    
    # Create base SVM
    base_svm = SVC(kernel='rbf', probability=True, random_state=random_state)
    
    # Setup GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_svm,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    print("Starting GridSearchCV for binary classification...")
    grid_search.fit(X_train_scaled, y_train_binary)
    
    print("Binary GridSearchCV completed!")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Get the best model
    best_binary_svm = grid_search.best_estimator_
    
    # Evaluate performance
    train_score = best_binary_svm.score(X_train_scaled, y_train_binary)
    
    print(f"Binary SVM Training Accuracy: {train_score:.4f}")
    
    return best_binary_svm, scaler, grid_search


def compute_entailment_scores(binary_svm, scaler, X_data, y_data, dataset_name=""):
    """Compute continuous entailment scores for all samples using binary SVM"""
    print("\n" + "="*60)
    print(f"STEP 2: COMPUTING CONTINUOUS ENTAILMENT for {dataset_name.upper()} SCORES")
    print("="*60)
    
    # Scale the full dataset
    X_scaled = scaler.transform(X_data)
    
    # Get decision function values (signed distances from hyperplane)
    entailment_scores = binary_svm.decision_function(X_scaled)
    
    print(f"Computed entailment scores for {len(entailment_scores)} samples")
    print(f"Score range: [{np.min(entailment_scores):.3f}, {np.max(entailment_scores):.3f}]")
    
    # Analyze score distributions by class
    class_names = ['Entailment', 'Neutral', 'Contradiction']
    
    print("\nEntailment Score Distribution by Class:")
    for class_idx in range(3):
        class_scores = entailment_scores[y_data == class_idx]
        print(f"  {class_names[class_idx]:12s}: mean={np.mean(class_scores):6.3f}, "
              f"std={np.std(class_scores):5.3f}, "
              f"range=[{np.min(class_scores):6.3f}, {np.max(class_scores):6.3f}]")
    
    return entailment_scores


def optimize_thresholds(train_scores, y_train):
    """Find optimal thresholds for 3-way classification based on entailment scores"""
    print("\n" + "="*60)
    print("STEP 3: OPTIMIZING CLASSIFICATION THRESHOLDS")
    print("="*60)
    
    print("Searching for optimal thresholds...")
    
    # Grid search over threshold values
    score_min, score_max = np.min(train_scores), np.max(train_scores)
    entailment_thresholds = np.linspace(score_min, score_max, 50)
    contradiction_thresholds = np.linspace(score_min, score_max, 50)
    
    best_accuracy = 0
    best_thresholds = None
    best_predictions = None
    
    for e_thresh in entailment_thresholds:
        for c_thresh in contradiction_thresholds:
            if e_thresh >= c_thresh:  # Ensure logical ordering
                continue
                
            # Apply thresholds
            y_pred = apply_thresholds(train_scores, e_thresh, c_thresh)
            
            # Calculate accuracy
            accuracy = np.mean(y_pred == y_train)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_thresholds = (e_thresh, c_thresh)
    
    print(f"Best threshold optimization accuracy: {best_accuracy:.4f}")
    print(f"Best thresholds: Entailment < {best_thresholds[0]:.3f}, Contradiction > {best_thresholds[1]:.3f}")
    
    return best_thresholds


def apply_thresholds(scores, entailment_threshold, contradiction_threshold):
    """Apply thresholds to convert continuous scores to discrete classes"""
    predictions = np.ones(len(scores))  # Default to neutral (1)
    
    # Low scores -> Entailment (0)
    predictions[scores < entailment_threshold] = 0
    
    # High scores -> Contradiction (2)
    predictions[scores > contradiction_threshold] = 2
    
    return predictions.astype(int)

def validate_hierarchical_classifier(entailment_scores, y_full, thresholds, sample_size):
    """Comprehensive validation of hierarchical classification approach"""
    print("\n" + "="*60)
    print("HIERARCHICAL CLASSIFIER VALIDATION RESULTS")
    print("="*60)
    
    # Apply thresholds to get final predictions
    y_pred = apply_thresholds(entailment_scores, thresholds[0], thresholds[1])
    
    # Calculate accuracy manually
    accuracy = np.mean(y_pred == y_full)
    print(f"Overall Test Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_full, y_pred, 
                              target_names=['Entailment', 'Neutral', 'Contradiction']))
    
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_full, y_pred)
    print("                Predicted")
    print("               E   N   C")
    class_names = ['E', 'N', 'C']
    for i, (actual_class, row) in enumerate(zip(class_names, conf_matrix)):
        print(f"Actual {actual_class} {row[0]:4d} {row[1]:3d} {row[2]:3d}")
    
    # Save results
    save_hierarchical_results(entailment_scores, y_full, y_pred, thresholds, accuracy, sample_size)
    
    return y_pred, accuracy


def save_hierarchical_results(entailment_scores, y_true, y_pred, thresholds, accuracy, sample_size):
    """Save hierarchical classification results to file"""
    os.makedirs('svm_validation_results', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    class_names = ['Entailment', 'Neutral', 'Contradiction']
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Prepare results dictionary
    results = {
        'timestamp': timestamp,
        'experiment_config': {
            'approach': 'hierarchical_classification',
            'sample_size': sample_size,
            'embedding_space': 'lattice_containment',
            'method': 'binary_svm_with_thresholds',
            'test_size': 0.2,
            'random_state': 42
        },
        'thresholds': {
            'entailment_threshold': float(thresholds[0]),
            'contradiction_threshold': float(thresholds[1])
        },
        'performance_metrics': {
            'test_accuracy': float(accuracy),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        },
        'entailment_scores_stats': {
            'mean': float(np.mean(entailment_scores)),
            'std': float(np.std(entailment_scores)),
            'min': float(np.min(entailment_scores)),
            'max': float(np.max(entailment_scores)),
            'range': float(np.max(entailment_scores) - np.min(entailment_scores))
        }
    }
    
    # Save as JSON
    json_filename = f'entailment_surfaces/svm_validation_results/binary_hierarchical_svm_{timestamp}_n{sample_size}.json'
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as readable text file
    txt_filename = f'entailment_surfaces/svm_validation_results/binary_hierarchical_svm_{timestamp}_n{sample_size}.txt'
    with open(txt_filename, 'w') as f:
        f.write("HIERARCHICAL SVM VALIDATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Approach: Hierarchical Classification\n")
        f.write(f"Sample Size: {sample_size}\n")
        f.write(f"Embedding Space: lattice_containment\n")
        f.write(f"Method: Binary SVM + Thresholds\n\n")
        
        f.write("THRESHOLDS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Entailment Threshold: > {thresholds[0]:.3f}\n")
        f.write(f"Contradiction Threshold: < {thresholds[1]:.3f}\n")
        f.write(f"Neutral Range: [{thresholds[1]:.3f}, {thresholds[0]:.3f}]\n\n")
        
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        
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
        f.write("\n")
        
        f.write("ENTAILMENT SCORES STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean: {np.mean(entailment_scores):.3f}\n")
        f.write(f"Std: {np.std(entailment_scores):.3f}\n")
        f.write(f"Range: [{np.min(entailment_scores):.3f}, {np.max(entailment_scores):.3f}]\n")
    
    print(f"\nResults saved to:")
    print(f"  JSON: {json_filename}")
    print(f"  TXT:  {txt_filename}")


def analyze_lattice_embeddings(embeddings, labels):
    """Analyze distribution of lattice containment embeddings by class"""
    print("\n" + "="*50)
    print("LATTICE CONTAINMENT EMBEDDING ANALYSIS")
    print("="*50)
    
    class_names = ['Entailment', 'Neutral', 'Contradiction']
    
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Total samples: {len(embeddings)}")
    
    for class_idx in range(3):
        class_embeddings = embeddings[labels == class_idx]
        print(f"\n{class_names[class_idx]} Class:")
        print(f"  Count: {len(class_embeddings)}")
        print(f"  Mean norm: {np.mean(np.linalg.norm(class_embeddings, axis=1)):.4f}")
        print(f"  Std norm: {np.std(np.linalg.norm(class_embeddings, axis=1)):.4f}")
        print(f"  Mean per dimension: {np.mean(class_embeddings, axis=0)[:5]}...")  # First 5 dims
        print(f"  Std per dimension: {np.std(class_embeddings, axis=0)[:5]}...")   # First 5 dims


def main():
    # Configuration
    data_path = 'data/processed/snli_full_standard_SBERT.pt'
    sample_size = 20000  # Start with same size for comparison
    
    print("Starting Binary SVM Teacher Training Pipeline")
    print("="*50)
    
    # Load data
    print(f"About to load data from: {data_path}")
    flush_output()
    data = load_snli_data(data_path, sample_size)
    print("Data loaded successfully!")
    flush_output()

    print("About to generate embeddings...")
    flush_output()
    # Generate lattice containment embedding space
    X = generate_lattice_containment_embeddings(
        data['premise_embeddings'], 
        data['hypothesis_embeddings']
    )
    
    print("Preparing labels...")
    flush_output()
    # Prepare labels
    y = prepare_labels(data['labels'])
    
    print("Analyzing lattice embeddings...")
    flush_output()
    # Analyze lattice embeddings
    analyze_lattice_embeddings(X, y)
    
    print("Creating binary dataset and training E vs C SVM...")
    flush_output()
    # 2. Perform a SINGLE Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Train Binary SVM on the TRAINING data only
    X_train_binary, y_train_binary = create_binary_ec_dataset(X_train, y_train)
    binary_svm, scaler, _ = train_binary_svm(X_train_binary, y_train_binary) 

    # 4. Compute scores for BOTH train and test sets using the single trained SVM
    train_scores = compute_entailment_scores(binary_svm, scaler, X_train, y_train, "training dataset")
    test_scores = compute_entailment_scores(binary_svm, scaler, X_test, y_test, "test dataset")
    
    # 5. Find thresholds on the TRAINING scores only
    optimal_thresholds = optimize_thresholds(train_scores, y_train)

    # 6. Final validation on the HELD-OUT test set
    print("\n" + "="*60)
    print("FINAL HIERARCHICAL CLASSIFIER VALIDATION ON TEST SET")
    print("="*60)
    y_pred, final_accuracy = validate_hierarchical_classifier(test_scores, y_test, optimal_thresholds, sample_size)
    
    print("\nPIPELINE COMPLETE")
    
    print("\n" + "="*60)
    print("HIERARCHICAL SVM TRAINING COMPLETE")
    print("="*60)
    print(f"Final Test Accuracy: {final_accuracy:.4f}")
    print(f"Improvement over baseline: {final_accuracy - 0.656:.4f}")
    
    return binary_svm, scaler, test_scores, optimal_thresholds, grid_search

if __name__ == "__main__":
    binary_svm, scaler, test_scores, thresholds, grid_search = main()
