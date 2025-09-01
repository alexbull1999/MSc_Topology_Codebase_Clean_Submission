import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import sys
import os
import json
from datetime import datetime

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


def generate_lattice_containment_embeddings(premise_embeddings, hypothesis_embeddings):
    """Generate lattice containment embedding space for each premise-hypothesis pair"""
    print("Generating lattice containment embedding space")
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    epsilon = 1e-8
    
    # Convert to tensors and move to GPU if available
    if not torch.is_tensor(premise_embeddings):
        premise_embeddings = torch.tensor(premise_embeddings)
    if not torch.is_tensor(hypothesis_embeddings):
        hypothesis_embeddings = torch.tensor(hypothesis_embeddings)
    
    # Move to GPU
    premise_embeddings = premise_embeddings.to(device)
    hypothesis_embeddings = hypothesis_embeddings.to(device)
    
    print(f"Processing {len(premise_embeddings)} samples on {device}")
    
    # Vectorized computation - all samples at once
    lattice_embeddings = (premise_embeddings * hypothesis_embeddings) / (
        torch.abs(premise_embeddings) + torch.abs(hypothesis_embeddings) + epsilon
    )
    
    # Move back to CPU and convert to numpy
    return lattice_embeddings.cpu().numpy()


def prepare_labels(labels):
    """Convert labels to numerical format"""
    label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    
    if torch.is_tensor(labels):
        # Assuming labels are already numerical 0,1,2
        return labels.numpy()
    else:
        # If labels are strings, map them
        return np.array([label_map.get(label, label) for label in labels])


def train_sgd_teacher(X, y, test_size=0.2, random_state=42):
    """Train SGD on containment distance features"""
    print("Splitting data into train/test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print("Scaling features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training SGD classifier")
    sgd = SGDClassifier(loss='hinge', random_state=random_state, max_iter=1000, tol=1e-3)
    sgd.fit(X_train_scaled, y_train)
    
    print("Evaluating SGD performance")
    train_score = sgd.score(X_train_scaled, y_train)
    test_score = sgd.score(X_test_scaled, y_test)
    
    print(f"SGD Training Accuracy: {train_score:.4f}")
    print(f"SGD Test Accuracy: {test_score:.4f}")
    
    return sgd, scaler, X_train_scaled, X_test_scaled, y_train, y_test


def extract_decision_boundaries(sgd, X_scaled):
    """Extract signed distances from SGD decision boundaries"""
    print("Extracting decision function values")
    # SGD doesn't have decision_function for multiclass, use predict_proba and convert
    if hasattr(sgd, 'decision_function'):
        decision_values = sgd.decision_function(X_scaled)
    else:
        # For multiclass, we can use the distances or probabilities
        # Using the raw scores if available
        decision_values = sgd._predict_proba_lr(X_scaled)
    return decision_values


def validate_sgd_teacher(sgd, scaler, X_test_scaled, y_test, sample_size):
    """Comprehensive validation of SGD teacher performance"""
    print("\n" + "="*50)
    print("SGD TEACHER VALIDATION RESULTS")
    print("="*50)
    
    y_pred = sgd.predict(X_test_scaled)

    # Calculate test accuracy manually to verify
    manual_accuracy = np.mean(y_pred == y_test)
    sgd_accuracy = sgd.score(X_test_scaled, y_test)
    
    print(f"Manual Test Accuracy: {manual_accuracy:.4f}")
    print(f"SGD Test Accuracy: {sgd_accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Entailment', 'Neutral', 'Contradiction']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    if hasattr(sgd, 'decision_function'):
        decision_values = sgd.decision_function(X_test_scaled)
    else:
        # For SGD, use predict_proba as proxy for decision values
        decision_values = sgd.predict_proba(X_test_scaled)
    
    print(f"\nDecision Function Shape: {decision_values.shape}")
    print(f"Decision Function Stats:")
    print(f"  Mean: {np.mean(decision_values, axis=0)}")
    print(f"  Std: {np.std(decision_values, axis=0)}")
    print(f"  Min: {np.min(decision_values, axis=0)}")
    print(f"  Max: {np.max(decision_values, axis=0)}")

    # Save results to files
    json_file, txt_file = save_validation_results(
        sgd, scaler, X_test_scaled, y_test, decision_values, y_pred, sample_size
    )
    
    return decision_values, y_pred

def save_validation_results(sgd, scaler, X_test_scaled, y_test, decision_values, y_pred, sample_size):
    """Save SGD validation results to file"""
    # Create results directory
    os.makedirs('entailment_surfaces/sgd_validation_results', exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate metrics
    test_score = sgd.score(X_test_scaled, y_test)
    
    class_names = ['Entailment', 'Neutral', 'Contradiction']
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Prepare results dictionary
    results = {
        'timestamp': timestamp,
        'experiment_config': {
            'sample_size': sample_size,
            'embedding_space': 'lattice_containment',
            'classifier': 'SGD',
            'loss_function': 'hinge',
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
    json_filename = f'entailment_surfaces/sgd_validation_results/sgd_validation_{timestamp}_n{sample_size}.json'
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as readable text file
    txt_filename = f'entailment_surfaces/sgd_validation_results/sgd_validation_{timestamp}_n{sample_size}.txt'
    with open(txt_filename, 'w') as f:
        f.write("SVM TEACHER VALIDATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Sample Size: {sample_size}\n")
        f.write(f"Embedding Space: lattice_containment\n")
        f.write(f"SVM Kernel: linear\n\n")
        
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Test Accuracy: {test_score:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write("                Predicted\n")
        f.write("               E   N   C\n")
        for i, (actual_class, row) in enumerate(zip(class_names, conf_matrix)):
            f.write(f"Actual {actual_class[0]:1s} {row[0]:4d} {row[1]:3d} {row[2]:3d}\n")
        f.write("\n")
        
        f.write("Classification Report:\n")
        for class_name in class_names:
            metrics = class_report[class_name]
            f.write(f"{class_name:12s}: precision={metrics['precision']:.3f}, "
                   f"recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}\n")
        f.write("\n")
        
        f.write("DECISION FUNCTION STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Shape: {decision_values.shape}\n")
        f.write(f"Mean per class: {np.mean(decision_values, axis=0)}\n")
        f.write(f"Std per class:  {np.std(decision_values, axis=0)}\n")
        f.write(f"Min per class:  {np.min(decision_values, axis=0)}\n")
        f.write(f"Max per class:  {np.max(decision_values, axis=0)}\n")
    
    print(f"\nResults saved to:")
    print(f"  JSON: {json_filename}")
    print(f"  TXT:  {txt_filename}")
    
    return json_filename, txt_filename


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
    
    print("Starting SGD Teacher Training Pipeline")
    print("="*50)
    
    # Load data
    print(f"About to load data from: {data_path}")
    flush_output()
    data = load_snli_data(data_path)
    sample_size = len(data)
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
    
    print("Training SGD...")
    flush_output()
    # Train SVM teacher
    sgd, scaler, X_train_scaled, X_test_scaled, y_train, y_test = train_sgd_teacher(X, y)
    
    print("Validating SGD...")
    flush_output()
    # Validate SVM teacher
    decision_values, y_pred = validate_sgd_teacher(sgd, scaler, X_test_scaled, y_test, sample_size)
    
    print("\n" + "="*50)
    print("SGD TEACHER TRAINING COMPLETE")
    print("="*50)
    
    return sgd, scaler, decision_values

if __name__ == "__main__":
    sgd_model, scaler, decision_values = main()
