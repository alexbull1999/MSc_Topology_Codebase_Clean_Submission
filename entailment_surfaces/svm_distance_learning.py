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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.order_embeddings_asymmetry import OrderEmbeddingModel

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

def generate_order_asymmetry_embeddings(premise_embeddings, hypothesis_embeddings, batch_size=1000):
    """Generate order asymmetry embedding space for each premise-hypothesis pair"""
    print("Generating order asymmetry embedding space")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    order_model_path="models/enhanced_order_embeddings_snli_SBERT_full.pt"

    checkpoint = torch.load(order_model_path, map_location=device, weights_only=False)

    model_config = checkpoint['model_config']
    order_model = OrderEmbeddingModel(
        bert_dim=model_config['bert_dim'],
        order_dim=model_config['order_dim'],
        asymmetry_weight=model_config.get('asymmetry_weight', 0.2)
    )
    order_model.load_state_dict(checkpoint['model_state_dict'])
    order_model.to(device)
    order_model.eval()
    
    total_samples = len(premise_embeddings)
    print(f"Processing {total_samples} samples in batches of {batch_size}")
    
    # Process in batches to avoid memory issues
    all_order_asymmetry_embeddings = []
    
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
            premise_order_batch = order_model(premise_batch)
            hypothesis_order_batch = order_model(hypothesis_batch)
            asymmetry_vectors_batch = torch.abs(premise_order_batch - hypothesis_order_batch)
            # Move back to CPU and store
            all_order_asymmetry_embeddings.append(asymmetry_vectors_batch.cpu().numpy())

        # Clear GPU memory
        del premise_batch, premise_order_batch, hypothesis_batch, hypothesis_order_batch, asymmetry_vectors_batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Concatenate all batches
    return np.concatenate(all_order_asymmetry_embeddings, axis=0)


def prepare_labels(labels):
    """Convert labels to numerical format"""
    label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    
    if torch.is_tensor(labels):
        # Assuming labels are already numerical 0,1,2
        return labels.numpy()
    else:
        # If labels are strings, map them
        return np.array([label_map.get(label, label) for label in labels])


# def train_svm_teacher_gridsearch(X, y, test_size=0.2, random_state=42):
#     """Train SVM on lattice containment embeddings with GridSearchCV optimization"""
#     print("Splitting data into train/test sets")
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state, stratify=y
#     )
    
#     print("Scaling features")
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     print("Setting up GridSearchCV for RBF SVM hyperparameter optimization")
    
#     # Define parameter grid
#     param_grid = {
#         'C': [1],
#         'gamma': ['scale']
#     }
    
#     print(f"Parameter grid: {param_grid}")
#     print(f"Total combinations to test: {len(param_grid['C']) * len(param_grid['gamma'])}")
    
#     # Create base SVM
#     base_svm = SVC(kernel='rbf', probability=True, random_state=random_state)
    
#     # Setup GridSearchCV
#     grid_search = GridSearchCV(
#         estimator=base_svm,
#         param_grid=param_grid,
#         cv=2,  # 1-fold (now grid search complete, was 3)
#         scoring='accuracy',
#         n_jobs=-1,  # Use all available cores
#         verbose=2   # Print progress
#     )
    
#     print("Starting GridSearchCV (this may take several minutes)...")
#     grid_search.fit(X_train_scaled, y_train)
    
#     print("GridSearchCV completed!")
#     print(f"Best parameters: {grid_search.best_params_}")
#     print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
#     # Get the best model
#     best_svm = grid_search.best_estimator_
    
#     print("Evaluating best SVM performance")
#     train_score = best_svm.score(X_train_scaled, y_train)
#     test_score = best_svm.score(X_test_scaled, y_test)
    
#     print(f"Best SVM Training Accuracy: {train_score:.4f}")
#     print(f"Best SVM Test Accuracy: {test_score:.4f}")
    
#     return best_svm, scaler, X_train_scaled, X_test_scaled, y_train, y_test, grid_search

def train_svm_teacher(X, y, test_size=0.2, random_state=42):
    """Train SVM on containment distance features"""
    print("Splitting data into train/test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print("Scaling features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training SVM classifier")
    svm = SVC(kernel='rbf', gamma='scale', C=1, probability=True, random_state=random_state)
    svm.fit(X_train_scaled, y_train)
    
    print("Evaluating SVM performance")
    train_score = svm.score(X_train_scaled, y_train)
    test_score = svm.score(X_test_scaled, y_test)
    
    print(f"SVM Training Accuracy: {train_score:.4f}")
    print(f"SVM Test Accuracy: {test_score:.4f}")
    
    return svm, scaler, X_train_scaled, X_test_scaled, y_train, y_test


def extract_decision_boundaries(svm, X_scaled):
    """Extract signed distances from SVM decision boundaries"""
    print("Extracting decision function values")
    decision_values = svm.decision_function(X_scaled)
    return decision_values


# def validate_svm_teacher(svm, scaler, X_test_scaled, y_test, sample_size, grid_search):
#     """Comprehensive validation of SVM teacher performance"""
#     print("\n" + "="*50)
#     print("SVM TEACHER VALIDATION RESULTS")
#     print("="*50)
    
#     y_pred = svm.predict(X_test_scaled)

#     # Calculate test accuracy manually to verify
#     manual_accuracy = np.mean(y_pred == y_test)
#     svm_accuracy = svm.score(X_test_scaled, y_test)
    
#     print(f"Manual Test Accuracy: {manual_accuracy:.4f}")
#     print(f"SVM Test Accuracy: {svm_accuracy:.4f}")
#     print(f"Best GridSearch CV Score: {grid_search.best_score_:.4f}")
#     print(f"Best Parameters: {grid_search.best_params_}")

#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred, 
#                               target_names=['Entailment', 'Neutral', 'Contradiction']))
    
#     print("\nConfusion Matrix:")
#     print(confusion_matrix(y_test, y_pred))
    
#     # Extract decision function values
#     decision_values = svm.decision_function(X_test_scaled)
    
#     print(f"\nDecision Function Shape: {decision_values.shape}")
#     print(f"Decision Function Stats:")
#     print(f"  Mean: {np.mean(decision_values, axis=0)}")
#     print(f"  Std: {np.std(decision_values, axis=0)}")
#     print(f"  Min: {np.min(decision_values, axis=0)}")
#     print(f"  Max: {np.max(decision_values, axis=0)}")

#     # Save results to files
#     json_file, txt_file = save_validation_results(
#         svm, scaler, X_test_scaled, y_test, decision_values, y_pred, sample_size, grid_search
#     )
    
#     return decision_values, y_pred

def validate_svm_teacher(svm, scaler, X_test_scaled, y_test, sample_size):
    """Comprehensive validation of SVM teacher performance"""
    print("\n" + "="*50)
    print("SVM TEACHER VALIDATION RESULTS")
    print("="*50)
    
    # X_test_scaled is already scaled, so use directly
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
    print(confusion_matrix(y_test, y_pred))
    
    # Extract decision function values
    decision_values = svm.decision_function(X_test_scaled)
    
    print(f"\nDecision Function Shape: {decision_values.shape}")
    print(f"Decision Function Stats:")
    print(f"  Mean: {np.mean(decision_values, axis=0)}")
    print(f"  Std: {np.std(decision_values, axis=0)}")
    print(f"  Min: {np.min(decision_values, axis=0)}")
    print(f"  Max: {np.max(decision_values, axis=0)}")
    
    # Save results to files
    json_file, txt_file = save_validation_results(
        svm, scaler, X_test_scaled, y_test, decision_values, y_pred, sample_size
    )
    
    return decision_values, y_pred

# def save_validation_results(svm, scaler, X_test_scaled, y_test, decision_values, y_pred, sample_size, grid_search):
#     """Save SVM validation results to file"""
#     # Create results directory
#     os.makedirs('entailment_surfaces/svm_validation_results', exist_ok=True)
    
#     # Generate timestamp for unique filename
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     # Calculate metrics
#     test_score = svm.score(X_test_scaled, y_test)
    
#     class_names = ['Entailment', 'Neutral', 'Contradiction']
#     conf_matrix = confusion_matrix(y_test, y_pred)
#     class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
#     # Prepare results dictionary
#     results = {
#         'timestamp': timestamp,
#         'experiment_config': {
#             'sample_size': sample_size,
#             'embedding_space': 'lattice_containment',
#             'svm_kernel': 'rbf',
#             'optimization': 'GridSearchCV',
#             'test_size': 0.2,
#             'random_state': 42
#         },
#         'gridsearch_results': {
#             'best_params': grid_search.best_params_,
#             'best_cv_score': float(grid_search.best_score_),
#             'param_grid': {
#                 'C': [1],
#                 'gamma': ['scale']
#             },
#         },
#         'performance_metrics': {
#             'test_accuracy': float(test_score),
#             'confusion_matrix': conf_matrix.tolist(),
#             'classification_report': class_report
#         },
#         'decision_function_stats': {
#             'shape': list(decision_values.shape),
#             'mean_per_class': np.mean(decision_values, axis=0).tolist(),
#             'std_per_class': np.std(decision_values, axis=0).tolist(),
#             'min_per_class': np.min(decision_values, axis=0).tolist(),
#             'max_per_class': np.max(decision_values, axis=0).tolist()
#         }
#     }
    
#     # Save as JSON
#     json_filename = f'entailment_surfaces/svm_validation_results/svm_optimized_kernel_full_{timestamp}_n{sample_size}.json'
#     with open(json_filename, 'w') as f:
#         json.dump(results, f, indent=2)
    
#     # Save as readable text file
#     txt_filename = f'entailment_surfaces/svm_validation_results/svm_optimized_kernel_full_{timestamp}_n{sample_size}.txt'
#     with open(txt_filename, 'w') as f:
#         f.write("SVM TEACHER VALIDATION RESULTS\n")
#         f.write("=" * 50 + "\n\n")
#         f.write(f"Timestamp: {timestamp}\n")
#         f.write(f"Sample Size: {sample_size}\n")
#         f.write(f"Embedding Space: lattice_containment\n")
#         f.write(f"SVM Kernel: rbf\n")
#         f.write(f"Optimization: GridSearchCV\n\n")

#         f.write("GRIDSEARCH RESULTS\n")
#         f.write("-" * 30 + "\n")
#         f.write(f"Best Parameters: {grid_search.best_params_}\n")
#         f.write(f"Best CV Score: {grid_search.best_score_:.4f}\n")
#         f.write(f"Parameter Grid: C={[0.1, 1, 10, 100]}, gamma=['scale', 'auto', 0.01, 0.1, 1]\n\n")
        
#         f.write("PERFORMANCE METRICS\n")
#         f.write("-" * 30 + "\n")
#         f.write(f"Test Accuracy: {test_score:.4f}\n\n")
        
#         f.write("Confusion Matrix:\n")
#         f.write("                Predicted\n")
#         f.write("               E   N   C\n")
#         for i, (actual_class, row) in enumerate(zip(class_names, conf_matrix)):
#             f.write(f"Actual {actual_class[0]:1s} {row[0]:4d} {row[1]:3d} {row[2]:3d}\n")
#         f.write("\n")
        
#         f.write("Classification Report:\n")
#         for class_name in class_names:
#             metrics = class_report[class_name]
#             f.write(f"{class_name:12s}: precision={metrics['precision']:.3f}, "
#                    f"recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}\n")
#         f.write("\n")
        
#         f.write("DECISION FUNCTION STATISTICS\n")
#         f.write("-" * 30 + "\n")
#         f.write(f"Shape: {decision_values.shape}\n")
#         f.write(f"Mean per class: {np.mean(decision_values, axis=0)}\n")
#         f.write(f"Std per class:  {np.std(decision_values, axis=0)}\n")
#         f.write(f"Min per class:  {np.min(decision_values, axis=0)}\n")
#         f.write(f"Max per class:  {np.max(decision_values, axis=0)}\n")
    
#     print(f"\nResults saved to:")
#     print(f"  JSON: {json_filename}")
#     print(f"  TXT:  {txt_filename}")
    
#     return json_filename, txt_filename

def save_validation_results(svm, scaler, X_test_scaled, y_test, decision_values, y_pred, sample_size):
    """Save SVM validation results to file"""
    # Create results directory
    os.makedirs('entailment_surfaces/svm_validation_results', exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate metrics
    test_score = svm.score(X_test_scaled, y_test)
    
    class_names = ['Entailment', 'Neutral', 'Contradiction']
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Prepare results dictionary
    results = {
        'timestamp': timestamp,
        'experiment_config': {
            'sample_size': sample_size,
            'embedding_space': 'lattice_containment',
            'svm_kernel': 'rbf',
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
    json_filename = f'entailment_surfaces/svm_validation_results/svm_validation_order_asymmetry__{timestamp}_n{sample_size}.json'
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as readable text file
    txt_filename = f'entailment_surfaces/svm_validation_results/svm_validation_order_asymmetry_{timestamp}_n{sample_size}.txt'
    with open(txt_filename, 'w') as f:
        f.write("SVM TEACHER VALIDATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Sample Size: {sample_size}\n")
        f.write(f"Embedding Space: lattice_containment\n")
        f.write(f"SVM Kernel: rbf\n\n")
        
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
            metrics = class_report[class_name]  # Remove .lower() since keys are already correct
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
    """Comprehensive validation of SVM teacher performance"""
    print("\n" + "="*50)
    print("SVM TEACHER VALIDATION RESULTS")
    print("="*50)
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = svm.predict(X_test_scaled)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Entailment', 'Neutral', 'Contradiction']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Extract decision function values
    decision_values = svm.decision_function(X_test_scaled)
    
    print(f"\nDecision Function Shape: {decision_values.shape}")
    print(f"Decision Function Stats:")
    print(f"  Mean: {np.mean(decision_values, axis=0)}")
    print(f"  Std: {np.std(decision_values, axis=0)}")
    print(f"  Min: {np.min(decision_values, axis=0)}")
    print(f"  Max: {np.max(decision_values, axis=0)}")
    
    return decision_values, y_pred


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
    sample_size = 20000
    
    print("Starting SVM Teacher Training Pipeline")
    print("="*50)
    
    # Load data
    print(f"About to load data from: {data_path}")
    flush_output()
    data = load_snli_data(data_path, sample_size)
    print("Data loaded successfully!")
    flush_output()

    print("About to generate embeddings...")
    flush_output()
    # Generate lattice containment / order asymmetry embedding space
    X = generate_order_asymmetry_embeddings(
        data['premise_embeddings'], 
        data['hypothesis_embeddings']
    )
    
    print("Preparing labels...")
    flush_output()
    # Prepare labels
    y = prepare_labels(data['labels'])
    
    print("Analyzing lattice / order_asymmetry embeddings...")
    flush_output()
    # Analyze lattice embeddings
    analyze_lattice_embeddings(X, y)
    
    print("Training SVM...")
    flush_output()
    # Train SVM teacher
    svm, scaler, X_train_scaled, X_test_scaled, y_train, y_test = train_svm_teacher(X, y)
    
    print("Validating SVM...")
    flush_output()
    # Validate SVM teacher
    decision_values, y_pred = validate_svm_teacher(svm, scaler, X_test_scaled, y_test, sample_size)
    
    print("\n" + "="*50)
    print("SVM TEACHER TRAINING COMPLETE")
    print("="*50)
    
    return svm, scaler, decision_values

if __name__ == "__main__":
    svm_model, scaler, decision_values = main()
