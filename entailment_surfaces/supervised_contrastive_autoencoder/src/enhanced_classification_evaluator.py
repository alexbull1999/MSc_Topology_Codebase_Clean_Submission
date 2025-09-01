"""
Enhanced Classification Evaluation Module
Implements Random Forest, SVM with RBF, and Neural Network classifiers
for evaluating topological autoencoder latent representations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Import existing evaluator and dependencies
from evaluator_global import GlobalContrastiveEvaluator
from contrastive_autoencoder_model_global import ContrastiveAutoencoder
from data_loader_global import GlobalDataLoader
from attention_autoencoder_model import AttentionAutoencoder


class LatentNeuralClassifier(nn.Module):
    """
    Small neural network classifier for latent representations
    """
    def __init__(self, latent_dim, num_classes=3, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


class EnhancedClassificationEvaluator:
    """
    Enhanced classification evaluator using multiple advanced methods
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
        
    def subsample_for_efficiency(self, X, y, max_samples=None, method='random_forest'):
        """
        Optionally subsample data only if computationally necessary
        """
        if max_samples is None:
            # Set conservative limits only for SVM
            if method == 'svm_rbf' and len(X) > 100000:
                max_samples = 100000  # SVM is quadratic, so limit for very large datasets
                print(f"  Note: SVM limited to {max_samples} samples for computational efficiency")
            else:
                return X, y  # Use all data for RF and NN
        
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sub = X[indices]
            y_sub = y[indices]
            print(f"  Subsampled to {len(X_sub)} samples for {method}")
            return X_sub, y_sub
        else:
            return X, y
    
    def evaluate_random_forest(self, X_train, y_train, X_val, y_val):
        """
        Evaluate using Random Forest classifier
        """
        print("  Training Random Forest classifier...")
        
        # Configure Random Forest
        rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        # Train classifier
        rf_classifier.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_classifier.predict(X_val)
        y_pred_proba = rf_classifier.predict_proba(X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        balanced_acc = balanced_accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average='macro')
        f1_per_class = f1_score(y_val, y_pred, average=None)
        
        # Get feature importances (interesting for latent space analysis)
        feature_importances = rf_classifier.feature_importances_
        
        # Cross-validation on subset for robustness
        cv_subset_size = min(10000, len(X_train))
        if cv_subset_size < len(X_train):
            cv_indices = np.random.choice(len(X_train), cv_subset_size, replace=False)
            X_cv = X_train[cv_indices]
            y_cv = y_train[cv_indices]
        else:
            X_cv = X_train
            y_cv = y_train
            
        cv_scores = cross_val_score(rf_classifier, X_cv, y_cv, cv=3, scoring='accuracy')
        
        results = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_macro': f1_macro,
            'f1_per_class': f1_per_class.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importances_stats': {
                'mean': feature_importances.mean(),
                'std': feature_importances.std(),
                'max': feature_importances.max(),
                'top_5_features': np.argsort(feature_importances)[-5:].tolist()
            }
        }
        
        print(f"    Random Forest Accuracy: {accuracy:.4f}")
        print(f"    Random Forest F1-macro: {f1_macro:.4f}")
        print(f"    Random Forest CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return results
    
    def evaluate_svm_rbf(self, X_train, y_train, X_val, y_val):
        """
        Evaluate using SVM with RBF kernel
        """
        print("  Training SVM with RBF kernel...")
        
        # Scale features (important for SVM)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Configure SVM
        svm_classifier = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,  # Enable probability estimates
            random_state=42
        )
        
        # Train classifier
        svm_classifier.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = svm_classifier.predict(X_val_scaled)
        y_pred_proba = svm_classifier.predict_proba(X_val_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        balanced_acc = balanced_accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average='macro')
        f1_per_class = f1_score(y_val, y_pred, average=None)
        
        # Cross-validation on subset
        cv_subset_size = min(10000, len(X_train_scaled))
        if cv_subset_size < len(X_train_scaled):
            cv_indices = np.random.choice(len(X_train_scaled), cv_subset_size, replace=False)
            X_cv = X_train_scaled[cv_indices]
            y_cv = y_train[cv_indices]
        else:
            X_cv = X_train_scaled
            y_cv = y_train
            
        cv_scores = cross_val_score(svm_classifier, X_cv, y_cv, cv=3, scoring='accuracy')
        
        results = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_macro': f1_macro,
            'f1_per_class': f1_per_class.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'num_support_vectors': svm_classifier.n_support_.tolist(),
            'total_support_vectors': svm_classifier.support_vectors_.shape[0]
        }
        
        print(f"    SVM RBF Accuracy: {accuracy:.4f}")
        print(f"    SVM RBF F1-macro: {f1_macro:.4f}")
        print(f"    SVM RBF CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return results
    
    def evaluate_neural_network(self, X_train, y_train, X_val, y_val, epochs=50):
        """
        Evaluate using small neural network classifier
        """
        print("  Training Neural Network classifier...")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # Create model
        latent_dim = X_train.shape[1]
        model = LatentNeuralClassifier(latent_dim).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"      Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Training accuracy
            train_outputs = model(X_train_tensor)
            train_pred = torch.argmax(train_outputs, dim=1)
            train_accuracy = (train_pred == y_train_tensor).float().mean().item()
            
            # Validation predictions
            val_outputs = model(X_val_tensor)
            val_pred = torch.argmax(val_outputs, dim=1).cpu().numpy()
            val_proba = torch.softmax(val_outputs, dim=1).cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, val_pred)
        balanced_acc = balanced_accuracy_score(y_val, val_pred)
        f1_macro = f1_score(y_val, val_pred, average='macro')
        f1_per_class = f1_score(y_val, val_pred, average=None)
        
        results = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_macro': f1_macro,
            'f1_per_class': f1_per_class.tolist(),
            'train_accuracy': train_accuracy,
            'final_loss': loss.item(),
            'model_parameters': sum(p.numel() for p in model.parameters())
        }
        
        print(f"    Neural Network Accuracy: {accuracy:.4f}")
        print(f"    Neural Network F1-macro: {f1_macro:.4f}")
        print(f"    Neural Network Train Acc: {train_accuracy:.4f}")
        
        return results
    
    def comprehensive_classification_evaluation(self, train_latent, train_labels, val_latent, val_labels):
        """
        Run comprehensive classification evaluation with all three methods
        
        Args:
            train_latent: Training latent representations [N, latent_dim]
            train_labels: Training labels [N]
            val_latent: Validation latent representations [M, latent_dim]  
            val_labels: Validation labels [M]
            
        Returns:
            Dictionary with results from all classifiers
        """
        print("Starting comprehensive classification evaluation...")
        print("=" * 60)
        
        # Convert to numpy if needed
        if isinstance(train_latent, torch.Tensor):
            X_train = train_latent.cpu().numpy()
        else:
            X_train = train_latent
            
        if isinstance(train_labels, torch.Tensor):
            y_train = train_labels.cpu().numpy()
        else:
            y_train = train_labels
            
        if isinstance(val_latent, torch.Tensor):
            X_val = val_latent.cpu().numpy()
        else:
            X_val = val_latent
            
        if isinstance(val_labels, torch.Tensor):
            y_val = val_labels.cpu().numpy()
        else:
            y_val = val_labels
        
        print(f"Dataset info:")
        print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Class distribution (train): {np.bincount(y_train)}")
        print(f"  Class distribution (val): {np.bincount(y_val)}")
        
        # Use all data unless computationally prohibitive
        print(f"  Using all {len(X_train)} training samples")
        X_train_sub, y_train_sub = self.subsample_for_efficiency(X_train, y_train, method='random_forest')
        X_val_sub, y_val_sub = X_val, y_val  # Always use all validation data
        
        # Evaluate with each method
        print("\n1. Random Forest Classification:")
        rf_results = self.evaluate_random_forest(X_train_sub, y_train_sub, X_val_sub, y_val_sub)
        
        print("\n2. SVM with RBF Kernel Classification:")
        X_train_svm, y_train_svm = self.subsample_for_efficiency(X_train, y_train, method='svm_rbf')
        svm_results = self.evaluate_svm_rbf(X_train_svm, y_train_svm, X_val_sub, y_val_sub)
        
        print("\n3. Neural Network Classification:")
        nn_results = self.evaluate_neural_network(X_train_sub, y_train_sub, X_val_sub, y_val_sub)
        
        # Combine results
        self.results = {
            'random_forest': rf_results,
            'svm_rbf': svm_results,
            'neural_network': nn_results,
            'dataset_info': {
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'latent_dim': X_train.shape[1],
                'num_classes': len(np.unique(y_train)),
                'class_names': ['entailment', 'neutral', 'contradiction']
            }
        }
        
        print("\n" + "=" * 60)
        print("ENHANCED CLASSIFICATION SUMMARY")
        print("=" * 60)
        
        methods = ['random_forest', 'svm_rbf', 'neural_network']
        method_names = ['Random Forest', 'SVM RBF', 'Neural Network']
        
        print(f"{'Method':<15} {'Accuracy':<10} {'F1-Macro':<10} {'Balanced Acc':<12}")
        print("-" * 50)
        
        for method, name in zip(methods, method_names):
            acc = self.results[method]['accuracy']
            f1 = self.results[method]['f1_macro']
            bal_acc = self.results[method]['balanced_accuracy']
            print(f"{name:<15} {acc:<10.4f} {f1:<10.4f} {bal_acc:<12.4f}")
        
        # Find best method
        best_method = max(methods, key=lambda x: self.results[x]['accuracy'])
        best_acc = self.results[best_method]['accuracy']
        print(f"\nBest performing method: {best_method.replace('_', ' ').title()} ({best_acc:.4f})")
        
        print("=" * 60)
        
        return self.results
    
    def print_detailed_results(self):
        """
        Print detailed results for all classification methods
        """
        if not self.results:
            print("No evaluation results available. Run comprehensive_classification_evaluation() first.")
            return
        
        print("\n" + "=" * 80)
        print("DETAILED CLASSIFICATION RESULTS")
        print("=" * 80)
        
        for method_key, method_name in [('random_forest', 'Random Forest'), 
                                       ('svm_rbf', 'SVM with RBF Kernel'), 
                                       ('neural_network', 'Neural Network')]:
            
            results = self.results[method_key]
            print(f"\n{method_name.upper()}:")
            print("-" * 40)
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  Balanced Accuracy: {results['balanced_accuracy']:.4f}")
            print(f"  F1-Macro: {results['f1_macro']:.4f}")
            print(f"  F1 per class: {[f'{f:.4f}' for f in results['f1_per_class']]}")
            
            if 'cv_mean' in results:
                print(f"  Cross-validation: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
            
            if method_key == 'random_forest':
                print(f"  Feature importance (mean): {results['feature_importances_stats']['mean']:.6f}")
                print(f"  Top 5 important features: {results['feature_importances_stats']['top_5_features']}")
            
            elif method_key == 'svm_rbf':
                print(f"  Total support vectors: {results['total_support_vectors']}")
                print(f"  Support vectors per class: {results['num_support_vectors']}")
            
            elif method_key == 'neural_network':
                print(f"  Training accuracy: {results['train_accuracy']:.4f}")
                print(f"  Model parameters: {results['model_parameters']:,}")
        
        print("=" * 80)


# Integration function for the existing evaluator
def integrate_enhanced_classification(evaluator_instance, train_dataloader, val_dataloader):
    """
    Integration function to add enhanced classification to existing evaluator
    
    Args:
        evaluator_instance: Instance of GlobalContrastiveEvaluator
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        
    Returns:
        Enhanced classification results
    """
    print("Running enhanced classification evaluation...")
    
    # Extract latent representations (reuse existing method)
    train_latent, train_labels = evaluator_instance.extract_latent_representations(train_dataloader)
    val_latent, val_labels = evaluator_instance.extract_latent_representations(val_dataloader)
    
    # Create enhanced evaluator
    enhanced_evaluator = EnhancedClassificationEvaluator(device=evaluator_instance.device)
    
    # Run comprehensive evaluation
    enhanced_results = enhanced_evaluator.comprehensive_classification_evaluation(
        train_latent, train_labels, val_latent, val_labels
    )
    
    # Print detailed results
    enhanced_evaluator.print_detailed_results()
    
    return enhanced_results


def load_data(config):
    """
    Load and prepare data
    """
    print("Loading data...")
    print("=" * 40)
    
    # Create data loader
    data_loader = GlobalDataLoader(
        train_path=config['train_path'],
        val_path=config['val_path'], 
        test_path=config['test_path'],
        embedding_type=config['embedding_type'],
        sample_size=config['sample_size'],
        random_state=config['random_state']
    )
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = data_loader.load_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        batch_size=config['batch_size'],
        balanced_sampling=config['balanced_sampling']
    )
    
    print(f"Data loading completed!")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """
    Main evaluation script - loads trained model and runs enhanced classification
    """
    print("Running Enhanced Classification Evaluation on Trained Model")
    print("=" * 70)
    
    # Configuration - modify these paths as needed
    MODEL_PATH = "entailment_surfaces/supervised_contrastive_autoencoder/experiments/FIXED_DECODERS/global_concat_pure_reconstruction_20250724_123815_no_attention/checkpoints/best_model.pt"  # Update this path
    DATA_CONFIG = {
        'train_path': 'data/processed/snli_full_standard_SBERT.pt',
        'val_path': 'data/processed/snli_full_standard_SBERT_validation.pt',
        'test_path': 'data/processed/snli_full_standard_SBERT_test.pt',
        'embedding_type': 'concat',  
        'batch_size': 1020,
        'sample_size': None,
        'balanced_sampling': True,
        'random_state': 42
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check if model path exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please update MODEL_PATH to point to your trained model checkpoint")
        print("\nExample paths to try:")
        print("- experiments/*/checkpoints/best_model.pt")
        print("- checkpoints/best_model.pt")
        sys.exit(1)
    
    try:
        # Load trained model
        print(f"Loading model from: {MODEL_PATH}")
        
        # Create model instance (adjust parameters to match your trained model)
        model = ContrastiveAutoencoder(
            input_dim=1536,  # Adjust if different
            latent_dim=100,  # Adjust to match your trained model
            hidden_dims=[],  # Adjust to match your model
            dropout_rate=0.2
        )

        
        # Load model weights
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print("Model loaded successfully")
        
        # Load data
        print("\nLoading SNLI data...")
        train_loader, val_loader, test_loader = load_data(DATA_CONFIG)
        print(f"Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
        
        # Create evaluator with loaded model
        print("\nCreating evaluator...")
        evaluator = GlobalContrastiveEvaluator(model, device=device)
        
        # Run enhanced classification evaluation
        print("\n" + "=" * 70)
        print("RUNNING ENHANCED CLASSIFICATION EVALUATION")
        print("=" * 70)
        
        enhanced_results = integrate_enhanced_classification(evaluator, train_loader, val_loader)
        
        # Save results
        results_dir = "enhanced_classification_results"
        os.makedirs(results_dir, exist_ok=True)
        
        import json
        from datetime import datetime
        
        results_file = os.path.join(results_dir, f"enhanced_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Make results JSON serializable
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = make_serializable(enhanced_results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nEnhanced classification results saved to: {results_file}")
        
        # Print final summary
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        best_method = max(['random_forest', 'svm_rbf', 'neural_network'], 
                         key=lambda x: enhanced_results[x]['accuracy'])
        best_accuracy = enhanced_results[best_method]['accuracy']
        
        print(f" Best performing method: {best_method.replace('_', ' ').title()}")
        print(f" Best accuracy: {best_accuracy:.4f}")
        
        # Compare with baseline k-NN (if available)
        if hasattr(evaluator, 'evaluation_results') and evaluator.evaluation_results:
            baseline_acc = evaluator.evaluation_results.get('classification', {}).get('accuracy', 'N/A')
            if isinstance(baseline_acc, float):
                improvement = best_accuracy - baseline_acc
                print(f" Improvement over k-NN baseline: +{improvement:.4f} ({improvement/baseline_acc*100:.1f}%)")
        
        print("\nFor detailed results, check the saved JSON file or run with detailed printing enabled.")
        
    except Exception as e:
        print(f"\n ERROR during evaluation: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check that MODEL_PATH points to a valid checkpoint file")
        print("2. Ensure model architecture parameters match your trained model")
        print("3. Verify that all dependencies are installed")
        print("4. Check that SNLI data is available in the expected location")
        
        import traceback
        print(f"\nFull error traceback:")
        traceback.print_exc()
        sys.exit(1)
