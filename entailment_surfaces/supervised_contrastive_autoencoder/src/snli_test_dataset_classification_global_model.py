"""
Test Saved Contrastive Autoencoder on SNLI Test Set
Evaluates classification accuracy on completely unseen test data
"""

import torch
import numpy as np
import os
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add src directory to path for imports
sys.path.append('entailment_surfaces/supervised_contrastive_autoencoder/src')

from contrastive_autoencoder_model_global import ContrastiveAutoencoder
from data_loader_global import GlobalDataLoader
from evaluator_global import GlobalContrastiveEvaluator


def load_saved_model(model_path, device='cuda'):
    """
    Load the saved contrastive autoencoder model
    
    Args:
        model_path: Path to saved model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model instance
    """
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model configuration (you may need to adjust these based on your saved model)
    model_config = {
        'input_dim': 1536,  # SBERT concat dimension
        'latent_dim': 75,
        'hidden_dims': [512, 256],
        'dropout_rate': 0.2
    }
    
    # Create model instance
    model = ContrastiveAutoencoder(**model_config)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"   Best epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"   Best validation loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    return model


def evaluate_on_test_set(model, test_loader, train_loader, device='cuda'):
    """
    Evaluate model classification performance on test set
    
    Args:
        model: Trained contrastive autoencoder
        test_loader: Test data loader
        train_loader: Training data loader (for k-NN training)
        device: Compute device
        
    Returns:
        Dictionary with test results
    """
    print("Evaluating on SNLI Test Set...")
    print("=" * 50)
    
    # Create evaluator
    evaluator = GlobalContrastiveEvaluator(model, device)
    
    # Extract latent representations for training (k-NN training)
    print("Extracting training representations for k-NN classifier...")
    train_latent, train_labels = evaluator.extract_latent_representations(train_loader)
    
    # Extract latent representations for test set (k-NN evaluation)
    print("Extracting test representations...")
    test_latent, test_labels = evaluator.extract_latent_representations(test_loader)
    
    # Convert to numpy
    X_train = train_latent.cpu().numpy()
    y_train = train_labels.cpu().numpy()
    X_test = test_latent.cpu().numpy()
    y_test = test_labels.cpu().numpy()
    
    # Subsample training data for efficiency (if needed)
    if len(X_train) > 50000:
        print("⚡ Subsampling training data for efficiency...")
        train_indices = np.random.choice(len(X_train), 50000, replace=False)
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
    
    print(f"Training k-NN classifier on {len(X_train)} training samples...")
    print(f"Testing on {len(X_test)} test samples...")
    
    # Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = knn.predict(X_test)
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(
        y_test, y_pred,
        target_names=['entailment', 'neutral', 'contradiction'],
        output_dict=True
    )
    
    # Extract per-class F1 scores
    per_class_f1 = [
        class_report.get('entailment', {}).get('f1-score', 0.0),
        class_report.get('neutral', {}).get('f1-score', 0.0),
        class_report.get('contradiction', {}).get('f1-score', 0.0)
    ]
    
    # Print results
    print("\nTEST SET RESULTS")
    print("=" * 50)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Per-class F1 Scores:")
    print(f"   Entailment:    {per_class_f1[0]:.4f}")
    print(f"   Neutral:       {per_class_f1[1]:.4f}")
    print(f"   Contradiction: {per_class_f1[2]:.4f}")
    print(f"   Average F1:    {np.mean(per_class_f1):.4f}")
    
    print(f"\nConfusion Matrix:")
    print("    Predicted:")
    print("      E    N    C")
    for i, true_class in enumerate(['E', 'N', 'C']):
        row = f"{true_class}: {conf_matrix[i]}"
        print(f"T {row}")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['entailment', 'neutral', 'contradiction']
    ))
    
    return {
        'test_accuracy': test_accuracy,
        'per_class_f1': per_class_f1,
        'average_f1': np.mean(per_class_f1),
        'confusion_matrix': conf_matrix.tolist(),
        'detailed_report': class_report,
        'num_test_samples': len(X_test),
        'num_train_samples': len(X_train)
    }


def main():
    """Main execution function"""
    print("TESTING SAVED CONTRASTIVE AUTOENCODER ON SNLI TEST SET")
    print("=" * 60)
    
    # Configuration
    model_path = "entailment_surfaces/supervised_contrastive_autoencoder/experiments/global_concat_test_20250712_180422_BEST_NOLEAKAGE/checkpoints/best_model.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data paths (adjust these to match your setup)
    data_config = {
        'train_path': 'data/processed/snli_full_standard_SBERT.pt',
        'val_path': 'data/processed/snli_full_standard_SBERT_validation.pt',  
        'test_path': 'data/processed/snli_full_standard_SBERT_test.pt',
        'embedding_type': 'concat',  # SBERT concat
        'sample_size': None,
        'batch_size': 1020
    }
    
    print(f"Using device: {device}")
    print(f"Model path: {model_path}")
    
    try:
        # Load the saved model
        model = load_saved_model(model_path, device)
        
        # Load data
        print("\nLoading SNLI data...")
        data_loader = GlobalDataLoader(**data_config)
        train_dataset, val_dataset, test_dataset = data_loader.load_data()
        
        # Create data loaders
        train_loader, val_loader, test_loader = data_loader.get_dataloaders(
            batch_size=1020, 
            balanced_sampling=False  # Use regular sampling for testing
        )
        
        print(f"Data loaded successfully!")
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Val samples: {len(val_dataset)}")
        print(f"   Test samples: {len(test_dataset)}")
        
        # Evaluate on test set
        test_results = evaluate_on_test_set(model, test_loader, train_loader, device)
        
        # Save results
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"test_set_evaluation_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'model_path': model_path,
                'test_results': test_results,
                'data_config': data_config
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Summary
        print(f"\n🎯 FINAL SUMMARY")
        print(f"=" * 30)
        print(f"Validation Accuracy (previous): ~81.67%")
        print(f"Test Accuracy (unseen data):    {test_results['test_accuracy']*100:.2f}%")
        
        if test_results['test_accuracy'] > 0.95:
            print("OUTSTANDING: >95% accuracy on unseen test data!")
        elif test_results['test_accuracy'] > 0.90:
            print("EXCELLENT: >90% accuracy on unseen test data!")
        elif test_results['test_accuracy'] > 0.80:
            print("GOOD: >80% accuracy on unseen test data!")
        else:
            print("Lower than expected - investigate potential issues")
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n🎉 Test evaluation completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())