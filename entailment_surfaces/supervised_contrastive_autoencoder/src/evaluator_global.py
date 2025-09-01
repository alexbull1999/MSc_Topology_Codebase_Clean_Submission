"""
Model Evaluator for Global Dataset Contrastive Training
Clean implementation for comprehensive evaluation
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, accuracy_score,
    classification_report, confusion_matrix
)
from attention_autoencoder_model import AttentionAutoencoder
from data_loader_global import GlobalDataLoader
from contrastive_autoencoder_model_global import ContrastiveAutoencoder


class GlobalContrastiveEvaluator:
    """
    Comprehensive evaluator for the global dataset contrastive autoencoder
    """
    
    def __init__(self, model, device='cuda'):
        """
        Initialize evaluator
        
        Args:
            model: Trained ContrastiveAutoencoder model
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.evaluation_results = None
        
        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()
        
        print(f"GlobalContrastiveEvaluator initialized on {device}")
    
    def extract_latent_representations(self, dataloader):
        """
        Extract latent representations for entire dataset
        
        Args:
            dataloader: DataLoader for the dataset
            
        Returns:
            latent_representations: Tensor of latent vectors [N, latent_dim]
            labels: Tensor of labels [N]
        """
        all_latents = []
        all_labels = []
        
        print("Extracting latent representations...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['labels']
                
                # Get latent representations
                latent, _ = self.model(embeddings)
                
                all_latents.append(latent.cpu())
                all_labels.append(labels)
                
                if batch_idx % 50 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        latent_representations = torch.cat(all_latents, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        print(f"Extracted representations: {latent_representations.shape}")
        return latent_representations, labels
    
    def evaluate_clustering(self, latent_representations, labels):
        """
        Evaluate clustering performance in latent space
        
        Args:
            latent_representations: Latent vectors [N, latent_dim]
            labels: True labels [N]
            
        Returns:
            Dictionary with clustering metrics
        """
        print("Evaluating clustering performance...")
        
        # Convert to numpy for sklearn
        latent_np = latent_representations.numpy()
        labels_np = labels.numpy()

        if len(latent_np) > 50000:  # Only subsample if dataset is large
            indices = np.random.choice(len(latent_np), 50000, replace=False)
            latent_np = latent_np[indices]
            labels_np = labels_np[indices]
            print(f"Subsampled to {len(latent_np)} points for clustering evaluation")
        
        # K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_predictions = kmeans.fit_predict(latent_np)
        
        # Calculate clustering metrics
        silhouette = silhouette_score(latent_np, labels_np)
        adjusted_rand = adjusted_rand_score(labels_np, cluster_predictions)
        
        # Calculate clustering accuracy (best permutation)
        clustering_accuracy = self._calculate_clustering_accuracy(labels_np, cluster_predictions)
        
        # Calculate class centroids and distances
        class_centroids = {}
        class_names = ['entailment', 'neutral', 'contradiction']
        
        for i, class_name in enumerate(class_names):
            class_mask = labels_np == i
            if np.any(class_mask):
                centroid = np.mean(latent_np[class_mask], axis=0)
                class_centroids[class_name] = centroid
        
        # Calculate inter-class distances
        inter_class_distances = {}
        for i, class1 in enumerate(class_names):
            for j, class2 in enumerate(class_names):
                if i < j and class1 in class_centroids and class2 in class_centroids:
                    dist = np.linalg.norm(class_centroids[class1] - class_centroids[class2])
                    inter_class_distances[f"{class1}_to_{class2}"] = dist
        
        clustering_results = {
            'silhouette_score': silhouette,
            'adjusted_rand_score': adjusted_rand,
            'clustering_accuracy': clustering_accuracy,
            'class_centroids': {k: v.tolist() for k, v in class_centroids.items()},
            'inter_class_distances': inter_class_distances,
            'kmeans_predictions': cluster_predictions.tolist()
        }
        
        print(f"Clustering Results:")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Adjusted Rand Score: {adjusted_rand:.4f}")
        print(f"  Clustering Accuracy: {clustering_accuracy:.4f}")
        
        return clustering_results
    
    def _calculate_clustering_accuracy(self, true_labels, cluster_predictions):
        """
        Calculate clustering accuracy by finding best label permutation
        """
        from itertools import permutations
        
        best_accuracy = 0
        n_classes = len(np.unique(true_labels))
        
        # Try all possible permutations of cluster labels
        for perm in permutations(range(n_classes)):
            # Map cluster predictions to true labels using this permutation
            mapped_predictions = np.array([perm[pred] for pred in cluster_predictions])
            accuracy = accuracy_score(true_labels, mapped_predictions)
            best_accuracy = max(best_accuracy, accuracy)
        
        return best_accuracy
    
    def evaluate_classification(self, train_dataloader, val_dataloader):
        """
        Evaluate classification performance using latent representations
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            
        Returns:
            Dictionary with classification metrics
        """
        print("Evaluating classification performance...")
        
        # Extract latent representations for training
        print("  Extracting training representations...")
        train_latent, train_labels = self.extract_latent_representations(train_dataloader)
        
        # Extract latent representations for validation
        print("  Extracting validation representations...")
        val_latent, val_labels = self.extract_latent_representations(val_dataloader)
        
        # Convert to numpy
        X_train = train_latent.numpy()
        y_train = train_labels.numpy()
        X_val = val_latent.numpy()
        y_val = val_labels.numpy()

        # Subsample training to 50k, validation to 10k
        if len(X_train) > 50000:
            train_indices = np.random.choice(len(X_train), 50000, replace=False)
            X_train = X_train[train_indices]
            y_train = y_train[train_indices]
            print(f"Subsampled training to {len(X_train)} samples")

        if len(X_val) > 10000:
            val_indices = np.random.choice(len(X_val), 10000, replace=False)
            X_val = X_val[val_indices]
            y_val = y_val[val_indices]
            print(f"Subsampled validation to {len(X_val)} samples")
        
        print(f"  Training on {len(X_train)} samples, evaluating on {len(X_val)} samples")
        
        # Train k-NN classifier on training set
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        
        # Make predictions on validation set
        y_pred = knn.predict(X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        conf_matrix = confusion_matrix(y_val, y_pred)
        class_report = classification_report(y_val, y_pred, 
                                           target_names=['entailment', 'neutral', 'contradiction'],
                                           output_dict=True)
        
        classification_results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }
        
        print(f"Classification Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        
        # Fix the per-class F1 score extraction
        per_class_f1 = [
            class_report.get('entailment', {}).get('f1-score', 0.0),
            class_report.get('neutral', {}).get('f1-score', 0.0), 
            class_report.get('contradiction', {}).get('f1-score', 0.0)
        ]
        
        print(f"  Per-class F1: {per_class_f1}")
        
        return classification_results
    
    def evaluate_reconstruction(self, dataloader):
        """
        Evaluate reconstruction quality
        
        Args:
            dataloader: DataLoader for the dataset
            
        Returns:
            Dictionary with reconstruction metrics
        """
        print("Evaluating reconstruction quality...")
        
        total_mse = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                embeddings = batch['embeddings'].to(self.device)
                
                # Forward pass
                latent, reconstructed = self.model(embeddings)
                
                # Calculate MSE
                mse = torch.mean((reconstructed - embeddings) ** 2)
                total_mse += mse.item() * embeddings.size(0)
                total_samples += embeddings.size(0)
        
        average_mse = total_mse / total_samples
        
        reconstruction_results = {
            'average_mse': average_mse,
            'total_samples': total_samples
        }
        
        print(f"Reconstruction Results:")
        print(f"  Average MSE: {average_mse:.6f}")
        
        return reconstruction_results
    
    def evaluate_separation_quality(self, latent_representations, labels):
        """
        Evaluate separation quality in latent space
        
        Args:
            latent_representations: Latent vectors [N, latent_dim]
            labels: True labels [N]
            
        Returns:
            Dictionary with separation metrics
        """
        print("Evaluating separation quality...")
        
        # Convert to torch if needed
        if isinstance(latent_representations, np.ndarray):
            latent_representations = torch.from_numpy(latent_representations)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        # SUBSAMPLE if too large (same as clustering)
        if len(latent_representations) > 20000:
            indices = torch.randperm(len(latent_representations))[:20000]
            latent_representations = latent_representations[indices]
            labels = labels[indices]
            print(f"Subsampled to {len(latent_representations)} points for separation evaluation")
        
        # Move to CPU to save GPU memory
        latent_representations = latent_representations.cpu()
        labels = labels.cpu()

        # Compute pairwise distances
        distances = torch.cdist(latent_representations, latent_representations, p=2)
        
        # Create masks
        labels_expanded = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels_expanded, labels_expanded.T).float()
        mask_no_diagonal = 1 - torch.eye(len(labels))
        mask_positive = mask_positive * mask_no_diagonal
        mask_negative = (1 - torch.eq(labels_expanded, labels_expanded.T).float()) * mask_no_diagonal
        
        # Get positive and negative distances
        pos_distances = distances[mask_positive.bool()]
        neg_distances = distances[mask_negative.bool()]
        
        if len(pos_distances) == 0 or len(neg_distances) == 0:
            return {'error': 'No valid distances found'}
        
        # Calculate separation metrics
        pos_mean = pos_distances.mean().item()
        neg_mean = neg_distances.mean().item()
        if pos_mean == 0.0:
            print("  Warning: No positive distance separation detected (likely pure reconstruction model)")
            separation_ratio = float('inf') if neg_mean > 0 else 1.0
        else:
            separation_ratio = neg_mean / pos_mean
        gap = neg_distances.min().item() - pos_distances.max().item()
        
        separation_results = {
            'pos_mean': pos_mean,
            'pos_std': pos_distances.std().item(),
            'neg_mean': neg_mean,
            'neg_std': neg_distances.std().item(),
            'separation_ratio': separation_ratio,
            'gap': gap,
            'perfect_separation': gap > 0
        }
        
        print(f"Separation Results:")
        print(f"  Positive distances: {pos_mean:.3f} ± {pos_distances.std().item():.3f}")
        print(f"  Negative distances: {neg_mean:.3f} ± {neg_distances.std().item():.3f}")
        print(f"  Separation ratio: {separation_ratio:.2f}x")
        print(f"  Gap: {gap:.3f}")
        print(f"  Perfect separation: {'Yes' if gap > 0 else 'No'}")
        
        return separation_results
    
    def comprehensive_evaluation(self, train_dataloader, val_dataloader, test_dataloader):
        """
        Run comprehensive evaluation of the model
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            test_dataloader: Test data loader
            
        Returns:
            Dictionary with all evaluation results
        """
        print("Starting comprehensive evaluation...")
        print("=" * 60)
        
        # Extract latent representations for test set (main evaluation)
        test_latent, test_labels = self.extract_latent_representations(test_dataloader)
        
        # Run evaluations
        clustering_results = self.evaluate_clustering(test_latent, test_labels)
        classification_results = self.evaluate_classification(train_dataloader, val_dataloader)
        reconstruction_results = self.evaluate_reconstruction(test_dataloader)
        separation_results = self.evaluate_separation_quality(test_latent, test_labels)
        
        # Combine results
        self.evaluation_results = {
            'clustering': clustering_results,
            'classification': classification_results,
            'reconstruction': reconstruction_results,
            'separation': separation_results,
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'num_test_samples': len(test_latent),
                'latent_dimension': test_latent.shape[1],
                'model_type': type(self.model).__name__
            }
        }
        
        print("\nComprehensive evaluation completed!")
        return self.evaluation_results
    
    def print_summary(self):
        """Print a summary of evaluation results"""
        if self.evaluation_results is None:
            print("No evaluation results available. Run comprehensive_evaluation() first.")
            return
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        # Clustering performance
        clustering = self.evaluation_results['clustering']
        print(f"Clustering Performance:")
        print(f"  Silhouette Score: {clustering['silhouette_score']:.4f}")
        print(f"  Clustering Accuracy: {clustering['clustering_accuracy']:.4f}")
        print(f"  Adjusted Rand Score: {clustering['adjusted_rand_score']:.4f}")
        
        # Classification performance
        classification = self.evaluation_results['classification']
        print(f"\nClassification Performance:")
        print(f"  Accuracy: {classification['accuracy']:.4f}")
        
        # Separation quality
        separation = self.evaluation_results['separation']
        if 'error' not in separation:
            print(f"\nSeparation Quality:")
            print(f"  Separation Ratio: {separation['separation_ratio']:.2f}x")
            print(f"  Gap: {separation['gap']:.3f}")
            print(f"  Perfect Separation: {separation['perfect_separation']}")
        
        # Reconstruction quality
        reconstruction = self.evaluation_results['reconstruction']
        print(f"\nReconstruction Quality:")
        print(f"  Average MSE: {reconstruction['average_mse']:.6f}")
        
        print("="*60)
    
    def save_evaluation_results(self, save_dir):
        """
        Save evaluation results to file
        
        Args:
            save_dir: Directory to save results
        """
        if self.evaluation_results is None:
            print("No evaluation results to save. Run comprehensive_evaluation() first.")
            return None
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Convert the results
        serializable_results = convert_numpy_types(self.evaluation_results)
        
        # Save results as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(save_dir, f'evaluation_results_{timestamp}.json')
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Evaluation results saved to: {results_file}")
        return results_file


def load_data(config):
    """
    Load and prepare data
    """
    print("Loading data...")
    print("=" * 40)
    
    # Create data loader
    data_loader = GlobalDataLoader(
        train_path=config['data']['train_path'],
        val_path=config['data']['val_path'], 
        test_path=config['data']['test_path'],
        embedding_type=config['data']['embedding_type'],
        sample_size=config['data']['sample_size'],
        random_state=config['data']['random_state']
    )
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = data_loader.load_data()
        
    # Create data loaders
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        batch_size=config['data']['batch_size'],
        balanced_sampling=config['data']['balanced_sampling']
    )
    
    print(f"Data loading completed!")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def test_evaluator():
    """Test evaluator with synthetic model and data"""
    print("Testing GlobalContrastiveEvaluator...")
    
    
    # Create model
    model = ContrastiveAutoencoder(input_dim=1536, latent_dim=75, hidden_dims=[1024, 768, 512, 256, 128])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Load best model for evaluation
    gw_model_path = "entailment_surfaces/supervised_contrastive_autoencoder/experiments/FIXED_DECODERS/H0H1_signature_moor_lifted_autoencoder_no_attention_20250728_152242/checkpoints/best_model.pt"
    if os.path.exists(gw_model_path):
        print(f"Loading best model from {gw_model_path}")
        checkpoint = torch.load(gw_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = GlobalContrastiveEvaluator(model, device)
    
    config = {
        # Data configuration
        'data': {
            'train_path': 'data/processed/snli_full_standard_SBERT.pt',
            'val_path': 'data/processed/snli_full_standard_SBERT_validation.pt',
            'test_path': 'data/processed/snli_full_standard_SBERT_test.pt',
            'embedding_type': 'concat',  # 'lattice', 'concat', 'difference', 'cosine_concat'
            'batch_size': 1020,
            'sample_size': None,  # Use all data
            'balanced_sampling': True,
            'random_state': 42
        }
    }

    train_loader, val_loader, test_loader = load_data(config)

    # Run comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    results = evaluator.comprehensive_evaluation(test_loader, test_loader, test_loader)
    
    # Print summary
    evaluator.print_summary()
    
    # Test saving results
    save_dir = 'test_evaluation_results'
    results_file = evaluator.save_evaluation_results(save_dir)
    
    # Clean up
    import shutil
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    
    print("✅ Evaluator test completed!")


if __name__ == "__main__":
    test_evaluator()