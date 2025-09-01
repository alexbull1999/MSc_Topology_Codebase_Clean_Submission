"""
Evaluation Module for Supervised Contrastive Autoencoder
Handles clustering metrics, classification performance, and generative capabilities
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
import json


class ContrastiveAutoencoderEvaluator:
    """
    Comprehensive evaluator for the supervised contrastive autoencoder
    """
    
    def __init__(self, model, device='cuda'):
        """
        Initialize evaluator
        
        Args:
            model: Trained ContrastiveAutoencoder
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Store evaluation results
        self.evaluation_results = {}
        
        print(f"Evaluator initialized on device: {device}")
    
    def extract_latent_representations(self, dataloader):
        """
        Extract latent representations for all samples in dataloader
        
        Args:
            dataloader: DataLoader with samples to encode
            
        Returns:
            latent_representations: Tensor of latent vectors [N, latent_dim]
            labels: Tensor of true labels [N]
        """
        print("Extracting latent representations...")
        
        all_latent = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['labels']
                
                # Encode to latent space
                latent, _ = self.model(embeddings)
                
                all_latent.append(latent.cpu())
                all_labels.append(labels)
        
        # Concatenate all batches
        latent_representations = torch.cat(all_latent, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        print(f"Extracted {len(latent_representations)} latent representations")
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
        
        # K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_predictions = kmeans.fit_predict(latent_np)
        
        # Calculate clustering metrics
        silhouette = silhouette_score(latent_np, labels_np)
        adjusted_rand = adjusted_rand_score(labels_np, cluster_predictions)
        
        # Calculate clustering accuracy (best permutation)
        clustering_accuracy = self._calculate_clustering_accuracy(labels_np, cluster_predictions)
        
        # Calculate class centroids
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
                if i < j:
                    dist = np.linalg.norm(class_centroids[class1] - class_centroids[class2])
                    inter_class_distances[f"{class1}_to_{class2}"] = dist
        
        clustering_results = {
            'silhouette_score': silhouette,
            'adjusted_rand_score': adjusted_rand,
            'clustering_accuracy': clustering_accuracy,
            'class_centroids': class_centroids,
            'inter_class_distances': inter_class_distances,
            'kmeans_predictions': cluster_predictions
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
        Train classifier on training set, evaluate on validation set
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            
        Returns:
            Dictionary with classification metrics
        """
        print("Evaluating classification performance...")
        
        # Extract latent representations for training
        print("Extracting training latent representations...")
        train_latent, train_labels = self.extract_latent_representations(train_dataloader)
        
        # Extract latent representations for validation
        print("Extracting validation latent representations...")
        val_latent, val_labels = self.extract_latent_representations(val_dataloader)
        
        # Convert to numpy
        X_train = train_latent.numpy()
        y_train = train_labels.numpy()
        X_val = val_latent.numpy()
        y_val = val_labels.numpy()
        
        print(f"Training on {len(X_train)} samples, evaluating on {len(X_val)} samples")
        
        # Train k-NN classifier on training set
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        
        # Make predictions on validation set
        y_pred = knn.predict(X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        
        # Get classification report
        class_names = ['entailment', 'neutral', 'contradiction']
        report = classification_report(y_val, y_pred, target_names=class_names, output_dict=True)
        
        # Get confusion matrix
        conf_matrix = confusion_matrix(y_val, y_pred)
        
        classification_results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'y_true': y_val.tolist(),
            'y_pred': y_pred.tolist(),
            'train_samples': len(X_train),
            'val_samples': len(X_val)
        }
        
        print(f"Classification Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Per-class F1 scores:")
        for class_name in class_names:
            f1_score = report[class_name]['f1-score']
            print(f"    {class_name}: {f1_score:.4f}")
        
        return classification_results
    
    def evaluate_reconstruction(self, dataloader):
        """
        Evaluate reconstruction quality
        
        Args:
            dataloader: DataLoader with samples to reconstruct
            
        Returns:
            Dictionary with reconstruction metrics
        """
        print("Evaluating reconstruction quality...")
        
        total_mse = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                embeddings = batch['embeddings'].to(self.device)
                
                # Forward pass
                _, reconstructed = self.model(embeddings)
                
                # Calculate MSE
                mse = F.mse_loss(reconstructed, embeddings)
                total_mse += mse.item() * embeddings.size(0)
                total_samples += embeddings.size(0)
        
        avg_mse = total_mse / total_samples
        
        reconstruction_results = {
            'average_mse': avg_mse,
            'total_samples': total_samples
        }
        
        print(f"Reconstruction Results:")
        print(f"  Average MSE: {avg_mse:.6f}")
        
        return reconstruction_results
    
    def evaluate_clustering_phase1_style(self, latent_representations, labels, group_size=1000, num_groups=10):
        """
        Evaluate clustering using Phase 1 methodology with groups
        Tests whether groups of samples maintain the topological separation
        
        Args:
            latent_representations: Latent vectors [N, latent_dim]
            labels: True labels [N]
            group_size: Size of each group (1000 to match Phase 1)
            num_groups: Number of groups to test per class
            
        Returns:
            Dictionary with Phase 1 style clustering metrics
        """
        print("Evaluating Phase 1 style clustering (group-level)...")
        
        # Convert to numpy
        latent_np = latent_representations.numpy()
        labels_np = labels.numpy()
        
        # Sample groups for each class
        class_groups = {}
        class_names = ['entailment', 'neutral', 'contradiction']
        
        for class_idx, class_name in enumerate(class_names):
            class_samples = latent_np[labels_np == class_idx]
            
            if len(class_samples) < group_size:
                print(f"Warning: Not enough {class_name} samples for grouping ({len(class_samples)} < {group_size})")
                continue
            
            # Create multiple groups for this class
            groups = []
            for group_num in range(num_groups):
                start_idx = group_num * group_size
                end_idx = start_idx + group_size
                
                if end_idx <= len(class_samples):
                    group = class_samples[start_idx:end_idx]
                    groups.append(group)
                else:
                    # If not enough samples for full group, take remaining samples
                    if start_idx < len(class_samples):
                        group = class_samples[start_idx:]
                        groups.append(group)
                    break
            
            class_groups[class_name] = groups
            print(f"Created {len(groups)} groups for {class_name} (group size: {group_size})")
        
        # Compute group-level statistics (similar to your PH-dim analysis)
        group_statistics = {}
        
        for class_name, groups in class_groups.items():
            class_stats = []
            
            for group in groups:
                # Compute group-level metrics
                group_centroid = np.mean(group, axis=0)
                group_variance = np.var(group)
                group_mean_pairwise_distance = np.mean([
                    np.linalg.norm(group[i] - group[j]) 
                    for i in range(len(group)) 
                    for j in range(i+1, len(group))
                ])
                
                class_stats.append({
                    'centroid': group_centroid,
                    'variance': group_variance,
                    'mean_pairwise_distance': group_mean_pairwise_distance
                })
            
            group_statistics[class_name] = class_stats
        
        # Test clustering of group representatives (centroids)
        if len(class_groups) == 3:  # All classes have groups
            # Collect all group centroids
            all_centroids = []
            centroid_labels = []
            
            for class_idx, (class_name, groups) in enumerate(class_groups.items()):
                for group_stats in group_statistics[class_name]:
                    all_centroids.append(group_stats['centroid'])
                    centroid_labels.append(class_idx)
            
            all_centroids = np.array(all_centroids)
            centroid_labels = np.array(centroid_labels)
            
            # Cluster the group centroids
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            centroid_cluster_predictions = kmeans.fit_predict(all_centroids)
            
            # Calculate clustering metrics for group centroids
            centroid_silhouette = silhouette_score(all_centroids, centroid_labels)
            centroid_adjusted_rand = adjusted_rand_score(centroid_labels, centroid_cluster_predictions)
            centroid_clustering_accuracy = self._calculate_clustering_accuracy(
                centroid_labels, centroid_cluster_predictions
            )
            
            phase1_results = {
                'group_size': group_size,
                'num_groups_per_class': {name: len(groups) for name, groups in class_groups.items()},
                'group_statistics': {
                    name: {
                        'mean_variance': np.mean([stats['variance'] for stats in class_stats]),
                        'mean_pairwise_distance': np.mean([stats['mean_pairwise_distance'] for stats in class_stats]),
                        'std_variance': np.std([stats['variance'] for stats in class_stats]),
                        'std_pairwise_distance': np.std([stats['mean_pairwise_distance'] for stats in class_stats])
                    }
                    for name, class_stats in group_statistics.items()
                },
                'centroid_clustering': {
                    'silhouette_score': centroid_silhouette,
                    'adjusted_rand_score': centroid_adjusted_rand,
                    'clustering_accuracy': centroid_clustering_accuracy,
                    'num_centroids': len(all_centroids)
                }
            }
            
            print(f"Phase 1 Style Clustering Results:")
            print(f"  Group-level Silhouette Score: {centroid_silhouette:.4f}")
            print(f"  Group-level Clustering Accuracy: {centroid_clustering_accuracy:.4f}")
            print(f"  Group-level Adjusted Rand Score: {centroid_adjusted_rand:.4f}")
            print(f"  Total groups tested: {len(all_centroids)}")
            
        else:
            phase1_results = {
                'error': 'Insufficient samples for Phase 1 style evaluation',
                'available_classes': list(class_groups.keys())
            }
        
        return phase1_results

    def evaluate_generative_capabilities(self, latent_representations, labels):
        """
        Evaluate generative capabilities and manifold structure
        
        Args:
            latent_representations: Latent vectors [N, latent_dim]
            labels: True labels [N]
            
        Returns:
            Dictionary with generative evaluation results
        """
        print("Evaluating generative capabilities...")
        
        # Calculate class centroids
        class_centroids = {}
        class_names = ['entailment', 'neutral', 'contradiction']
        
        for i, class_name in enumerate(class_names):
            class_mask = labels == i
            if torch.any(class_mask):
                centroid = torch.mean(latent_representations[class_mask], dim=0)
                class_centroids[class_name] = centroid
        
        # Test transformation: contradiction -> entailment
        print("Testing contradiction -> entailment transformation...")
        
        # Find contradiction samples
        contradiction_mask = labels == 2
        contradiction_samples = latent_representations[contradiction_mask]
        
        if len(contradiction_samples) > 0:
            # Take first 10 contradiction samples
            test_samples = contradiction_samples[:min(10, len(contradiction_samples))]
            
            # Transform toward entailment centroid
            entailment_centroid = class_centroids['entailment']
            
            transformations = []
            for sample in test_samples:
                # Generate interpolation path
                path = self._generate_interpolation_path(
                    sample, entailment_centroid, num_steps=10
                )
                transformations.append(path)
            
            # Evaluate transformation quality
            transformation_quality = self._evaluate_transformation_quality(
                transformations, class_centroids
            )
            
            generative_results = {
                'class_centroids': {k: v.tolist() for k, v in class_centroids.items()},
                'transformation_quality': transformation_quality,
                'num_test_samples': len(test_samples)
            }
            
            print(f"Generative Results:")
            print(f"  Test samples: {len(test_samples)}")
            print(f"  Transformation quality: {transformation_quality:.4f}")
            
            return generative_results
        
        else:
            print("No contradiction samples found for transformation testing")
            return {'error': 'No contradiction samples available'}
    
    def _generate_interpolation_path(self, start_point, end_point, num_steps=10):
        """Generate interpolation path between two points"""
        alphas = torch.linspace(0, 1, num_steps)
        path = []
        
        for alpha in alphas:
            interpolated = (1 - alpha) * start_point + alpha * end_point
            path.append(interpolated)
        
        return path
    
    def _evaluate_transformation_quality(self, transformations, class_centroids):
        """
        Evaluate quality of transformations
        
        Args:
            transformations: List of interpolation paths
            class_centroids: Dictionary of class centroids
            
        Returns:
            Quality score (higher is better)
        """
        total_quality = 0
        
        for path in transformations:
            # Final point should be close to entailment centroid
            final_point = path[-1]
            distance_to_entailment = torch.norm(final_point - class_centroids['entailment'])
            
            # Quality is inverse of distance (closer = better)
            quality = 1.0 / (1.0 + distance_to_entailment.item())
            total_quality += quality
        
        return total_quality / len(transformations)
    
    def comprehensive_evaluation(self, train_dataloader, val_dataloader, test_dataloader):
        """
        Run comprehensive evaluation of the model
        
        Args:
            train_dataloader: DataLoader for training data (for classification training)
            val_dataloader: DataLoader for validation data (for classification evaluation)
            test_dataloader: DataLoader for test data (for final evaluation)
            
        Returns:
            Dictionary with all evaluation results
        """
        print("Starting comprehensive evaluation...")
        print("=" * 50)
        
        # Extract latent representations for test set (main evaluation)
        test_latent_representations, test_labels = self.extract_latent_representations(test_dataloader)
        
        # Run evaluations
        clustering_results = self.evaluate_clustering(test_latent_representations, test_labels)
        phase1_clustering_results = self.evaluate_clustering_phase1_style(test_latent_representations, test_labels)
        classification_results = self.evaluate_classification(train_dataloader, val_dataloader)
        reconstruction_results = self.evaluate_reconstruction(test_dataloader)
        generative_results = self.evaluate_generative_capabilities(test_latent_representations, test_labels)
        
        # Combine results
        self.evaluation_results = {
            'clustering_individual': clustering_results,
            'clustering_phase1_style': phase1_clustering_results,
            'classification': classification_results,
            'reconstruction': reconstruction_results,
            'generative': generative_results,
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'num_test_samples': len(test_latent_representations),
                'latent_dimension': test_latent_representations.shape[1]
            }
        }
        
        print("\nComprehensive evaluation completed!")
        return self.evaluation_results
    
    def save_evaluation_results(self, save_dir='evaluation_results'):
        """
        Save evaluation results to file
        
        Args:
            save_dir: Directory to save results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'evaluation_results_{timestamp}.json'
        filepath = os.path.join(save_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        results_to_save = self._prepare_results_for_json(self.evaluation_results)
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"Evaluation results saved to: {filepath}")
        return filepath
    
    def _prepare_results_for_json(self, results):
        """Prepare results for JSON serialization by converting numpy arrays and scalars"""
        if isinstance(results, dict):
            return {k: self._prepare_results_for_json(v) for k, v in results.items()}
        elif isinstance(results, list):
            return [self._prepare_results_for_json(item) for item in results]
        elif isinstance(results, np.ndarray):
            return results.tolist()
        elif isinstance(results, torch.Tensor):
            return results.tolist()
        elif isinstance(results, (np.float32, np.float64)):
            return float(results)
        elif isinstance(results, (np.int32, np.int64)):
            return int(results)
        elif isinstance(results, np.bool_):
            return bool(results)
        else:
            return results

    
    def print_summary(self):
        """Print a summary of evaluation results"""
        if not self.evaluation_results:
            print("No evaluation results available. Run comprehensive_evaluation() first.")
            return
        
        print("\nEVALUATION SUMMARY")
        print("=" * 50)
        
        # Individual clustering performance
        clustering_individual = self.evaluation_results['clustering_individual']
        print(f"Individual Clustering Performance:")
        print(f"  Silhouette Score: {clustering_individual['silhouette_score']:.4f}")
        print(f"  Clustering Accuracy: {clustering_individual['clustering_accuracy']:.4f}")
        print(f"  Adjusted Rand Score: {clustering_individual['adjusted_rand_score']:.4f}")
        
        # Phase 1 style clustering performance
        if 'clustering_phase1_style' in self.evaluation_results:
            phase1_clustering = self.evaluation_results['clustering_phase1_style']
            if 'error' not in phase1_clustering:
                print(f"\nPhase 1 Style Clustering Performance:")
                centroid_clustering = phase1_clustering['centroid_clustering']
                print(f"  Group-level Silhouette Score: {centroid_clustering['silhouette_score']:.4f}")
                print(f"  Group-level Clustering Accuracy: {centroid_clustering['clustering_accuracy']:.4f}")
                print(f"  Group-level Adjusted Rand Score: {centroid_clustering['adjusted_rand_score']:.4f}")
                print(f"  Total groups tested: {centroid_clustering['num_centroids']}")
            else:
                print(f"\nPhase 1 Style Clustering: {phase1_clustering['error']}")
        
        # Classification performance
        classification = self.evaluation_results['classification']
        print(f"\nClassification Performance:")
        print(f"  Accuracy: {classification['accuracy']:.4f}")
        
        # Reconstruction quality
        reconstruction = self.evaluation_results['reconstruction']
        print(f"\nReconstruction Quality:")
        print(f"  Average MSE: {reconstruction['average_mse']:.6f}")
        
        # Generative capabilities
        if 'generative' in self.evaluation_results and 'error' not in self.evaluation_results['generative']:
            generative = self.evaluation_results['generative']
            print(f"\nGenerative Capabilities:")
            print(f"  Transformation Quality: {generative['transformation_quality']:.4f}")
        
        print("\n" + "=" * 50)


# def test_evaluator():
#     """Test evaluator functionality"""
#     print("Testing Evaluator Module")
#     print("=" * 40)
    
#     # Import required components
#     from contrastive_autoencoder_model import ContrastiveAutoencoder
    
#     # Create model
#     model = ContrastiveAutoencoder(input_dim=768, latent_dim=75)
    
#     # Create evaluator
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     evaluator = ContrastiveAutoencoderEvaluator(model, device)
    
#     # Create synthetic test data
#     batch_size = 32
#     num_samples = 160
    
#     synthetic_embeddings = torch.randn(num_samples, 768)
#     synthetic_labels = torch.randint(0, 3, (num_samples,))
    
#     # Create dataset and dataloader
#     from torch.utils.data import DataLoader
    
#     class SyntheticDataset:
#         def __init__(self, embeddings, labels):
#             self.embeddings = embeddings
#             self.labels = labels
        
#         def __len__(self):
#             return len(self.embeddings)
        
#         def __getitem__(self, idx):
#             return {
#                 'embeddings': self.embeddings[idx],
#                 'labels': self.labels[idx]
#             }
    
#     test_dataset = SyntheticDataset(synthetic_embeddings, synthetic_labels)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     # Run comprehensive evaluation
#     results = evaluator.comprehensive_evaluation(test_loader, test_loader, test_loader)
    
#     # Print summary
#     evaluator.print_summary()
    
#     # Test saving results
#     evaluator.save_evaluation_results('entailment_surfaces/supervised_contrastive_autoencoder/test_evaluation_results')
    
#     print("\nEvaluator testing completed!")


# if __name__ == "__main__":
#     test_evaluator()