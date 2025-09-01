"""
Evaluator for InfoNCE + Order Embeddings autoencoder
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.cluster import KMeans


class InfoNCEOrderEvaluator:
    """
    Evaluator for InfoNCE + Order Embeddings approach
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def extract_features(self, data_loader):
        """Extract latent features and labels"""
        self.model.eval()
        
        all_latent_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                premise_embeddings = batch['premise_embedding'].to(self.device)
                hypothesis_embeddings = batch['hypothesis_embedding'].to(self.device)
                labels = batch['label']
                
                # Concatenate premise and hypothesis embeddings for the model
                premise_hyp_concat = torch.cat([premise_embeddings, hypothesis_embeddings], dim=1)
                
                # Get latent features
                latent_features, _ = self.model(premise_hyp_concat)
                
                all_latent_features.append(latent_features.cpu())
                all_labels.append(labels)
        
        features = torch.cat(all_latent_features, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        
        return features, labels
    
    def evaluate_classification(self, train_loader, val_loader):
        """
        Evaluate classification performance using k-NN on latent representations
        """
        print("Evaluating classification performance...")
        
        # Extract training features
        print("  Extracting training representations...")
        train_features, train_labels = self.extract_features(train_loader)
        
        # Extract validation features  
        print("  Extracting validation representations...")
        val_features, val_labels = self.extract_features(val_loader)
        
        # Subsample if datasets are large
        if len(train_features) > 50000:
            indices = np.random.choice(len(train_features), 50000, replace=False)
            train_features = train_features[indices]
            train_labels = train_labels[indices]
            print(f"  Subsampled training to {len(train_features)} samples")
        
        if len(val_features) > 10000:
            indices = np.random.choice(len(val_features), 10000, replace=False)
            val_features = val_features[indices]
            val_labels = val_labels[indices]
            print(f"  Subsampled validation to {len(val_features)} samples")
        
        # Train k-NN classifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(train_features, train_labels)
        
        # Make predictions
        predictions = knn.predict(val_features)
        
        # Calculate metrics
        accuracy = accuracy_score(val_labels, predictions)
        conf_matrix = confusion_matrix(val_labels, predictions)
        class_report = classification_report(
            val_labels, predictions,
            target_names=['entailment', 'neutral', 'contradiction'],
            output_dict=True
        )
        
        classification_results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }
        
        print(f"Classification Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        
        # Extract per-class F1 scores
        per_class_f1 = [
            class_report.get('entailment', {}).get('f1-score', 0.0),
            class_report.get('neutral', {}).get('f1-score', 0.0),
            class_report.get('contradiction', {}).get('f1-score', 0.0)
        ]
        print(f"  Per-class F1: {per_class_f1}")
        
        return classification_results
    
    def evaluate_clustering(self, features, labels):
        """
        Evaluate clustering performance using k-means
        """
        print("Evaluating clustering performance...")
        
        # Subsample if dataset is large
        if len(features) > 50000:
            indices = np.random.choice(len(features), 50000, replace=False)
            features = features[indices]
            labels = labels[indices]
            print(f"  Subsampled to {len(features)} points for clustering")
        
        # K-means clustering
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, adjusted_rand_score
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_predictions = kmeans.fit_predict(features)
        
        # Calculate metrics
        silhouette = silhouette_score(features, labels)
        adjusted_rand = adjusted_rand_score(labels, cluster_predictions)
        
        # Calculate clustering accuracy (best permutation)
        clustering_accuracy = self._calculate_clustering_accuracy(labels, cluster_predictions)
        
        clustering_results = {
            'clustering_accuracy': clustering_accuracy,
            'silhouette_score': silhouette,
            'adjusted_rand_score': adjusted_rand
        }
        
        print(f"Clustering Results:")
        print(f"  Clustering Accuracy: {clustering_accuracy:.4f}")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Adjusted Rand Score: {adjusted_rand:.4f}")
        
        return clustering_results
    
    def _calculate_clustering_accuracy(self, true_labels, cluster_predictions):
        """Calculate clustering accuracy by finding best label permutation"""
        from itertools import permutations
        from sklearn.metrics import accuracy_score
        
        best_accuracy = 0
        n_classes = len(np.unique(true_labels))
        
        # Try all possible permutations of cluster labels
        for perm in permutations(range(n_classes)):
            # Map cluster predictions to true labels using this permutation
            mapped_predictions = np.array([perm[pred] for pred in cluster_predictions])
            accuracy = accuracy_score(true_labels, mapped_predictions)
            best_accuracy = max(best_accuracy, accuracy)
        
        return best_accuracy
    
    def evaluate(self, data_loader):
        """
        Full evaluation including both clustering and classification
        """
        features, labels = self.extract_features(data_loader)
        
        # Clustering evaluation
        clustering_results = self.evaluate_clustering(features, labels)
        
        # For comprehensive evaluation, we'd need separate train/val loaders for classification
        # For now, just return clustering + separation metrics
        
        # Distance-based metrics (separation quality)
        pos_distances = []
        neg_distances = []
        
        print("Calculating separation metrics...")
        
        # Subsample for distance calculation if too large
        if len(features) > 20000:
            indices = np.random.choice(len(features), 20000, replace=False)
            features_sub = features[indices]
            labels_sub = labels[indices]
        else:
            features_sub = features
            labels_sub = labels
        
        for i in range(len(features_sub)):
            for j in range(i+1, len(features_sub)):
                dist = np.linalg.norm(features_sub[i] - features_sub[j])
                if labels_sub[i] == labels_sub[j]:
                    pos_distances.append(dist)
                else:
                    neg_distances.append(dist)
        
        if pos_distances and neg_distances:
            separation_ratio = np.mean(neg_distances) / np.mean(pos_distances)
            gap = np.min(neg_distances) - np.max(pos_distances)
        else:
            separation_ratio = 0
            gap = 0
        
        results = {
            **clustering_results,
            'separation_ratio': separation_ratio,
            'pos_distance_mean': np.mean(pos_distances) if pos_distances else 0,
            'neg_distance_mean': np.mean(neg_distances) if neg_distances else 0,
            'gap': gap,
            'num_samples': len(features)
        }
        
        print(f"Separation Results:")
        print(f"  Separation Ratio: {separation_ratio:.2f}x")
        print(f"  Gap: {gap:.3f}")
        
        return results
    
    def evaluate_detailed(self, test_loader, train_loader=None, val_loader=None):
        """
        Detailed evaluation including both clustering and classification (if train/val provided)
        """
        print("Running detailed evaluation...")
        
        # Always do clustering evaluation on test set
        test_features, test_labels = self.extract_features(test_loader)
        clustering_results = self.evaluate_clustering(test_features, test_labels)
        
        # Add separation metrics
        separation_results = self.evaluate_separation_metrics(test_features, test_labels)
        
        results = {**clustering_results, **separation_results}
        
        # If train/val loaders provided, also do classification evaluation
        if train_loader is not None and val_loader is not None:
            classification_results = self.evaluate_classification(train_loader, val_loader)
            results.update(classification_results)
        
        return results
    
    def evaluate_separation_metrics(self, features, labels):
        """Calculate separation quality metrics"""
        print("Calculating separation metrics...")
        
        # Subsample if too large
        if len(features) > 20000:
            indices = np.random.choice(len(features), 20000, replace=False)
            features = features[indices]
            labels = labels[indices]
        
        pos_distances = []
        neg_distances = []
        
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                dist = np.linalg.norm(features[i] - features[j])
                if labels[i] == labels[j]:
                    pos_distances.append(dist)
                else:
                    neg_distances.append(dist)
        
        if pos_distances and neg_distances:
            separation_ratio = np.mean(neg_distances) / np.mean(pos_distances)
            gap = np.min(neg_distances) - np.max(pos_distances)
            
            return {
                'separation_ratio': separation_ratio,
                'pos_distance_mean': np.mean(pos_distances),
                'neg_distance_mean': np.mean(neg_distances), 
                'gap': gap
            }
        else:
            return {
                'separation_ratio': 0,
                'pos_distance_mean': 0,
                'neg_distance_mean': 0,
                'gap': 0
            }