"""
k-NN Classifier Optimization for Trained Contrastive Autoencoder
Test how different k-NN hyperparameters affect classification accuracy
"""

import sys
import os
import json
import torch
import numpy as np
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import itertools
from pathlib import Path
from contrastive_autoencoder_model_global import ContrastiveAutoencoder
from data_loader_global import GlobalDataLoader


class KNNClassifierOptimizer:
    """
    Optimize k-NN classifier hyperparameters for trained autoencoder
    """
    
    def __init__(self, model_path, config_path):
        """
        Initialize with trained model and configuration
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to model configuration file
        """
        self.model_path = model_path
        self.config_path = config_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load config first, then model (model loading depends on config)
        self.config = self._load_config()
        self.model = self._load_model()
        
        # Results storage
        self.results = []
        self.best_params = None
        self.best_accuracy = 0.0
        
        print(f"KNNClassifierOptimizer initialized on {self.device}")
        print(f"Model loaded from: {model_path}")
        print(f"Config loaded from: {config_path}")
        
    def _load_model(self):
        """Load trained model from checkpoint using config specifications"""
        print(f"Loading model from {self.model_path}")
        
        # Load config first to get model specifications
        config = self._load_config()
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model parameters from checkpoint
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            model_state_dict = checkpoint
        
        # Create model using config specifications (like in full_pipeline_global.py)
        model_config = config['model']
        print(f"Creating model with config: {model_config}")
        
        # Create model with exact config specifications
        model = ContrastiveAutoencoder(
            input_dim=model_config['input_dim'],
            latent_dim=model_config['latent_dim'],
            hidden_dims=model_config['hidden_dims'],
            dropout_rate=model_config['dropout_rate']
        )
        
        # Load weights
        model.load_state_dict(model_state_dict)
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully:")
        print(f"  Input dim: {model_config['input_dim']}")
        print(f"  Latent dim: {model_config['latent_dim']}")
        print(f"  Hidden dims: {model_config['hidden_dims']}")
        print(f"  Dropout rate: {model_config['dropout_rate']}")
        
        return model
    
    def _load_config(self):
        """Load configuration from file"""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        return config
    
    def extract_latent_representations(self, dataloader):
        """
        Extract latent representations from trained model
        
        Args:
            dataloader: DataLoader for dataset
            
        Returns:
            latent_features: numpy array of latent representations
            labels: numpy array of labels
        """
        print("Extracting latent representations...")
        
        all_latents = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['labels']
                
                # Get latent representations
                latent, _ = self.model(embeddings)
                
                all_latents.append(latent.cpu().numpy())
                all_labels.append(labels.numpy())
                
                if batch_idx % 20 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        # Concatenate all batches
        latent_features = np.concatenate(all_latents, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        print(f"Extracted {len(latent_features)} latent representations")
        return latent_features, labels
    
    def run_hyperparameter_search(self):
        """
        Run hyperparameter search over k-NN parameters
        """
        print("="*70)
        print("k-NN CLASSIFIER HYPERPARAMETER OPTIMIZATION")
        print("="*70)
        
        # Load data
        print("Loading data...")
        data_loader = GlobalDataLoader(
            train_path=self.config['data']['train_path'],
            val_path=self.config['data']['val_path'],
            test_path=self.config['data']['test_path'],
            embedding_type=self.config['data']['embedding_type'],
            batch_size=self.config['data']['batch_size'],
            sample_size=self.config['data']['sample_size'],
            random_state=self.config['data']['random_state']
        )
        
        # Get dataloaders
        train_dataset, val_dataset, test_dataset = data_loader.load_data()
        train_loader, val_loader, test_loader = data_loader.get_dataloaders(
            batch_size=self.config['data']['batch_size'],
            balanced_sampling=self.config['data']['balanced_sampling']
        )
        
        # Extract latent representations
        print("Extracting training representations...")
        X_train, y_train = self.extract_latent_representations(train_loader)
        
        print("Extracting validation representations...")
        X_val, y_val = self.extract_latent_representations(val_loader)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Define search space
        k_values = [3, 5, 7, 10, 15]
        weights = ['uniform', 'distance']
        metrics = ['euclidean', 'cosine', 'minkowski']
        
        print(f"\nSearch space:")
        print(f"  k_values: {k_values}")
        print(f"  weights: {weights}")
        print(f"  metrics: {metrics}")
        print(f"  Total combinations: {len(k_values) * len(weights) * len(metrics)}")
        
        # Current baseline (your best model result)
        baseline_accuracy = 0.8317
        print(f"  Baseline accuracy: {baseline_accuracy:.4f}")
        
        experiment_count = 0
        total_experiments = len(k_values) * len(weights) * len(metrics)
        
        for k in k_values:
            for weight in weights:
                for metric in metrics:
                    experiment_count += 1
                    
                    print(f"\n[{experiment_count}/{total_experiments}] Testing k={k}, weights={weight}, metric={metric}")
                    
                    try:
                        # Create and train classifier
                        classifier = KNeighborsClassifier(
                            n_neighbors=k,
                            weights=weight,
                            metric=metric
                        )
                        
                        classifier.fit(X_train, y_train)
                        y_pred = classifier.predict(X_val)
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_val, y_pred)
                        improvement = accuracy - baseline_accuracy
                        
                        # Get per-class performance
                        class_report = classification_report(
                            y_val, y_pred,
                            target_names=['entailment', 'neutral', 'contradiction'],
                            output_dict=True
                        )
                        
                        # Store result
                        result = {
                            'experiment_id': experiment_count,
                            'k': k,
                            'weights': weight,
                            'metric': metric,
                            'accuracy': accuracy,
                            'improvement': improvement,
                            'per_class_f1': {
                                'entailment': class_report['entailment']['f1-score'],
                                'neutral': class_report['neutral']['f1-score'],
                                'contradiction': class_report['contradiction']['f1-score']
                            },
                            'confusion_matrix': confusion_matrix(y_val, y_pred).tolist(),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        self.results.append(result)
                        
                        # Track best
                        if accuracy > self.best_accuracy:
                            self.best_accuracy = accuracy
                            self.best_params = {'k': k, 'weights': weight, 'metric': metric}
                            print(f"  NEW BEST: {accuracy:.4f} (+{improvement:.4f})")
                        else:
                            print(f"  Result: {accuracy:.4f} (+{improvement:.4f})")
                            
                        # Show per-class F1 scores
                        print(f"    Per-class F1: E={class_report['entailment']['f1-score']:.3f}, "
                              f"N={class_report['neutral']['f1-score']:.3f}, "
                              f"C={class_report['contradiction']['f1-score']:.3f}")
                        
                    except Exception as e:
                        print(f"  FAILED: {e}")
                        continue
        
        print(f"\n{'='*70}")
        print("HYPERPARAMETER SEARCH COMPLETED")
        print(f"{'='*70}")
        print(f"Best accuracy: {self.best_accuracy:.4f}")
        print(f"Best improvement: +{self.best_accuracy - baseline_accuracy:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        return self.best_params, self.best_accuracy
    
    def save_results(self, output_file=None):
        """
        Save all results to JSON file
        
        Args:
            output_file: Optional output filename
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"knn_optimization_results_{timestamp}.json"
        
        # Sort results by accuracy
        sorted_results = sorted(self.results, key=lambda x: x['accuracy'], reverse=True)
        
        summary = {
            'optimization_summary': {
                'baseline_accuracy': 0.8317,
                'best_accuracy': self.best_accuracy,
                'total_improvement': self.best_accuracy - 0.8317,
                'best_parameters': self.best_params,
                'total_experiments': len(self.results),
                'model_path': self.model_path,
                'config_path': self.config_path
            },
            'top_5_results': sorted_results[:5],
            'all_results': sorted_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save readable summary
        summary_file = output_file.replace('.json', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("k-NN CLASSIFIER OPTIMIZATION RESULTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Baseline accuracy: 83.17%\n")
            f.write(f"Best accuracy: {self.best_accuracy:.4f}\n")
            f.write(f"Total improvement: +{self.best_accuracy - 0.8317:.4f}\n\n")
            f.write(f"Best parameters:\n")
            for param, value in self.best_params.items():
                f.write(f"  {param}: {value}\n")
            f.write(f"\nTop 10 Results:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Rank':<4} {'Accuracy':<10} {'Improvement':<12} {'k':<4} {'Weights':<10} {'Metric':<12}\n")
            f.write("-" * 70 + "\n")
            
            for i, result in enumerate(sorted_results[:10]):
                f.write(f"{i+1:<4} {result['accuracy']:<10.4f} {result['improvement']:<12.4f} "
                       f"{result['k']:<4} {result['weights']:<10} {result['metric']:<12}\n")
        
        print(f"\nResults saved to: {output_file}")
        print(f"Summary saved to: {summary_file}")
        
        return output_file, summary_file


def main():
    """
    Main execution function
    """
    # You'll need to update these paths to point to your best model
    model_path = "entailment_surfaces/supervised_contrastive_autoencoder/experiments/coarse_embeddingcosine_concat_hiddendims[1024, 768, 512, 256, 128]_dropout0.2_optimAdam_lr0.0001_20250715_204239/checkpoints/best_model.pt"
    config_path = "entailment_surfaces/supervised_contrastive_autoencoder/experiments/coarse_embeddingcosine_concat_hiddendims[1024, 768, 512, 256, 128]_dropout0.2_optimAdam_lr0.0001_20250715_204239/config.json"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Please update the model_path variable to point to your best model checkpoint")
        return
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        print("Please update the config_path variable to point to your experiment config")
        return
    
    # Create optimizer
    optimizer = KNNClassifierOptimizer(model_path, config_path)
    
    # Run optimization
    best_params, best_accuracy = optimizer.run_hyperparameter_search()
    
    # Save results
    optimizer.save_results()
    
    print(f"\nOptimization complete!")
    print(f"Best k-NN parameters: {best_params}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Improvement over baseline: +{best_accuracy - 0.8317:.4f}")


if __name__ == "__main__":
    main()