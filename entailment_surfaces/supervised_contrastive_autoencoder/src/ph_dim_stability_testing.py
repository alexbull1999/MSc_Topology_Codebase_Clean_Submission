
"""
PH Dimension Stability Testing Script

Tests the stability of PH dimensions across multiple runs for different models.
Extracts latent embeddings and calculates PH dimensions using multiple random seeds.
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import json
from datetime import datetime
import time
from contrastive_autoencoder_model_global import ContrastiveAutoencoder
from attention_autoencoder_model import AttentionAutoencoder
from data_loader_global import GlobalDataLoader

# Add project paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import your topology functions
from phd_method.src_phd.topology import ph_dim_from_distance_matrix


class PHDimensionStabilityTester:
    """
    Test PH dimension stability across multiple runs for different models
    """
    
    def __init__(self, 
                 data_config: Dict,
                 n_runs: int = 10,
                 min_points: int = 200,
                 max_points: int = 1000,
                 point_jump: int = 50,
                 seed_base: int = 42):
        """
        Initialize stability tester
        
        Args:
            data_config: Configuration for data loading
            n_runs: Number of random runs to test stability
            min_points: Minimum points for PH calculation
            max_points: Maximum points for PH calculation  
            point_jump: Point jump for PH calculation
            seed_base: Base seed for reproducibility
        """
        self.data_config = data_config
        self.n_runs = n_runs
        self.min_points = min_points
        self.max_points = max_points
        self.point_jump = point_jump
        self.seed_base = seed_base
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load test data for PH dimension calculation"""
        print("Loading data...")
        
        # Create data loader
        data_loader = GlobalDataLoader(
            train_path=self.data_config['train_path'],
            val_path=self.data_config['val_path'],
            test_path=self.data_config['test_path'],
            embedding_type=self.data_config['embedding_type'],
            sample_size=self.data_config.get('sample_size', 5000)
        )
        
        # Load datasets
        train_dataset, val_dataset, test_dataset = data_loader.load_data()
        
        # Create data loaders
        train_loader, val_loader, test_loader = data_loader.get_dataloaders(
            batch_size=3000,  # Large batch to get full classes
            balanced_sampling=True
        )
        
        # Store the data loader for extracting embeddings
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        print(f"Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    def extract_latent_embeddings_sample(self, model_path: str, seed: int) -> Dict[str, torch.Tensor]:
        """
        Extract a random sample of latent embeddings for each class from a trained model
        
        Args:
            model_path: Path to trained model checkpoint
            seed: Random seed for sampling reproducibility
            
        Returns:
            Dictionary mapping class names to latent embeddings
        """
        print(f"  Extracting latent embeddings (seed={seed})")
        
        # Set random seed for reproducible sampling
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model (you may need to adjust these parameters based on your model)
        model = AttentionAutoencoder(
            input_dim=1536,  # Adjust based on your embedding dimension
            latent_dim=75,   # Adjust based on your latent dimension
            hidden_dims=[1024, 768, 512, 256, 128]  # Adjust based on your architecture
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # Extract ALL embeddings by class first
        all_class_embeddings = {'entailment': [], 'neutral': [], 'contradiction': []}
        class_names = ['entailment', 'neutral', 'contradiction']
        
        with torch.no_grad():
            # Use training data for consistency
            for batch in self.train_loader:
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Get latent representations
                latent_features = model.encode(embeddings)
                
                # Group by class
                for class_idx, class_name in enumerate(class_names):
                    class_mask = (labels == class_idx)
                    if class_mask.sum() > 0:
                        all_class_embeddings[class_name].append(latent_features[class_mask].cpu())
        
        # Combine all embeddings for each class
        combined_embeddings = {}
        for class_name in class_names:
            if all_class_embeddings[class_name]:
                combined = torch.cat(all_class_embeddings[class_name], dim=0)
                combined_embeddings[class_name] = combined
            else:
                combined_embeddings[class_name] = torch.empty((0, model.latent_dim))
        
        # Now sample from each class
        sampled_embeddings = {}
        for class_name, embeddings in combined_embeddings.items():
            n_samples = len(embeddings)
            if n_samples < self.min_points:
                sampled_embeddings[class_name] = embeddings
            else:
                # Sample up to max_points, but at least min_points
                sample_size = min(self.max_points, n_samples)
                sample_size = max(sample_size, self.min_points)
                
                # Random sampling with current seed
                indices = torch.randperm(n_samples)[:sample_size]
                sampled_embeddings[class_name] = embeddings[indices]
        
        return sampled_embeddings
    
    def calculate_ph_dimension(self, embeddings: torch.Tensor) -> float:
        """
        Calculate PH dimension for given embeddings
        
        Args:
            embeddings: Tensor of embeddings [n_samples, dim]
            
        Returns:
            PH dimension value
        """
        if len(embeddings) < self.min_points:
            return np.nan
        
        try:
            # Convert to numpy
            embeddings_np = embeddings.numpy()
            
            # Compute distance matrix
            distance_matrix = pairwise_distances(embeddings_np, metric='euclidean')
            
            # Calculate PH dimension
            ph_dim = ph_dim_from_distance_matrix(
                distance_matrix,
                min_points=self.min_points,
                max_points=min(self.max_points, len(embeddings)),
                point_jump=self.point_jump,
                h_dim=0,  # H0 dimension
                alpha=1.0
            )
            
            return ph_dim
            
        except Exception as e:
            print(f"    Error calculating PH dimension: {e}")
            return np.nan
    
    def test_model_stability(self, model_path: str, model_name: str) -> Dict:
        """
        Test PH dimension stability for a single model across multiple runs
        
        Args:
            model_path: Path to model checkpoint
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary containing stability results
        """
        print(f"\n{'='*60}")
        print(f"TESTING STABILITY: {model_name}")
        print(f"{'='*60}")
        
        # Test stability across multiple runs with different samples
        results = {
            'model_name': model_name,
            'model_path': model_path,
            'n_runs': self.n_runs,
            'class_results': {}
        }
        
        # We'll collect results for each class across all runs
        class_ph_dimensions = {'entailment': [], 'neutral': [], 'contradiction': []}
        
        # Run multiple times with different seeds (different samples each time)
        for run_idx in range(self.n_runs):
            seed = self.seed_base + run_idx
            print(f"\n  Run {run_idx+1}/{self.n_runs} (seed={seed})")
            
            # Extract DIFFERENT sample of latent embeddings for this run
            latent_embeddings = self.extract_latent_embeddings_sample(model_path, seed)
            
            # Calculate PH dimension for each class in this run
            for class_name, embeddings in latent_embeddings.items():
                if len(embeddings) < self.min_points:
                    ph_dim = np.nan
                    print(f"    {class_name}: SKIP (only {len(embeddings)} samples)")
                else:
                    ph_dim = self.calculate_ph_dimension(embeddings)
                    print(f"    {class_name}: {ph_dim:.4f} ({len(embeddings)} samples)")
                
                class_ph_dimensions[class_name].append(ph_dim)
        
        # Calculate statistics for each class
        for class_name in ['entailment', 'neutral', 'contradiction']:
            ph_dimensions = np.array(class_ph_dimensions[class_name])
            valid_dims = ph_dimensions[~np.isnan(ph_dimensions)]
            
            print(f"\n  {class_name.upper()} STABILITY ANALYSIS:")
            
            if len(valid_dims) > 0:
                stats = {
                    'mean': float(np.mean(valid_dims)),
                    'std': float(np.std(valid_dims)),
                    'min': float(np.min(valid_dims)),
                    'max': float(np.max(valid_dims)),
                    'range': float(np.max(valid_dims) - np.min(valid_dims)),
                    'cv': float(np.std(valid_dims) / np.mean(valid_dims)) if np.mean(valid_dims) != 0 else np.inf,
                    'valid_runs': len(valid_dims),
                    'all_values': valid_dims.tolist()
                }
                
                print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"    Coefficient of Variation: {stats['cv']:.4f}")
                print(f"    Valid runs: {stats['valid_runs']}/{self.n_runs}")
                
                # Assess stability
                if stats['cv'] < 0.1:
                    stability = "EXCELLENT"
                elif stats['cv'] < 0.3:
                    stability = "GOOD"
                elif stats['cv'] < 0.5:
                    stability = "MODERATE"
                else:
                    stability = "POOR"
                
                print(f"    Stability: {stability}")
                stats['stability'] = stability
                
            else:
                stats = {'error': 'All runs failed'}
                print(f"    ERROR: All runs failed")
            
            results['class_results'][class_name] = stats
        
        return results
    
    def test_multiple_models(self, model_configs: List[Dict]) -> List[Dict]:
        """
        Test stability across multiple models
        
        Args:
            model_configs: List of dicts with 'path' and 'name' keys
            
        Returns:
            List of results for each model
        """
        all_results = []
        
        print(f"Testing PH dimension stability across {len(model_configs)} models")
        print(f"Number of runs per model: {self.n_runs}")
        print(f"Each run uses a DIFFERENT random sample of latent embeddings")
        
        for config in model_configs:
            if not os.path.exists(config['path']):
                print(f"Warning: Model not found: {config['path']}")
                continue
                
            results = self.test_model_stability(config['path'], config['name'])
            all_results.append(results)
        
        return all_results
    
    def save_results(self, results: List[Dict], output_path: str):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_data = {
            'timestamp': timestamp,
            'test_config': {
                'n_runs': self.n_runs,
                'min_points': self.min_points,
                'max_points': self.max_points,
                'point_jump': self.point_jump,
                'seed_base': self.seed_base,
                'note': 'Each run uses different random sample of latent embeddings'
            },
            'results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def print_summary(self, results: List[Dict]):
        """Print summary of all results"""
        print(f"\n{'='*80}")
        print("PH DIMENSION STABILITY SUMMARY")
        print(f"{'='*80}")
        
        for result in results:
            print(f"\nModel: {result['model_name']}")
            print("-" * 50)
            
            for class_name, stats in result['class_results'].items():
                if 'error' in stats:
                    print(f"  {class_name}: ERROR")
                else:
                    print(f"  {class_name}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                          f"(CV: {stats['cv']:.3f}, {stats['stability']})")


def main():
    """Main execution function"""
    
    # =================================================================
    # CONFIGURATION - EDIT THESE VALUES MANUALLY
    # =================================================================
    
    # Data configuration
    data_config = {
        'train_path': 'data/processed/snli_full_standard_SBERT.pt',
        'val_path': 'data/processed/snli_full_standard_SBERT_validation.pt',
        'test_path': 'data/processed/snli_full_standard_SBERT_test.pt',
        'embedding_type': 'concat',
        'sample_size': 5000
    }
    
    model_path = "entailment_surfaces/supervised_contrastive_autoencoder/experiments/FIXED_DECODERS/pure_moor_topological_autoencoder_attention_20250724_222029/checkpoints/best_model.pt"
    model_name = "moor_pure_topological_attention"
    
    # Test configuration
    n_runs = 10  # Number of random runs for stability testing
    output_dir = 'entailment_surfaces/supervised_contrastive_autoencoder/phd_stability_results'
    
    # =================================================================
    # END CONFIGURATION
    # =================================================================
    
    print("PH Dimension Stability Testing")
    print("="*50)
    print(f"Number of runs per model: {n_runs}")
    print(f"Output directory: {output_dir}")
    print("FIXED: Each run now uses different random sample of latent embeddings")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tester
    tester = PHDimensionStabilityTester(
        data_config=data_config,
        n_runs=n_runs
    )
    
    # Test all models
    results = tester.test_model_stability(model_path, model_name)
    
    # Save results
    output_path = os.path.join(output_dir, f'{model_name}_stability_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    tester.save_results(results, output_path)
    
    # Print summary
    tester.print_summary([results])


if __name__ == "__main__":
    main()