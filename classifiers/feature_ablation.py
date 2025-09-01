
"""
Feature Ablation Analysis for TDA Neural Network Classifiers

This script systematically tests different feature combinations to determine
which features help vs hinder classification performance.

With the current 14D enhanced feature vector there are:
- 10 geometric/asymmetric features 
- 4 topological (landmark_tda) features

Usage:
    python feature_ablation.py --data_path path/to/data.pt [options]
"""

import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import your existing classifier components
from neural_classifier_landmark_asymmetry import (
    EnhancedTDANeuralClassifier, 
    EnhancedFeatureNormalizer
)

@dataclass
class AblationResult:
    """Store results for each feature combination"""
    feature_set: str
    feature_indices: List[int]
    num_features: int
    accuracy_mean: float
    accuracy_std: float
    f1_mean: float
    f1_std: float
    improvement_over_baseline: float

class FeatureAblationAnalyzer:
    """
    Systematically test different feature combinations to identify 
    which features contribute positively vs negatively to performance.
    """
    
    def __init__(self, data_path: str, random_seed: int = 42, output_dir: str = "results"):
        self.data_path = data_path
        self.random_seed = random_seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'feature_ablation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Feature groupings based on your current implementation
        self.feature_groups = {
            'all_features': list(range(14)),  # All 14 features
            'geometric_only': list(range(10)),  # First 10 are geometric
            'topological_only': list(range(10, 14)),  # Last 4 are TDA
            'core_geometric': [0, 1, 2, 3, 4],  # Core cone/order features
            'asymmetric_only': [5, 6, 7, 8, 9],  # Asymmetric features
            'persistence_only': [10, 11],  # H0/H1 persistence
            'tda_structure': [12, 13],  # Max persistence + feature count
            'no_tda': list(range(10)),  # Geometric without TDA
            'no_asymmetric': [0, 1, 2, 3, 4] + list(range(10, 14)),  # Remove asymmetric
            'minimal_set': [0, 1, 10, 11],  # Minimal effective set
        }
        
        self.feature_names = [
            # Geometric features (0-9)
            'cone_energy', 'order_energy', 'hyperbolic_distance', 
            'entailment_score', 'violation_count',
            'forward_cone', 'backward_cone', 'cone_asymmetry',
            'forward_energy', 'backward_energy',
            # Topological features (10-13)  
            'h0_persistence', 'h1_persistence', 
            'h1_max_persistence', 'h1_feature_count'
        ]
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load the enhanced classifier data"""
        self.logger.info(f"Loading data from: {self.data_path}")
        
        try:
            data = torch.load(self.data_path, map_location='cpu')  # Load to CPU first
            features = data['features'].numpy()
            
            # Convert labels to numeric
            labels = data['labels']
            label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
            numeric_labels = np.array([label_to_idx[label] for label in labels])
            
            self.logger.info(f"Loaded data: {features.shape[0]} samples, {features.shape[1]} features")
            self.logger.info(f"Class distribution: {np.bincount(numeric_labels)}")
            
            return features, numeric_labels
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def evaluate_feature_subset(self, 
                              features: np.ndarray, 
                              labels: np.ndarray,
                              feature_indices: List[int],
                              feature_set_name: str,
                              n_folds: int = 5) -> AblationResult:
        """
        Evaluate a specific subset of features using cross-validation
        """
        self.logger.info(f"Evaluating feature set: {feature_set_name}")
        self.logger.info(f"Features ({len(feature_indices)}): {[self.feature_names[i] for i in feature_indices]}")
        
        # Select feature subset
        X_subset = features[:, feature_indices]
        
        # Cross-validation setup
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        fold_accuracies = []
        fold_f1_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_subset, labels)):
            self.logger.info(f"  Fold {fold + 1}/{n_folds}")
            
            X_train, X_val = X_subset[train_idx], X_subset[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            # Normalize features
            normalizer = EnhancedFeatureNormalizer()
            normalizer.fit(X_train, [self.feature_names[i] for i in feature_indices])
            X_train_norm = normalizer.transform(X_train)
            X_val_norm = normalizer.transform(X_val)
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train_norm)
            y_train_tensor = torch.LongTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val_norm)
            y_val_tensor = torch.LongTensor(y_val)
            
            # Create and train model
            model = EnhancedTDANeuralClassifier(input_dim=len(feature_indices))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            
            # Training loop (simplified for speed)
            model.train()
            for epoch in range(50):  # Reduced epochs for ablation speed
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                _, predicted = torch.max(val_outputs.data, 1)
                
                accuracy = accuracy_score(y_val, predicted.numpy())
                f1 = f1_score(y_val, predicted.numpy(), average='macro')
                
                fold_accuracies.append(accuracy)
                fold_f1_scores.append(f1)
                
                self.logger.info(f"    Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Calculate statistics
        accuracy_mean = np.mean(fold_accuracies)
        accuracy_std = np.std(fold_accuracies)
        f1_mean = np.mean(fold_f1_scores)
        f1_std = np.std(fold_f1_scores)
        
        self.logger.info(f"  Results: {accuracy_mean*100:.2f}±{accuracy_std*100:.2f}% accuracy")
        
        return AblationResult(
            feature_set=feature_set_name,
            feature_indices=feature_indices,
            num_features=len(feature_indices),
            accuracy_mean=accuracy_mean,
            accuracy_std=accuracy_std,
            f1_mean=f1_mean,
            f1_std=f1_std,
            improvement_over_baseline=0.0  # Will be calculated later
        )
    
    def run_comprehensive_ablation(self) -> pd.DataFrame:
        """
        Run ablation study on all feature combinations
        """
        self.logger.info("Starting comprehensive feature ablation analysis...")
        
        # Load data
        features, labels = self.load_data()
        
        # Test each feature combination
        for i, (feature_set_name, feature_indices) in enumerate(self.feature_groups.items()):
            self.logger.info(f"[{i+1}/{len(self.feature_groups)}] Testing: {feature_set_name}")
            
            result = self.evaluate_feature_subset(
                features, labels, feature_indices, feature_set_name
            )
            self.results.append(result)
        
        # Convert to DataFrame and calculate improvements
        results_df = pd.DataFrame([
            {
                'Feature Set': r.feature_set,
                'Num Features': r.num_features,
                'Accuracy (%)': f"{r.accuracy_mean*100:.2f} ± {r.accuracy_std*100:.2f}",
                'F1 Score': f"{r.f1_mean:.3f} ± {r.f1_std:.3f}",
                'Accuracy Mean': r.accuracy_mean,
                'F1 Mean': r.f1_mean,
                'Features Used': [self.feature_names[i] for i in r.feature_indices]
            }
            for r in self.results
        ])
        
        # Calculate improvement over baseline (geometric_only)
        baseline_acc = results_df[results_df['Feature Set'] == 'geometric_only']['Accuracy Mean'].iloc[0]
        results_df['Improvement vs Geometric (%)'] = (results_df['Accuracy Mean'] - baseline_acc) * 100
        
        # Sort by accuracy
        results_df = results_df.sort_values('Accuracy Mean', ascending=False)
        
        return results_df
    
    def analyze_feature_importance(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze which specific features contribute most to performance
        """
        analysis = {
            'best_performance': results_df.iloc[0],
            'geometric_vs_all': None,
            'tda_contribution': None,
            'asymmetric_contribution': None,
            'minimal_effective_set': None
        }
        
        # Compare geometric-only vs all features
        geometric_acc = results_df[results_df['Feature Set'] == 'geometric_only']['Accuracy Mean'].iloc[0]
        all_features_acc = results_df[results_df['Feature Set'] == 'all_features']['Accuracy Mean'].iloc[0]
        analysis['geometric_vs_all'] = {
            'geometric_only': geometric_acc,
            'all_features': all_features_acc,
            'improvement': (all_features_acc - geometric_acc) * 100,
            'tda_helps': all_features_acc > geometric_acc
        }
        
        # TDA contribution analysis
        no_tda_acc = results_df[results_df['Feature Set'] == 'no_tda']['Accuracy Mean'].iloc[0]
        tda_only_acc = results_df[results_df['Feature Set'] == 'topological_only']['Accuracy Mean'].iloc[0]
        analysis['tda_contribution'] = {
            'without_tda': no_tda_acc,
            'tda_only': tda_only_acc,
            'with_tda': all_features_acc,
            'tda_improvement': (all_features_acc - no_tda_acc) * 100,
            'tda_standalone_viability': tda_only_acc
        }
        
        return analysis
    
    def create_visualization(self, results_df: pd.DataFrame) -> None:
        """
        Create visualizations of ablation results
        """
        plt.style.use('default')  # Ensure compatibility
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Performance by feature set
        feature_sets = results_df['Feature Set']
        accuracies = results_df['Accuracy Mean'] * 100
        
        ax1.barh(feature_sets, accuracies)
        ax1.set_xlabel('Accuracy (%)')
        ax1.set_title('Performance by Feature Set')
        ax1.grid(True, alpha=0.3)
        
        # 2. Number of features vs performance
        ax2.scatter(results_df['Num Features'], results_df['Accuracy Mean'] * 100)
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Features vs Performance')
        ax2.grid(True, alpha=0.3)
        
        # 3. Improvement over baseline
        improvements = results_df['Improvement vs Geometric (%)']
        colors = ['green' if x > 0 else 'red' for x in improvements]
        ax3.barh(feature_sets, improvements, color=colors, alpha=0.7)
        ax3.set_xlabel('Improvement over Geometric-Only (%)')
        ax3.set_title('Feature Set Improvements')
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        
        # 4. F1 Score comparison
        f1_scores = results_df['F1 Mean']
        ax4.barh(feature_sets, f1_scores)
        ax4.set_xlabel('F1 Score')
        ax4.set_title('F1 Score by Feature Set')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'feature_ablation_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Visualization saved to: {plot_path}")
        
        # Don't show plot in headless environment
        # plt.show()


def main():
    """
    Main execution function
    """
    parser = argparse.ArgumentParser(description='Feature Ablation Analysis for TDA Neural Classifier')
    parser.add_argument('--data_path', type=str, 
                       default="results/tda_integration/landmark_tda_features/enhanced_neural_network_features_snli_10k.pt",
                       help='Path to the enhanced neural network features file (.pt)')
    parser.add_argument('--output_dir', type=str, default='results/feature_ablation',
                       help='Output directory for results')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save visualization plots')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print(f"Starting feature ablation analysis...")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.random_seed}")
    print(f"Cross-validation folds: {args.n_folds}")
    
    # Check if data file exists
    if not Path(args.data_path).exists():
        print(f"ERROR: Data file not found: {args.data_path}")
        return 1
    
    analyzer = FeatureAblationAnalyzer(args.data_path, args.random_seed, args.output_dir)
    
    # Run comprehensive ablation study
    results_df = analyzer.run_comprehensive_ablation()
    
    # Display results
    print("\n" + "="*80)
    print("FEATURE ABLATION ANALYSIS RESULTS")
    print("="*80)
    print(results_df[['Feature Set', 'Num Features', 'Accuracy (%)', 'F1 Score']].to_string(index=False))
    
    # Analyze feature importance
    analysis = analyzer.analyze_feature_importance(results_df)
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    print(f"Best performing feature set: {analysis['best_performance']['Feature Set']}")
    print(f"Best accuracy: {analysis['best_performance']['Accuracy (%)']}")
    
    tda_analysis = analysis['tda_contribution']
    if tda_analysis['tda_improvement'] > 0:
        print(f"\nTDA FEATURES HELP: +{tda_analysis['tda_improvement']:.2f}% improvement")
        print(f"Without TDA: {tda_analysis['without_tda']*100:.2f}%")
        print(f"With TDA: {tda_analysis['with_tda']*100:.2f}%")
    else:
        print(f"\nTDA FEATURES HINDER: {tda_analysis['tda_improvement']:.2f}% decrease")
        print(f"Without TDA: {tda_analysis['without_tda']*100:.2f}%")
        print(f"With TDA: {tda_analysis['with_tda']*100:.2f}%")
    
    print(f"\nTDA-only performance: {tda_analysis['tda_standalone_viability']*100:.2f}%")
    
    # Create visualizations
    if args.save_plots:
        analyzer.create_visualization(results_df)
    
    # Save detailed results
    results_file = Path(args.output_dir) / 'feature_ablation_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    # Save analysis summary
    summary = {
        'best_feature_set': analysis['best_performance']['Feature Set'],
        'best_accuracy': float(analysis['best_performance']['Accuracy Mean']),
        'tda_helps': bool(analysis['tda_contribution']['tda_improvement'] > 0),
        'tda_improvement_percent': float(analysis['tda_contribution']['tda_improvement']),
        'geometric_vs_all': {
            'geometric_only': float(analysis['geometric_vs_all']['geometric_only']),
            'all_features': float(analysis['geometric_vs_all']['all_features']),
            'improvement': float(analysis['geometric_vs_all']['improvement']),
            'tda_helps': bool(analysis['geometric_vs_all']['tda_helps'])
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    
    summary_file = Path(args.output_dir) / 'analysis_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")
    
    # Return success
    return 0

if __name__ == "__main__":
    exit(main())