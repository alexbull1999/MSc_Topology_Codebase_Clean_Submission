"""
Blind Test Evaluation Script

Evaluates trained binary classifier models on blind test data for unbiased performance assessment.
Supports both individual fold evaluation and ensemble prediction.

Usage:
    python evaluate_blind_test.py --blind_data_path blind_tests/snli_10k_test_asymmetry_input.pt --models_dir results/overnight_binary_hyperparam_search_normalizer/final_binary_model
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Import your classifier
from binary_neural_classifier_landmark_asymmetry import (
    BinaryTDANeuralClassifier, 
    EnhancedFeatureNormalizer
)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResults:
    """Results from model evaluation"""
    model_name: str
    accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    roc_auc: float
    confusion_matrix: np.ndarray
    classification_report: str
    predictions: np.ndarray
    probabilities: np.ndarray

class BlindTestEvaluator:
    """
    Evaluates trained binary classifiers on blind test data
    """
    
    def __init__(self, models_dir: str, device: str = None):
        """
        Initialize evaluator
        
        Args:
            models_dir: Directory containing trained model checkpoints
            device: Device to use for inference
        """
        self.models_dir = Path(models_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models = {}
        self.normalizers = {}
        
        logger.info(f"Initializing blind test evaluator on {self.device}")


    def load_models(self) -> None:
        """
        Load all fold models from the models directory
        """
        logger.info(f"Loading models from {self.models_dir}")
        
        model_files = list(self.models_dir.glob("binary_model_fold_*.pt"))
        model_files.sort()  # Ensure consistent ordering
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {self.models_dir}")
        
        for model_file in model_files:
            fold_name = model_file.stem  # e.g., "binary_model_fold_1"
            
            logger.info(f"Loading {fold_name}...")
            
            # Load checkpoint
            checkpoint = torch.load(model_file, map_location=self.device)
            
            # Get model configuration
            config = checkpoint['config']
            fold_results = checkpoint['fold_results']

            # Extract normalizer from checkpoint
            normalizer = checkpoint.get('normalizer', None)
            if normalizer is not None:
                logger.info(f"  Found normalizer in {fold_name}")
            else:
                logger.warning(f"  No normalizer found in {fold_name}")
            
            # Create model with correct architecture
            # Infer input_dim from the saved model state
            state_dict = checkpoint['model_state_dict']
            input_dim = state_dict['layer1.weight'].shape[1]  # Get input dimension from first layer
            
            model = BinaryTDANeuralClassifier(
                input_dim=input_dim,
                dropout_rate=config['dropout_rate']
            )
            
            # Load trained weights
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            self.models[fold_name] = {
                'model': model,
                'config': config,
                'fold_results': fold_results,
                'input_dim': input_dim,
                'normalizer': normalizer
            }
            
            logger.info(f"{fold_name} loaded (input_dim={input_dim}, val_acc={fold_results['val_accuracy']:.2f}%)")
        
        logger.info(f"Loaded {len(self.models)} models successfully")


    def load_blind_test_data(self, blind_data_path: str) -> Tuple[torch.Tensor, np.ndarray, List[str]]:
        """
        Load blind test data
        
        Args:
            blind_data_path: Path to blind test data file
            
        Returns:
            Tuple of (features, binary_labels, string_labels)
        """
        logger.info(f"Loading blind test data from {blind_data_path}")
        
        blind_data = torch.load(blind_data_path, map_location='cpu')
        
        # Extract data
        features = blind_data['features']  # torch.Tensor [n_samples, n_features]
        binary_labels = np.array(blind_data['binary_labels'])  # [n_samples]
        string_labels = blind_data['labels']  # List[str]
        
        logger.info(f"Loaded blind test data:")
        logger.info(f"  - Samples: {len(binary_labels)}")
        logger.info(f"  - Features: {features.shape[1]}D")
        logger.info(f"  - Feature type: {blind_data.get('feature_type', 'unknown')}")
        
        # Class distribution
        n_entailment = np.sum(binary_labels == 0)
        n_non_entailment = np.sum(binary_labels == 1)
        logger.info(f"  - Entailment: {n_entailment} ({n_entailment/len(binary_labels)*100:.1f}%)")
        logger.info(f"  - Non-entailment: {n_non_entailment} ({n_non_entailment/len(binary_labels)*100:.1f}%)")
        
        return features, binary_labels, string_labels



    def normalize_features(self, features: torch.Tensor, fold_name: str = None) -> torch.Tensor:
        """
        Normalize features using the exact normalizer from training
        
        Args:
            features: Raw features to normalize
            fold_name: Specific fold to use normalizer from (if None, uses first available)
            
        Returns:
            Normalized features
        """
        features_np = features.numpy()
        
        # Get normalizer
        normalizer = None
        
        if fold_name and fold_name in self.models:
            normalizer = self.models[fold_name].get('normalizer')
        
        # If no specific normalizer, use any available one (they should be the same if global)
        if normalizer is None:
            for model_info in self.models.values():
                if 'normalizer' in model_info and model_info['normalizer'] is not None:
                    normalizer = model_info['normalizer']
                    break
        
        if normalizer is not None:
            logger.info("Using exact normalizer from training")
            # Use the exact normalizer from training
            normalized_features = normalizer.transform(features_np)
        else:
            logger.warning("No saved normalizer found - using approximation")
            # Fallback to approximation (same as before)
            from binary_neural_classifier_landmark_asymmetry import EnhancedFeatureNormalizer
            normalizer = EnhancedFeatureNormalizer(scaler_type='standard')
            normalized_features = normalizer.fit_transform(features_np, [])
        
        return torch.from_numpy(normalized_features).float()

    def evaluate_single_model(self, model_info: Dict, features: torch.Tensor, 
                            labels: np.ndarray, model_name: str) -> EvaluationResults:
        """
        Evaluate a single model on test data
        
        Args:
            model_info: Dictionary containing model and config
            features: Test features
            labels: Test labels
            model_name: Name of the model
            
        Returns:
            EvaluationResults object
        """
        model = model_info['model']
        
        # Move features to device
        features = features.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            # Get raw logits
            logits = model(features)
            
            # Get probabilities
            probabilities = F.softmax(logits, dim=1).cpu().numpy()
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        precision_macro = precision_score(labels, predictions, average='macro')
        recall_macro = recall_score(labels, predictions, average='macro')
        
        # ROC AUC (using probability of positive class)
        roc_auc = roc_auc_score(labels, probabilities[:, 1])
        
        # Confusion matrix and classification report
        conf_matrix = confusion_matrix(labels, predictions)
        class_names = ['entailment', 'non-entailment']
        class_report = classification_report(labels, predictions, target_names=class_names)
        
        return EvaluationResults(
            model_name=model_name,
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            roc_auc=roc_auc,
            confusion_matrix=conf_matrix,
            classification_report=class_report,
            predictions=predictions,
            probabilities=probabilities
        )

    
    def evaluate_ensemble(self, features: torch.Tensor, labels: np.ndarray) -> EvaluationResults:
        """
        Evaluate ensemble of all models
        
        Args:
            features: Test features
            labels: Test labels
            
        Returns:
            EvaluationResults for ensemble
        """
        logger.info("Evaluating ensemble of all models...")
        
        features = features.to(self.device)
        all_probabilities = []
        
        # Get predictions from all models
        for fold_name, model_info in self.models.items():
            model = model_info['model']
            
            with torch.no_grad():
                logits = model(features)
                probabilities = F.softmax(logits, dim=1).cpu().numpy()
                all_probabilities.append(probabilities)
        
        # Average probabilities across models
        ensemble_probabilities = np.mean(all_probabilities, axis=0)
        ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, ensemble_predictions)
        f1_macro = f1_score(labels, ensemble_predictions, average='macro')
        f1_weighted = f1_score(labels, ensemble_predictions, average='weighted')
        precision_macro = precision_score(labels, ensemble_predictions, average='macro')
        recall_macro = recall_score(labels, ensemble_predictions, average='macro')
        roc_auc = roc_auc_score(labels, ensemble_probabilities[:, 1])
        
        conf_matrix = confusion_matrix(labels, ensemble_predictions)
        class_names = ['entailment', 'non-entailment']
        class_report = classification_report(labels, ensemble_predictions, target_names=class_names)
        
        return EvaluationResults(
            model_name="ensemble",
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            roc_auc=roc_auc,
            confusion_matrix=conf_matrix,
            classification_report=class_report,
            predictions=ensemble_predictions,
            probabilities=ensemble_probabilities
        )

    def evaluate_all(self, blind_data_path: str, include_ensemble: bool = True) -> Dict[str, EvaluationResults]:
        """
        Evaluate all models on blind test data
        
        Args:
            blind_data_path: Path to blind test data
            include_ensemble: Whether to include ensemble evaluation
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info("="*80)
        logger.info("STARTING BLIND TEST EVALUATION")
        logger.info("="*80)
        
        # Load data
        features, labels, string_labels = self.load_blind_test_data(blind_data_path)
        
        # Normalize features
        normalized_features = self.normalize_features(features)
        
        # Evaluate each model
        results = {}
        
        for fold_name, model_info in self.models.items():
            logger.info(f"\nEvaluating {fold_name}...")
            
            result = self.evaluate_single_model(
                model_info, normalized_features, labels, fold_name
            )
            results[fold_name] = result
            
            logger.info(f"  Accuracy: {result.accuracy:.4f} ({result.accuracy*100:.2f}%)")
            logger.info(f"  F1-Macro: {result.f1_macro:.4f}")
            logger.info(f"  ROC-AUC:  {result.roc_auc:.4f}")
        
        # Evaluate ensemble
        if include_ensemble and len(self.models) > 1:
            logger.info(f"\nEvaluating ensemble...")
            ensemble_result = self.evaluate_ensemble(normalized_features, labels)
            results['ensemble'] = ensemble_result
            
            logger.info(f"  Accuracy: {ensemble_result.accuracy:.4f} ({ensemble_result.accuracy*100:.2f}%)")
            logger.info(f"  F1-Macro: {ensemble_result.f1_macro:.4f}")
            logger.info(f"  ROC-AUC:  {ensemble_result.roc_auc:.4f}")
        
        return results
    
    def print_summary(self, results: Dict[str, EvaluationResults]) -> None:
        """
        Print summary of all evaluation results
        """
        logger.info("\n" + "="*80)
        logger.info("BLIND TEST EVALUATION SUMMARY")
        logger.info("="*80)
        
        # Individual model results
        fold_results = {k: v for k, v in results.items() if k.startswith('binary_model_fold')}
        
        if fold_results:
            accuracies = [r.accuracy for r in fold_results.values()]
            f1_scores = [r.f1_macro for r in fold_results.values()]
            
            logger.info(f"\nIndividual Fold Results:")
            logger.info(f"  Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
            logger.info(f"  Range: {np.min(accuracies):.4f} - {np.max(accuracies):.4f}")
            logger.info(f"  Mean F1-Macro: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
            
            logger.info(f"\nPer-Fold Details:")
            for fold_name, result in fold_results.items():
                fold_num = fold_name.split('_')[-1]
                logger.info(f"  Fold {fold_num}: Acc={result.accuracy:.4f}, F1={result.f1_macro:.4f}, AUC={result.roc_auc:.4f}")
        
        # Ensemble results
        if 'ensemble' in results:
            ensemble = results['ensemble']
            logger.info(f"\nEnsemble Results:")
            logger.info(f"  Accuracy: {ensemble.accuracy:.4f} ({ensemble.accuracy*100:.2f}%)")
            logger.info(f"  F1-Macro: {ensemble.f1_macro:.4f}")
            logger.info(f"  F1-Weighted: {ensemble.f1_weighted:.4f}")
            logger.info(f"  Precision: {ensemble.precision_macro:.4f}")
            logger.info(f"  Recall: {ensemble.recall_macro:.4f}")
            logger.info(f"  ROC-AUC: {ensemble.roc_auc:.4f}")
        
        # Compare with training results
        logger.info(f"\n" + "="*50)
        logger.info("PERFORMANCE COMPARISON")
        logger.info("="*50)
        
        # Your training accuracy was ~79.27%
        training_acc = 79.27
        
        if fold_results:
            test_acc = np.mean([r.accuracy * 100 for r in fold_results.values()])
            generalization_gap = training_acc - test_acc
            
            logger.info(f"Training Accuracy (CV): {training_acc:.2f}%")
            logger.info(f"Blind Test Accuracy:    {test_acc:.2f}%")
            logger.info(f"Generalization Gap:     {generalization_gap:+.2f}%")
            
            if abs(generalization_gap) < 2:
                logger.info("✓ EXCELLENT: Very small generalization gap!")
            elif abs(generalization_gap) < 5:
                logger.info("✓ GOOD: Reasonable generalization gap")
            else:
                logger.warning("⚠ ATTENTION: Large generalization gap - investigate!")
        
        if 'ensemble' in results:
            ensemble_acc = results['ensemble'].accuracy * 100
            improvement = ensemble_acc - test_acc if fold_results else 0
            logger.info(f"Ensemble Improvement:   {improvement:+.2f}%")

    
    def save_results(self, results: Dict[str, EvaluationResults], output_dir: str) -> None:
        """
        Save evaluation results to files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary statistics
        summary = {}
        
        for model_name, result in results.items():
            summary[model_name] = {
                'accuracy': float(result.accuracy),
                'f1_macro': float(result.f1_macro),
                'f1_weighted': float(result.f1_weighted),
                'precision_macro': float(result.precision_macro),
                'recall_macro': float(result.recall_macro),
                'roc_auc': float(result.roc_auc),
                'confusion_matrix': result.confusion_matrix.tolist()
            }
        
        # Save to JSON
        with open(output_dir / 'blind_evaluation_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed classification reports
        for model_name, result in results.items():
            report_file = output_dir / f'{model_name}_classification_report.txt'
            with open(report_file, 'w') as f:
                f.write(f"Classification Report - {model_name}\n")
                f.write("="*50 + "\n")
                f.write(result.classification_report)
                f.write(f"\nAccuracy: {result.accuracy:.4f}\n")
                f.write(f"ROC-AUC: {result.roc_auc:.4f}\n")
        
        logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models on blind test data')
    parser.add_argument('--blind_data_path',
                       type=str,
                       default='blind_tests/snli_10k_test_asymmetry_input.pt',
                       help='Path to blind test data file')
    parser.add_argument('--models_dir',
                       type=str,
                       default='results/overnight_binary_hyperparam_search_normalizer/final_binary_model',
                       help='Directory containing trained model checkpoints')
    parser.add_argument('--output_dir',
                       type=str,
                       default='results/overnight_binary_hyperparm_search_normalizer/blind_evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--device',
                       type=str,
                       default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--no_ensemble',
                       action='store_true',
                       help='Skip ensemble evaluation')
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = BlindTestEvaluator(
            models_dir=args.models_dir,
            device=args.device
        )
        
        # Load models
        evaluator.load_models()
        
        # Evaluate all models
        results = evaluator.evaluate_all(
            blind_data_path=args.blind_data_path,
            include_ensemble=not args.no_ensemble
        )
        
        # Print summary
        evaluator.print_summary(results)
        
        # Save results
        evaluator.save_results(results, args.output_dir)
        
        logger.info("\nBlind test evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()

    