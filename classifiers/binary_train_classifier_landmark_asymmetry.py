"""
Binary Enhanced TDA Neural Classifier Training Pipeline

Adaptation of the original training script to handle binary classification (entailment vs non-entailment).
Minimal changes from the original - just updated to work with binary labels.

Usage:
    python binary_train_classifier.py --data_path results/tda_integration/enhanced_landmark_tda_features/enhanced_neural_network_features_snli_10k.pt
"""

import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import the binary classes we created
from binary_neural_classifier_landmark_asymmetry import (
    BinaryTDANeuralClassifier,
    EnhancedFeatureNormalizer,
    load_enhanced_classifier_data,
    prepare_binary_training_data
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('binary_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_classifier_data(data_path: str) -> Dict:
    """Use the enhanced data loader"""
    return load_enhanced_classifier_data(data_path)


def prepare_training_data(features_matrix: np.ndarray, labels: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert to binary labels: entailment vs non-entailment"""
    X = torch.FloatTensor(features_matrix)

    # Binary label conversion: entailment (0) vs non-entailment (1)
    binary_labels = []
    for label in labels:
        if label == 'entailment':
            binary_labels.append(0)  # entailment
        else:  # neutral or contradiction
            binary_labels.append(1)  # non-entailment

    y = torch.LongTensor(binary_labels)

    # Log class distribution
    entailment_count = np.sum(np.array(binary_labels) == 0)
    non_entailment_count = np.sum(np.array(binary_labels) == 1)
    logger.info(f"Binary class distribution:")
    logger.info(f"  Entailment: {entailment_count} samples ({entailment_count / len(binary_labels) * 100:.1f}%)")
    logger.info(
        f"  Non-entailment: {non_entailment_count} samples ({non_entailment_count / len(binary_labels) * 100:.1f}%)")

    logger.info(f"Prepared binary training data: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


@dataclass
class HyperparameterGrid:
    """Unchanged from original"""
    learning_rates: List[float] = None
    batch_sizes: List[int] = None
    dropout_rates: List[float] = None
    weight_decays: List[float] = None

    def __post_init__(self):
        if self.learning_rates is None:
            self.learning_rates = [1e-5, 5e-4, 1e-3, 5e-3]
        if self.batch_sizes is None:
            self.batch_sizes = [32, 64, 128, 256]
        if self.dropout_rates is None:
            self.dropout_rates = [0.1, 0.3, 0.5]
        if self.weight_decays is None:
            self.weight_decays = [1e-5, 1e-4, 1e-3, 5e-3]

    def get_all_combinations(self) -> List[Dict]:
        combinations = []
        for lr, bs, dr, wd in product(
                self.learning_rates, self.batch_sizes,
                self.dropout_rates, self.weight_decays
        ):
            combinations.append({
                'learning_rate': lr,
                'batch_size': bs,
                'dropout_rate': dr,
                'weight_decay': wd
            })
        return combinations

    def get_random_combinations(self, n_combinations: int, seed: int = 42) -> List[Dict]:
        all_combinations = self.get_all_combinations()
        np.random.seed(seed)

        if n_combinations >= len(all_combinations):
            return all_combinations

        indices = np.random.choice(len(all_combinations), n_combinations, replace=False)
        return [all_combinations[i] for i in indices]


@dataclass
class TrainingResults:
    """Unchanged from original"""
    fold: int
    train_accuracy: float
    val_accuracy: float
    train_f1_macro: float
    val_f1_macro: float
    train_loss: float
    val_loss: float
    epochs_trained: int
    training_time: float
    val_f1_per_class: List[float]
    confusion_matrix: List[List[int]]
    classification_report: str


@dataclass
class HyperparameterResult:
    """Unchanged from original"""
    hyperparameters: Dict
    cv_mean_accuracy: float
    cv_std_accuracy: float
    cv_mean_f1: float
    cv_std_f1: float
    fold_results: List[TrainingResults]
    total_training_time: float

    def get_score(self) -> float:
        variance_penalty = min(self.cv_std_accuracy / 100, 0.1)
        return self.cv_mean_accuracy - (variance_penalty * 100)


@dataclass
class TrainingConfig:
    """Enhanced config for binary classification"""
    learning_rate: float = 5e-4
    batch_size: int = 32
    dropout_rate: float = 0.3
    weight_decay: float = 1e-3
    max_epochs: int = 100
    patience: int = 15
    min_delta: float = 1e-4
    use_scheduler: bool = True
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    gradient_clip_norm: float = 1.0
    use_class_weights: bool = True
    n_folds: int = 5
    random_seed: int = 42


class EarlyStopping:
    """Unchanged from original"""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None

    def should_stop(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.patience_counter = 0
            self.best_model_state = model.state_dict().copy()
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.patience

    def restore_best_model(self, model: nn.Module):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """Unchanged from original - works for binary too"""
    class_counts = torch.bincount(labels)
    total_samples = len(labels)
    n_classes = len(class_counts)
    weights = total_samples / (n_classes * class_counts.float())
    return weights


def train_single_fold(
        model: BinaryTDANeuralClassifier,  # Use binary classifier
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: torch.device,
        fold: int,
        hyperparameter_search: bool = False
) -> TrainingResults:
    """Minimal changes from original - just updated class names in logging"""

    if not hyperparameter_search:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"TRAINING FOLD {fold + 1}")
        logger.info(f"{'=' * 50}")

    start_time = time.time()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    if config.use_class_weights:
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.tolist())
        all_labels = torch.tensor(all_labels)
        class_weights = compute_class_weights(all_labels).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    if config.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.min_lr,
        )

    early_stopping = EarlyStopping(config.patience, config.min_delta)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_accuracy = 0.0

    # Training loop - unchanged from original
    for epoch in range(config.max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()

            if config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

        if config.use_scheduler:
            scheduler.step(avg_val_loss)

        # Progress logging
        log_frequency = 25 if hyperparameter_search else 10
        if (epoch + 1) % log_frequency == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            if not hyperparameter_search:
                logger.info(f"Epoch {epoch + 1:3d}/{config.max_epochs} | "
                            f"Train Loss: {avg_train_loss:.4f} | "
                            f"Val Loss: {avg_val_loss:.4f} | "
                            f"Train Acc: {train_accuracy:.2f}% | "
                            f"Val Acc: {val_accuracy:.2f}% | "
                            f"LR: {current_lr:.6f}")

        if early_stopping.should_stop(avg_val_loss, model):
            if not hyperparameter_search:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    early_stopping.restore_best_model(model)

    # Final evaluation - unchanged from original
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_accuracy_final = accuracy_score(all_labels, all_predictions) * 100
    val_f1_macro = f1_score(all_labels, all_predictions, average='macro')
    val_f1_per_class = f1_score(all_labels, all_predictions, average=None).tolist()

    # Training set metrics
    model.eval()
    train_predictions = []
    train_labels_list = []

    with torch.no_grad():
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())

    train_accuracy_final = accuracy_score(train_labels_list, train_predictions) * 100
    train_f1_macro = f1_score(train_labels_list, train_predictions, average='macro')

    # Updated class names for binary classification
    class_names = ['entailment', 'non-entailment']
    classification_report_str = classification_report(
        all_labels, all_predictions, target_names=class_names
    )
    conf_matrix = confusion_matrix(all_labels, all_predictions).tolist()

    training_time = time.time() - start_time

    if not hyperparameter_search:
        logger.info(f"\nFold {fold + 1} Results:")
        logger.info(f"Training Time: {training_time:.2f} seconds")
        logger.info(f"Epochs Trained: {epoch + 1}")
        logger.info(f"Final Train Accuracy: {train_accuracy_final:.2f}%")
        logger.info(f"Final Val Accuracy: {val_accuracy_final:.2f}%")
        logger.info(f"Validation F1-Macro: {val_f1_macro:.4f}")
        logger.info(f"Best Val Accuracy: {best_val_accuracy:.2f}%")
        logger.info(f"\nDetailed Classification Report:")
        logger.info(f"\n{classification_report_str}")

    return TrainingResults(
        fold=fold,
        train_accuracy=train_accuracy_final,
        val_accuracy=val_accuracy_final,
        train_f1_macro=train_f1_macro,
        val_f1_macro=val_f1_macro,
        train_loss=train_losses[-1],
        val_loss=val_losses[-1],
        epochs_trained=epoch + 1,
        training_time=training_time,
        val_f1_per_class=val_f1_per_class,
        confusion_matrix=conf_matrix,
        classification_report=classification_report_str
    )


def run_cross_validation(
        features: np.ndarray,
        labels: np.ndarray,
        config: TrainingConfig,
        results_dir: Path,
        save_models: bool = True,
        hyperparameter_search: bool = False
) -> List[TrainingResults]:
    """Minimal changes from original - just use binary classifier"""

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    skf = StratifiedKFold(
        n_splits=config.n_folds,
        shuffle=True,
        random_state=config.random_seed
    )

    all_fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        if not hyperparameter_search:
            logger.info(f"\nPreparing fold {fold + 1}/{config.n_folds}")

        normalizer = EnhancedFeatureNormalizer()

        X_train_raw, X_val_raw = features[train_idx], features[val_idx]
        y_train_np, y_val_np = labels[train_idx], labels[val_idx]

        X_train = normalizer.fit_transform(X_train_raw, [])
        X_val = normalizer.transform(X_val_raw)

        X_train_tensor, y_train_tensor = torch.from_numpy(X_train).float(), torch.from_numpy(y_train_np).long()
        X_val_tensor, y_val_tensor = torch.from_numpy(X_val).float(), torch.from_numpy(y_val_np).long()

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        # Create binary model with correct input dimension
        model = BinaryTDANeuralClassifier(
            input_dim=features.shape[1],
            dropout_rate=config.dropout_rate,
        ).to(device)

        result = train_single_fold(model, train_loader, val_loader, config, device, fold, hyperparameter_search)
        all_fold_results.append(result)

        if save_models and not hyperparameter_search:
            model_path = results_dir / f"binary_model_fold_{fold + 1}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'normalizer': normalizer,
                'config': asdict(config),
                'fold_results': asdict(result)
            }, model_path)
            logger.info(f"Saved binary model checkpoint: {model_path}")

    return all_fold_results


def run_hyperparameter_search(
        features: np.ndarray,
        labels: np.ndarray,
        hyperparameter_grid: HyperparameterGrid,
        base_config: TrainingConfig,
        results_dir: Path,
        max_combinations: Optional[int] = None,
        use_random_search: bool = True
) -> List[HyperparameterResult]:
    """Unchanged from original"""

    logger.info(f"\n{'=' * 70}")
    logger.info(f"STARTING BINARY HYPERPARAMETER SEARCH")
    logger.info(f"{'=' * 70}")

    if use_random_search and max_combinations:
        combinations = hyperparameter_grid.get_random_combinations(max_combinations)
        logger.info(f"Random search: {len(combinations)} combinations (max: {max_combinations})")
    else:
        combinations = hyperparameter_grid.get_all_combinations()
        if max_combinations:
            combinations = combinations[:max_combinations]
        logger.info(f"Grid search: {len(combinations)} combinations")

    logger.info(f"Total hyperparameter combinations to test: {len(combinations)}")

    all_results = []
    best_score = -np.inf
    best_result = None

    for i, hyperparams in enumerate(combinations):
        logger.info(f"\n{'-' * 50}")
        logger.info(f"COMBINATION {i + 1}/{len(combinations)}")
        logger.info(f"{'-' * 50}")
        logger.info(f"Hyperparameters: {hyperparams}")

        start_time = time.time()

        config = TrainingConfig(
            learning_rate=hyperparams['learning_rate'],
            batch_size=hyperparams['batch_size'],
            dropout_rate=hyperparams['dropout_rate'],
            weight_decay=hyperparams['weight_decay'],
            max_epochs=base_config.max_epochs,
            patience=base_config.patience,
            min_delta=base_config.min_delta,
            use_scheduler=base_config.use_scheduler,
            scheduler_patience=base_config.scheduler_patience,
            scheduler_factor=base_config.scheduler_factor,
            min_lr=base_config.min_lr,
            gradient_clip_norm=base_config.gradient_clip_norm,
            use_class_weights=base_config.use_class_weights,
            n_folds=base_config.n_folds,
            random_seed=base_config.random_seed
        )

        try:
            fold_results = run_cross_validation(features, labels, config, results_dir,
                                                save_models=False, hyperparameter_search=True)

            val_accuracies = [r.val_accuracy for r in fold_results]
            val_f1_scores = [r.val_f1_macro for r in fold_results]

            cv_mean_accuracy = np.mean(val_accuracies)
            cv_std_accuracy = np.std(val_accuracies)
            cv_mean_f1 = np.mean(val_f1_scores)
            cv_std_f1 = np.std(val_f1_scores)

            training_time = time.time() - start_time

            hp_result = HyperparameterResult(
                hyperparameters=hyperparams,
                cv_mean_accuracy=cv_mean_accuracy,
                cv_std_accuracy=cv_std_accuracy,
                cv_mean_f1=cv_mean_f1,
                cv_std_f1=cv_std_f1,
                fold_results=fold_results,
                total_training_time=training_time
            )

            all_results.append(hp_result)

            score = hp_result.get_score()
            if score > best_score:
                best_score = score
                best_result = hp_result
                logger.info(f"NEW BEST RESULT!")

            logger.info(f"Results: Accuracy = {cv_mean_accuracy:.2f}% ± {cv_std_accuracy:.2f}%")
            logger.info(f"         F1-Macro = {cv_mean_f1:.4f} ± {cv_std_f1:.4f}")
            logger.info(f"         Score = {score:.2f}")
            logger.info(f"         Time = {training_time:.1f}s")

        except Exception as e:
            logger.error(f"Failed to train with hyperparameters {hyperparams}: {e}")
            continue

    all_results.sort(key=lambda x: x.get_score(), reverse=True)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"BINARY HYPERPARAMETER SEARCH COMPLETED")
    logger.info(f"{'=' * 70}")

    logger.info(f"\nTop 5 Hyperparameter Combinations:")
    for i, result in enumerate(all_results[:5]):
        logger.info(f"{i + 1}. Score: {result.get_score():.2f} | "
                    f"Accuracy: {result.cv_mean_accuracy:.2f}% ± {result.cv_std_accuracy:.2f}% | "
                    f"Params: {result.hyperparameters}")

    if best_result:
        logger.info(f"\nBEST HYPERPARAMETERS:")
        logger.info(f"Parameters: {best_result.hyperparameters}")
        logger.info(f"Validation Accuracy: {best_result.cv_mean_accuracy:.2f}% ± {best_result.cv_std_accuracy:.2f}%")
        logger.info(f"Validation F1-Macro: {best_result.cv_mean_f1:.4f} ± {best_result.cv_std_f1:.4f}")
        logger.info(f"Optimization Score: {best_result.get_score():.2f}")

    return all_results


def summarize_cv_results(results: List[TrainingResults]) -> Dict:
    """Enhanced summary with binary classification baseline comparison"""

    logger.info(f"\n{'=' * 60}")
    logger.info(f"BINARY CROSS-VALIDATION SUMMARY")
    logger.info(f"{'=' * 60}")

    val_accuracies = [r.val_accuracy for r in results]
    val_f1_macros = [r.val_f1_macro for r in results]
    train_accuracies = [r.train_accuracy for r in results]
    training_times = [r.training_time for r in results]
    epochs_trained = [r.epochs_trained for r in results]

    summary = {
        'validation_accuracy': {
            'mean': np.mean(val_accuracies),
            'std': np.std(val_accuracies),
            'min': np.min(val_accuracies),
            'max': np.max(val_accuracies),
            'values': val_accuracies
        },
        'validation_f1_macro': {
            'mean': np.mean(val_f1_macros),
            'std': np.std(val_f1_macros),
            'min': np.min(val_f1_macros),
            'max': np.max(val_f1_macros),
            'values': val_f1_macros
        },
        'training_accuracy': {
            'mean': np.mean(train_accuracies),
            'std': np.std(train_accuracies),
            'values': train_accuracies
        },
        'training_time_seconds': {
            'mean': np.mean(training_times),
            'total': np.sum(training_times),
            'values': training_times
        },
        'epochs_trained': {
            'mean': np.mean(epochs_trained),
            'values': epochs_trained
        }
    }

    # Print summary
    logger.info(
        f"Binary Validation Accuracy: {summary['validation_accuracy']['mean']:.2f}% ± {summary['validation_accuracy']['std']:.2f}%")
    logger.info(f"Range: {summary['validation_accuracy']['min']:.2f}% - {summary['validation_accuracy']['max']:.2f}%")
    logger.info(
        f"Binary Validation F1-Macro: {summary['validation_f1_macro']['mean']:.4f} ± {summary['validation_f1_macro']['std']:.4f}")
    logger.info(f"Total Training Time: {summary['training_time_seconds']['total']:.2f} seconds")
    logger.info(f"Average Epochs: {summary['epochs_trained']['mean']:.1f}")

    # Print per-fold results
    logger.info(f"\nPer-Fold Results:")
    for i, result in enumerate(results):
        logger.info(f"Fold {i + 1}: Val Acc = {result.val_accuracy:.2f}%, "
                    f"F1 = {result.val_f1_macro:.4f}, "
                    f"Epochs = {result.epochs_trained}")

    # Diagnostic checks
    logger.info(f"\nDiagnostic Analysis:")

    avg_train_acc = summary['training_accuracy']['mean']
    avg_val_acc = summary['validation_accuracy']['mean']
    overfitting_gap = avg_train_acc - avg_val_acc

    if overfitting_gap > 10:
        logger.warning(f"Potential overfitting detected! Train-Val gap: {overfitting_gap:.2f}%")
    else:
        logger.info(f"Train-Val gap is reasonable: {overfitting_gap:.2f}%")

    val_acc_std = summary['validation_accuracy']['std']
    if val_acc_std > 5:
        logger.warning(f"High variance across folds: {val_acc_std:.2f}%")
    else:
        logger.info(f"Low variance across folds: {val_acc_std:.2f}%")

    # Enhanced success criteria with 3-way baseline comparison
    three_way_baseline = 58.9  # Your 3-way classification accuracy
    improvement = avg_val_acc - three_way_baseline

    logger.info(f"\nPERFORMANCE COMPARISON:")
    logger.info(f"Binary Classification Accuracy: {avg_val_acc:.2f}%")
    logger.info(f"3-Way Classification Baseline: {three_way_baseline:.2f}%")
    logger.info(f"Improvement: +{improvement:.2f} percentage points")

    if avg_val_acc > 90:
        logger.info(f"EXCELLENT: Binary accuracy > 90%! Outstanding performance!")
    elif avg_val_acc > 85:
        logger.info(f"VERY GOOD: Binary accuracy > 85%! Strong binary classification!")
    elif avg_val_acc > 80:
        logger.info(f"GOOD: Binary accuracy > 80%! Solid improvement over 3-way!")
    elif avg_val_acc > 75:
        logger.info(f"DECENT: Binary accuracy > 75%! Reasonable improvement!")
    elif avg_val_acc > three_way_baseline:
        logger.info(f"MODEST: Binary accuracy improved over 3-way baseline!")
    else:
        logger.warning(f"CONCERNING: Binary accuracy not better than 3-way - investigate!")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train Binary TDA Neural Classifier")
    parser.add_argument('--data_path', type=str,
                        default='results/tda_integration/landmark_tda_features/enhanced_neural_network_features_snli_10k.pt',
                        help='Path to enhanced neural network data file')
    parser.add_argument('--results_dir', type=str,
                        default='results/binary_classifier_training_roberta',
                        help='Directory to save binary training results')
    parser.add_argument('--config_file', type=str,
                        default=None,
                        help='Path to JSON config file (optional)')

    # Hyperparameter search options
    parser.add_argument('--hyperparameter_search', action='store_true',
                        help='Run hyperparameter search instead of single config training')
    parser.add_argument('--max_combinations', type=int, default=20,
                        help='Maximum number of hyperparameter combinations to test')
    parser.add_argument('--use_random_search', action='store_true', default=True,
                        help='Use random search instead of grid search')
    parser.add_argument('--final_training', action='store_true',
                        help='Train final model with best hyperparameters')

    args = parser.parse_args()

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load or create config
    if args.config_file and Path(args.config_file).exists():
        with open(args.config_file) as f:
            config_dict = json.load(f)
        config = TrainingConfig(**config_dict)
        logger.info(f"Loaded config from {args.config_file}")
    else:
        config = TrainingConfig()
        logger.info("Using default configuration")

    # Save config
    config_path = results_dir / 'binary_training_config.json'
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    logger.info(f"Saved config to {config_path}")

    try:
        # Load and prepare data
        logger.info(f"\nLoading data from {args.data_path}")
        classifier_data = load_classifier_data(args.data_path)

        features_matrix = classifier_data['features'].numpy()
        labels_list = classifier_data['labels']

        # Convert to binary labels using our function
        _, binary_labels_tensor = prepare_training_data(features_matrix, labels_list)
        numeric_labels = binary_labels_tensor.numpy()

        logger.info(
            f"Loaded {len(numeric_labels)} samples with {features_matrix.shape[1]} features each for binary classification.")

        if args.hyperparameter_search:
            # Run hyperparameter search
            logger.info(f"Running binary hyperparameter search with up to {args.max_combinations} combinations")

            hp_grid = HyperparameterGrid()

            hp_results = run_hyperparameter_search(
                features_matrix, numeric_labels, hp_grid, config, results_dir,
                max_combinations=args.max_combinations,
                use_random_search=args.use_random_search
            )

            # Save hyperparameter search results
            hp_results_path = results_dir / 'binary_hyperparameter_search_results.json'
            hp_results_dict = {
                'search_config': {
                    'max_combinations': args.max_combinations,
                    'use_random_search': args.use_random_search,
                    'base_config': asdict(config),
                    'classification_type': 'binary'
                },
                'results': []
            }

            for result in hp_results:
                hp_results_dict['results'].append({
                    'hyperparameters': result.hyperparameters,
                    'cv_mean_accuracy': result.cv_mean_accuracy,
                    'cv_std_accuracy': result.cv_std_accuracy,
                    'cv_mean_f1': result.cv_mean_f1,
                    'cv_std_f1': result.cv_std_f1,
                    'optimization_score': result.get_score(),
                    'total_training_time': result.total_training_time
                })

            with open(hp_results_path, 'w') as f:
                json.dump(hp_results_dict, f, indent=2)

            logger.info(f"\nBinary hyperparameter search results saved to {hp_results_path}")

            # Optionally train final model with best hyperparameters
            if args.final_training and hp_results:
                best_result = hp_results[0]
                logger.info(f"\nTraining final binary model with best hyperparameters...")
                logger.info(f"Best hyperparameters: {best_result.hyperparameters}")

                final_config = TrainingConfig(
                    learning_rate=best_result.hyperparameters['learning_rate'],
                    batch_size=best_result.hyperparameters['batch_size'],
                    dropout_rate=best_result.hyperparameters['dropout_rate'],
                    weight_decay=best_result.hyperparameters['weight_decay'],
                    max_epochs=config.max_epochs,
                    patience=config.patience,
                    min_delta=config.min_delta,
                    use_scheduler=config.use_scheduler,
                    scheduler_patience=config.scheduler_patience,
                    scheduler_factor=config.scheduler_factor,
                    min_lr=config.min_lr,
                    gradient_clip_norm=config.gradient_clip_norm,
                    use_class_weights=config.use_class_weights,
                    n_folds=config.n_folds,
                    random_seed=config.random_seed
                )

                final_results_dir = results_dir / 'final_binary_model'
                final_results_dir.mkdir(exist_ok=True)

                cv_results = run_cross_validation(features_matrix, numeric_labels,
                                                  final_config, final_results_dir, save_models=True)
                summary = summarize_cv_results(cv_results)

                final_results_path = final_results_dir / 'final_binary_cv_results.json'
                detailed_results = {
                    'summary': summary,
                    'fold_results': [asdict(r) for r in cv_results],
                    'config': asdict(final_config),
                    'best_hyperparameters': best_result.hyperparameters,
                    'classification_type': 'binary'
                }

                with open(final_results_path, 'w') as f:
                    json.dump(detailed_results, f, indent=2, default=str)

                logger.info(f"Final binary model results saved to {final_results_path}")

        else:
            # Run single configuration training
            logger.info(f"Training binary classifier with single configuration")
            logger.info(f"Binary Training Configuration:")
            for key, value in asdict(config).items():
                logger.info(f"  {key}: {value}")

            cv_results = run_cross_validation(features_matrix, numeric_labels, config, results_dir)

            summary = summarize_cv_results(cv_results)

            results_path = results_dir / 'binary_cv_results.json'
            detailed_results = {
                'summary': summary,
                'fold_results': [asdict(r) for r in cv_results],
                'config': asdict(config),
                'classification_type': 'binary'
            }

            with open(results_path, 'w') as f:
                json.dump(detailed_results, f, indent=2, default=str)

            logger.info(f"\nDetailed binary results saved to {results_path}")
            logger.info(f"Binary model checkpoints saved in {results_dir}")

        logger.info(f"\nBinary classifier training completed successfully!")
        logger.info(f"Expected improvement: Binary classification should perform much better")
        logger.info(f"than 3-way classification since your geometric features excel at")
        logger.info(f"distinguishing entailment from non-entailment relationships!")

    except Exception as e:
        logger.error(f"Binary training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()