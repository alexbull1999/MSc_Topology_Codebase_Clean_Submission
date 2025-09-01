import torch
import numpy as np
from itertools import product
import json
from datetime import datetime
import os
import sys
from pathlib import Path
import logging

from lattice_metric_discovery import SubsumptionMetrics, LatticeClassTester, LatticeDiscoveryAnalyzer, ClassTestResult

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.order_embeddings_asymmetry_3classloss import train_order_embeddings, OrderEmbeddingModel, EntailmentDataset
from torch.utils.data import DataLoader


def flush_output():
    """Force output to appear immediately in SLURM"""
    sys.stdout.flush()
    sys.stderr.flush()

class ComprehensiveLatticeEvaluator:
    """Comprehensive evaluation of lattice metrics with detailed statistics"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.lattice_analyzer = LatticeDiscoveryAnalyzer()
    
    def _create_order_embedding_data(self, model, processed_data_path, sample_size=10000):
        """
        Create order embeddings from SBERT data for lattice metric evaluation
        """
        # Load original SBERT data
        processed_data = torch.load(processed_data_path)
        dataset = EntailmentDataset(processed_data)
    
        # Sample for efficiency
        if len(dataset) > sample_size:
            indices = torch.randperm(len(dataset))[:sample_size]
            sampled_premise = processed_data['premise_embeddings'][indices]
            sampled_hypothesis = processed_data['hypothesis_embeddings'][indices]
            sampled_labels = [processed_data['labels'][i] for i in indices]
        else:
            sampled_premise = processed_data['premise_embeddings']
            sampled_hypothesis = processed_data['hypothesis_embeddings']
            sampled_labels = processed_data['labels']
    
        # Convert string labels to integers if needed
        if isinstance(sampled_labels[0], str):
            label_to_int = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
            sampled_labels = [label_to_int[label] for label in sampled_labels]
    
        # Generate order embeddings using the trained model
        model.eval()
        with torch.no_grad():
            premise_order = model(sampled_premise.to(self.device)).cpu().numpy()
            hypothesis_order = model(sampled_hypothesis.to(self.device)).cpu().numpy()
    
        return premise_order, hypothesis_order, np.array(sampled_labels)


    def evaluate_model_comprehensive(self, model, processed_data_path):
        """
        Use existing lattice metric implementations for comprehensive evaluation
        """
        # Create order embedding data
        premise_order, hypothesis_order, labels = self._create_order_embedding_data(model, processed_data_path)
        
        # Use existing LatticeClassTester for all metrics
        results = self.lattice_analyzer.test_embedding_space(
            premise_embeddings=premise_order,
            hypothesis_embeddings=hypothesis_order, 
            labels=labels,
            space_name="order_embeddings"
        )
        
        return results

    
def comprehensive_hyperparameter_search():
    """Enhanced hyperparameter search with comprehensive evaluation"""
    
    # Expanded search space
    search_params = {
        'order_dim': [75, 100, 125],
        'asymmetry_weight': [1.0, 1.5, 1.9],
        'margin': [1.0, 1.5, 2.0],
        'lr': [1e-3, 1e-4, 1e-5]
    }
    
    # Your SBERT processed data
    processed_data_path = "data/processed/snli_full_standard_SBERT.pt"

    train_data_path, eval_data_path = create_train_eval_split(processed_data_path)


    evaluator = ComprehensiveLatticeEvaluator()
    results = []
    best_score_so_far = -1
    best_result_so_far = None
    total_combinations = len(list(product(*search_params.values())))

    for i, (order_dim, asymmetry_weight, margin, lr) in enumerate(product(*search_params.values())):
        print(f"\Combination {i+1}/{total_combinations}")
        print(f"order_dim={order_dim}, asymmetry_weight={asymmetry_weight}")
        print(f"margin={margin}, lr={lr}")
        flush_output()
        

        model, trainer = train_order_embeddings(
                processed_data_path=train_data_path,
                epochs=100,  # Reasonable for comparison
                batch_size=32,
                order_dim=order_dim,
                asymmetry_weight=asymmetry_weight,
                margin=margin,
                lr=lr,
                random_seed=42
            )


        evaluator = ComprehensiveLatticeEvaluator()
        comprehensive_stats = evaluator.evaluate_model_comprehensive(model, eval_data_path)

        # Training performance
        final_val_loss = trainer.val_losses[-1] if trainer.val_losses else float('inf')
        final_train_loss = trainer.train_losses[-1] if trainer.train_losses else float('inf')
            
        # Overall performance score (weighted combination of key metrics)
        overall_score = calculate_overall_performance_score(comprehensive_stats)
            
        # Store comprehensive results
        result = {
            'hyperparameters': {
                'order_dim': order_dim,
                'asymmetry_weight': asymmetry_weight,
                'margin': margin,
                'lr': lr
            },
            'training_performance': {
                'final_val_loss': final_val_loss,
                'final_train_loss': final_train_loss,
                'converged': len(trainer.val_losses) < 30  # Early stopping indicator
            },
            'lattice_metrics': comprehensive_stats,
            'overall_score': overall_score,
            'timestamp': datetime.now().isoformat()
        }
            
        results.append(result)
            
        # Print quick summary
        print(f"Overall Score: {overall_score:.4f}")
        print(f"Val Loss: {final_val_loss:.4f}")
        flush_output()
            
        # Print best metric so far
        # Track best results in memory (lighter I/O)
        if result['overall_score'] > best_score_so_far:
            best_score_so_far = result['overall_score']
            best_result_so_far = result.copy()
            print(f"NEW BEST! (Score: {overall_score:.4f})")
            flush_output()                
            # Save both config AND model weights for best performer
            save_best_checkpoint_and_model(best_result_so_far, model, trainer, 
                                            f"best_config_lattice_OE")

    
    # Final analysis
    best_config, analysis = analyze_comprehensive_results(results)
    
    return best_config, results


def calculate_overall_performance_score(comprehensive_stats):
    """Calculate weighted overall performance score from lattice metrics"""
    
    score = 0.0
    weights = {
        'containment_proxy_score': 0.2,
        'asymmetric_energy_score': 0.2,
        'lattice_height_score': 0.2,
        'subsumption_distance_score': 0.2
    }
    
    # Extract the class results from ClassTestResult objects
    for metric_name, weight in weights.items():
        try:
            # Access the attributes directly from ClassTestResult objects
            ent_mean = getattr(comprehensive_stats['entailment'], metric_name)
            neu_mean = getattr(comprehensive_stats['neutral'], metric_name)
            con_mean = getattr(comprehensive_stats['contradiction'], metric_name)
            
            # Get standard deviations (replace '_score' with '_std')
            std_attr_name = metric_name.replace('_score', '_std')
            ent_std = getattr(comprehensive_stats['entailment'], std_attr_name)
            neu_std = getattr(comprehensive_stats['neutral'], std_attr_name)
            con_std = getattr(comprehensive_stats['contradiction'], std_attr_name)
            
            # CORRECT GAP-TO-STD CALCULATION: Average Gap / Average Std
            gap1 = neu_mean - ent_mean  # entailment -> neutral gap
            gap2 = con_mean - neu_mean  # neutral -> contradiction gap
            avg_gap = (gap1 + gap2) / 2
            avg_std = (ent_std + neu_std + con_std) / 3
            
            # This is the REAL separability measure we care about
            correct_gap_to_std = avg_gap / (avg_std + 1e-8)
            
            # Check monotonic progression (either increasing or decreasing)
            means_list = [ent_mean, neu_mean, con_mean]
            is_monotonic = (all(means_list[i] <= means_list[i+1] for i in range(2)) or 
                          all(means_list[i] >= means_list[i+1] for i in range(2)))
            
            # SNR calculations
            snr_ent = abs(ent_mean) / (ent_std + 1e-8)
            snr_neu = abs(neu_mean) / (neu_std + 1e-8)
            snr_con = abs(con_mean) / (con_std + 1e-8)
            avg_snr = np.mean([snr_ent, snr_neu, snr_con])
            min_snr = min([snr_ent, snr_neu, snr_con])

            # Calculate separation absolute
            total_range = max(ent_mean, neu_mean, con_mean) - min(ent_mean, neu_mean, con_mean)
            
            # Composite score for this metric
            metric_score = (
                correct_gap_to_std * 0.7 +           # Primary concern: overlap
                (2.0 if is_monotonic else 0.0) +    # BONUS: Monotonic progression (fixed bonus)
                min(avg_snr / 5.0, 0.5) * 0.3  +      # SECONDARY: Signal quality (30%, capped)
                min(total_range, 2.0) * 0.1         # Absolute separation
            )
            
            score += metric_score * weight

            if correct_gap_to_std >= 2.2:
                print("PHASE 2 READY: A metric exceeds target threshold!")
                score += 5.0  # Bonus for meeting phase requirements
            elif correct_gap_to_std >= 1.8:
                print("CLOSE TO PHASE 2: Metric approaching threshold")
                score += 2.0  # Smaller bonus for getting close
            
            print(f"  {metric_name}:")
            print(f"    Means: {ent_mean:.4f} → {neu_mean:.4f} → {con_mean:.4f}")
            print(f"    Range: {total_range:.4f}, Avg Std: {avg_std:.4f}")
            print(f"    Gap/Std Ratio: {correct_gap_to_std:.4f}")
            print(f"    Monotonic: {is_monotonic}")
            print(f"    Metric Score: {metric_score:.4f}")
            
        except Exception as e:
            print(f"Warning: Could not calculate score for {metric_name}: {e}")
            continue
    
    return score


def create_train_eval_split(processed_data_path, train_ratio=0.8, seed=42, max_samples=10000):
    """
    Create proper train/eval split to avoid data contamination
    
    Returns:
        train_data_path, eval_data_path: Paths to split datasets
    """
    print(f"📂 Loading data from {processed_data_path}")
    processed_data = torch.load(processed_data_path)
    
    # Set seed for reproducible splits
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    total_samples = len(processed_data['labels'])
    # Sample if dataset is too large
    if total_samples > max_samples:
        print(f"Sampling {max_samples} from {total_samples} for efficiency")
        sample_indices = torch.randperm(total_samples)[:max_samples]
        processed_data = {
            'premise_embeddings': processed_data['premise_embeddings'][sample_indices],
            'hypothesis_embeddings': processed_data['hypothesis_embeddings'][sample_indices],
            'labels': [processed_data['labels'][i] for i in sample_indices],
            'texts': {
                'premises': [processed_data['texts']['premises'][i] for i in sample_indices],
                'hypotheses': [processed_data['texts']['hypotheses'][i] for i in sample_indices]
            } if 'texts' in processed_data else None
        }
        total_samples = max_samples


    indices = torch.randperm(total_samples)
    
    train_size = int(train_ratio * total_samples)
    train_indices = indices[:train_size]
    eval_indices = indices[train_size:]
    
    print(f"Split: {len(train_indices)} train, {len(eval_indices)} eval samples")
    
    # Create train split
    train_data = {
        'premise_embeddings': processed_data['premise_embeddings'][train_indices],
        'hypothesis_embeddings': processed_data['hypothesis_embeddings'][train_indices],
        'labels': [processed_data['labels'][i] for i in train_indices],
        'texts': {
            'premises': [processed_data['texts']['premises'][i] for i in train_indices],
            'hypotheses': [processed_data['texts']['hypotheses'][i] for i in train_indices]
        } if 'texts' in processed_data else None
    }
    
    # Create eval split  
    eval_data = {
        'premise_embeddings': processed_data['premise_embeddings'][eval_indices],
        'hypothesis_embeddings': processed_data['hypothesis_embeddings'][eval_indices],
        'labels': [processed_data['labels'][i] for i in eval_indices],
        'texts': {
            'premises': [processed_data['texts']['premises'][i] for i in eval_indices],
            'hypotheses': [processed_data['texts']['hypotheses'][i] for i in eval_indices]
        } if 'texts' in processed_data else None
    }
    
    # Save splits
    train_path = processed_data_path.replace('.pt', '_train_OELattice_split.pt')
    eval_path = processed_data_path.replace('.pt', '_eval_OELattice_split.pt')
    
    torch.save(train_data, train_path)
    torch.save(eval_data, eval_path)
    
    print(f"Train split saved to: {train_path}")
    print(f"Eval split saved to: {eval_path}")
    
    return train_path, eval_path


def analyze_comprehensive_results(results):
    """Comprehensive analysis of all results"""
    
    if not results:
        return None, None
    
    # Sort by overall score
    sorted_results = sorted(results, key=lambda x: x['overall_score'], reverse=True)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print("="*80)
    
    print("\nTOP 5 CONFIGURATIONS:")
    print("-" * 80)
    
    for i, result in enumerate(sorted_results[:5]):
        hp = result['hyperparameters']
        perf = result['training_performance']
        overall = result['overall_score']
        
        print(f"\n{i+1}.RANK {i+1} (Score: {overall:.4f})")
        print(f"   Dimensions: {hp['order_dim']}D")
        print(f"   Asymmetry Weight: {hp['asymmetry_weight']}")
        print(f"   Margin: {hp['margin']}, LR: {hp['lr']}")
        print(f"   Val Loss: {perf['final_val_loss']:.4f}")
        
        # Show key metrics for best performers
        if i < 3:
            print(f"   Key Metrics:")
            for metric_name in ['containment_proxy', 'asymmetric_energy']:
                if metric_name in result['lattice_metrics']:
                    stats = result['lattice_metrics'][metric_name]
                    print(f"      {metric_name}:")
                    print(f"        Gap-to-Std: {stats.get('gap_to_std_ratio', 0):.2f}")
                    print(f"        Monotonic: {stats.get('is_monotonic', False)}")
                    print(f"        Avg SNR: {stats.get('avg_snr', 0):.2f}")
    
    # Best configuration analysis
    best = sorted_results[0]
    
    print(f"\nRECOMMENDED CONFIGURATION:")
    print("-" * 50)
    best_hp = best['hyperparameters']
    for param, value in best_hp.items():
        print(f"   {param}: {value}")
    
    print(f"\nEXPECTED PERFORMANCE:")
    print("-" * 30)
    print(f"   Overall Score: {best['overall_score']:.4f}")
    print(f"   Validation Loss: {best['training_performance']['final_val_loss']:.4f}")
    
    return best, {
        'total_tested': len(results),
        'top_5': sorted_results[:5],
    }

def save_best_checkpoint_and_model(best_result, model, trainer, prefix):
    """Save both configuration and model weights for best performer"""
    results_dir = Path("entailment_surfaces/results/hyperparameter_search_lattice_OE_model")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save configuration
    config_path = results_dir / f"{prefix}_config.json"
    with open(config_path, 'w') as f:
        json.dump(best_result, f, indent=2, default=str)
    
    # Save model weights (following your existing model saving pattern)
    model_path = results_dir / f"{prefix}_model.pt"
    
    # Create checkpoint in same format as your existing order embedding saves
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'bert_dim': model.bert_dim,
            'order_dim': model.order_dim,
            'asymmetry_weight': model.asymmetry_weight
        },
        'hyperparameters': best_result['hyperparameters'],
        'best_val_loss': best_result['training_performance']['final_val_loss'],
        'overall_score': best_result['overall_score'],
        'training_history': {
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'energy_rankings': trainer.energy_rankings[-1] if trainer.energy_rankings else None
        },
        'timestamp': timestamp
    }
    
    torch.save(checkpoint, model_path)
    
    print(f"Config saved to: {config_path}")
    print(f"Model saved to: {model_path}")
    print(f"Score: {best_result['overall_score']:.4f}")
    
    return config_path, model_path


if __name__ == "__main__":
    print("Starting Comprehensive Order Embeddings Hyperparameter Search")
    print("This will test all combinations with full lattice metric evaluation...")
    
    best_config, all_results = comprehensive_hyperparameter_search()
    
    if best_config:
        print(f"\nSearch completed successfully!")
        print(f"Best configuration found with score: {best_config['overall_score']:.4f}")
    
        
    else:
        print("Search failed. Check error messages above.")