import torch
import numpy as np
from typing import Dict, List, Tuple
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from topology import calculate_ph_dim, fast_ripser


class SubtokenPHDClassifier:
    """
    Test classification performance using subtoken PHD approach.
    Compares individual sample PHD to global class baselines.
    """
    
    def __init__(self, 
                 min_points_individual: int = 200,  # Lowered for individual samples
                 max_points_individual: int = 1000,  # Lowered for individual samples
                 point_jump: int = 25,  # Smaller steps
                 h_dim: int = 0,  # Start with H0
                 alpha: float = 1.0,
                 seed: int = 42):
        """
        Initialize classifier with parameters suitable for individual samples
        """
        self.min_points_individual = min_points_individual
        self.max_points_individual = max_points_individual
        self.point_jump = point_jump
        self.h_dim = h_dim
        self.alpha = alpha
        self.seed = seed
        
        # Will store global baselines and individual results
        self.global_baselines = {}
        self.individual_phds = []
        self.true_labels = []
        self.predicted_labels = []
        
        print(f"Subtoken PHD Classifier Initialized:")
        print(f"  Individual sample params: {min_points_individual}-{max_points_individual} points")
        print(f"  Homology dimension: H{h_dim}")
        print(f"  Point jump: {point_jump}")

    def load_global_baselines(self, baselines_path: str):
        """Load pre-computed global PHD baselines"""
        print(f"Loading global baselines from: {baselines_path}")
        
        try:
            baseline_data = torch.load(baselines_path)
            
            if 'class_phds' in baseline_data:
                # Handle single-dimension baselines
                self.global_baselines = baseline_data['class_phds']
        except Exception as e:
            print(f"Error loading baselines: {e}")
            raise

    def compute_individual_phd(self, pointcloud: torch.Tensor) -> float:
        """
        Compute PHD for an individual sample's point cloud
        
        Args:
            pointcloud: Point cloud tensor [n_points, embed_dim]
            
        Returns:
            PHD value or None if computation fails
        """
        n_points = pointcloud.shape[0]
        pointcloud_np = pointcloud.detach().cpu().numpy()

        dynamic_max_points = min(self.max_points_individual, int(n_points))
        if dynamic_max_points <= self.min_points_individual:
            dynamic_max_points = n_points - 1  # Use almost all points

        try:
            # Use fast_ripser with reduced parameters for individual samples
            phd_value = calculate_ph_dim(W=pointcloud_np, min_points=self.min_points_individual, max_points=dynamic_max_points, 
                                        point_jump=self.point_jump, h_dim=self.h_dim, print_error=True, metric="euclidean", alpha=self.alpha,
                                        seed=self.seed)
            return phd_value
            
        except Exception as e:
            print(f"    Error computing PHD: {e}")
            return None


    def classify_sample(self, individual_phd: float) -> str:
        """
        Classify sample based on closest global baseline
        
        Args:
            individual_phd: PHD value for individual sample
            
        Returns:
            Predicted class name
        """
        if individual_phd is None:
            return "unknown"
        
        # Find closest baseline
        min_distance = float('inf')
        predicted_class = "unknown"
        
        for class_name, baseline_phd in self.global_baselines.items():
            distance = abs(individual_phd - baseline_phd)
            if distance < min_distance:
                min_distance = distance
                predicted_class = class_name
                
        return predicted_class

    def test_classification(self, subtoken_data: Dict, test_size: float = 0.3) -> Dict:
        """
        Test classification performance on subtoken data
        
        Args:
            subtoken_data: Loaded subtoken data with pointclouds and labels
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with results and metrics
        """
        print(f"\nTesting classification performance...")
        
        pointclouds = subtoken_data['pointclouds']
        labels = subtoken_data['labels']
        
        # Split data into train/test
        train_indices, test_indices = train_test_split(
            range(len(pointclouds)), 
            test_size=test_size, 
            stratify=labels,
            random_state=self.seed
        )
        
        print(f"Using {len(test_indices)} samples for testing")
        
        # Test on held-out samples
        test_results = []
        successful_classifications = 0
        failed_computations = 0
        
        for i, idx in enumerate(test_indices):
            if i % 50 == 0:
                print(f"  Testing sample {i+1}/{len(test_indices)}")
            
            pointcloud = pointclouds[idx]
            true_label = labels[idx]
            
            # Compute individual PHD
            individual_phd = self.compute_individual_phd(pointcloud)
            
            if individual_phd is not None:
                # Classify based on closest baseline
                predicted_label = self.classify_sample(individual_phd)
                
                test_results.append({
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'individual_phd': individual_phd,
                    'n_points': pointcloud.shape[0]
                })
                
                self.individual_phds.append(individual_phd)
                self.true_labels.append(true_label)
                self.predicted_labels.append(predicted_label)
                
                successful_classifications += 1
            else:
                failed_computations += 1
        
        print(f"\nClassification Summary:")
        print(f"  Successful classifications: {successful_classifications}")
        print(f"  Failed computations: {failed_computations}")
        print(f"  Success rate: {successful_classifications/(successful_classifications + failed_computations)*100:.1f}%")
        
        return {
            'test_results': test_results,
            'successful_classifications': successful_classifications,
            'failed_computations': failed_computations,
            'global_baselines': self.global_baselines
        }


    def analyze_results(self, results: Dict):
        """Analyze and display classification results"""
        if len(self.true_labels) == 0:
            print("No successful classifications to analyze")
            return
        
        print(f"\n{'='*60}")
        print("CLASSIFICATION PERFORMANCE ANALYSIS")
        print(f"{'='*60}")
        
        # Basic accuracy
        correct = sum(1 for true, pred in zip(self.true_labels, self.predicted_labels) if true == pred)
        accuracy = correct / len(self.true_labels)
        print(f"Overall Accuracy: {accuracy:.3f} ({correct}/{len(self.true_labels)})")
        
        # Classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(self.true_labels, self.predicted_labels))
        
        # Confusion matrix
        cm = confusion_matrix(self.true_labels, self.predicted_labels)
        print(f"\nConfusion Matrix:")
        classes = sorted(set(self.true_labels))
        print(f"{'':>15}", end="")
        for cls in classes:
            print(f"{cls:>12}", end="")
        print()
        
        for i, cls in enumerate(classes):
            print(f"{cls:>15}", end="")
            for j in range(len(classes)):
                print(f"{cm[i][j]:>12}", end="")
            print()

    def analyze_phd_distributions(self):
        """Analyze the distribution of individual PHD values vs baselines"""
        if len(self.individual_phds) == 0:
            print("No individual PHD values to analyze")
            return
        
        print(f"\n{'='*60}")
        print("PHD DISTRIBUTION ANALYSIS")
        print(f"{'='*60}")
        
        # Group individual PHDs by true label
        phd_by_class = {}
        for phd, label in zip(self.individual_phds, self.true_labels):
            if label not in phd_by_class:
                phd_by_class[label] = []
            phd_by_class[label].append(phd)
        
        # Compare distributions to baselines
        print(f"Individual PHD Statistics vs Global Baselines:")
        print(f"{'Class':>15} {'Global':>10} {'Ind.Mean':>10} {'Ind.Std':>10} {'Min':>8} {'Max':>8}")
        print("-" * 65)
        
        for class_name in sorted(phd_by_class.keys()):
            individual_phds = phd_by_class[class_name]
            global_baseline = self.global_baselines.get(class_name, 0)
            
            mean_individual = np.mean(individual_phds)
            std_individual = np.std(individual_phds)
            min_individual = np.min(individual_phds)
            max_individual = np.max(individual_phds)
            
            print(f"{class_name:>15} {global_baseline:>10.3f} {mean_individual:>10.3f} "
                  f"{std_individual:>10.3f} {min_individual:>8.3f} {max_individual:>8.3f}")
        
        # Check if individual PHDs are meaningful
        all_individual = self.individual_phds
        individual_range = max(all_individual) - min(all_individual)
        baseline_range = max(self.global_baselines.values()) - min(self.global_baselines.values())
        
        print(f"\nRange Comparison:")
        print(f"  Individual PHD range: {individual_range:.6f}")
        print(f"  Global baseline range: {baseline_range:.6f}")
        print(f"  Range ratio (ind/global): {individual_range/baseline_range:.3f}")


    def save_results(self, results: Dict, output_path: str):
        """Save classification results"""
        save_data = {
            'results': results,
            'classification_params': {
                'min_points_individual': self.min_points_individual,
                'max_points_individual': self.max_points_individual,
                'point_jump': self.point_jump,
                'h_dim': self.h_dim,
                'alpha': self.alpha,
                'seed': self.seed
            },
            'predictions': {
                'true_labels': self.true_labels,
                'predicted_labels': self.predicted_labels,
                'individual_phds': self.individual_phds
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(save_data, output_path)
        print(f"\nResults saved to: {output_path}")
            

def main():
    """Main function to test subtoken PHD classification"""
    
    # Initialize classifier with reduced parameters for individual samples
    classifier = SubtokenPHDClassifier(
        min_points_individual=10,  # Reduced from 200
        max_points_individual=100,  # Reduced from 1000
        point_jump=5,  # Reduced from 50
        h_dim=0,  # Test H0 first
        alpha=1.0,
        seed=42
    )
    
    # Define paths
    subtoken_data_path = "phd_method/phd_data/processed/snli_10k_enhanced_multilayer.pt"
    baselines_path = "phd_method/class_phd_results/snli_10k_subset_enhanced_multilayer_hdim0.pt"
    results_output_path = "phd_method/individual_classification_results/enhanced_subtoken_classification_test_results.pt"
    
    try:
        # Load global baselines
        classifier.load_global_baselines(baselines_path)
        
        # Load subtoken data
        print(f"Loading subtoken data from: {subtoken_data_path}")
        subtoken_data = torch.load(subtoken_data_path)
        
        # Test classification
        results = classifier.test_classification(subtoken_data, test_size=0.3)
        
        # Analyze results
        classifier.analyze_results(results)
        classifier.analyze_phd_distributions()
        
        # Save results
        classifier.save_results(results, results_output_path)

    except FileNotFoundError as e:
        print(f"ERROR: Required file not found - {e}")
        print("Please ensure you have run:")
        print("1. subtoken_text_processing_phd.py")
        print("2. phd_computation.py")
        
    except Exception as e:
        print(f"ERROR during classification test: {e}")
        raise

if __name__ == "__main__":
    main()

    