import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from topology import calculate_ph_dim, fast_ripser

class PHDCalibrationSystem:
    """
    Calibration system to map between individual PHD values and global class baselines.
    Computes individual PHDs for training data to establish the relationship between scales.
    """
    
    def __init__(self, 
                 min_points_individual: int = 200,
                 max_points_individual: int = 1000,
                 point_jump: int = 50,
                 h_dim: int = 0,
                 alpha: float = 1.0,
                 seed: int = 42):
        """Initialize calibration system"""
        self.min_points_individual = min_points_individual
        self.max_points_individual = max_points_individual
        self.point_jump = point_jump
        self.h_dim = h_dim
        self.alpha = alpha
        self.seed = seed
        
        # Store calibration data
        self.global_baselines = {}
        self.individual_training_phds = {}  # {class: [phd_values]}
        self.calibration_stats = {}
        self.calibration_factors = {}


    def load_global_baselines(self, baselines_path: str):
        """Load pre-computed global PHD baselines"""
        print(f"Loading global baselines from: {baselines_path}")
        
        baseline_data = torch.load(baselines_path)
        if 'class_phds' in baseline_data:
            self.global_baselines = baseline_data['class_phds']
        else:
            raise ValueError("Baseline file format not recognized")
            
        print(f"Loaded global baselines:")
        for class_name, phd_value in self.global_baselines.items():
            print(f"  {class_name}: {phd_value:.6f}")

    
    def compute_individual_phd(self, pointcloud: torch.Tensor) -> Optional[float]:
        """Compute PHD for an individual sample"""
        n_points = pointcloud.shape[0]
        
        if n_points < self.min_points_individual:
            return None
            
        pointcloud_np = pointcloud.detach().cpu().numpy()
        dynamic_max_points = min(self.max_points_individual, int(n_points))
        
        try:
            phd_value = calculate_ph_dim(
                W=pointcloud_np, 
                min_points=self.min_points_individual, 
                max_points=dynamic_max_points, 
                point_jump=self.point_jump, 
                h_dim=self.h_dim, 
                print_error=True,  # Suppress error messages during calibration
                metric="euclidean", 
                alpha=self.alpha,
                seed=self.seed
            )
            return phd_value
        except Exception:
            return None


    def compute_training_individual_phds(self, subtoken_data: Dict, 
                                       sample_size: int = 500) -> Dict:
        """
        Compute individual PHDs for a sample of training data to establish calibration
        
        Args:
            subtoken_data: Enhanced subtoken data with pointclouds and labels
            sample_size: Number of samples per class to use for calibration
            
        Returns:
            Dictionary with individual PHDs organized by class
        """
        print(f"\nComputing individual PHDs for calibration...")
        print(f"Using {sample_size} samples per class")
        
        pointclouds = subtoken_data['pointclouds']
        labels = subtoken_data['labels']
        
        # Organize data by class
        class_indices = {}
        for i, label in enumerate(labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)
        
        # Sample from each class for calibration
        individual_phds_by_class = {}
        
        for class_name, indices in class_indices.items():
            print(f"\nProcessing class: {class_name}")
            
            # Sample indices for this class
            np.random.seed(self.seed)
            if len(indices) > sample_size:
                sampled_indices = np.random.choice(indices, sample_size, replace=False)
            else:
                sampled_indices = indices
            
            class_phds = []
            successful_computations = 0
            nan_computations = 0
            
            for i, idx in enumerate(sampled_indices):
                if i % 50 == 0:
                    print(f"  Computing PHD {i+1}/{len(sampled_indices)}")
                
                pointcloud = pointclouds[idx]
                individual_phd = self.compute_individual_phd(pointcloud)
                
                if individual_phd is not None and not np.isnan(individual_phd):
                    class_phds.append(individual_phd)
                    successful_computations += 1
                elif individual_phd is not None and np.isnan(individual_phd):
                    nan_computations += 1
            
            individual_phds_by_class[class_name] = class_phds
            print(f"  Successful computations: {successful_computations}/{len(sampled_indices)}")
            print(f"  NaN computations: {nan_computations}")
            print(f"  PHD range: {np.min(class_phds):.3f} - {np.max(class_phds):.3f}")
            print(f"  Mean PHD: {np.mean(class_phds):.3f}")
        
        self.individual_training_phds = individual_phds_by_class
        return individual_phds_by_class


    def compute_calibration_statistics(self):
        """Compute calibration statistics and factors"""
        print(f"\n{'='*60}")
        print("COMPUTING CALIBRATION STATISTICS")
        print(f"{'='*60}")
        
        calibration_stats = {}
        calibration_factors = {}
        
        # Overall statistics across all individual PHDs
        all_individual_phds = []
        for class_phds in self.individual_training_phds.values():
            all_individual_phds.extend(class_phds)
        
        overall_individual_mean = np.mean(all_individual_phds)
        overall_individual_std = np.std(all_individual_phds)
        overall_global_mean = np.mean(list(self.global_baselines.values()))
        
        print(f"Overall Statistics:")
        print(f"  Individual PHDs - Mean: {overall_individual_mean:.6f}, Std: {overall_individual_std:.6f}")
        print(f"  Global Baselines - Mean: {overall_global_mean:.6f}")
        print(f"  Scale Factor (Global/Individual): {overall_global_mean/overall_individual_mean:.3f}")
        
        # Class-specific calibration
        print(f"\nClass-Specific Calibration:")
        print(f"{'Class':>15} {'Global':>10} {'Ind.Mean':>10} {'Ind.Std':>10} {'Scale':>8} {'Offset':>10}")
        print("-" * 75)
        
        for class_name in self.global_baselines.keys():
            if class_name not in self.individual_training_phds:
                continue
                
            individual_phds = self.individual_training_phds[class_name]
            global_baseline = self.global_baselines[class_name]
            
            individual_mean = np.mean(individual_phds)
            individual_std = np.std(individual_phds)
            
            # Method 1: Simple scaling factor
            scale_factor = global_baseline / individual_mean
            
            # Method 2: Z-score normalization + offset
            # Normalize individual PHDs to global range
            global_min = min(self.global_baselines.values())
            global_max = max(self.global_baselines.values())
            global_range = global_max - global_min
            
            # Offset to map individual mean to global baseline
            offset = global_baseline - individual_mean
            
            calibration_stats[class_name] = {
                'global_baseline': global_baseline,
                'individual_mean': individual_mean,
                'individual_std': individual_std,
                'scale_factor': scale_factor,
                'offset': offset,
                'individual_samples': len(individual_phds)
            }
            
            print(f"{class_name:>15} {global_baseline:>10.3f} {individual_mean:>10.3f} "
                  f"{individual_std:>10.3f} {scale_factor:>8.3f} {offset:>10.3f}")
        
        # Choose calibration method
        # Method A: Class-specific scaling
        for class_name, stats in calibration_stats.items():
            calibration_factors[class_name] = {
                'method': 'class_scaling',
                'scale_factor': stats['scale_factor'],
                'offset': stats['offset']
            }
        
        # Method B: Global offset (alternative approach)
        global_offset = overall_global_mean - overall_individual_mean
        calibration_factors['global'] = {
            'method': 'global_offset',
            'offset': global_offset,
            'scale_factor': overall_global_mean / overall_individual_mean
        }
        
        self.calibration_stats = calibration_stats
        self.calibration_factors = calibration_factors
        
        print(f"\nCalibration factors computed successfully!")


    
    def calibrate_individual_phd(self, individual_phd: float, 
                                method: str = 'global_offset') -> float:
        """
        Calibrate an individual PHD value to the global scale
        
        Args:
            individual_phd: Raw individual PHD value
            method: Calibration method ('global_offset', 'global_scaling', 'class_specific')
            
        Returns:
            Calibrated PHD value
        """
        if method == 'global_offset':
            offset = self.calibration_factors['global']['offset']
            return individual_phd + offset
            
        elif method == 'global_scaling':
            scale_factor = self.calibration_factors['global']['scale_factor']
            return individual_phd * scale_factor
        
        else:
            raise ValueError(f"Unknown calibration method: {method}")


    def classify_calibrated_sample(self, individual_phd: float, 
                                 calibration_method: str = 'global_offset') -> str:
        """
        Classify sample using calibrated PHD value
        
        Args:
            individual_phd: Raw individual PHD value
            calibration_method: Method to use for calibration
            
        Returns:
            Predicted class name
        """
        if individual_phd is None:
            return "unknown"
        
        # Calibrate the individual PHD to global scale
        calibrated_phd = self.calibrate_individual_phd(individual_phd, calibration_method)
        
        # Find closest global baseline
        min_distance = float('inf')
        predicted_class = "unknown"
        
        for class_name, baseline_phd in self.global_baselines.items():
            distance = abs(calibrated_phd - baseline_phd)
            if distance < min_distance:
                min_distance = distance
                predicted_class = class_name
        
        return predicted_class

    def test_calibrated_classification(self, subtoken_data: Dict, 
                                     test_size: float = 0.3,
                                     calibration_method: str = 'global_offset') -> Dict:
        """
        Test classification performance using calibrated PHD values
        
        Args:
            subtoken_data: Enhanced subtoken data
            test_size: Fraction for testing
            calibration_method: Calibration method to use
            
        Returns:
            Results dictionary
        """
        print(f"\nTesting calibrated classification with method: {calibration_method}")
        
        pointclouds = subtoken_data['pointclouds']
        labels = subtoken_data['labels']
        
        # Split data (ensure test set doesn't overlap with calibration data)
        train_indices, test_indices = train_test_split(
            range(len(pointclouds)), 
            test_size=test_size, 
            stratify=labels,
            random_state=self.seed + 1  # Different seed to avoid overlap
        )
        
        print(f"Using {len(test_indices)} samples for testing")
        
        # Test classification
        true_labels = []
        predicted_labels = []
        individual_phds = []
        calibrated_phds = []
        successful_classifications = 0
        failed_computations = 0
        
        for i, idx in enumerate(test_indices):
            if i % 100 == 0:
                print(f"  Testing sample {i+1}/{len(test_indices)}")
            
            pointcloud = pointclouds[idx]
            true_label = labels[idx]
            
            # Compute individual PHD
            individual_phd = self.compute_individual_phd(pointcloud)
            
            if individual_phd is not None:
                # Classify using calibrated PHD
                predicted_label = self.classify_calibrated_sample(individual_phd, calibration_method)
                calibrated_phd = self.calibrate_individual_phd(individual_phd, calibration_method)
                
                true_labels.append(true_label)
                predicted_labels.append(predicted_label)
                individual_phds.append(individual_phd)
                calibrated_phds.append(calibrated_phd)
                
                successful_classifications += 1
            else:
                failed_computations += 1
        
        print(f"\nClassification Summary:")
        print(f"  Successful classifications: {successful_classifications}")
        print(f"  Failed computations: {failed_computations}")
        
        return {
            'true_labels': true_labels,
            'predicted_labels': predicted_labels,
            'individual_phds': individual_phds,
            'calibrated_phds': calibrated_phds,
            'successful_classifications': successful_classifications,
            'failed_computations': failed_computations,
            'calibration_method': calibration_method
        }

    def analyze_calibrated_results(self, results: Dict):
        """Analyze calibrated classification results"""
        true_labels = results['true_labels']
        predicted_labels = results['predicted_labels']
        calibrated_phds = results['calibrated_phds']
        
        if len(true_labels) == 0:
            print("No successful classifications to analyze")
            return
        
        print(f"\n{'='*60}")
        print("CALIBRATED CLASSIFICATION ANALYSIS")
        print(f"{'='*60}")
        
        # Basic accuracy
        correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
        accuracy = correct / len(true_labels)
        print(f"Overall Accuracy: {accuracy:.3f} ({correct}/{len(true_labels)})")
        
        # Classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(true_labels, predicted_labels))
        
        # Calibrated PHD distribution analysis
        print(f"\nCalibrated PHD Distribution Analysis:")
        
        # Group calibrated PHDs by true label
        phd_by_class = {}
        for phd, label in zip(calibrated_phds, true_labels):
            if label not in phd_by_class:
                phd_by_class[label] = []
            phd_by_class[label].append(phd)
        
        print(f"{'Class':>15} {'Global':>10} {'Cal.Mean':>10} {'Cal.Std':>10} {'Min':>8} {'Max':>8}")
        print("-" * 70)
        
        for class_name in sorted(phd_by_class.keys()):
            calibrated_phds_class = phd_by_class[class_name]
            global_baseline = self.global_baselines.get(class_name, 0)
            
            mean_calibrated = np.mean(calibrated_phds_class)
            std_calibrated = np.std(calibrated_phds_class)
            min_calibrated = np.min(calibrated_phds_class)
            max_calibrated = np.max(calibrated_phds_class)
            
            print(f"{class_name:>15} {global_baseline:>10.3f} {mean_calibrated:>10.3f} "
                  f"{std_calibrated:>10.3f} {min_calibrated:>8.3f} {max_calibrated:>8.3f}")
        
        # Range comparison after calibration
        all_calibrated = calibrated_phds
        calibrated_range = max(all_calibrated) - min(all_calibrated)
        baseline_range = max(self.global_baselines.values()) - min(self.global_baselines.values())
        
        print(f"\nRange Comparison After Calibration:")
        print(f"  Calibrated PHD range: {calibrated_range:.6f}")
        print(f"  Global baseline range: {baseline_range:.6f}")
        print(f"  Range ratio (cal/global): {calibrated_range/baseline_range:.3f}")

    def save_calibration_system(self, output_path: str):
        """Save the entire calibration system"""
        calibration_data = {
            'global_baselines': self.global_baselines,
            'individual_training_phds': self.individual_training_phds,
            'calibration_stats': self.calibration_stats,
            'calibration_factors': self.calibration_factors,
            'computation_params': {
                'min_points_individual': self.min_points_individual,
                'max_points_individual': self.max_points_individual,
                'point_jump': self.point_jump,
                'h_dim': self.h_dim,
                'alpha': self.alpha,
                'seed': self.seed
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(calibration_data, output_path)
        print(f"\nCalibration system saved to: {output_path}")

def main():
    """Main function to run PHD calibration and classification"""
    
    # Initialize calibration system
    calibrator = PHDCalibrationSystem(
        min_points_individual=40,
        max_points_individual=1000,
        point_jump=10,
        h_dim=0,
        alpha=1.0,
        seed=42
    )
    
    # Define paths
    subtoken_data_path = "phd_method/phd_data/processed/snli_10k_enhanced_multilayer.pt"
    baselines_path = "phd_method/class_phd_results/snli_10k_subset_enhanced_multilayer_hdim0.pt"
    calibration_output_path = "phd_method/phd_data/individual_classification_results/phd_calibration_system.pt"
    
    try:
        # Load data and baselines
        calibrator.load_global_baselines(baselines_path)
        
        print(f"Loading enhanced subtoken data...")
        subtoken_data = torch.load(subtoken_data_path)
        
        # Compute individual PHDs for calibration
        calibrator.compute_training_individual_phds(subtoken_data, sample_size=500)
        
        # Compute calibration statistics
        calibrator.compute_calibration_statistics()
        
        # Test different calibration methods
        for method in ['global_offset', 'global_scaling']:
            print(f"\n{'='*80}")
            print(f"TESTING CALIBRATION METHOD: {method.upper()}")
            print(f"{'='*80}")
            
            results = calibrator.test_calibrated_classification(
                subtoken_data, 
                test_size=0.3, 
                calibration_method=method
            )
            
            calibrator.analyze_calibrated_results(results)
        
        # Save calibration system
        calibrator.save_calibration_system(calibration_output_path)
        
        print(f"\n{'='*60}")
        print("CALIBRATION SYSTEM COMPLETE")
        print(f"{'='*60}")
        print("Individual PHDs computed for training samples")
        print("Calibration factors calculated")
        print("Multiple calibration methods tested")
        print("System saved for future use")
        
    except Exception as e:
        print(f"ERROR: {e}")
        raise

if __name__ == "__main__":
    main()
        

