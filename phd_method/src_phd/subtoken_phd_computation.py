import torch
import numpy as np
from typing import Dict, List, Tuple
import os
import sys
from pathlib import Path
from topology import calculate_ph_dim, distance_matrix, ph_dim_from_distance_matrix, fast_ripser

class PHDClassComputation:
    """Compute PH Dimension for each entailment class"""

    def __init__(self,
                 min_points: int=200,
                 max_points: int=1000,
                 point_jump: int=50,
                 h_dim: int=0,
                 alpha: float=1.0,
                 seed: int=42):
        """Initialize PHD computation parameters
        Args:
            min_points: Minimum number of points for PHD calculation
            max_points: Maximum number of points for PHD calculation
            point_jump: Step size between point counts
            h_dim: Homology dimension (0=connected components, 1=loops, etc.)
            alpha: Alpha parameter for weight persistence sum
            seed: Random seed
        """
        self.min_points = min_points
        self.max_points = max_points
        self.point_jump = point_jump
        self.h_dim = h_dim
        self.alpha = alpha
        self.seed = seed

        print(f"PHD Computation Initialized:")
        print(f"Point range: {self.min_points} to {self.max_points} (step: {self.point_jump})")
        print(f"Homology dimension: {self.h_dim}")
        print(f"Alpha: {self.alpha}, Seed: {self.seed}")


    def combine_class_pointclouds(self, class_pointclouds: List[torch.Tensor]) -> torch.Tensor:
        """
        Combine all point clouds from a class into one large tensor for PHD computation
        
        Args:
            class_pointclouds: List of point cloud tensors for one class
            
        Returns:
            Combined tensor containing all points from the class
        """
        print(f"    Combining {len(class_pointclouds)} point clouds...")
        
        # Concatenate all point clouds for this class
        combined_pointcloud = torch.cat(class_pointclouds, dim=0)
        
        print(f"    Combined shape: {combined_pointcloud.shape}")
        print(f"    Total points: {combined_pointcloud.shape[0]:,}")
        
        return combined_pointcloud


    def compute_class_phd(self, class_embeddings: torch.Tensor, class_name: str, use_fast_ripser: bool = False) -> float:
        """Compute PHD for a single class of embeddings:
        Args:
            class_embeddings: Tensor of concatenated embeddings for one class [n_samples, embed_dim]
            class_name: Name of the entailment class
            use_fast_ripser: Choice of method
        Returns:
            float: persistent homology dimension
        """
        print(f"Computing PHD for class {class_name}...")
        print(f"Input shape: {class_embeddings.shape}")

        #Convert to numpy, ensuring enough samples
        embeddings_np = class_embeddings.detach().cpu().numpy()
        n_samples = embeddings_np.shape[0]

        if n_samples < self.min_points:
            print(f"WARNING: Only {n_samples} samples but need minimum {self.min_points}")
            exit(1)

        # Limit to max_points if necessary (for computational efficiency)
        # if n_samples > self.max_points:
        #     print(f"Subsampling from {n_samples:,} to {self.max_points:,} points for efficiency")
        #     np.random.seed(self.seed)
        #     indices = np.random.choice(n_samples, self.max_points, replace=False)
        #     embeddings_np = embeddings_np[indices]

        try:
            if use_fast_ripser:
                # Method 1: Using fast_ripser (computing from distance matrix)
                print(f"Using fast_ripser method")
                phd_value = fast_ripser(w=embeddings_np,
                                        min_points=self.min_points,
                                        max_points=self.max_points,
                                        point_jump=self.point_jump,
                                        h_dim=self.h_dim,
                                        alpha=self.alpha,
                                        seed=self.seed,
                                        metric="euclidean"
                                        )
            else:
                #Method 2: Direction computation
                print(f"Using direct PHD computation")
                phd_value = calculate_ph_dim(W=embeddings_np,
                                             min_points=self.min_points,
                                             max_points=self.max_points,
                                             point_jump=self.point_jump,
                                             h_dim=self.h_dim,
                                             print_error=True,
                                             metric="euclidean",
                                             alpha=self.alpha,
                                             seed=self.seed
                                             )
                print(f"PHD Value for '{class_name}' = {phd_value:.6f}")
            return phd_value
        except Exception as e:
            print(e)
            raise


    def compute_all_class_phds(self, processed_data: Dict) -> Dict[str, float]:
        """Compute PHD for all entailment classes
        Args:
            processed_data: Output from text_processing_phd.py containing class_embeddings
        Returns:
            Dict mapping class names to their PHD values
        """
        if "class_pointclouds" not in processed_data:
            raise ValueError("processed_data must contain 'class_pointclouds' - run subtoken_text_processing_phd.py first")

        class_pointclouds = processed_data["class_pointclouds"]
        class_phds = {}

        #Compute PHD for each class
        for class_name, pointclouds in class_pointclouds.items():
            print(f"Processing class: {class_name}")
            combined_embeddings = self.combine_class_pointclouds(pointclouds)

            phd_value = self.compute_class_phd(class_embeddings=combined_embeddings, class_name=class_name)
            if phd_value is not None:
                class_phds[class_name] = phd_value
            else:
                raise ValueError(f"Class '{class_name}' has no PHD value")

        # Summary
        print("=" * 40)
        print("PHD COMPUTATION SUMMARY")
        print("=" * 40)
        for class_name, phd in class_phds.items():
            print(f"{class_name:>15}: {phd:.6f}")

        return class_phds


    def analyze_phd_patterns(self, class_phds: Dict[str, float]) -> Dict:
        print("=" * 40)
        print("PHD PATTERN ANALYSIS")
        print("=" * 40)

        for i, (class_name, phd) in enumerate(sorted(class_phds.items())):
            print(f"  {i + 1}. {class_name}: {phd:.6f}")


    def save_phd_results(self, class_phds: Dict[str, float], output_path: str):
        results = {
            "class_phds": class_phds,
            "computation_params": {
                "min_points": self.min_points,
                "max_points": self.max_points,
                "point_jump": self.point_jump,
                "h_dim": self.h_dim,
                "alpha": self.alpha,
                "seed": self.seed
            }
        }

        torch.save(results, output_path)

def test_phd_computation():
    #Load processed data from text_processing_phd.py
    data_path = "phd_method/phd_data/processed/snli_10k_enhanced_multilayer.pt"
    if not os.path.exists(data_path):
        print(f"Processed data not found at {data_path}")
        print("Please run text_processing_phd.py first!")
        return

    processed_data = torch.load(data_path)
    phd_compute = PHDClassComputation()
    class_phds = phd_compute.compute_all_class_phds(processed_data=processed_data)
    phd_compute.analyze_phd_patterns(class_phds=class_phds)
    output_path = "phd_method/class_phd_results/snli_10k_subset_enhanced_multilayer_hdim0.pt"
    phd_compute.save_phd_results(class_phds=class_phds, output_path=output_path)

if __name__ == "__main__":
    test_phd_computation()


















