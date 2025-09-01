import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
import warnings

try:
    import persim
    PERSIM_AVAILABLE = True
    print("Persim library available for bottleneck distance computation")
except ImportError:
    PERSIM_AVAILABLE = False
    print("WARNING: Persim not available. Install with: pip install persim")
    print("Note: This is required for bottleneck distance computation")

class PersistencePrototypeCreator:
    """
    Creates prototype persistence diagrams for each class based on the similarity analysis.
    
    Given the excellent stability results (CV < 0.05), we can confidently create
    representative prototypes for regularization.
    """
    
    def __init__(self, diagrams_data: Dict):
        self.diagrams_data = diagrams_data
        self.prototypes = {}
        self.bottleneck_prototypes = {}
        print("Initialized prototype creator")
    
    def _clean_diagram(self, diagram: np.ndarray) -> np.ndarray:
        """Remove infinite and invalid points from persistence diagram"""
        if diagram.size == 0:
            return np.array([]).reshape(0, 2)
        
        # Remove infinite points
        finite_mask = np.isfinite(diagram).all(axis=1)
        clean_diagram = diagram[finite_mask]
        
        # Remove points where death <= birth (shouldn't happen but just in case)
        if clean_diagram.size > 0:
            valid_mask = clean_diagram[:, 1] > clean_diagram[:, 0]
            clean_diagram = clean_diagram[valid_mask]
        
        return clean_diagram

    def _compute_bottleneck_median(self, diagrams: List[np.ndarray], max_diagrams: int = 10) -> np.ndarray:
        """
        Compute median Persistence Diagram using bottleneck distance.
        
        This finds the persistence diagram that minimizes the sum of bottleneck 
        distances to all other diagrams - the theoretically correct "average".
        """
        if not PERSIM_AVAILABLE:
            raise ImportError("Persim library required for bottleneck distance computation")
        
        print(f"    Computing bottleneck distance Fréchet mean with {len(diagrams)} diagrams...")
        
        # Clean all diagrams
        cleaned_diagrams = [self._clean_diagram(d) for d in diagrams]
        non_empty_diagrams = [d for d in cleaned_diagrams if d.size > 0]
        
        if not non_empty_diagrams:
            return np.array([]).reshape(0, 2)
        
        if len(non_empty_diagrams) == 1:
            return non_empty_diagrams[0]

        np.random.seed(42)

        if len(non_empty_diagrams) > max_diagrams:
            print(f"      Subsampling to {max_diagrams} diagrams for efficiency...")
            indices = np.random.choice(len(non_empty_diagrams), max_diagrams, replace=False)
            non_empty_diagrams = [non_empty_diagrams[i] for i in indices]
        
        try:
            print("      Computing pairwise bottleneck distances...")
            
            # Compute bottleneck distance from each diagram to all others
            min_total_distance = float('inf')
            best_diagram_idx = 0
            
            for i, candidate_diagram in enumerate(non_empty_diagrams):
                total_distance = 0.0
                
                # Compute sum of bottleneck distances to all other diagrams
                for j, other_diagram in enumerate(non_empty_diagrams):
                    if i != j:
                        # Compute bottleneck distance
                        distance = persim.bottleneck(candidate_diagram, other_diagram)
                        total_distance += distance
                
                print(f"        Diagram {i}: total bottleneck distance = {total_distance:.6f}")
                
                # Keep track of the diagram with minimum total distance
                if total_distance < min_total_distance:
                    min_total_distance = total_distance
                    best_diagram_idx = i
            
            bottleneck_median = non_empty_diagrams[best_diagram_idx]
            
            print(f"      Selected diagram {best_diagram_idx} as Bottleneck Median")
            print(f"      Total bottleneck distance: {min_total_distance:.6f}")
            print(f"      Mean bottleneck distance: {min_total_distance/(len(non_empty_diagrams)-1):.6f}")
            
            if len(bottleneck_median) > 0:
                persistences = bottleneck_median[:, 1] - bottleneck_median[:, 0]
                print(f"      Features: {len(bottleneck_median)}")
                print(f"      Total persistence: {np.sum(persistences):.4f}")
                print(f"      Max persistence: {np.max(persistences):.6f}")
            
            return bottleneck_median
            
        except Exception as e:
            print(f"      Bottleneck Fréchet mean computation failed: {e}")
            print(f"      Error details: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("      Falling back to robust centroid method...")
            return self._compute_robust_average_prototype(non_empty_diagrams)
    
    def _compute_centroid_prototype(self, diagrams: List[np.ndarray]) -> np.ndarray:
        """
        Method 1: Simple centroid-based averaging
        Works well when diagrams have similar structure (which your analysis confirms)
        """
        print("    Using centroid-based averaging...")
        
        # Clean all diagrams
        cleaned_diagrams = [self._clean_diagram(d) for d in diagrams]
        
        # Remove empty diagrams
        non_empty_diagrams = [d for d in cleaned_diagrams if d.size > 0]
        
        if not non_empty_diagrams:
            return np.array([]).reshape(0, 2)
        
        # Find common number of features (use median)
        feature_counts = [len(d) for d in non_empty_diagrams]
        target_features = int(np.median(feature_counts))
        
        print(f"      Target features: {target_features} (median of {np.min(feature_counts)}-{np.max(feature_counts)})")
        
        # Pad or truncate diagrams to common size
        normalized_diagrams = []
        for diagram in non_empty_diagrams:
            if len(diagram) >= target_features:
                # Sort by persistence and take top features
                persistences = diagram[:, 1] - diagram[:, 0]
                top_indices = np.argsort(persistences)[-target_features:]
                normalized_diagrams.append(diagram[top_indices])
            else:
                # Pad with zeros (will be filtered out later)
                padding = np.zeros((target_features - len(diagram), 2))
                normalized_diagrams.append(np.vstack([diagram, padding]))
        
        # Compute centroid
        if normalized_diagrams:
            centroid = np.mean(normalized_diagrams, axis=0)
            # Remove zero-persistence points
            valid_mask = centroid[:, 1] > centroid[:, 0]
            centroid = centroid[valid_mask]
            return centroid
        else:
            return np.array([]).reshape(0, 2)
    
    def _compute_medoid_prototype(self, diagrams: List[np.ndarray]) -> np.ndarray:
        """
        Method 2: Medoid-based prototype (most representative diagram)
        More robust to outliers
        """
        print("    Using medoid-based selection...")
        
        # Clean all diagrams
        cleaned_diagrams = [self._clean_diagram(d) for d in diagrams]
        non_empty_diagrams = [d for d in cleaned_diagrams if d.size > 0]
        
        if len(non_empty_diagrams) <= 1:
            return non_empty_diagrams[0] if non_empty_diagrams else np.array([]).reshape(0, 2)
        
        # Compute simple signature distances (fast approximation)
        signatures = []
        for diagram in non_empty_diagrams:
            if diagram.size > 0:
                persistences = diagram[:, 1] - diagram[:, 0]
                sig = np.array([
                    np.sum(persistences),
                    np.mean(persistences),
                    np.std(persistences),
                    len(persistences)
                ])
                signatures.append(sig)
            else:
                signatures.append(np.zeros(4))
        
        # Find medoid (diagram with minimum average distance to all others)
        signatures = np.array(signatures)
        distances = pdist(signatures, metric='cosine')
        distance_matrix = squareform(distances)
        
        # Find index of medoid
        avg_distances = np.mean(distance_matrix, axis=1)
        medoid_idx = np.argmin(avg_distances)
        
        print(f"      Selected medoid: diagram {medoid_idx} with avg distance {avg_distances[medoid_idx]:.3f}")
        
        return non_empty_diagrams[medoid_idx]
    
    def _compute_robust_average_prototype(self, diagrams: List[np.ndarray]) -> np.ndarray:
        """
        Method 3: Robust averaging with outlier removal
        Best approach given your high stability
        """
        print("    Using robust averaging with outlier removal...")
        
        # Clean all diagrams
        cleaned_diagrams = [self._clean_diagram(d) for d in diagrams]
        non_empty_diagrams = [d for d in cleaned_diagrams if d.size > 0]
        
        if not non_empty_diagrams:
            return np.array([]).reshape(0, 2)
        
        # Compute signatures for outlier detection
        signatures = []
        for diagram in non_empty_diagrams:
            persistences = diagram[:, 1] - diagram[:, 0]
            sig = np.array([
                np.sum(persistences),
                np.mean(persistences),
                np.std(persistences),
                len(persistences),
                np.max(persistences)
            ])
            signatures.append(sig)
        
        signatures = np.array(signatures)
        
        # Remove outliers using IQR method
        Q1 = np.percentile(signatures, 25, axis=0)
        Q3 = np.percentile(signatures, 75, axis=0)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find inliers
        inlier_mask = np.all((signatures >= lower_bound) & (signatures <= upper_bound), axis=1)
        inlier_diagrams = [non_empty_diagrams[i] for i in range(len(non_empty_diagrams)) if inlier_mask[i]]
        
        print(f"      Kept {len(inlier_diagrams)}/{len(non_empty_diagrams)} diagrams after outlier removal")
        
        if not inlier_diagrams:
            inlier_diagrams = non_empty_diagrams  # Fallback if all removed
        
        # Now compute centroid on inliers
        feature_counts = [len(d) for d in inlier_diagrams]
        target_features = int(np.median(feature_counts))
        
        # Normalize to common size
        normalized_diagrams = []
        for diagram in inlier_diagrams:
            if len(diagram) >= target_features:
                persistences = diagram[:, 1] - diagram[:, 0]
                top_indices = np.argsort(persistences)[-target_features:]
                normalized_diagrams.append(diagram[top_indices])
            else:
                # Pad with the most persistent feature repeated
                if len(diagram) > 0:
                    most_persistent = diagram[np.argmax(diagram[:, 1] - diagram[:, 0])]
                    padding = np.tile(most_persistent, (target_features - len(diagram), 1))
                    normalized_diagrams.append(np.vstack([diagram, padding]))
        
        # Compute robust centroid
        if normalized_diagrams:
            centroid = np.mean(normalized_diagrams, axis=0)
            valid_mask = centroid[:, 1] > centroid[:, 0]
            centroid = centroid[valid_mask]
            return centroid
        else:
            return np.array([]).reshape(0, 2)
    
    def create_prototypes(self, method: str = 'robust') -> Dict[str, Dict[str, np.ndarray]]:
        """
        Create prototype persistence diagrams for each class
        
        Args:
            method: 'centroid', 'medoid', 'robust', or 'bottleneck' 
        
        Returns:
            Dictionary with prototypes for each class and dimension
        """
        print(f"\nCreating prototypes using {method} method...")
        print("="*50)
        
        prototypes = {}
        
        for class_name, data in self.diagrams_data.items():
            print(f"\n--- Creating prototypes for {class_name.upper()} ---")
            
            class_prototypes = {}
            
            # Process H0 and H1 separately
            for dim_name in ['H0', 'H1']:
                print(f"  Processing {dim_name}...")
                diagrams = data[dim_name]
                
                if method == 'centroid':
                    prototype = self._compute_centroid_prototype(diagrams)
                elif method == 'medoid':
                    prototype = self._compute_medoid_prototype(diagrams)
                elif method == 'robust':
                    prototype = self._compute_robust_average_prototype(diagrams)
                elif method == 'bottleneck':
                    prototype = self._compute_bottleneck_median(diagrams)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                class_prototypes[dim_name] = prototype
                print(f"    {dim_name} prototype: {len(prototype)} features")
                
                if len(prototype) > 0:
                    persistences = prototype[:, 1] - prototype[:, 0]
                    print(f"    Total persistence: {np.sum(persistences):.4f}")
                    print(f"    Max persistence: {np.max(persistences):.4f}")
            
            prototypes[class_name] = class_prototypes
        
        if method == 'bottleneck':
            self.bottleneck_prototypes = prototypes
        else:
            self.prototypes = prototypes
        return prototypes
    
    def save_prototypes(self, save_path: str, method: str = None):
        """Save prototypes to file"""
        if method == 'bottleneck':
            prototypes_to_save = self.bottleneck_prototypes
        else:
            prototypes_to_save = self.prototypes

        with open(save_path, 'wb') as f:
            pickle.dump(prototypes_to_save, f)
        print(f"\nPrototypes saved to {save_path}")
    
    
    def visualize_prototypes(self, save_path: str = None, method: str = None):
        """Create visualizations showing H0 (top row) and H1 (bottom row) prototypes"""

        if method == 'bottleneck':
            prototypes_to_viz = self.bottleneck_prototypes
        else:
            prototypes_to_viz = self.prototypes

        if not prototypes_to_viz:
            print("No prototypes to visualize. Run create_prototypes() first.")
            return

        # Create 2x3 subplot grid: H0 on top row, H1 on bottom row
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Persistence Diagram Prototypes ({method} method)', fontsize=16)

        class_names = list(prototypes_to_viz.keys())

        # Find global limits for each dimension separately
        all_h0_data = []
        all_h1_data = []
        for class_name in class_names:
            h0_data = prototypes_to_viz[class_name]['H0']
            h1_data = prototypes_to_viz[class_name]['H1']
            if len(h0_data) > 0:
                all_h0_data.append(h0_data)
            if len(h1_data) > 0:
                all_h1_data.append(h1_data)

        # Set limits for H0
        if all_h0_data:
            all_h0_points = np.vstack(all_h0_data)
            h0_min = np.min(all_h0_points)
            h0_max = np.max(all_h0_points)
            h0_padding = (h0_max - h0_min) * 0.05
            h0_plot_min = -0.1  # Move y-axis down to show points at origin
            h0_plot_max = h0_max + h0_padding
        else:
            h0_plot_min, h0_plot_max = -0.1, 1

        # Set limits for H1
        if all_h1_data:
            all_h1_points = np.vstack(all_h1_data)
            h1_min = np.min(all_h1_points)
            h1_max = np.max(all_h1_points)
            h1_padding = (h1_max - h1_min) * 0.05
            h1_plot_min = max(0, h1_min - h1_padding)
            h1_plot_max = h1_max + h1_padding
        else:
            h1_plot_min, h1_plot_max = 0, 1

        for class_idx, class_name in enumerate(class_names):
            # H0 plots (top row)
            ax_h0 = axes[0, class_idx]
            h0_prototype = prototypes_to_viz[class_name]['H0']

            if len(h0_prototype) > 0:
                ax_h0.scatter(h0_prototype[:, 0], h0_prototype[:, 1], alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
                ax_h0.plot([h0_plot_min, h0_plot_max], [h0_plot_min, h0_plot_max], 'k--', alpha=0.5, linewidth=1)
                ax_h0.set_xlim(h0_plot_min, h0_plot_max)
                ax_h0.set_ylim(h0_plot_min, h0_plot_max)
            else:
                ax_h0.text(0.5, 0.5, 'Empty\nH0 Diagram', ha='center', va='center', transform=ax_h0.transAxes, fontsize=14)
                ax_h0.set_xlim(0, 1)
                ax_h0.set_ylim(0, 1)

            ax_h0.set_xlabel('Birth', fontsize=12)
            ax_h0.set_ylabel('Death', fontsize=12)
            ax_h0.set_title(f'{class_name.title()} H0', fontsize=14, fontweight='bold')
            ax_h0.grid(True, alpha=0.3)
            ax_h0.set_aspect('equal')

            # H1 plots (bottom row)
            ax_h1 = axes[1, class_idx]
            h1_prototype = prototypes_to_viz[class_name]['H1']

            if len(h1_prototype) > 0:
                ax_h1.scatter(h1_prototype[:, 0], h1_prototype[:, 1], alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
                ax_h1.plot([h1_plot_min, h1_plot_max], [h1_plot_min, h1_plot_max], 'k--', alpha=0.5, linewidth=1)
                ax_h1.set_xlim(h1_plot_min, h1_plot_max)
                ax_h1.set_ylim(h1_plot_min, h1_plot_max)
            else:
                ax_h1.text(0.5, 0.5, 'Empty\nH1 Diagram', ha='center', va='center', transform=ax_h1.transAxes, fontsize=14)
                ax_h1.set_xlim(0, 1)
                ax_h1.set_ylim(0, 1)

            ax_h1.set_xlabel('Birth', fontsize=12)
            ax_h1.set_ylabel('Death', fontsize=12)
            ax_h1.set_title(f'{class_name.title()} H1', fontsize=14, fontweight='bold')
            ax_h1.grid(True, alpha=0.3)
            ax_h1.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPrototype visualization saved to {save_path}")

        plt.show()
    
    def get_prototype_summary(self, method) -> str:
        """Generate a summary report of the prototypes"""
        if method == 'bottleneck':
            prototypes_to_summarize = self.bottleneck_prototypes
        else:
            prototypes_to_summarize = self.prototypes
            
        if not prototypes_to_summarize:
            return f"No {method} prototypes created yet."
        
        
        report = []
        report.append("="*60)
        report.append("PERSISTENCE DIAGRAM PROTOTYPES SUMMARY")
        report.append("="*60)
        
        for class_name, class_prototypes in prototypes_to_summarize.items():
            report.append(f"\n{class_name.upper()} CLASS PROTOTYPES:")
            
            for dim_name, prototype in class_prototypes.items():
                report.append(f"  {dim_name}:")
                
                if len(prototype) > 0:
                    persistences = prototype[:, 1] - prototype[:, 0]
                    report.append(f"    Features: {len(prototype)}")
                    report.append(f"    Total persistence: {np.sum(persistences):.4f}")
                    report.append(f"    Max persistence: {np.max(persistences):.4f}")
                    report.append(f"    Mean persistence: {np.mean(persistences):.4f}")
                else:
                    report.append(f"    Empty diagram")
        
        return "\n".join(report)



def load_and_visualize_bottleneck_prototypes(prototypes_path):
    """Load bottleneck prototypes from pkl file and visualize them"""
    
    
    # Load the prototypes
    with open(prototypes_path, 'rb') as f:
        bottleneck_prototypes = pickle.load(f)
    
    print("Loaded bottleneck prototypes:")
    for class_name, prototypes in bottleneck_prototypes.items():
        print(f"{class_name}:")
        for dim, diagram in prototypes.items():
            print(f"  {dim}: {len(diagram)} features")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Bottleneck Distance Prototypes', fontsize=16)
    
    class_names = list(bottleneck_prototypes.keys())
    
    # Find global limits
    all_h0_data = []
    all_h1_data = []
    for class_name in class_names:
        h0_data = bottleneck_prototypes[class_name]['H0']
        h1_data = bottleneck_prototypes[class_name]['H1']
        if len(h0_data) > 0:
            all_h0_data.append(h0_data)
        if len(h1_data) > 0:
            all_h1_data.append(h1_data)
    
    # Set limits for H0
    if all_h0_data:
        all_h0_points = np.vstack(all_h0_data)
        h0_min = np.min(all_h0_points)
        h0_max = np.max(all_h0_points)
        h0_padding = (h0_max - h0_min) * 0.05
        h0_plot_min = max(0, h0_min - h0_padding)
        h0_plot_max = h0_max + h0_padding
    else:
        h0_plot_min, h0_plot_max = 0, 1
    
    # Set limits for H1
    if all_h1_data:
        all_h1_points = np.vstack(all_h1_data)
        h1_min = np.min(all_h1_points)
        h1_max = np.max(all_h1_points)
        h1_padding = (h1_max - h1_min) * 0.05
        h1_plot_min = max(0, h1_min - h1_padding)
        h1_plot_max = h1_max + h1_padding
    else:
        h1_plot_min, h1_plot_max = 0, 1
    
    for class_idx, class_name in enumerate(class_names):
        # H0 plots (top row)
        ax_h0 = axes[0, class_idx]
        h0_prototype = bottleneck_prototypes[class_name]['H0']
        
        if len(h0_prototype) > 0:
            ax_h0.scatter(h0_prototype[:, 0], h0_prototype[:, 1], 
                         alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
            ax_h0.plot([h0_plot_min, h0_plot_max], [h0_plot_min, h0_plot_max], 
                      'k--', alpha=0.5, linewidth=1)
            ax_h0.set_xlim(h0_plot_min, h0_plot_max)
            ax_h0.set_ylim(h0_plot_min, h0_plot_max)
        else:
            ax_h0.text(0.5, 0.5, 'Empty\nH0 Diagram', ha='center', va='center', 
                      transform=ax_h0.transAxes, fontsize=14)
            ax_h0.set_xlim(0, 1)
            ax_h0.set_ylim(0, 1)
        
        ax_h0.set_xlabel('Birth', fontsize=12)
        ax_h0.set_ylabel('Death', fontsize=12)
        ax_h0.set_title(f'{class_name.title()} H0', fontsize=14, fontweight='bold')
        ax_h0.grid(True, alpha=0.3)
        ax_h0.set_aspect('equal')
        
        # H1 plots (bottom row)
        ax_h1 = axes[1, class_idx]
        h1_prototype = bottleneck_prototypes[class_name]['H1']
        
        if len(h1_prototype) > 0:
            ax_h1.scatter(h1_prototype[:, 0], h1_prototype[:, 1], 
                         alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
            ax_h1.plot([h1_plot_min, h1_plot_max], [h1_plot_min, h1_plot_max], 
                      'k--', alpha=0.5, linewidth=1)
            ax_h1.set_xlim(h1_plot_min, h1_plot_max)
            ax_h1.set_ylim(h1_plot_min, h1_plot_max)
        else:
            ax_h1.text(0.5, 0.5, 'Empty\nH1 Diagram', ha='center', va='center', 
                      transform=ax_h1.transAxes, fontsize=14)
            ax_h1.set_xlim(0, 1)
            ax_h1.set_ylim(0, 1)
        
        ax_h1.set_xlabel('Birth', fontsize=12)
        ax_h1.set_ylabel('Death', fontsize=12)
        ax_h1.set_title(f'{class_name.title()} H1', fontsize=14, fontweight='bold')
        ax_h1.grid(True, alpha=0.3)
        ax_h1.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    return bottleneck_prototypes


def main():
    """Main function to create prototypes"""
    print("Creating persistence diagram prototypes...")
    
    # Load the collected diagrams
    DIAGRAMS_PATH = 'entailment_surfaces/supervised_contrastive_autoencoder/src/persistence_diagrams/collected_diagrams_LATTICE_COSINE.pkl'
    
    if not Path(DIAGRAMS_PATH).exists():
        print(f"Error: {DIAGRAMS_PATH} not found!")
        return
    
    with open(DIAGRAMS_PATH, 'rb') as f:
        all_diagrams = pickle.load(f)
    
    # Create prototypes
    creator = PersistencePrototypeCreator(all_diagrams)
    
    methods = ['bottleneck']
    for method in methods:
        prototypes = creator.create_prototypes(method=method)
    
        # Save prototypes
        PROTOTYPES_PATH = f'entailment_surfaces/supervised_contrastive_autoencoder/src/persistence_diagrams/prototypes_{method}_LATTICE_COSINE.pkl'
        creator.save_prototypes(PROTOTYPES_PATH, method=method)
    
        # Generate summary
        summary = creator.get_prototype_summary(method=method)
        print("\n" + summary)
    
        # Save summary
        SUMMARY_PATH = f'entailment_surfaces/supervised_contrastive_autoencoder/src/persistence_diagrams/prototype__{method}_summary_LATTICE_COSINE.txt'
        with open(SUMMARY_PATH, 'w') as f:
            f.write(summary)
    
        # Create visualizations
        VIZ_PATH = f'entailment_surfaces/supervised_contrastive_autoencoder/src/persistence_diagrams/prototype__{method}_visualizations_LATTICE_COSINE.png'
        creator.visualize_prototypes(VIZ_PATH, method=method)
    
        print(f"\nPrototype creation complete!")
        print(f"Files created:")
        print(f"  - Prototypes: {PROTOTYPES_PATH}")
        print(f"  - Summary: {SUMMARY_PATH}")
        print(f"  - Visualizations: {VIZ_PATH}")


if __name__ == '__main__':
    main()
    # prototypes_path = 'entailment_surfaces/supervised_contrastive_autoencoder/src/persistence_diagrams/prototypes_bottleneck.pkl'
    
    # if not Path(prototypes_path).exists():
    #     print(f"Error: {prototypes_path} not found!")
    #     print("Run the prototype creation first.")
    # else:
    #     print("Loading and visualizing bottleneck prototypes...")
        
    #     # Load the prototypes
    #     with open(prototypes_path, 'rb') as f:
    #         bottleneck_prototypes = pickle.load(f)
        
    #     # Create a dummy creator object just for visualization
    #     creator = PersistencePrototypeCreator({})  # Empty data, we just want the viz method
    #     creator.bottleneck_prototypes = bottleneck_prototypes
        
    #     # Create visualization with the fixed method
    #     viz_path = 'entailment_surfaces/supervised_contrastive_autoencoder/src/persistence_diagrams/bottleneck_prototypes_visualization.png'
    #     creator.visualize_prototypes(viz_path, method='bottleneck')
        
    #     print(f"Bottleneck prototypes visualization saved to {viz_path}")
