import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Tuple
import os
from sklearn.decomposition import PCA

# Updated imports for asymmetric versions
from entailment_cones_asymmetry import HyperbolicEntailmentCones, EnhancedHyperbolicConeEmbeddingPipeline
from order_embeddings_asymmetry import EntailmentDataset
from torch.utils.data import DataLoader


class ImprovedConeVisualizer:
    """
    Enhanced visualizer for real hyperbolic entailment cone data with asymmetric features
    """

    def __init__(self, cone_pipeline: 'EnhancedHyperbolicConeEmbeddingPipeline'):
        self.cone_pipeline = cone_pipeline
        self.cone_computer = cone_pipeline.cone_computer
        
        # Add validation for hyperbolic_pipeline
        if self.cone_pipeline.hyperbolic_pipeline is None:
            raise RuntimeError("Hyperbolic pipeline failed to initialize. Check model path and dependencies.")
        
        self.device = self.cone_pipeline.hyperbolic_pipeline.device

    def select_representative_examples(self, dataset, n_examples=5):
        """
        Select representative examples that show clear energy differences using enhanced features
        """
        examples_by_label = {'entailment': [], 'neutral': [], 'contradiction': []}
        energies_by_label = {'entailment': [], 'neutral': [], 'contradiction': []}

        # Collect all examples and their energies
        for item in dataset:
            label_str = item['label_str']
            premise_emb = item['premise_emb'].to(self.device)
            hypothesis_emb = item['hypothesis_emb'].to(self.device)

            # Get enhanced hyperbolic embeddings and cone energy
            results = self.cone_pipeline.hyperbolic_pipeline.compute_enhanced_hyperbolic_energies(
                premise_emb.unsqueeze(0), hypothesis_emb.unsqueeze(0)
            )

            premise_hyp = results['premise_hyperbolic'][0]
            hypothesis_hyp = results['hypothesis_hyperbolic'][0]

            # Compute cone energy using the enhanced cone computer
            cone_energy = self.cone_computer.cone_membership_energy(
                premise_hyp.unsqueeze(0), hypothesis_hyp.unsqueeze(0)
            ).item()

            examples_by_label[label_str].append((premise_hyp, hypothesis_hyp, cone_energy))
            energies_by_label[label_str].append(cone_energy)

        # Select representative examples (median energy for each type)
        selected_examples = {}
        for label in ['entailment', 'neutral', 'contradiction']:
            if examples_by_label[label]:
                energies = energies_by_label[label]
                examples = examples_by_label[label]

                # Sort by energy and pick median, min, max for variety
                sorted_pairs = sorted(zip(energies, examples), key=lambda x: x[0])

                median_idx = len(sorted_pairs) // 2
                selected_examples[label] = {
                    'low': sorted_pairs[0][1],  # Lowest energy
                    'median': sorted_pairs[median_idx][1],  # Median energy
                    'high': sorted_pairs[-1][1] if len(sorted_pairs) > 1 else sorted_pairs[0][1]  # Highest energy
                }

                print(f"{label.capitalize()} energy range: {sorted_pairs[0][0]:.3f} - {sorted_pairs[-1][0]:.3f}")

        return selected_examples

    def create_pca_projection_visualization(self, examples_by_label, save_path):
        """
        Use PCA to create 2D projections that preserve cone relationships
        """
        # Collect all embeddings for PCA
        all_premises = []
        all_hypotheses = []
        labels = []
        energies = []

        for label, variants in examples_by_label.items():
            for variant_name, (premise, hypothesis, energy) in variants.items():
                all_premises.append(premise.cpu().numpy())
                all_hypotheses.append(hypothesis.cpu().numpy())
                labels.append(f"{label}_{variant_name}")
                energies.append(energy)

        # Combine all embeddings for PCA
        all_embeddings = np.vstack([all_premises, all_hypotheses])

        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(all_embeddings)

        # Split back into premises and hypotheses
        n_examples = len(all_premises)
        premises_2d = embeddings_2d[:n_examples]
        hypotheses_2d = embeddings_2d[n_examples:]

        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        colors = {'entailment': 'green', 'neutral': 'blue', 'contradiction': 'red'}

        for i, label in enumerate(['entailment', 'neutral', 'contradiction']):
            ax = axes[i]

            # Find examples for this label
            label_indices = [j for j, l in enumerate(labels) if l.startswith(label)]

            if not label_indices:
                continue

            # Plot all variants for this label
            for idx in label_indices:
                premise_2d = premises_2d[idx]
                hypothesis_2d = hypotheses_2d[idx]
                energy = energies[idx]
                variant = labels[idx].split('_')[1]

                # Plot points
                color = colors[label]
                alpha = 0.7 if variant == 'median' else 0.4
                size = 120 if variant == 'median' else 80

                ax.scatter(*premise_2d, s=size, c='orange', marker='o',
                           edgecolors='black', linewidth=1, alpha=alpha, zorder=5)
                ax.scatter(*hypothesis_2d, s=size, c=color, marker='^',
                           edgecolors='black', linewidth=1, alpha=alpha, zorder=5)

                # Draw connection
                ax.plot([premise_2d[0], hypothesis_2d[0]],
                        [premise_2d[1], hypothesis_2d[1]],
                        color=color, linewidth=2, alpha=alpha)

                # Add energy annotation for median example
                if variant == 'median':
                    ax.annotate(f'E={energy:.3f}',
                                xy=hypothesis_2d, xytext=(5, 5),
                                textcoords='offset points', fontsize=10,
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            ax.set_title(f'{label.title()}\n(Enhanced PCA Projection)', fontsize=12, fontweight='bold')
            ax.set_xlabel('PC1 (Cone-Preserving)', fontsize=10)
            ax.set_ylabel('PC2', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

        plt.suptitle('Enhanced Hyperbolic Entailment Cones: PCA-Projected Real Data with Asymmetric Features', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved enhanced PCA projection visualization to {save_path}")

    def create_energy_distribution_plot(self, dataset, save_path):
        """
        Create distribution plots showing enhanced cone energy differences
        """
        energies_by_label = {'entailment': [], 'neutral': [], 'contradiction': []}
        enhanced_features_by_label = {
            'entailment': {'forward_cone': [], 'backward_cone': [], 'asymmetry': []},
            'neutral': {'forward_cone': [], 'backward_cone': [], 'asymmetry': []},
            'contradiction': {'forward_cone': [], 'backward_cone': [], 'asymmetry': []}
        }

        # Collect all energies and enhanced features
        for item in dataset:
            label_str = item['label_str']
            premise_emb = item['premise_emb'].to(self.device)
            hypothesis_emb = item['hypothesis_emb'].to(self.device)

            # Get enhanced results
            results = self.cone_pipeline.compute_enhanced_cone_energies(
                premise_emb.unsqueeze(0), hypothesis_emb.unsqueeze(0)
            )

            # Basic cone energy
            cone_energy = results['cone_energies'][0].item()
            energies_by_label[label_str].append(cone_energy)

            # Enhanced asymmetric features
            if 'forward_cone_energies' in results:
                enhanced_features_by_label[label_str]['forward_cone'].append(
                    results['forward_cone_energies'][0].item()
                )
            if 'backward_cone_energies' in results:
                enhanced_features_by_label[label_str]['backward_cone'].append(
                    results['backward_cone_energies'][0].item()
                )
            if 'cone_asymmetries' in results:
                enhanced_features_by_label[label_str]['asymmetry'].append(
                    results['cone_asymmetries'][0].item()
                )

        # Create enhanced distribution plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Main cone energies (top left)
        colors = {'entailment': 'green', 'neutral': 'blue', 'contradiction': 'red'}
        ax = axes[0, 0]
        for label, energies in energies_by_label.items():
            ax.hist(energies, bins=20, alpha=0.7, label=f'{label.title()} (n={len(energies)})',
                     color=colors[label], edgecolor='black')
        ax.set_xlabel('Cone Violation Energy')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Enhanced Cone Violation Energies')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Box plot (top right)
        ax = axes[0, 1]
        data_for_box = [energies_by_label['entailment'],
                        energies_by_label['neutral'],
                        energies_by_label['contradiction']]

        bp = ax.boxplot(data_for_box, labels=['Entailment', 'Neutral', 'Contradiction'],
                         patch_artist=True)

        for patch, color in zip(bp['boxes'], ['green', 'blue', 'red']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Cone Violation Energy')
        ax.set_title('Energy Distribution by Relationship Type')
        ax.grid(True, alpha=0.3)

        # Add statistics
        for i, (label, energies) in enumerate(energies_by_label.items()):
            mean_energy = np.mean(energies)
            std_energy = np.std(energies)
            ax.text(i + 1, max(energies) * 0.9, f'μ={mean_energy:.3f}\nσ={std_energy:.3f}',
                     ha='center', va='top', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Forward vs Backward cone energies (bottom left)
        ax = axes[1, 0]
        for label in ['entailment', 'neutral', 'contradiction']:
            forward_energies = enhanced_features_by_label[label]['forward_cone']
            backward_energies = enhanced_features_by_label[label]['backward_cone']
            
            if forward_energies and backward_energies:
                ax.scatter(forward_energies, backward_energies, 
                          c=colors[label], label=label.title(), alpha=0.6, s=30)

        ax.set_xlabel('Forward Cone Energy')
        ax.set_ylabel('Backward Cone Energy')
        ax.set_title('Forward vs Backward Cone Energies')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, transform=ax.transAxes)  # diagonal line

        # Asymmetry measures (bottom right)
        ax = axes[1, 1]
        asymmetry_data = []
        asymmetry_labels = []
        
        for label in ['entailment', 'neutral', 'contradiction']:
            asymmetries = enhanced_features_by_label[label]['asymmetry']
            if asymmetries:
                asymmetry_data.append(asymmetries)
                asymmetry_labels.append(label.title())

        if asymmetry_data:
            bp = ax.boxplot(asymmetry_data, labels=asymmetry_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], ['green', 'blue', 'red'][:len(asymmetry_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_ylabel('Cone Asymmetry Measure')
        ax.set_title('Asymmetric Feature Distribution')
        ax.grid(True, alpha=0.3)

        plt.suptitle('Enhanced Cone Violation Energy Analysis: Real Data with Asymmetric Features', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved enhanced energy distribution plot to {save_path}")


def create_improved_real_data_visualizations():
    """
    Create improved visualizations using real data with enhanced asymmetric features
    """
    try:
        # Updated paths for asymmetric versions
        processed_data_path = "data/processed/snli_10k_subset_balanced.pt"
        model_path = "models/enhanced_order_embeddings_snli_10k_asymmetry.pt"

        print("\nCreating Improved Real Data Cone Visualizations with Asymmetric Features")
        print("=" * 80)

        # Check if files exist
        if not os.path.exists(processed_data_path):
            print(f"Error: Processed data file not found: {processed_data_path}")
            print("Available data files:")
            data_dir = "data/processed/"
            if os.path.exists(data_dir):
                for file in os.listdir(data_dir):
                    if file.endswith('.pt'):
                        print(f"  - {os.path.join(data_dir, file)}")
            return
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            print("Available model files:")
            model_dir = "models/"
            if os.path.exists(model_dir):
                for file in os.listdir(model_dir):
                    if file.endswith('.pt'):
                        print(f"  - {os.path.join(model_dir, file)}")
            return

        # Load enhanced pipeline and data with better error handling
        try:
            # Use the enhanced pipeline for asymmetric features
            pipeline = EnhancedHyperbolicConeEmbeddingPipeline(model_path)
            
            # Validate pipeline initialization
            if pipeline.hyperbolic_pipeline is None:
                print("Error: Enhanced hyperbolic pipeline failed to initialize properly")
                print("This could be due to:")
                print("1. Missing enhanced model file")
                print("2. Incompatible enhanced model format")  
                print("3. Missing dependencies for asymmetric features")
                return
                
        except Exception as e:
            print(f"Failed to create EnhancedHyperbolicConeEmbeddingPipeline: {e}")
            import traceback
            traceback.print_exc()
            return

        try:
            # Load with explicit weights_only parameter
            processed_data = torch.load(processed_data_path, weights_only=False)
            dataset = EntailmentDataset(processed_data)
            print(f"Loaded enhanced dataset with {len(dataset)} examples")
        except Exception as e:
            print(f"Failed to load processed data: {e}")
            return

        try:
            visualizer = ImprovedConeVisualizer(pipeline)
        except RuntimeError as e:
            print(f"Failed to create enhanced visualizer: {e}")
            return

        # Create output directory
        os.makedirs("plots/real_data_cone_visualizations", exist_ok=True)

        # 1. Select representative examples
        print("Selecting representative examples with enhanced features...")
        try:
            examples = visualizer.select_representative_examples(dataset)
        except Exception as e:
            print(f"Failed to select representative examples: {e}")
            import traceback
            traceback.print_exc()
            return

        # 2. Create enhanced PCA projection visualization
        print("Creating enhanced PCA projection visualization...")
        try:
            visualizer.create_pca_projection_visualization(
                examples,
                "plots/real_data_cone_visualizations/enhanced_pca_projection_comparison_snli_10k_asymmetry.png"
            )
        except Exception as e:
            print(f"Failed to create enhanced PCA visualization: {e}")
            import traceback
            traceback.print_exc()

        # 3. Create enhanced energy distribution plot
        print("Creating enhanced energy distribution analysis...")
        try:
            visualizer.create_energy_distribution_plot(
                dataset,
                "plots/real_data_cone_visualizations/enhanced_energy_distribution_analysis_snli_10k_asymmetry.png"
            )
        except Exception as e:
            print(f"Failed to create enhanced energy distribution plot: {e}")
            import traceback
            traceback.print_exc()

        print("Enhanced real data visualizations with asymmetric features completed!")

    except Exception as e:
        print(f"Error creating enhanced visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    create_improved_real_data_visualizations()