import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Tuple
import os
from sklearn.decomposition import PCA

from entailment_cones import HyperbolicEntailmentCones, HyperbolicConeEmbeddingPipeline
from order_embeddings import EntailmentDataset
from torch.utils.data import DataLoader


class ImprovedConeVisualizer:
    """
    Enhanced visualizer for real hyperbolic entailment cone data
    """

    def __init__(self, cone_pipeline: 'HyperbolicConeEmbeddingPipeline'):
        self.cone_pipeline = cone_pipeline
        self.cone_computer = cone_pipeline.cone_computer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def select_representative_examples(self, dataset, n_examples=5):
        """
        Select representative examples that show clear energy differences
        """
        examples_by_label = {'entailment': [], 'neutral': [], 'contradiction': []}
        energies_by_label = {'entailment': [], 'neutral': [], 'contradiction': []}

        # Collect all examples and their energies
        for item in dataset:
            label_str = item['label_str']
            premise_emb = item['premise_emb'].to(self.cone_pipeline.hyperbolic_pipeline.device)
            hypothesis_emb = item['hypothesis_emb'].to(self.cone_pipeline.hyperbolic_pipeline.device)

            # Get hyperbolic embeddings and cone energy
            results = self.cone_pipeline.hyperbolic_pipeline.compute_hyperbolic_energies(
                premise_emb.unsqueeze(0), hypothesis_emb.unsqueeze(0)
            )

            premise_hyp = results['premise_hyperbolic'][0]
            hypothesis_hyp = results['hypothesis_hyperbolic'][0]

            # Compute cone energy
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

            ax.set_title(f'{label.title()}\n(PCA Projection)', fontsize=12, fontweight='bold')
            ax.set_xlabel('PC1 (Cone-Preserving)', fontsize=10)
            ax.set_ylabel('PC2', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

        plt.suptitle('Hyperbolic Entailment Cones: PCA-Projected Real Data', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PCA projection visualization to {save_path}")

    def create_energy_distribution_plot(self, dataset, save_path):
        """
        Create distribution plots showing cone energy differences
        """
        energies_by_label = {'entailment': [], 'neutral': [], 'contradiction': []}

        # Collect all energies
        for item in dataset:
            label_str = item['label_str']
            premise_emb = item['premise_emb'].to(self.cone_pipeline.hyperbolic_pipeline.device)
            hypothesis_emb = item['hypothesis_emb'].to(self.cone_pipeline.hyperbolic_pipeline.device)

            results = self.cone_pipeline.hyperbolic_pipeline.compute_hyperbolic_energies(
                premise_emb.unsqueeze(0), hypothesis_emb.unsqueeze(0)
            )

            premise_hyp = results['premise_hyperbolic'][0]
            hypothesis_hyp = results['hypothesis_hyperbolic'][0]

            cone_energy = self.cone_computer.cone_membership_energy(
                premise_hyp.unsqueeze(0), hypothesis_hyp.unsqueeze(0)
            ).item()

            energies_by_label[label_str].append(cone_energy)

        # Create distribution plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Histogram
        colors = {'entailment': 'green', 'neutral': 'blue', 'contradiction': 'red'}
        for label, energies in energies_by_label.items():
            ax1.hist(energies, bins=20, alpha=0.7, label=f'{label.title()} (n={len(energies)})',
                     color=colors[label], edgecolor='black')

        ax1.set_xlabel('Cone Violation Energy')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Cone Violation Energies')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        data_for_box = [energies_by_label['entailment'],
                        energies_by_label['neutral'],
                        energies_by_label['contradiction']]

        bp = ax2.boxplot(data_for_box, labels=['Entailment', 'Neutral', 'Contradiction'],
                         patch_artist=True)

        for patch, color in zip(bp['boxes'], ['green', 'blue', 'red']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_ylabel('Cone Violation Energy')
        ax2.set_title('Energy Distribution by Relationship Type')
        ax2.grid(True, alpha=0.3)

        # Add statistics
        for i, (label, energies) in enumerate(energies_by_label.items()):
            mean_energy = np.mean(energies)
            std_energy = np.std(energies)
            ax2.text(i + 1, max(energies) * 0.9, f'μ={mean_energy:.3f}\nσ={std_energy:.3f}',
                     ha='center', va='top', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        plt.suptitle('Cone Violation Energy Analysis: Real Data', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved energy distribution plot to {save_path}")


def create_improved_real_data_visualizations():
    """
    Create improved visualizations using real data
    """
    try:
        processed_data_path = "data/processed/snli_10k_subset_balanced.pt"
        model_path = "models/enhanced_order_embeddings_snli_10k_asymmetry.pt"

        print("\nCreating Improved Real Data Cone Visualizations")
        print("=" * 60)

        # Load pipeline and data
        pipeline = HyperbolicConeEmbeddingPipeline(model_path)
        visualizer = ImprovedConeVisualizer(pipeline)

        processed_data = torch.load(processed_data_path)
        dataset = EntailmentDataset(processed_data)

        # Create output directory
        os.makedirs("plots/real_data_cone_visualizations", exist_ok=True)

        # 1. Select representative examples
        print("Selecting representative examples...")
        examples = visualizer.select_representative_examples(dataset)

        # 2. Create PCA projection visualization
        print("Creating PCA projection visualization...")
        visualizer.create_pca_projection_visualization(
            examples,
            "plots/real_data_cone_visualizations/pca_projection_comparison_snli_10k_asymmetry.png"
        )

        # 3. Create energy distribution plot
        print("Creating energy distribution analysis...")
        visualizer.create_energy_distribution_plot(
            dataset,
            "plots/real_data_cone_visualizations/energy_distribution_analysis_snli_10k_asymmetry.png"
        )

        print("Improved real data visualizations completed!")

    except Exception as e:
        print(f"Error creating improved visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    create_improved_real_data_visualizations()