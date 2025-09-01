import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Tuple
import os

from entailment_cones import HyperbolicEntailmentCones, HyperbolicConeEmbeddingPipeline
from order_embeddings import EntailmentDataset
from torch.utils.data import DataLoader

class ConeVisualiser:
    """
    Create visualisations of our entailment cones
    """
    def __init__(self, cone_computer: 'HyperbolicEntailmentCones'):
        self.cone_computer = cone_computer

    def visualise_2d_cone(self, premise: torch.Tensor, hypothesis: torch.Tensor, relationship_type: str,
                          save_path: str):
        """
            Create 2D visualization of a single entailment cone

            Args:
                premise: Premise embedding in Poincaré ball [2D]
                hypothesis: Hypothesis embedding in Poincaré ball [2D]
                relationship_type: 'entailment', 'neutral', or 'contradiction'
                save_path: Path to save the plot
        """

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Draw Poincaré ball boundary
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2, linestyle='--')
        ax.add_patch(circle)

        # Extract 2D coordinates
        premise_2d = premise[:2].detach().numpy()
        hypothesis_2d = hypothesis[:2].detach().numpy()

        # Compute cone properties
        aperture = self.cone_computer.cone_aperture(premise.unsqueeze(0)).item()
        cone_energy = self.cone_computer.cone_membership_energy(premise.unsqueeze(0), hypothesis.unsqueeze(0)).item()

        # Draw cone
        self.draw_cone_2d(ax, premise_2d, aperture)

        # Plot points
        colors = {'entailment': 'green', 'neutral': 'blue', 'contradiction': 'red'}
        color = colors.get(relationship_type, 'gray')

        ax.scatter(*premise_2d, s=200, c='orange', marker='o', label=f'Premise (apex)', edgecolors='black', linewidth=2, zorder=5)
        ax.scatter(*hypothesis_2d, s=200, c=color, marker='^', label=f'Hypothesis ({relationship_type})', edgecolors='black', linewidth=2, zorder=5)

        # Draw line from origin to premise (cone axis)
        ax.plot([0, premise_2d[0]], [0, premise_2d[1]], 'k--', alpha=0.5, linewidth=1)

        # Draw line from premise to hypothesis
        ax.plot([premise_2d[0], hypothesis_2d[0]], [premise_2d[1], hypothesis_2d[1]], color=color, linewidth=3, alpha=0.7)

        # Annotations
        ax.text(premise_2d[0] + 0.05, premise_2d[1] + 0.05, 'P', fontsize=14, fontweight='bold')
        ax.text(hypothesis_2d[0] + 0.05, hypothesis_2d[1] + 0.05, 'H', fontsize=14, fontweight='bold')

        # Title and labels
        ax.set_title(f'{relationship_type.title()} Relationship\n'
                     f'Cone Aperture: {np.degrees(aperture):.1f}°, '
                     f'Violation Energy: {cone_energy:.3f}',
                     fontsize=16, fontweight='bold')

        ax.set_xlabel('Hyperbolic Dimension 1', fontsize=12)
        ax.set_ylabel('Hyperbolic Dimension 2', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)

        # Add text box with details
        textstr = f'''Cone Properties:
        • Aperture: {np.degrees(aperture):.1f}°
        • Violation Energy: {cone_energy:.3f}
        • Premise Norm: {torch.norm(premise):.3f}
        • Hypothesis Norm: {torch.norm(hypothesis):.3f}'''

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 2D cone visualization to {save_path}")

    def draw_cone_2d(self, ax, premise_2d: np.ndarray, aperture: float):
        """Draw 2d representation of the cone - CORRECTED VERSION"""
        # Cone direction (from origin to premise)
        premise_norm = np.linalg.norm(premise_2d)
        if premise_norm < 1e-6:
            return

        cone_axis = premise_2d / premise_norm

        # Calculate the angle of the cone axis
        axis_angle = np.arctan2(cone_axis[1], cone_axis[0])

        # Create cone boundaries by rotating from the axis direction
        theta_start = axis_angle - aperture  # Rotate counterclockwise by aperture
        theta_end = axis_angle + aperture  # Rotate clockwise by aperture

        # Create the cone sector from the premise point
        theta_range = np.linspace(theta_start, theta_end, 50)

        # Extend to circle boundary
        t_max = 0.95  # Stay inside Poincaré ball

        # Create cone shape
        cone_x = [premise_2d[0]]  # Start at premise
        cone_y = [premise_2d[1]]

        # Add arc points
        for theta in theta_range:
            x = t_max * np.cos(theta)
            y = t_max * np.sin(theta)
            cone_x.append(x)
            cone_y.append(y)

        # Close back to premise
        cone_x.append(premise_2d[0])
        cone_y.append(premise_2d[1])

        # Fill the cone region (this should contain entailment hypotheses)
        ax.fill(cone_x, cone_y, alpha=0.2, color='orange', label='Entailment Cone')

        # Draw cone boundary lines
        boundary_1_x = t_max * np.cos(theta_start)
        boundary_1_y = t_max * np.sin(theta_start)
        boundary_2_x = t_max * np.cos(theta_end)
        boundary_2_y = t_max * np.sin(theta_end)

        ax.plot([premise_2d[0], boundary_1_x], [premise_2d[1], boundary_1_y],
                'orange', linewidth=2, alpha=0.8)
        ax.plot([premise_2d[0], boundary_2_x], [premise_2d[1], boundary_2_y],
                'orange', linewidth=2, alpha=0.8)


    def create_comparison_plots(self, examples: Dict[str, Tuple[torch.Tensor, torch.Tensor]], save_path: str):
        """
        Create side-by-side comparison of entailment, neutral, and contradiction cones

        Args:
            examples: Dict with 'entailment', 'neutral', 'contradiction' keys
            Each value is (premise, hypothesis) tuple
            save_path: Path to save the plot
        """

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        relationship_types = ['entailment', 'neutral', 'contradiction']
        colors = {'entailment': 'green', 'neutral': 'blue', 'contradiction': 'red'}

        for i, rel_type in enumerate(relationship_types):
            if rel_type not in examples:
                continue

            premise, hypothesis = examples[rel_type]
            ax = axes[i]

            # Draw Poincaré ball boundary
            circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2, linestyle='--')
            ax.add_patch(circle)

            # Extract 2D coordinates
            premise_2d = premise[:2].detach().numpy()
            hypothesis_2d = hypothesis[:2].detach().numpy()

            # Compute cone properties
            aperture = self.cone_computer.cone_aperture(premise.unsqueeze(0)).item()
            cone_energy = self.cone_computer.cone_membership_energy(
                premise.unsqueeze(0), hypothesis.unsqueeze(0)
            ).item()

            # Draw cone
            self.draw_cone_2d(ax, premise_2d, aperture)

            # Plot points
            color = colors[rel_type]
            ax.scatter(*premise_2d, s=150, c='orange', marker='o',
                       edgecolors='black', linewidth=2, zorder=5)
            ax.scatter(*hypothesis_2d, s=150, c=color, marker='^',
                       edgecolors='black', linewidth=2, zorder=5)

            # Draw connections
            ax.plot([0, premise_2d[0]], [0, premise_2d[1]], 'k--', alpha=0.5, linewidth=1)
            ax.plot([premise_2d[0], hypothesis_2d[0]], [premise_2d[1], hypothesis_2d[1]],
                    color=color, linewidth=3, alpha=0.7)

            # Title and formatting
            ax.set_title(f'{rel_type.title()}\nEnergy: {cone_energy:.3f}',
                         fontsize=14, fontweight='bold')
            ax.set_aspect('equal')
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.grid(True, alpha=0.3)

            if i == 0:
                ax.set_ylabel('Hyperbolic Dimension 2', fontsize=12)
            ax.set_xlabel('Hyperbolic Dimension 1', fontsize=12)

        plt.suptitle('Hyperbolic Entailment Cones: Relationship Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

        plt.show()

    def create_energy_based_comparison_plots(self, examples: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                                             save_path: str):
        """
        Create comparison plots that show cone membership based on actual computed energies (for real data)
        rather than trying to draw geometric cones that don't always translate to 2D
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        relationship_types = ['entailment', 'neutral', 'contradiction']
        colors = {'entailment': 'green', 'neutral': 'blue', 'contradiction': 'red'}

        for i, rel_type in enumerate(relationship_types):
            if rel_type not in examples:
                continue

            premise, hypothesis = examples[rel_type]
            ax = axes[i]

            # Draw Poincaré ball boundary
            circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2, linestyle='--')
            ax.add_patch(circle)

            # Extract 2D coordinates
            premise_2d = premise[:2].detach().numpy()
            hypothesis_2d = hypothesis[:2].detach().numpy()

            # Compute cone energy
            cone_energy = self.cone_computer.cone_membership_energy(
                premise.unsqueeze(0), hypothesis.unsqueeze(0)
            ).item()

            # Determine cone membership based on energy
            is_inside_cone = cone_energy < 0.5  # Threshold for "inside"
            membership_status = "Inside Cone" if is_inside_cone else "Outside Cone"

            # Create energy-based background coloring
            if is_inside_cone:
                # Green background for inside cone
                ax.add_patch(plt.Circle((0, 0), 1, fill=True, color='lightgreen', alpha=0.2, zorder=1))
                status_text = f"✓ {membership_status}"
                status_color = 'green'
            else:
                # Red background for outside cone
                ax.add_patch(plt.Circle((0, 0), 1, fill=True, color='lightcoral', alpha=0.2, zorder=1))
                status_text = f"✗ {membership_status}"
                status_color = 'red'

            # Plot points with enhanced visibility
            color = colors[rel_type]

            # Premise point (cone apex)
            ax.scatter(*premise_2d, s=300, c='orange', marker='o',
                       label='Premise (cone apex)', edgecolors='black', linewidth=3, zorder=5)

            # Hypothesis point with energy-based styling
            edge_color = 'darkgreen' if is_inside_cone else 'darkred'
            ax.scatter(*hypothesis_2d, s=300, c=color, marker='^',
                       label=f'Hypothesis ({rel_type})', edgecolors=edge_color, linewidth=4, zorder=5)

            # Draw connections
            ax.plot([0, premise_2d[0]], [0, premise_2d[1]], 'k--', alpha=0.7, linewidth=3,
                    label='Cone axis (to premise)', zorder=3)
            ax.plot([premise_2d[0], hypothesis_2d[0]], [premise_2d[1], hypothesis_2d[1]],
                    color=color, linewidth=4, alpha=0.8, zorder=4)

            # Add origin point
            ax.scatter(0, 0, s=150, c='black', marker='x', linewidth=4, zorder=5)

            # Enhanced annotations
            ax.text(premise_2d[0] + 0.08, premise_2d[1] + 0.08, 'P', fontsize=20, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))
            ax.text(hypothesis_2d[0] + 0.08, hypothesis_2d[1] + 0.08, 'H', fontsize=20, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))
            ax.text(0.05, -0.05, 'O', fontsize=16, fontweight='bold')

            # Comprehensive title with energy and membership info
            ax.set_title(f'{rel_type.title()} Relationship\n'
                         f'Cone Energy: {cone_energy:.3f}\n'
                         f'{status_text}',
                         fontsize=14, fontweight='bold', color=status_color)

            # Add energy level indicator
            energy_level = "Very Low" if cone_energy < 0.1 else "Low" if cone_energy < 0.5 else "Medium" if cone_energy < 1.5 else "High"
            ax.text(0.02, 0.02, f'Energy Level: {energy_level}', transform=ax.transAxes,
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))

            # Add cone membership indicator with clear visual feedback
            membership_y = 0.95
            if is_inside_cone:
                ax.text(0.02, membership_y, '✓ Hypothesis IN Cone', transform=ax.transAxes,
                        fontsize=12, fontweight='bold', color='green',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
            else:
                ax.text(0.02, membership_y, '✗ Hypothesis OUT of Cone', transform=ax.transAxes,
                        fontsize=12, fontweight='bold', color='red',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))

            # Set axis properties
            ax.set_xlabel('Hyperbolic Dimension 1', fontsize=12)
            if i == 0:
                ax.set_ylabel('Hyperbolic Dimension 2', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)

            # Add border color based on cone membership
            for spine in ax.spines.values():
                spine.set_linewidth(5)
                spine.set_color(status_color)

            if i == 0:
                ax.legend(fontsize=11, loc='lower right')

        # Enhanced overall title with energy progression
        entail_energy = None
        neutral_energy = None
        contra_energy = None

        if 'entailment' in examples:
            entail_energy = self.cone_computer.cone_membership_energy(
                examples['entailment'][0].unsqueeze(0), examples['entailment'][1].unsqueeze(0)
            ).item()
        if 'neutral' in examples:
            neutral_energy = self.cone_computer.cone_membership_energy(
                examples['neutral'][0].unsqueeze(0), examples['neutral'][1].unsqueeze(0)
            ).item()
        if 'contradiction' in examples:
            contra_energy = self.cone_computer.cone_membership_energy(
                examples['contradiction'][0].unsqueeze(0), examples['contradiction'][1].unsqueeze(0)
            ).item()

        hierarchy_status = "✓ VALID" if (entail_energy and neutral_energy and contra_energy and
                                         entail_energy < neutral_energy < contra_energy) else "✗ Invalid"

        plt.suptitle(f'Hyperbolic Entailment Cones: Energy-Based Membership Analysis\n'
                     f'Energy Hierarchy: {entail_energy:.3f} < {neutral_energy:.3f} < {contra_energy:.3f} {hierarchy_status}',
                     fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved energy-based comparison plot to {save_path}")

def create_cone_visualisations():
    print("Creating hyperbolic entailment cone visualisations")
    print("=" * 60)

    # Initialize cone computer
    cone_computer = HyperbolicEntailmentCones(K=0.1, epsilon=0.1)
    visualizer = ConeVisualiser(cone_computer)

    # Create synthetic examples (2D for visualization)
    torch.manual_seed(42)

    # Example 1: Clear entailment (hypothesis in same direction as premise)
    premise_ent = torch.tensor([0.3, 0.2, 0.0, 0.0, 0.0], dtype=torch.float32)
    hypothesis_ent = torch.tensor([0.5, 0.3, 0.0, 0.0, 0.0], dtype=torch.float32)  # Same direction, further

    # Example 2: Neutral (hypothesis in different direction)
    premise_neu = torch.tensor([0.4, 0.1, 0.0, 0.0, 0.0], dtype=torch.float32)
    hypothesis_neu = torch.tensor([0.1, 0.4, 0.0, 0.0, 0.0], dtype=torch.float32)  # Perpendicular-ish

    # Example 3: Contradiction (hypothesis in opposite direction)
    premise_con = torch.tensor([0.5, 0.2, 0.0, 0.0, 0.0], dtype=torch.float32)
    hypothesis_con = torch.tensor([-0.3, -0.1, 0.0, 0.0, 0.0], dtype=torch.float32)  # Opposite

    # Ensure all points are in Poincaré ball
    for tensor in [premise_ent, hypothesis_ent, premise_neu, hypothesis_neu, premise_con, hypothesis_con]:
        norm = torch.norm(tensor)
        if norm >= 1.0:
            tensor.data = tensor / norm * 0.8

    # Create output directory
    os.makedirs("plots/cone_visualizations", exist_ok=True)

    visualizer.visualise_2d_cone(
        premise_ent, hypothesis_ent, 'entailment',
        "plots/cone_visualizations/entailment_cone.png"
    )

    visualizer.visualise_2d_cone(
        premise_neu, hypothesis_neu, 'neutral',
        "plots/cone_visualizations/neutral_cone.png"
    )

    visualizer.visualise_2d_cone(
        premise_con, hypothesis_con, 'contradiction',
        "plots/cone_visualizations/contradiction_cone.png"
    )

def visualize_real_data_cones():
    """
    Create visualizations using real data - using energy-based cone membership
    """
    try:
        from order_embeddings import EntailmentDataset
        from torch.utils.data import DataLoader

        processed_data_path = "data/processed/toy_embeddings_large.pt"
        model_path = "models/order_embeddings_large.pt"

        print("\nCreating Energy-Based Real Data Cone Visualizations")
        print("=" * 60)

        # Load pipeline and data
        pipeline = HyperbolicConeEmbeddingPipeline(model_path)
        visualizer = ConeVisualiser(pipeline.cone_computer)

        processed_data = torch.load(processed_data_path)
        dataset = EntailmentDataset(processed_data)

        # Use the existing smart selection logic from the previous version
        # [Include the candidate selection code from before]
        candidates_by_label = {'entailment': [], 'neutral': [], 'contradiction': []}

        print("Analyzing examples for energy-based visualization...")
        for i, item in enumerate(dataset):
            label_str = item['label_str']
            premise_emb = item['premise_emb'].to(pipeline.hyperbolic_pipeline.device)
            hypothesis_emb = item['hypothesis_emb'].to(pipeline.hyperbolic_pipeline.device)

            results = pipeline.hyperbolic_pipeline.compute_hyperbolic_energies(
                premise_emb.unsqueeze(0), hypothesis_emb.unsqueeze(0)
            )

            premise_hyp = results['premise_hyperbolic'][0]
            hypothesis_hyp = results['hypothesis_hyperbolic'][0]

            cone_energy = pipeline.cone_computer.cone_membership_energy(
                premise_hyp.unsqueeze(0), hypothesis_hyp.unsqueeze(0)
            ).item()

            candidates_by_label[label_str].append({
                'premise_hyp': premise_hyp,
                'hypothesis_hyp': hypothesis_hyp,
                'cone_energy': cone_energy,
                'index': i
            })

        # Select best examples based on energy
        selected_examples = {}
        for label in ['entailment', 'neutral', 'contradiction']:
            if candidates_by_label[label]:
                candidates = candidates_by_label[label]

                if label == 'entailment':
                    # Pick lowest energy (clearest entailment)
                    best = min(candidates, key=lambda x: x['cone_energy'])
                elif label == 'contradiction':
                    # Pick highest energy (clearest contradiction)
                    best = max(candidates, key=lambda x: x['cone_energy'])
                else:  # neutral
                    # Pick median energy
                    sorted_candidates = sorted(candidates, key=lambda x: x['cone_energy'])
                    best = sorted_candidates[len(sorted_candidates) // 2]

                selected_examples[label] = (best['premise_hyp'], best['hypothesis_hyp'])
                print(f"Selected {label}: energy = {best['cone_energy']:.3f}")

        if len(selected_examples) >= 2:
            visualizer.create_energy_based_comparison_plots(
                selected_examples,
                "plots/cone_visualizations/energy_based_real_data_comparison.png"
            )
            print("Energy-based real data visualizations completed!")
        else:
            print("Could not find sufficient examples")

    except Exception as e:
        print(f"Error creating energy-based visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_cone_visualisations()
    visualize_real_data_cones()


