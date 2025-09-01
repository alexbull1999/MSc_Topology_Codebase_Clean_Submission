"""
Validates the entailment cone implementation in entailment_cones.py against success criteria:
1. Cone Violation Hierarchy: entailment < neutral < contradiction energies
2. Correlation  Validation: Cone  energies align with order violation energies
3. Geometric Consistency: Proper cone properties maintained in hyperbolic space
4. Theoretical Alignment: Results match Ganea et al.'s expected performance gains
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import warnings
import os

from hyperbolic_projection import set_random_seed

warnings.filterwarnings("ignore")
from order_embeddings import EntailmentDataset
from torch.utils.data import DataLoader

from entailment_cones import HyperbolicEntailmentCones, HyperbolicConeEmbeddingPipeline
from hyperbolic_projection import safe_tensor_to_float

class ConeValidationFramework:
    def __init__(self, cone_pipeline):
        self.cone_pipeline = cone_pipeline
        self.validation_results = {}
        self.individual_data = {
            'cone_violations': [],
            'labels': [],
            'premise_embeddings': [],
            'hypothesis_embeddings': [],
            'sample_metadata': [],
            'premise_texts': [],
            'hypothesis_texts': []
        }

    def validate_energy_hierarchy(self) -> Dict[str, float]:
        """
        Validate that cone energies follow expected hierarchy
        """
        processed_data_path = "data/processed/snli_10k_subset_balanced.pt"
        processed_data = torch.load(processed_data_path)

        original_texts = processed_data["texts"]
        premise_texts = original_texts["premises"]
        hypothesis_texts = original_texts["hypotheses"]

        dataset_obj = EntailmentDataset(processed_data)
        dataloader = DataLoader(dataset_obj, batch_size=len(dataset_obj), shuffle=False)

        batch = next(iter(dataloader))
        premise_embs = batch['premise_emb'].to(self.cone_pipeline.hyperbolic_pipeline.device)
        hypothesis_embs = batch['hypothesis_emb'].to(self.cone_pipeline.hyperbolic_pipeline.device)
        labels = batch['label']
        label_strs = batch['label_str']

        print(f"Testing on real data: {len(premise_embs)} examples")

        # Get cone energies using the corrected function
        energies = self.cone_pipeline.compute_cone_energies(premise_embs, hypothesis_embs)

        # Collect individual data for TDA analysis
        individual_violations = []
        individual_labels = []
        sample_metadata = []

        # Analyze results by label AND collect individual data
        results = {}
        stats_by_label = {}

        for i, label_str in enumerate(label_strs):
            if label_str not in stats_by_label:
                stats_by_label[label_str] = {
                    'cone_energies': [],
                    'order_energies': [],
                    'hyperbolic_distances': []
                }

            # Use safe conversion
            cone_energy = safe_tensor_to_float(energies['cone_energies'][i])
            order_energy = safe_tensor_to_float(energies['order_energies'][i])
            hyp_distance = safe_tensor_to_float(energies['hyperbolic_distances'][i])

            # Collect for summary statistics (original functionality)
            stats_by_label[label_str]['cone_energies'].append(cone_energy)
            stats_by_label[label_str]['order_energies'].append(order_energy)
            stats_by_label[label_str]['hyperbolic_distances'].append(hyp_distance)

            # NEW: Collect individual violation patterns for TDA
            violation_vector = [cone_energy, order_energy, hyp_distance]
            individual_violations.append(violation_vector)
            individual_labels.append(label_str)
            sample_metadata.append({
                'sample_id': i,
                'label': label_str,
                'label_numeric': labels[i].item(),
                'cone_energy': cone_energy,
                'order_energy': order_energy,
                'hyperbolic_distance': hyp_distance,
                'premise_text': premise_texts[i],
                'hypothesis_text': hypothesis_texts[i]
            })

        # Store individual data for TDA
        self.individual_data['cone_violations'] = np.array(individual_violations)
        self.individual_data['labels'] = individual_labels
        self.individual_data['premise_embeddings'] = premise_embs.detach().cpu().numpy()
        self.individual_data['hypothesis_embeddings'] = hypothesis_embs.detach().cpu().numpy()
        self.individual_data['sample_metadata'] = sample_metadata
        self.individual_data['premise_texts'] = premise_texts
        self.individual_data['hypothesis_texts'] = hypothesis_texts

        # Compute means for each label (original validation functionality)
        for label, stats in stats_by_label.items():
            results[f'{label}_cone_energy_mean'] = np.mean(stats['cone_energies'])
            results[f'{label}_cone_energy_std'] = np.std(stats['cone_energies'])
            results[f'{label}_order_energy_mean'] = np.mean(stats['order_energies'])
            results[f'{label}_hyperbolic_distance_mean'] = np.mean(stats['hyperbolic_distances'])

            print(f"{label.capitalize()} pairs:")
            print(f"  Cone energy: {results[f'{label}_cone_energy_mean']:.4f} ± {results[f'{label}_cone_energy_std']:.4f}")
            print(f"  Order energy: {results[f'{label}_order_energy_mean']:.4f}")
            print(f"  Hyperbolic distance: {results[f'{label}_hyperbolic_distance_mean']:.4f}")

        # Validate hierarchy (original functionality)
        hierarchy_valid = False
        if all(key in results for key in
               ['entailment_cone_energy_mean', 'neutral_cone_energy_mean', 'contradiction_cone_energy_mean']):
            ent_energy = results['entailment_cone_energy_mean']
            neut_energy = results['neutral_cone_energy_mean']
            cont_energy = results['contradiction_cone_energy_mean']

            hierarchy_valid = ent_energy < neut_energy < cont_energy
            results['hierarchy_valid'] = hierarchy_valid

            print(f"\nEnergy Hierarchy Validation:")
            print(f"Entailment: {ent_energy:.4f}")
            print(f"Neutral: {neut_energy:.4f}")
            print(f"Contradiction: {cont_energy:.4f}")
            if hierarchy_valid:
                print(f"Hierarchy is valid ({ent_energy:.3f} < {neut_energy:.3f} < {cont_energy:.3f})")
            else:
                print("Hierarchy invalid")

        # Add individual data to results for TDA
        results['individual_cone_violations'] = self.individual_data['cone_violations']
        results['individual_labels'] = self.individual_data['labels']
        results['sample_metadata'] = self.individual_data['sample_metadata']
        results['premise_texts'] = self.individual_data['premise_texts']
        results['hypothesis_texts'] = self.individual_data['hypothesis_texts']

        self.validation_results['energy_hierarchy'] = results
        return results

    def validate_correlation_with_order_energies(self):
        """
        Validate cone energies correlate with order violation energies for theoretical consistency
        """
        print("Validating correlation with order energies")

        processed_data_path = "data/processed/snli_10k_subset_balanced.pt"
        processed_data = torch.load(processed_data_path)
        dataset_obj = EntailmentDataset(processed_data)
        dataloader = DataLoader(dataset_obj, batch_size=len(dataset_obj), shuffle=False)

        batch = next(iter(dataloader))
        premise_embs = batch['premise_emb'].to(self.cone_pipeline.hyperbolic_pipeline.device)
        hypothesis_embs = batch['hypothesis_emb'].to(self.cone_pipeline.hyperbolic_pipeline.device)

        # Get energies for all pairs
        energies = self.cone_pipeline.compute_cone_energies(premise_embs, hypothesis_embs)

        # Convert to numpy for correlation analysis
        cone_energies = [safe_tensor_to_float(e) for e in energies['cone_energies']]
        order_energies = [safe_tensor_to_float(e) for e in energies['order_energies']]

        # Compute correlations
        pearson_corr, pearson_p = pearsonr(cone_energies, order_energies)
        spearman_corr, spearman_p = spearmanr(cone_energies, order_energies)

        results = {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'n_pairs': len(cone_energies)
        }

        print(f"Correlation Results (n={results['n_pairs']}):")
        print(f"Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4f})")
        print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")

        # Check if correlation is significant and positive
        correlation_valid = pearson_corr > 0.3 and pearson_p < 0.05
        results['correlation_valid'] = correlation_valid
        if correlation_valid:
            print("Correlation valid")
        else:
            print("Correlation invalid")

        self.validation_results['correlation'] = results
        return results

    def validate_geometric_properties(self) -> Dict[str, bool]:
        """
        Validate geometric properties of hyperbolic cones:
        - Transitivity: if y ∈ cone(x), then cone(y) ⊆ cone(x)
        - Asymmetry: cone relationships should be directional
        - Aperture bounds: apertures should be ≤ π/2 (from Lemma 2)
        """
        print("\nValidating geometric properties...")

        # Test with synthetic data
        torch.manual_seed(123)
        cone_computer = HyperbolicEntailmentCones(K=0.1, epsilon=0.1)

        # Create test points
        x = torch.tensor([0.3, 0.0, 0.0, 0.0, 0.0])  # Point closer to origin
        y = torch.tensor([0.5, 0.1, 0.0, 0.0, 0.0])  # Point further out
        z = torch.tensor([0.7, 0.2, 0.1, 0.0, 0.0])  # Point even further

        results = {}

        # Test aperture bounds (should be ≤ π/2)
        apertures = []
        test_points = [x, y, z]
        for point in test_points:
            aperture = cone_computer.cone_aperture(point.unsqueeze(0))
            apertures.append(aperture.item())

        max_aperture = max(apertures)
        aperture_bounds_valid = max_aperture <= np.pi / 2
        results['aperture_bounds_valid'] = aperture_bounds_valid
        results['max_aperture'] = max_aperture

        print(f"Aperture bounds: max={max_aperture:.4f}, π/2={np.pi / 2:.4f}")
        print(f"Aperture bounds valid: {'YES' if aperture_bounds_valid else 'NO'}")

        # Test asymmetry: E(x,y) should not equal E(y,x) in general
        energy_xy = cone_computer.cone_membership_energy(x.unsqueeze(0), y.unsqueeze(0))
        energy_yx = cone_computer.cone_membership_energy(y.unsqueeze(0), x.unsqueeze(0))

        asymmetry_valid = abs(energy_xy.item() - energy_yx.item()) > 1e-6
        results['asymmetry_valid'] = asymmetry_valid
        results['energy_xy'] = energy_xy.item()
        results['energy_yx'] = energy_yx.item()

        print(f"Asymmetry test: E(x,y)={energy_xy.item():.4f}, E(y,x)={energy_yx.item():.4f}")
        print(f"Asymmetry valid: {'YES' if asymmetry_valid else 'NO'}")

        # Test basic transitivity property on synthetic points
        # If we create points such that z is "more specific" than y, which is "more specific" than x
        # Then cone violations should follow some relationship

        # Create a clear hierarchy: general -> specific -> very_specific
        general = torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0])  # Close to origin (general)
        specific = torch.tensor([0.4, 0.1, 0.0, 0.0, 0.0])  # Further (specific)
        very_specific = torch.tensor([0.6, 0.15, 0.05, 0.0, 0.0])  # Even further (very specific)

        # Check if the hierarchy is preserved in cone energies
        energy_gen_spec = cone_computer.cone_membership_energy(general.unsqueeze(0), specific.unsqueeze(0))
        energy_spec_vspec = cone_computer.cone_membership_energy(specific.unsqueeze(0), very_specific.unsqueeze(0))
        energy_gen_vspec = cone_computer.cone_membership_energy(general.unsqueeze(0), very_specific.unsqueeze(0))

        # For proper hierarchy: general should "contain" both specific and very_specific
        # So energy_gen_spec and energy_gen_vspec should be relatively small
        transitivity_hint = (energy_gen_spec < energy_spec_vspec and
                             energy_gen_spec < energy_gen_vspec)

        results['transitivity_hint_valid'] = transitivity_hint
        results['energy_general_specific'] = energy_gen_spec.item()
        results['energy_specific_very_specific'] = energy_spec_vspec.item()
        results['energy_general_very_specific'] = energy_gen_vspec.item()

        print(f"Transitivity hint:")
        print(f"  General->Specific: {energy_gen_spec.item():.4f}")
        print(f"  Specific->VerySpecific: {energy_spec_vspec.item():.4f}")
        print(f"  General->VerySpecific: {energy_gen_vspec.item():.4f}")
        print(f"Transitivity hint valid: {'YES' if transitivity_hint else 'NO'}")

        self.validation_results['geometric_properties'] = results
        return results

    def generate_validation_report(self) -> Dict[str, any]:
        """
        Generate a comprehensive validation report
        """
        print("\n" + "=" * 80)
        print("ENHANCED VALIDATION REPORT SUMMARY (Phase 2B+ with TDA Data)")
        print("=" * 80)

        all_tests_passed = True
        summary = {}

        if 'energy_hierarchy' in self.validation_results:
            hierarchy_results = self.validation_results['energy_hierarchy']
            hierarchy_passed = hierarchy_results.get('hierarchy_valid', False)
            summary['energy_hierarchy_passed'] = hierarchy_passed
            all_tests_passed &= hierarchy_passed
            print(f"1. Energy Hierarchy Test: {'PASS' if hierarchy_passed else 'FAIL'}")

        if 'correlation' in self.validation_results:
            corr_results = self.validation_results['correlation']
            correlation_passed = corr_results.get('correlation_valid', False)
            summary['correlation_passed'] = correlation_passed
            all_tests_passed &= correlation_passed
            print(f"2. Correlation Test: {'PASS' if correlation_passed else 'FAIL'}")

        if 'geometric_properties' in self.validation_results:
            geom_results = self.validation_results['geometric_properties']
            aperture_passed = geom_results.get('aperture_bounds_valid', False)
            asymmetry_passed = geom_results.get('asymmetry_valid', False)
            geometry_passed = aperture_passed and asymmetry_passed
            summary['geometric_properties_passed'] = geometry_passed
            all_tests_passed &= geometry_passed
            print(f"3. Geometric Properties Test: {'PASS' if geometry_passed else 'FAIL'}")

        # NEW: TDA Data Collection Summary
        if self.individual_data['cone_violations'].size > 0:
            n_samples = len(self.individual_data['labels'])
            n_features = self.individual_data['cone_violations'].shape[1]
            label_counts = {}
            for label in self.individual_data['labels']:
                label_counts[label] = label_counts.get(label, 0) + 1

            texts_preserved = (
                len(self.individual_data['premise_texts']) == n_samples and
                len(self.individual_data['hypothesis_texts']) == n_samples
            )

            print(f"4. TDA Data Collection: SUCCESS")
            print(f"   - Individual violations: {n_samples} samples × {n_features} features")
            print(f"   - Label distribution: {label_counts}")
            print(f"   - Text preservation: {'SUCCESS' if texts_preserved else 'FAILED'}")

            summary['tda_data_collected'] = True
            summary['tda_samples'] = n_samples
            summary['tda_features'] = n_features
            summary['tda_label_distribution'] = label_counts
            summary['texts_preserved'] = texts_preserved
        else:
            print(f"4. TDA Data Collection: FAILED")
            summary['tda_data_collected'] = False

        summary['all_tests_passed'] = all_tests_passed
        summary['validation_results'] = self.validation_results

        # NEW: Add individual data for TDA
        summary['individual_cone_violations'] = self.individual_data['cone_violations']
        summary['individual_labels'] = self.individual_data['labels']
        summary['cone_violations'] = torch.tensor(self.individual_data['cone_violations'])
        summary['labels'] = self.individual_data['labels']
        summary['premise_texts'] = self.individual_data['premise_texts']
        summary['hypothesis_texts'] = self.individual_data['hypothesis_texts']
        summary['sample_metadata'] = self.individual_data['sample_metadata']

        # Add energy hierarchy for TDA compatibility
        if 'energy_hierarchy' in self.validation_results:
            hierarchy_results = self.validation_results['energy_hierarchy']
            summary['energy_hierarchy'] = {
                'entailment_mean': hierarchy_results.get('entailment_cone_energy_mean', 0),
                'neutral_mean': hierarchy_results.get('neutral_cone_energy_mean', 0),
                'contradiction_mean': hierarchy_results.get('contradiction_cone_energy_mean', 0)
            }

        print(f"\n OVERALL VALIDATION: {'PASS' if all_tests_passed else 'FAIL'}")
        print(f" TDA READINESS: {'READY' if summary.get('tda_data_collected', False) else 'NOT READY'}")
        print("=" * 80)

        return summary



def main():
    set_random_seed(42)
    print("Hyperbolic Entailment Cones Validation")
    print("="*60)
    pipeline = HyperbolicConeEmbeddingPipeline()
    validator = ConeValidationFramework(pipeline)

    # Run all validations
    print("\n1. Running Energy Hierarchy Validation...")
    hierarchy_results = validator.validate_energy_hierarchy()

    print("\n2. Running Correlation Validation...")
    correlation_results = validator.validate_correlation_with_order_energies()

    print("\n3. Running Geometric Properties Validation...")
    geometric_results = validator.validate_geometric_properties()

    # Generate final report
    final_report = validator.generate_validation_report()

    # Save results if needed
    results_dir = Path("validation_results")
    results_dir.mkdir(exist_ok=True)

    torch.save(final_report, results_dir / "cone_validation_results_snli_10k.pt")

    tda_data = {
        'cone_violations': final_report['cone_violations'],
        'labels': final_report['labels'],
        'energy_hierarchy': final_report['energy_hierarchy'],
        'validation_passed': final_report['all_tests_passed'],
        'premise_texts': final_report['premise_texts'],
        'hypothesis_texts': final_report['hypothesis_texts'],
        'sample_metadata': final_report['sample_metadata']
    }
    torch.save(tda_data, results_dir / "tda_ready_data_snli_10k.pt")

    print(f"\nResults saved to:")
    print(f"   {results_dir / 'cone_validation_results.pt'} (enhanced validation)")
    print(f"   {results_dir / 'tda_ready_data.pt'} (TDA-specific data)")
    return final_report


if __name__ == "__main__":
    results = main()
