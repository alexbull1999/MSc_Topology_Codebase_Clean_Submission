"""
Enhanced Validation of entailment cone implementation with asymmetric features against success criteria:
1. Cone Violation Hierarchy: entailment < neutral < contradiction energies
2. Correlation Validation: Cone energies align with order violation energies
3. Geometric Consistency: Proper cone properties maintained in hyperbolic space
4. Theoretical Alignment: Results match Ganea et al.'s expected performance gains
5. NEW: Asymmetric Feature Analysis: Enhanced directional relationship modeling
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
from order_embeddings_asymmetry import EntailmentDataset
from torch.utils.data import DataLoader

# Updated imports for enhanced pipeline
from entailment_cones_asymmetry import HyperbolicEntailmentCones, EnhancedHyperbolicConeEmbeddingPipeline
from hyperbolic_projection_asymmetry import safe_tensor_to_float

class EnhancedConeValidationFramework:
    def __init__(self, cone_pipeline):
        self.cone_pipeline = cone_pipeline
        self.validation_results = {}
        self.individual_data = {
            'cone_violations': [],
            'enhanced_features': [],
            'labels': [],
            'premise_embeddings': [],
            'hypothesis_embeddings': [],
            'sample_metadata': [],
            'premise_texts': [],
            'hypothesis_texts': []
        }

    def validate_enhanced_energy_hierarchy(self) -> Dict[str, float]:
        """
        Validate that enhanced cone energies follow expected hierarchy with asymmetric analysis
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

        print(f"Testing enhanced pipeline on real data: {len(premise_embs)} examples")

        # Get enhanced cone energies (includes all asymmetric features)
        energies = self.cone_pipeline.compute_enhanced_cone_energies(premise_embs, hypothesis_embs)

        # Collect individual data for enhanced TDA analysis
        individual_violations = []
        enhanced_features = []
        individual_labels = []
        sample_metadata = []

        # Analyze results by label AND collect individual enhanced data
        results = {}
        stats_by_label = {}

        for i, label_str in enumerate(label_strs):
            if label_str not in stats_by_label:
                stats_by_label[label_str] = {
                    # Standard features
                    'cone_energies': [],
                    'order_energies': [],
                    'hyperbolic_distances': [],
                    # Enhanced asymmetric features
                    'forward_cone_energies': [],
                    'backward_cone_energies': [],
                    'cone_asymmetries': [],
                    'forward_energies': [],
                    'backward_energies': [],
                    'asymmetric_energies': [],
                    'asymmetry_measures': []
                }

            # Extract all enhanced features with safe conversion
            cone_energy = safe_tensor_to_float(energies['cone_energies'][i])
            order_energy = safe_tensor_to_float(energies['order_energies'][i])
            hyp_distance = safe_tensor_to_float(energies['hyperbolic_distances'][i])
            
            # Enhanced asymmetric features
            forward_cone = safe_tensor_to_float(energies['forward_cone_energies'][i])
            backward_cone = safe_tensor_to_float(energies['backward_cone_energies'][i])
            cone_asymmetry = safe_tensor_to_float(energies['cone_asymmetries'][i])
            forward_energy = safe_tensor_to_float(energies['forward_energies'][i])
            backward_energy = safe_tensor_to_float(energies['backward_energies'][i])
            asymmetric_energy = safe_tensor_to_float(energies['asymmetric_energies'][i])
            asymmetry_measure = safe_tensor_to_float(energies['asymmetry_measures'][i])

            # Collect for summary statistics (enhanced functionality)
            stats_by_label[label_str]['cone_energies'].append(cone_energy)
            stats_by_label[label_str]['order_energies'].append(order_energy)
            stats_by_label[label_str]['hyperbolic_distances'].append(hyp_distance)
            stats_by_label[label_str]['forward_cone_energies'].append(forward_cone)
            stats_by_label[label_str]['backward_cone_energies'].append(backward_cone)
            stats_by_label[label_str]['cone_asymmetries'].append(cone_asymmetry)
            stats_by_label[label_str]['forward_energies'].append(forward_energy)
            stats_by_label[label_str]['backward_energies'].append(backward_energy)
            stats_by_label[label_str]['asymmetric_energies'].append(asymmetric_energy)
            stats_by_label[label_str]['asymmetry_measures'].append(asymmetry_measure)

            # ENHANCED: Collect comprehensive feature vector for TDA
            # Original 3D violation vector
            violation_vector = [cone_energy, order_energy, hyp_distance]
            
            # Enhanced 10D feature vector for improved TDA analysis
            enhanced_feature_vector = [
                cone_energy, order_energy, hyp_distance,  # Original features
                forward_cone, backward_cone, cone_asymmetry,  # Cone directional features
                forward_energy, backward_energy, asymmetric_energy, asymmetry_measure  # Order directional features
            ]
            
            individual_violations.append(violation_vector)
            enhanced_features.append(enhanced_feature_vector)
            individual_labels.append(label_str)
            
            sample_metadata.append({
                'sample_id': i,
                'label': label_str,
                'label_numeric': labels[i].item(),
                # Standard features
                'cone_energy': cone_energy,
                'order_energy': order_energy,
                'hyperbolic_distance': hyp_distance,
                # Enhanced asymmetric features
                'forward_cone_energy': forward_cone,
                'backward_cone_energy': backward_cone,
                'cone_asymmetry': cone_asymmetry,
                'forward_energy': forward_energy,
                'backward_energy': backward_energy,
                'asymmetric_energy': asymmetric_energy,
                'asymmetry_measure': asymmetry_measure,
                # Text data
                'premise_text': premise_texts[i],
                'hypothesis_text': hypothesis_texts[i]
            })

        # Store enhanced individual data for TDA
        self.individual_data['cone_violations'] = np.array(individual_violations)
        self.individual_data['enhanced_features'] = np.array(enhanced_features)
        self.individual_data['labels'] = individual_labels
        self.individual_data['premise_embeddings'] = premise_embs.detach().cpu().numpy()
        self.individual_data['hypothesis_embeddings'] = hypothesis_embs.detach().cpu().numpy()
        self.individual_data['sample_metadata'] = sample_metadata
        self.individual_data['premise_texts'] = premise_texts
        self.individual_data['hypothesis_texts'] = hypothesis_texts

        # Compute enhanced statistics for each label
        for label, stats in stats_by_label.items():
            # Standard statistics
            results[f'{label}_cone_energy_mean'] = np.mean(stats['cone_energies'])
            results[f'{label}_cone_energy_std'] = np.std(stats['cone_energies'])
            results[f'{label}_order_energy_mean'] = np.mean(stats['order_energies'])
            results[f'{label}_order_energy_std'] = np.std(stats['order_energies'])
            results[f'{label}_hyperbolic_distance_mean'] = np.mean(stats['hyperbolic_distances'])
            
            # Enhanced asymmetric statistics
            results[f'{label}_forward_cone_mean'] = np.mean(stats['forward_cone_energies'])
            results[f'{label}_backward_cone_mean'] = np.mean(stats['backward_cone_energies'])
            results[f'{label}_cone_asymmetry_mean'] = np.mean(stats['cone_asymmetries'])
            results[f'{label}_forward_energy_mean'] = np.mean(stats['forward_energies'])
            results[f'{label}_backward_energy_mean'] = np.mean(stats['backward_energies'])
            results[f'{label}_asymmetric_energy_mean'] = np.mean(stats['asymmetric_energies'])
            results[f'{label}_asymmetry_measure_mean'] = np.mean(stats['asymmetry_measures'])

            print(f"\n{label.capitalize()} pairs (Enhanced Analysis):")
            print(f"  Standard Features:")
            print(f"    Cone energy: {results[f'{label}_cone_energy_mean']:.4f} ± {results[f'{label}_cone_energy_std']:.4f}")
            print(f"    Order energy: {results[f'{label}_order_energy_mean']:.4f} ± {results[f'{label}_order_energy_std']:.4f}")
            print(f"    Hyperbolic distance: {results[f'{label}_hyperbolic_distance_mean']:.4f}")
            print(f"  Enhanced Asymmetric Features:")
            print(f"    Forward cone: {results[f'{label}_forward_cone_mean']:.4f}")
            print(f"    Backward cone: {results[f'{label}_backward_cone_mean']:.4f}")
            print(f"    Cone asymmetry: {results[f'{label}_cone_asymmetry_mean']:.4f}")
            print(f"    Forward energy: {results[f'{label}_forward_energy_mean']:.4f}")
            print(f"    Backward energy: {results[f'{label}_backward_energy_mean']:.4f}")
            print(f"    Asymmetric energy: {results[f'{label}_asymmetric_energy_mean']:.4f}")
            print(f"    Asymmetry measure: {results[f'{label}_asymmetry_measure_mean']:.4f}")

        # Validate standard hierarchy
        hierarchy_valid = False
        if all(key in results for key in
               ['entailment_cone_energy_mean', 'neutral_cone_energy_mean', 'contradiction_cone_energy_mean']):
            ent_energy = results['entailment_cone_energy_mean']
            neut_energy = results['neutral_cone_energy_mean']
            cont_energy = results['contradiction_cone_energy_mean']

            hierarchy_valid = ent_energy < neut_energy < cont_energy
            results['hierarchy_valid'] = hierarchy_valid

            print(f"\nStandard Energy Hierarchy Validation:")
            print(f"Entailment: {ent_energy:.4f}")
            print(f"Neutral: {neut_energy:.4f}")
            print(f"Contradiction: {cont_energy:.4f}")
            if hierarchy_valid:
                print(f"Standard hierarchy is valid ({ent_energy:.3f} < {neut_energy:.3f} < {cont_energy:.3f})")
            else:
                print("Standard hierarchy invalid")

        # NEW: Validate asymmetric patterns
        asymmetric_patterns_valid = self._validate_asymmetric_patterns(results)
        results['asymmetric_patterns_valid'] = asymmetric_patterns_valid

        # Add enhanced individual data to results for TDA
        results['individual_cone_violations'] = self.individual_data['cone_violations']
        results['enhanced_features'] = self.individual_data['enhanced_features']
        results['individual_labels'] = self.individual_data['labels']
        results['sample_metadata'] = self.individual_data['sample_metadata']
        results['premise_texts'] = self.individual_data['premise_texts']
        results['hypothesis_texts'] = self.individual_data['hypothesis_texts']

        self.validation_results['enhanced_energy_hierarchy'] = results
        return results

    def _validate_asymmetric_patterns(self, results: Dict) -> bool:
        """Validate that asymmetric patterns match theoretical expectations"""
        try:
            # Expected patterns:
            # Entailment: High asymmetry (forward << backward)
            # Neutral: Low asymmetry (forward ≈ backward)
            # Contradiction: High asymmetry (forward >> backward)
            
            ent_asymmetry = results.get('entailment_asymmetry_measure_mean', 0)
            neu_asymmetry = results.get('neutral_asymmetry_measure_mean', 0)
            con_asymmetry = results.get('contradiction_asymmetry_measure_mean', 0)
            
            # Entailment and contradiction should have higher asymmetry than neutral
            asymmetry_pattern_valid = (ent_asymmetry > neu_asymmetry and con_asymmetry > neu_asymmetry)
            
            print(f"\nAsymmetric Pattern Validation:")
            print(f"Entailment asymmetry: {ent_asymmetry:.4f}")
            print(f"Neutral asymmetry: {neu_asymmetry:.4f}")
            print(f"Contradiction asymmetry: {con_asymmetry:.4f}")
            print(f"Asymmetric patterns valid: {'YES' if asymmetry_pattern_valid else 'NO'}")
            
            return asymmetry_pattern_valid
        except Exception as e:
            print(f"Error validating asymmetric patterns: {e}")
            return False

    def validate_enhanced_correlation_with_order_energies(self):
        """
        Validate enhanced cone energies correlate with order violation energies
        """
        print("\nValidating enhanced correlation with order energies")

        processed_data_path = "data/processed/snli_10k_subset_balanced.pt"
        processed_data = torch.load(processed_data_path)
        dataset_obj = EntailmentDataset(processed_data)
        dataloader = DataLoader(dataset_obj, batch_size=len(dataset_obj), shuffle=False)

        batch = next(iter(dataloader))
        premise_embs = batch['premise_emb'].to(self.cone_pipeline.hyperbolic_pipeline.device)
        hypothesis_embs = batch['hypothesis_emb'].to(self.cone_pipeline.hyperbolic_pipeline.device)

        # Get enhanced energies for all pairs
        energies = self.cone_pipeline.compute_enhanced_cone_energies(premise_embs, hypothesis_embs)

        # Convert to numpy for correlation analysis
        cone_energies = [safe_tensor_to_float(e) for e in energies['cone_energies']]
        order_energies = [safe_tensor_to_float(e) for e in energies['order_energies']]
        forward_cone_energies = [safe_tensor_to_float(e) for e in energies['forward_cone_energies']]
        backward_cone_energies = [safe_tensor_to_float(e) for e in energies['backward_cone_energies']]
        forward_order_energies = [safe_tensor_to_float(e) for e in energies['forward_energies']]
        backward_order_energies = [safe_tensor_to_float(e) for e in energies['backward_energies']]

        # Compute enhanced correlations
        # Standard correlations
        pearson_corr, pearson_p = pearsonr(cone_energies, order_energies)
        spearman_corr, spearman_p = spearmanr(cone_energies, order_energies)
        
        # Enhanced directional correlations
        forward_pearson, forward_p = pearsonr(forward_cone_energies, forward_order_energies)
        backward_pearson, backward_p = pearsonr(backward_cone_energies, backward_order_energies)

        results = {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'forward_pearson_correlation': forward_pearson,
            'forward_p_value': forward_p,
            'backward_pearson_correlation': backward_pearson,
            'backward_p_value': backward_p,
            'n_pairs': len(cone_energies)
        }

        print(f"Enhanced Correlation Results (n={results['n_pairs']}):")
        print(f"Standard:")
        print(f"  Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4f})")
        print(f"  Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")
        print(f"Enhanced Directional:")
        print(f"  Forward correlation: {forward_pearson:.4f} (p={forward_p:.4f})")
        print(f"  Backward correlation: {backward_pearson:.4f} (p={backward_p:.4f})")

        # Check if correlations are significant and positive
        standard_correlation_valid = pearson_corr > 0.3 and pearson_p < 0.05
        enhanced_correlation_valid = (forward_pearson > 0.2 and forward_p < 0.05 and 
                                    backward_pearson > 0.2 and backward_p < 0.05)
        
        results['standard_correlation_valid'] = standard_correlation_valid
        results['enhanced_correlation_valid'] = enhanced_correlation_valid
        results['correlation_valid'] = standard_correlation_valid and enhanced_correlation_valid
        
        if results['correlation_valid']:
            print("Enhanced correlation validation: PASSED")
        else:
            print("Enhanced correlation validation: FAILED")

        self.validation_results['enhanced_correlation'] = results
        return results

    def validate_geometric_properties(self) -> Dict[str, bool]:
        """
        Validate geometric properties of hyperbolic cones (unchanged from original)
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
        general = torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0])  # Close to origin (general)
        specific = torch.tensor([0.4, 0.1, 0.0, 0.0, 0.0])  # Further (specific)
        very_specific = torch.tensor([0.6, 0.15, 0.05, 0.0, 0.0])  # Even further (very specific)

        # Check if the hierarchy is preserved in cone energies
        energy_gen_spec = cone_computer.cone_membership_energy(general.unsqueeze(0), specific.unsqueeze(0))
        energy_spec_vspec = cone_computer.cone_membership_energy(specific.unsqueeze(0), very_specific.unsqueeze(0))
        energy_gen_vspec = cone_computer.cone_membership_energy(general.unsqueeze(0), very_specific.unsqueeze(0))

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

    def generate_enhanced_validation_report(self) -> Dict[str, any]:
        """
        Generate a comprehensive enhanced validation report with asymmetric features
        """
        print("\n" + "=" * 90)
        print("ENHANCED VALIDATION REPORT SUMMARY (with Asymmetric Features)")
        print("=" * 90)

        all_tests_passed = True
        summary = {}

        if 'enhanced_energy_hierarchy' in self.validation_results:
            hierarchy_results = self.validation_results['enhanced_energy_hierarchy']
            hierarchy_passed = hierarchy_results.get('hierarchy_valid', False)
            asymmetric_patterns_passed = hierarchy_results.get('asymmetric_patterns_valid', False)
            
            summary['energy_hierarchy_passed'] = hierarchy_passed
            summary['asymmetric_patterns_passed'] = asymmetric_patterns_passed
            
            enhanced_hierarchy_passed = hierarchy_passed and asymmetric_patterns_passed
            all_tests_passed &= enhanced_hierarchy_passed
            
            print(f"1. Standard Energy Hierarchy Test: {'PASS' if hierarchy_passed else 'FAIL'}")
            print(f"2. Asymmetric Patterns Test: {'PASS' if asymmetric_patterns_passed else 'FAIL'}")

        if 'enhanced_correlation' in self.validation_results:
            corr_results = self.validation_results['enhanced_correlation']
            correlation_passed = corr_results.get('correlation_valid', False)
            summary['enhanced_correlation_passed'] = correlation_passed
            all_tests_passed &= correlation_passed
            print(f"3. Enhanced Correlation Test: {'PASS' if correlation_passed else 'FAIL'}")

        if 'geometric_properties' in self.validation_results:
            geom_results = self.validation_results['geometric_properties']
            aperture_passed = geom_results.get('aperture_bounds_valid', False)
            asymmetry_passed = geom_results.get('asymmetry_valid', False)
            geometry_passed = aperture_passed and asymmetry_passed
            summary['geometric_properties_passed'] = geometry_passed
            all_tests_passed &= geometry_passed
            print(f"4. Geometric Properties Test: {'PASS' if geometry_passed else 'FAIL'}")

        # Enhanced TDA Data Collection Summary
        if self.individual_data['enhanced_features'].size > 0:
            n_samples = len(self.individual_data['labels'])
            n_standard_features = self.individual_data['cone_violations'].shape[1]
            n_enhanced_features = self.individual_data['enhanced_features'].shape[1]
            label_counts = {}
            for label in self.individual_data['labels']:
                label_counts[label] = label_counts.get(label, 0) + 1

            texts_preserved = (
                len(self.individual_data['premise_texts']) == n_samples and
                len(self.individual_data['hypothesis_texts']) == n_samples
            )

            print(f"5. Enhanced TDA Data Collection: SUCCESS")
            print(f"   - Standard features: {n_samples} samples × {n_standard_features} features")
            print(f"   - Enhanced features: {n_samples} samples × {n_enhanced_features} features")
            print(f"   - Label distribution: {label_counts}")
            print(f"   - Text preservation: {'SUCCESS' if texts_preserved else 'FAILED'}")

            summary['enhanced_tda_data_collected'] = True
            summary['tda_samples'] = n_samples
            summary['standard_features'] = n_standard_features
            summary['enhanced_features'] = n_enhanced_features
            summary['tda_label_distribution'] = label_counts
            summary['texts_preserved'] = texts_preserved
        else:
            print(f"5. Enhanced TDA Data Collection: FAILED")
            summary['enhanced_tda_data_collected'] = False

        summary['all_tests_passed'] = all_tests_passed
        summary['validation_results'] = self.validation_results

        # Enhanced individual data for TDA
        summary['individual_cone_violations'] = self.individual_data['cone_violations']
        summary['enhanced_features'] = self.individual_data['enhanced_features']
        summary['individual_labels'] = self.individual_data['labels']
        summary['cone_violations'] = torch.tensor(self.individual_data['cone_violations'])
        summary['enhanced_features_tensor'] = torch.tensor(self.individual_data['enhanced_features'])
        summary['labels'] = self.individual_data['labels']
        summary['premise_texts'] = self.individual_data['premise_texts']
        summary['hypothesis_texts'] = self.individual_data['hypothesis_texts']
        summary['sample_metadata'] = self.individual_data['sample_metadata']

        # Enhanced energy hierarchy for TDA compatibility
        if 'enhanced_energy_hierarchy' in self.validation_results:
            hierarchy_results = self.validation_results['enhanced_energy_hierarchy']
            summary['energy_hierarchy'] = {
                'entailment_mean': hierarchy_results.get('entailment_cone_energy_mean', 0),
                'neutral_mean': hierarchy_results.get('neutral_cone_energy_mean', 0),
                'contradiction_mean': hierarchy_results.get('contradiction_cone_energy_mean', 0)
            }
            
            # Enhanced asymmetric hierarchy
            summary['enhanced_energy_hierarchy'] = {
                'entailment_asymmetry': hierarchy_results.get('entailment_asymmetry_measure_mean', 0),
                'neutral_asymmetry': hierarchy_results.get('neutral_asymmetry_measure_mean', 0),
                'contradiction_asymmetry': hierarchy_results.get('contradiction_asymmetry_measure_mean', 0),
                'entailment_forward': hierarchy_results.get('entailment_forward_energy_mean', 0),
                'entailment_backward': hierarchy_results.get('entailment_backward_energy_mean', 0),
                'neutral_forward': hierarchy_results.get('neutral_forward_energy_mean', 0),
                'neutral_backward': hierarchy_results.get('neutral_backward_energy_mean', 0),
                'contradiction_forward': hierarchy_results.get('contradiction_forward_energy_mean', 0),
                'contradiction_backward': hierarchy_results.get('contradiction_backward_energy_mean', 0)
            }

        print(f"\n OVERALL ENHANCED VALIDATION: {'PASS' if all_tests_passed else 'FAIL'}")
        print(f" ENHANCED TDA READINESS: {'READY' if summary.get('enhanced_tda_data_collected', False) else 'NOT READY'}")
        print(f" FEATURE ENHANCEMENT: {summary.get('enhanced_features', 0)} features (vs {summary.get('standard_features', 0)} standard)")
        print("=" * 90)

        return summary


def main():
    set_random_seed(42)
    print("Enhanced Hyperbolic Entailment Cones Validation")
    print("="*70)
    
    try:
        # Use enhanced pipeline
        pipeline = EnhancedHyperbolicConeEmbeddingPipeline()
        validator = EnhancedConeValidationFramework(pipeline)
    except Exception as e:
        print(f"Failed to create enhanced pipeline: {e}")
        print("Make sure you have the enhanced order embeddings model trained!")
        return None

    # Run all enhanced validations
    print("\n1. Running Enhanced Energy Hierarchy Validation...")
    hierarchy_results = validator.validate_enhanced_energy_hierarchy()

    print("\n2. Running Enhanced Correlation Validation...")
    correlation_results = validator.validate_enhanced_correlation_with_order_energies()

    print("\n3. Running Geometric Properties Validation...")
    geometric_results = validator.validate_geometric_properties()

    # Generate enhanced final report
    final_report = validator.generate_enhanced_validation_report()

    # Save enhanced results
    results_dir = Path("validation_results")
    results_dir.mkdir(exist_ok=True)

    torch.save(final_report, results_dir / "enhanced_cone_validation_results_snli_10k.pt")

    # Enhanced TDA data with asymmetric features
    enhanced_tda_data = {
        'cone_violations': final_report['cone_violations'],
        'enhanced_features': final_report['enhanced_features_tensor'],
        'labels': final_report['labels'],
        'energy_hierarchy': final_report['energy_hierarchy'],
        'enhanced_energy_hierarchy': final_report.get('enhanced_energy_hierarchy', {}),
        'validation_passed': final_report['all_tests_passed'],
        'premise_texts': final_report['premise_texts'],
        'hypothesis_texts': final_report['hypothesis_texts'],
        'sample_metadata': final_report['sample_metadata'],
        'feature_names': [
            'cone_energy', 'order_energy', 'hyperbolic_distance',  # Standard features (0-2)
            'forward_cone', 'backward_cone', 'cone_asymmetry',     # Cone directional features (3-5)
            'forward_energy', 'backward_energy', 'asymmetric_energy', 'asymmetry_measure'  # Order directional features (6-9)
        ],
        'n_standard_features': 3,
        'n_enhanced_features': 10,
        'asymmetric_patterns_validated': final_report.get('asymmetric_patterns_passed', False)
    }
    
    torch.save(enhanced_tda_data, results_dir / "enhanced_tda_ready_data_snli_10k.pt")

    print(f"\nEnhanced results saved to:")
    print(f"   {results_dir / 'enhanced_cone_validation_results_snli_10k_asymmetry.pt'} (comprehensive validation)")
    print(f"   {results_dir / 'enhanced_tda_ready_data_snli_10k_asymmetry.pt'} (enhanced TDA-ready data)")
    
    print(f"\nEnhanced Feature Summary:")
    print(f"   Standard features: 3 (cone_energy, order_energy, hyperbolic_distance)")
    print(f"   Enhanced features: 10 (includes 7 asymmetric measurements)")
    print(f"   Expected TDA improvement: Significant due to richer feature space")
    
    return final_report


if __name__ == "__main__":
    results = main()