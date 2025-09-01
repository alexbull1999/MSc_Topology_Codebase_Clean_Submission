import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any


class SafeAnomalyExtractor:
    """
    Safely extracts anomalous entailment pairs by matching text content
    rather than relying on index ordering
    """

    def __init__(self, data_dir: str = "validation_results"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("anomaly_analysis")
        self.output_dir.mkdir(exist_ok=True)

        # Load data
        self._load_data()

    def _load_data(self):
        """Load required data files"""
        print("Loading data files...")

        # Load TDA-ready data with cone violations
        tda_data_path = self.data_dir / "tda_ready_data_SNLI_k=0.01.pt"
        if not tda_data_path.exists():
            raise FileNotFoundError(f"TDA data not found at {tda_data_path}")

        self.tda_data = torch.load(tda_data_path, map_location='cpu')
        print(f"Loaded TDA data: {len(self.tda_data['cone_violations'])} samples")
        # Verify we have all required data
        required_keys = ['cone_violations', 'labels', 'premise_texts', 'hypothesis_texts', 'sample_metadata']
        missing_keys = [key for key in required_keys if key not in self.tda_data]

        if missing_keys:
            raise KeyError(f"Missing required keys in TDA data: {missing_keys}")

        print(f"✓ Loaded TDA data: {len(self.tda_data['labels'])} samples with texts")

        # Extract cone violation energies from sample metadata for easier access
        self.sample_metadata = self.tda_data['sample_metadata']
        print(f"✓ Sample metadata contains {len(self.sample_metadata)} detailed records")

    def find_anomalous_entailments(self,
                                   cone_energy_threshold: float = 1,
                                   order_energy_threshold: float = 1,
                                   top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Find entailment examples with anomalously high cone violation energies

        Args:
            cone_energy_threshold: Minimum cone energy to be considered anomalous
            order_energy_threshold: Optional order energy threshold
            top_k: Maximum number of examples to return

        Returns:
            List of anomalous entailment examples with texts
        """
        print(f"\nFinding anomalous entailment examples...")
        print(f"Cone energy threshold: {cone_energy_threshold}")
        if order_energy_threshold:
            print(f"Order energy threshold: {order_energy_threshold}")

        anomalous_examples = []

        for sample in self.sample_metadata:
            # Only look at entailment examples
            if sample['label'] != 'entailment':
                continue

            cone_energy = sample['cone_energy']
            order_energy = sample['order_energy']

            # Check if anomalous based on thresholds
            is_anomalous = cone_energy > cone_energy_threshold
            if order_energy_threshold is not None:
                is_anomalous = is_anomalous and (order_energy > order_energy_threshold)

            if is_anomalous:
                example = {
                    'sample_id': sample['sample_id'],
                    'premise': sample['premise_text'],
                    'hypothesis': sample['hypothesis_text'],
                    'label': sample['label'],
                    'cone_energy': cone_energy,
                    'order_energy': order_energy,
                    'hyperbolic_distance': sample['hyperbolic_distance'],
                    'anomaly_score': cone_energy / cone_energy_threshold,
                    'potential_issue': 'High cone energy for entailment - possible mislabel'
                }
                anomalous_examples.append(example)

        # Sort by cone energy (highest first)
        anomalous_examples.sort(key=lambda x: x['cone_energy'], reverse=True)

        # Limit to top_k
        anomalous_examples = anomalous_examples[:top_k]

        print(f"Found {len(anomalous_examples)} anomalous entailment examples")
        return anomalous_examples

    def find_low_energy_non_entailments(self,
                                        cone_energy_threshold: float = 1,
                                        order_energy_threshold: float = 1,
                                        top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Find neutral/contradiction examples with suspiciously low energies
        These might be mislabeled entailments

        Args:
            cone_energy_threshold: Maximum cone energy to be considered suspicious
            order_energy_threshold: Optional order energy threshold
            top_k: Maximum number of examples to return

        Returns:
            List of potentially mislabeled examples
        """
        print(f"\nFinding low-energy non-entailment examples...")
        print(f"Cone energy threshold: {cone_energy_threshold}")
        if order_energy_threshold:
            print(f"Order energy threshold: {order_energy_threshold}")

        suspicious_examples = []

        for sample in self.sample_metadata:
            # Only look at neutral/contradiction examples
            if sample['label'] == 'entailment':
                continue

            cone_energy = sample['cone_energy']
            order_energy = sample['order_energy']

            # Check if suspiciously low energy
            is_suspicious = cone_energy < cone_energy_threshold
            if order_energy_threshold is not None:
                is_suspicious = is_suspicious and (order_energy < order_energy_threshold)

            if is_suspicious:
                example = {
                    'sample_id': sample['sample_id'],
                    'premise': sample['premise_text'],
                    'hypothesis': sample['hypothesis_text'],
                    'label': sample['label'],
                    'cone_energy': cone_energy,
                    'order_energy': order_energy,
                    'hyperbolic_distance': sample['hyperbolic_distance'],
                    'suspicion_score': cone_energy_threshold / max(cone_energy, 1e-6),
                    'potential_issue': f'Low cone energy for {sample["label"]} - possible entailment mislabel'
                }
                suspicious_examples.append(example)

        # Sort by cone energy (lowest first)
        suspicious_examples.sort(key=lambda x: x['cone_energy'])

        # Limit to top_k
        suspicious_examples = suspicious_examples[:top_k]

        print(f"Found {len(suspicious_examples)} potentially mislabeled non-entailment examples")
        return suspicious_examples

    def generate_anomaly_report(self,
                                anomalous_entailments: List[Dict],
                                suspicious_non_entailments: List[Dict]) -> str:
        """
        Generate a comprehensive anomaly analysis report

        Returns:
            Path to saved report
        """
        report = {
            'analysis_summary': {
                'total_samples': len(self.sample_metadata),
                'anomalous_entailments_found': len(anomalous_entailments),
                'suspicious_non_entailments_found': len(suspicious_non_entailments),
                'potential_mislabeling_rate': (len(anomalous_entailments) + len(suspicious_non_entailments)) / len(
                    self.sample_metadata) * 100
            },
            'anomalous_entailments': anomalous_entailments,
            'suspicious_non_entailments': suspicious_non_entailments,
            'methodology': {
                'description': 'Uses cone violation energies from hyperbolic entailment cones to identify potential annotation errors',
                'high_energy_entailments': 'Entailment pairs with high cone violations may be mislabeled',
                'low_energy_non_entailments': 'Neutral/contradiction pairs with low cone violations may be true entailments'
            }
        }

        # Save comprehensive report
        report_path = self.output_dir / "anomaly_analysis_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Save human-readable summary
        summary_path = self.output_dir / "anomaly_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("SNLI ENTAILMENT ANOMALY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Dataset: {report['analysis_summary']['total_samples']} samples\n")
            f.write(f"Anomalous entailments: {report['analysis_summary']['anomalous_entailments_found']}\n")
            f.write(f"Suspicious non-entailments: {report['analysis_summary']['suspicious_non_entailments_found']}\n")
            f.write(f"Potential mislabeling rate: {report['analysis_summary']['potential_mislabeling_rate']:.1f}%\n\n")

            f.write("TOP ANOMALOUS ENTAILMENTS (High cone energy - likely mislabeled):\n")
            f.write("-" * 70 + "\n")
            for i, example in enumerate(anomalous_entailments[:5], 1):
                f.write(f"{i}. Cone Energy: {example['cone_energy']:.4f}\n")
                f.write(f"   Premise: {example['premise']}\n")
                f.write(f"   Hypothesis: {example['hypothesis']}\n")
                f.write(f"   Label: {example['label']}\n\n")

            f.write("\nSUSPICIOUS NON-ENTAILMENTS (Low cone energy - might be entailments):\n")
            f.write("-" * 70 + "\n")
            for i, example in enumerate(suspicious_non_entailments[:5], 1):
                f.write(f"{i}. Cone Energy: {example['cone_energy']:.4f}\n")
                f.write(f"   Premise: {example['premise']}\n")
                f.write(f"   Hypothesis: {example['hypothesis']}\n")
                f.write(f"   Label: {example['label']}\n\n")

        return str(report_path)


def main():

    analyzer = SafeAnomalyExtractor()
    anomalous_entailments = analyzer.find_anomalous_entailments()
    suspicious_non_entailment = analyzer.find_low_energy_non_entailments()
    analyzer.generate_anomaly_report(anomalous_entailments, suspicious_non_entailment)

if __name__ == "__main__":
    main()