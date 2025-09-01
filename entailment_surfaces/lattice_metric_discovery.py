"""
Lattice-Based Entailment Surface Discovery - Phase 1
Systematic discovery of embedding space + metric combinations that exhibit De Morgan lattice properties
"""

import numpy as np
import torch
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize
from itertools import combinations, product
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass
from collections import defaultdict
import json
from datetime import datetime
from phdim_distance_metric_optimized import SurfaceDistanceMetricAnalyzer
import sys
import os
from pathlib import Path


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def flush_output():
    """Force output to appear immediately in SLURM"""
    sys.stdout.flush()
    sys.stderr.flush()


@dataclass
class ClassTestResult:
    """Results of lattice property testing"""
    space_name: str
    class_name: str
    # The 4 subsumption metrics we want to test
    containment_proxy_score: float
    asymmetric_energy_score: float
    lattice_height_score: float
    subsumption_distance_score: float
    #Std deviation for the 4 metrics
    containment_proxy_std: float
    asymmetric_energy_std: float
    lattice_height_std: float
    subsumption_distance_std: float
    #Signal to noise ratios
    containment_proxy_snr: float
    asymmetric_energy_snr: float
    lattice_height_snr: float
    subsumption_distance_snr: float
    num_tests: int


class SubsumptionMetrics:
    """Implementation of subsumption-aware distance metrics"""
    # NOTE -- THESE ARE DIFFERENT TO THE EMBEDDING SPACES WE ENDED UP TESTING IN PHDIM FILES, AS THESE METRICS
    # PROVIDE SCALAR DISTANE OUTPUTS BETWEEN INDIVIDUAL P-H PAIRS, WHEREAS EMBEDDING SPACES REQUIRE VECTOR REPRESENTATIONS
    # FOR PH-DIM CLUSTERING; HENCE WE TRANSFORM THEM INTO ELEMENT-WISE RELATIONSHIPS ACROSS ALL DIMENSIONS

    @staticmethod
    def containment_proxy_distance(p_emb: np.ndarray, h_emb: np.ndarray, use_threshold: bool = False) -> float:
        """
        Approximates subsumption via embedding containment.
        Based on: premise should 'contain' or 'underwrite' hypothesis
        """
        p_norm = np.linalg.norm(p_emb)
        h_norm = np.linalg.norm(h_emb)
    
        if p_norm == 0 or h_norm == 0:
            return 1.0
    
        overlap = np.dot(p_emb, h_emb) / (p_norm * h_norm)
    
        if use_threshold:
            # Adaptive threshold based on data distribution
            threshold = 0.1  # Could make this adaptive
            containment_score = max(0, overlap - threshold)
            return 1 - containment_score
        else:
            # Simple linear mapping
            return (1 - overlap) / 2  # Maps [-1,1] similarity to [0,1] distance
    

    @staticmethod
    def asymmetric_energy_distance(p_emb: np.ndarray, h_emb: np.ndarray) -> float:
        """
        Uses order violation energies for distance (SHOULD RENAME THIS TO ORDER VIOLATION ENERGY DISTANCE)
        Asymmetry energy is used elsewhere for forward vs backward energy difference which this does not measure
        """
        # Compute element-wise max(0, h - p) for order violation
        violation = np.maximum(0, h_emb - p_emb)
        return np.sum(violation ** 2)


    @staticmethod
    def lattice_height_distance(p_emb: np.ndarray, h_emb: np.ndarray) -> float:
        """
        Distance based on generality hierarchy (lattice height)
        """
        # Simple approximation: norm difference indicates generality difference
        p_generality = np.linalg.norm(p_emb)
        h_generality = np.linalg.norm(h_emb)
        
        # In lattice, more general concepts should have lower norm
        height_diff = h_generality - p_generality
        return max(0, height_diff)  # Only count "upward" movement in lattice


    @staticmethod
    def subsumption_distance(p_emb: np.ndarray, h_emb: np.ndarray) -> float:
        """
        Direct approximation of subsumption relation
        """
        # Compute how much p "covers" h in each dimension
        coverage = np.minimum(p_emb, h_emb) / (np.abs(h_emb) + 1e-8)
        coverage = np.nan_to_num(coverage, 0)
        
        # High coverage = low subsumption distance
        avg_coverage = np.mean(coverage)
        return 1 - avg_coverage

    
    
class LatticeClassTester:
    """Tests each entailment class with our 4 subsumption metrics"""
    
    def __init__(self):
        self.subsumption_metrics = SubsumptionMetrics()
    
    def test_single_class(self, premise_embeddings: np.ndarray, hypothesis_embeddings: np.ndarray, 
                         class_name: str) -> ClassTestResult:
        """Test a single entailment class with all 4 subsumption metrics"""
        
        n_samples = len(premise_embeddings)
        
        # Store scores for each metric
        containment_scores = []
        energy_scores = []
        height_scores = []
        subsumption_scores = []
        
        sample_size = n_samples
        sample_indices = range(n_samples)
        
        for idx in sample_indices:
            p_emb = premise_embeddings[idx]
            h_emb = hypothesis_embeddings[idx]
            
            # COMPUTE ALL 4 SUBSUMPTION METRICS
            # For each metric, we compute P->H distance (premise to hypothesis)
            containment_score = self.subsumption_metrics.containment_proxy_distance(p_emb, h_emb)
            energy_score = self.subsumption_metrics.asymmetric_energy_distance(p_emb, h_emb)
            height_score = self.subsumption_metrics.lattice_height_distance(p_emb, h_emb)
            subsumption_score = self.subsumption_metrics.subsumption_distance(p_emb, h_emb)
            
            containment_scores.append(containment_score)
            energy_scores.append(energy_score)
            height_scores.append(height_score)
            subsumption_scores.append(subsumption_score)
        
        # Check all required scores first
        if not containment_scores:
            raise ValueError("No proxy scores")
        if not energy_scores:
            raise ValueError("No asymmetric score")
        if not height_scores:
            raise ValueError("No lattice height score")
        if not subsumption_scores:
            raise ValueError("No subsumption score")

        #Calculate means and std_deviations
        containment_mean = np.mean(containment_scores)
        containment_std = np.std(containment_scores)

        energy_mean = np.mean(energy_scores)
        energy_std = np.std(energy_scores)

        height_mean = np.mean(height_scores)
        height_std = np.std(height_scores)
        
        subsumption_mean = np.mean(subsumption_scores)
        subsumption_std = np.std(subsumption_scores)

        # Calculate signal-to-noise ratios (mean / std)
        # Higher SNR = more consistent metric within class
        containment_snr = containment_mean / containment_std if containment_std > 0 else float('inf')
        energy_snr = energy_mean / energy_std if energy_std > 0 else float('inf')
        height_snr = height_mean / height_std if height_std > 0 else float('inf')
        subsumption_snr = subsumption_mean / subsumption_std if subsumption_std > 0 else float('inf')



        # Return average scores for this class
        return ClassTestResult(
            space_name="",  # Will be filled by caller
            class_name=class_name,
            containment_proxy_score=containment_mean,
            asymmetric_energy_score=energy_mean,
            lattice_height_score=height_mean,
            subsumption_distance_score=subsumption_mean,
            #Std devs
            containment_proxy_std=containment_std,
            asymmetric_energy_std=energy_std,
            lattice_height_std=height_std,
            subsumption_distance_std=subsumption_std,
            #SNRs
            containment_proxy_snr=containment_snr,
            asymmetric_energy_snr=energy_snr,
            lattice_height_snr=height_snr,
            subsumption_distance_snr=subsumption_snr,
            num_tests=len(sample_indices)
        )


class LatticeDiscoveryAnalyzer:
    """Main analyzer for testing each class individually"""
    
    def __init__(self):
        self.tester = LatticeClassTester()
    
    def test_embedding_space(self, premise_embeddings: np.ndarray, hypothesis_embeddings: np.ndarray, 
                           labels: np.ndarray, space_name: str) -> Dict[str, ClassTestResult]:
        """Test embedding space - return results by class"""
        
        print(f"Testing {space_name}...")
        flush_output()
        
        # Split data by class
        class_data = {}
        label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        
        for label_idx, class_name in label_map.items():
            mask = labels == label_idx
            if np.sum(mask) > 0:
                class_data[class_name] = {
                    'premise_embeddings': premise_embeddings[mask],
                    'hypothesis_embeddings': hypothesis_embeddings[mask]
                }
                print(f"  {class_name}: {np.sum(mask)} samples")
                flush_output()
        
        # Test each class with all 4 subsumption metrics
        results = {}
        for class_name, data in class_data.items():
            try:
                result = self.tester.test_single_class(
                    premise_embeddings=data['premise_embeddings'],
                    hypothesis_embeddings=data['hypothesis_embeddings'],
                    class_name=class_name
                )
                result.space_name = space_name
                results[class_name] = result
                
            except Exception as e:
                print(f"    Error testing {class_name}: {e}")
                flush_output()
        
        return results


    def save_results(self, all_results: Dict[str, Dict[str, ClassTestResult]], filename: str = None, results_dir: str = 'entailment_surfaces/results/class_lattice') -> str:
        """Save results organized by space -> class"""
    
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lattice_subsumption_metrics_results_{timestamp}_SBERT.json"
    
        # Convert to serializable format
        serializable_results = {}
        for space_name, space_results in all_results.items():
            serializable_results[space_name] = {}
            for class_name, result in space_results.items():
                serializable_results[space_name][class_name] = {
                    # Means
                    'containment_proxy_score': result.containment_proxy_score,
                    'asymmetric_energy_score': result.asymmetric_energy_score,
                    'lattice_height_score': result.lattice_height_score,
                    'subsumption_distance_score': result.subsumption_distance_score,
                    # Standard deviations
                    'containment_proxy_std': result.containment_proxy_std,
                    'asymmetric_energy_std': result.asymmetric_energy_std,
                    'lattice_height_std': result.lattice_height_std,
                    'subsumption_distance_std': result.subsumption_distance_std,
                    # Signal-to-noise ratios
                    'containment_proxy_snr': result.containment_proxy_snr,
                    'asymmetric_energy_snr': result.asymmetric_energy_snr,
                    'lattice_height_snr': result.lattice_height_snr,
                    'subsumption_distance_snr': result.subsumption_distance_snr,
                    'num_tests': result.num_tests
                }
    
        output = {
            'subsumption_metrics_by_class': serializable_results,
            'timestamp': datetime.now().isoformat()
        }
    
        # Create the directory if it doesn't exist
        Path(results_dir).mkdir(parents=True, exist_ok=True)
    
        # Create full file path
        full_path = Path(results_dir) / filename
    
        with open(full_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
    
        print(f"Results saved to {full_path}")
        return str(full_path)

def run_class_lattice_discovery(embedding_spaces_dict: Dict[str, Dict[str, np.ndarray]]):
    """
    Run subsumption metrics testing on each class individually
    
    Tests our 4 theoretically-grounded subsumption metrics to see which ones
    show entailment having different properties than neutral/contradiction
    """
    
    analyzer = LatticeDiscoveryAnalyzer()
    
    print("Testing 4 subsumption metrics on each entailment class...")
    flush_output()
    
    all_results = {}
    for space_name, space_data in embedding_spaces_dict.items():
        results = analyzer.test_embedding_space(
            premise_embeddings=space_data['premise_embeddings'],
            hypothesis_embeddings=space_data['hypothesis_embeddings'],
            labels=space_data['labels'],
            space_name=space_name
        )
        all_results[space_name] = results
    
    # Save results
    filename = analyzer.save_results(all_results)

    return all_results, filename

class ClassLatticeAnalyzer(SurfaceDistanceMetricAnalyzer):
    """Integrated analyzer for class-by-class testing"""
    
    def __init__(self, bert_data_path: str, order_model_path: str, 
                 results_dir: str = 'entailment_surfaces/results/class_lattice',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu', seed: int = 42):
        
        super().__init__(bert_data_path, order_model_path, results_dir, device, seed)
        self.lattice_analyzer = LatticeDiscoveryAnalyzer()
        print("Class Lattice Analyzer initialized")
    
    def extract_embeddings_by_class(self, max_samples_per_class: int = None):
        """Extract embeddings organized by class for individual testing"""
        
        premise_embs = self.bert_data['premise_embeddings']
        hypothesis_embs = self.bert_data['hypothesis_embeddings']
        labels = self.bert_data['labels']
        
        # Combine and organize
        label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        all_premise_embs = []
        all_hypothesis_embs = []
        all_labels = []
        
        for label_name, label_idx in label_map.items():
            mask = torch.tensor([l == label_name for l in labels], device=self.device, dtype=torch.bool)
            indices = torch.where(mask)[0]
            
            if max_samples_per_class and len(indices) > max_samples_per_class:
                perm = torch.randperm(len(indices), device=self.device)[:max_samples_per_class]
                indices = indices[perm]
            
            all_premise_embs.append(premise_embs[indices])
            all_hypothesis_embs.append(hypothesis_embs[indices])
            all_labels.extend([label_idx] * len(indices))
            print(f"  {label_name}: {len(indices)} samples")
        
        # Combine all
        combined_premise = torch.cat(all_premise_embs, dim=0).cpu().numpy()
        combined_hypothesis = torch.cat(all_hypothesis_embs, dim=0).cpu().numpy()
        combined_labels = np.array(all_labels)
        
        # Generate order embeddings
        print("Computing order embeddings...")
        flush_output()
        with torch.no_grad():
            premise_tensor = torch.from_numpy(combined_premise).to(self.device)
            hypothesis_tensor = torch.from_numpy(combined_hypothesis).to(self.device)
            premise_order = self.order_model(premise_tensor).cpu().numpy()
            hypothesis_order = self.order_model(hypothesis_tensor).cpu().numpy()
        
        embedding_spaces = {
            'bert_embeddings': {
                'premise_embeddings': combined_premise,
                'hypothesis_embeddings': combined_hypothesis,
                'labels': combined_labels
            },
            'order_embeddings': {
                'premise_embeddings': premise_order,
                'hypothesis_embeddings': hypothesis_order,
                'labels': combined_labels
            }
        }

        # Generate hyperbolic embeddings using cone pipeline
        print("Computing hyperbolic embeddings...")
        flush_output()
        with torch.no_grad():
            # Process in batches to avoid memory issues
            batch_size = 500
            premise_hyperbolic_list = []
            hypothesis_hyperbolic_list = []
                    
            for i in range(0, len(premise_tensor), batch_size):
                batch_premise = premise_tensor[i:i+batch_size]
                batch_hypothesis = hypothesis_tensor[i:i+batch_size]
                        
                # Use cone pipeline to get hyperbolic embeddings
                enhanced_results = self.cone_pipeline.compute_enhanced_cone_energies(
                    batch_premise, batch_hypothesis
                )
                        
                if 'premise_hyperbolic' in enhanced_results and enhanced_results['premise_hyperbolic'] is not None:
                    premise_hyperbolic_list.append(enhanced_results['premise_hyperbolic'])
                    hypothesis_hyperbolic_list.append(enhanced_results['hypothesis_hyperbolic'])
                else:
                    print(f"  Warning: batch {i//batch_size + 1} returned None for hyperbolic embeddings")
                    raise ValueError("No hyperbolic embeddings!")
                    
                if premise_hyperbolic_list and hypothesis_hyperbolic_list:
                    # Concatenate all batches
                    premise_hyperbolic = torch.cat(premise_hyperbolic_list, dim=0).cpu().numpy()
                    hypothesis_hyperbolic = torch.cat(hypothesis_hyperbolic_list, dim=0).cpu().numpy()
                        
                    embedding_spaces['hyperbolic_embeddings'] = {
                        'premise_embeddings': premise_hyperbolic,
                        'hypothesis_embeddings': hypothesis_hyperbolic,
                        'labels': combined_labels
                    }
                        
                    print(f"  Hyperbolic embeddings: {premise_hyperbolic.shape}")
                else:
                    print("  Could not generate hyperbolic embeddings - no valid batches")
                    raise 
                        
        return embedding_spaces
        
    
    def run_class_analysis(self, max_samples_per_class: int = None):
        """Run class-by-class lattice analysis"""
        
        print("Starting class-by-class subsumption metrics analysis")
        print("=" * 60)
        
        # Extract embeddings using the data pipeline
        embedding_spaces = self.extract_embeddings_by_class(max_samples_per_class)
        
        # Run the actual analysis
        all_results, filename = run_class_lattice_discovery(embedding_spaces)
        
        return all_results, filename

def run_integrated_class_discovery(bert_data_path: str, order_model_path: str, max_samples_per_class: int = None):
    """Main function to run integrated class discovery"""
    analyzer = ClassLatticeAnalyzer(bert_data_path, order_model_path)
    return analyzer.run_class_analysis(max_samples_per_class)

if __name__ == "__main__":
    print("Running Subsumption Metrics Analysis...")
    
    # Run the actual analysis
    results, filename = run_integrated_class_discovery(
        bert_data_path="data/processed/snli_full_standard_SBERT_test.pt",
        order_model_path="models/enhanced_order_embeddings_snli_SBERT_full_3way.pt",
    )
    
    print(f"\nAnalysis complete! Results saved to {filename}")