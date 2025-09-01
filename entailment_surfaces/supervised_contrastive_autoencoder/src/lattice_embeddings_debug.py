"""
Debug script to analyze lattice containment embeddings
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

# Add the project root to the path so we can find the data
project_root = Path(__file__).parent.parent.parent.parent  # Go up to MSc_Topology_Codebase
sys.path.append(str(project_root))

def find_data_file():
    """
    Try to find the SBERT data file in common locations
    """
    possible_paths = [
        "data/processed/snli_full_standard_SBERT.pt",
        "../../../data/processed/snli_full_standard_SBERT.pt", 
        "../../../../data/processed/snli_full_standard_SBERT.pt",
        str(project_root / "data" / "processed" / "snli_full_standard_SBERT.pt")
    ]
    
    for path in possible_paths:
        print(f"Checking: {path}")
        if os.path.exists(path):
            print(f"✅ Found data file: {path}")
            return path
    
    print("❌ Could not find SBERT data file in any of these locations:")
    for path in possible_paths:
        print(f"  {os.path.abspath(path)}")
    
    return None

def analyze_lattice_embeddings(premise_embeddings, hypothesis_embeddings, labels, title="Lattice Analysis"):
    """
    Analyze the quality of lattice containment embeddings
    """
    print(f"\n{title}")
    print("=" * 50)
    
    # Generate lattice embeddings using your current formula
    epsilon = 1e-8
    lattice_embeddings = (premise_embeddings * hypothesis_embeddings) / (
        torch.abs(premise_embeddings) + torch.abs(hypothesis_embeddings) + epsilon
    )
    
    print(f"Input premise stats:")
    print(f"  Mean: {premise_embeddings.mean().item():.6f}")
    print(f"  Std: {premise_embeddings.std().item():.6f}")
    print(f"  Range: [{premise_embeddings.min().item():.6f}, {premise_embeddings.max().item():.6f}]")
    
    print(f"\nInput hypothesis stats:")
    print(f"  Mean: {hypothesis_embeddings.mean().item():.6f}")
    print(f"  Std: {hypothesis_embeddings.std().item():.6f}")
    print(f"  Range: [{hypothesis_embeddings.min().item():.6f}, {hypothesis_embeddings.max().item():.6f}]")
    
    print(f"\nLattice embedding stats:")
    print(f"  Mean: {lattice_embeddings.mean().item():.6f}")
    print(f"  Std: {lattice_embeddings.std().item():.6f}")
    print(f"  Range: [{lattice_embeddings.min().item():.6f}, {lattice_embeddings.max().item():.6f}]")
    
    # Check for collapse
    if lattice_embeddings.std().item() < 0.01:
        print(f"⚠️  WARNING: Lattice embeddings have very low variance ({lattice_embeddings.std().item():.6f})")
        print("   This suggests feature collapse!")
    
    # Analyze per-class separation
    print(f"\nPer-class analysis:")
    for class_id in [0, 1, 2]:
        class_name = ['entailment', 'neutral', 'contradiction'][class_id]
        mask = labels == class_id
        if mask.any():
            class_embeddings = lattice_embeddings[mask]
            print(f"  {class_name}: mean={class_embeddings.mean().item():.6f}, std={class_embeddings.std().item():.6f}")
    
    # Check pairwise distances (same as your triplet debug)
    distances = torch.cdist(lattice_embeddings, lattice_embeddings, p=2)
    
    # Create masks for same/different class pairs
    labels_expanded = labels.contiguous().view(-1, 1)
    mask_positive = torch.eq(labels_expanded, labels_expanded.T).float()
    mask_no_diagonal = 1 - torch.eye(len(labels))
    mask_positive = mask_positive * mask_no_diagonal
    mask_negative = (1 - torch.eq(labels_expanded, labels_expanded.T).float()) * mask_no_diagonal
    
    pos_distances = distances[mask_positive.bool()]
    neg_distances = distances[mask_negative.bool()]
    
    print(f"\nDistance analysis:")
    print(f"  Positive distances: {pos_distances.mean().item():.4f} ± {pos_distances.std().item():.4f}")
    print(f"  Negative distances: {neg_distances.mean().item():.4f} ± {neg_distances.std().item():.4f}")
    print(f"  Separation ratio: {(neg_distances.mean() / pos_distances.mean()).item():.2f}x")
    print(f"  Gap: {(neg_distances.min() - pos_distances.max()).item():.4f}")
    
    if neg_distances.mean() / pos_distances.mean() < 1.5:
        print("⚠️  WARNING: Poor class separation in lattice embeddings!")
        print("   This will make contrastive learning very difficult.")
    
    return lattice_embeddings


def test_alternative_embeddings(premise_embeddings, hypothesis_embeddings, labels):
    """
    Test alternative embedding formulas
    """
    print(f"\nTesting Alternative Embedding Formulas")
    print("=" * 50)
    
    # Alternative 1: Simple concatenation
    concat_embeddings = torch.cat([premise_embeddings, hypothesis_embeddings], dim=1)
    print(f"1. Concatenation shape: {concat_embeddings.shape}")
    
    # Alternative 2: Element-wise difference
    diff_embeddings = premise_embeddings - hypothesis_embeddings
    analyze_lattice_embeddings(premise_embeddings, hypothesis_embeddings, labels, "2. Difference Embeddings")
    
    # Alternative 3: Cosine similarity as feature
    cos_sim = torch.cosine_similarity(premise_embeddings, hypothesis_embeddings, dim=1)
    cos_embeddings = torch.cat([premise_embeddings, hypothesis_embeddings, cos_sim.unsqueeze(1)], dim=1)
    print(f"3. Cosine similarity embeddings shape: {cos_embeddings.shape}")
    
    # Alternative 4: Your original lattice formula
    analyze_lattice_embeddings(premise_embeddings, hypothesis_embeddings, labels, "4. Original Lattice Formula")
    
    # Alternative 5: Improved lattice formula
    epsilon = 1e-8
    # Normalize first, then apply lattice operation
    premise_norm = torch.nn.functional.normalize(premise_embeddings, dim=1)
    hypothesis_norm = torch.nn.functional.normalize(hypothesis_embeddings, dim=1)
    
    improved_lattice = (premise_norm * hypothesis_norm) / (
        torch.abs(premise_norm) + torch.abs(hypothesis_norm) + epsilon
    )
    
    print(f"\n5. Improved Lattice Formula (normalized inputs):")
    print(f"  Mean: {improved_lattice.mean().item():.6f}")
    print(f"  Std: {improved_lattice.std().item():.6f}")
    print(f"  Range: [{improved_lattice.min().item():.6f}, {improved_lattice.max().item():.6f}]")


def test_with_synthetic_fallback():
    """
    Fallback test with synthetic SBERT-style embeddings
    """
    print("\nFalling back to synthetic SBERT-style test...")
    print("=" * 50)
    
    # Create realistic embeddings (normalized, typical SBERT range)
    torch.manual_seed(42)
    n_samples = 300  # 100 per class
    
    # Create realistic SBERT-style embeddings (mean ~0, std ~0.1)
    premise_embeddings = torch.randn(n_samples, 768) * 0.1
    hypothesis_embeddings = torch.randn(n_samples, 768) * 0.1
    
    # Create realistic labels
    labels = torch.cat([
        torch.zeros(100),      # entailment
        torch.ones(100),       # neutral  
        torch.full((100,), 2)  # contradiction
    ]).long()
    
    # Test current formula
    analyze_lattice_embeddings(premise_embeddings, hypothesis_embeddings, labels, "Synthetic SBERT-style Embeddings")
    
    # Test alternatives
    test_alternative_embeddings(premise_embeddings, hypothesis_embeddings, labels)
    """
    Suggest quick fixes for the data loader
    """
    print(f"\nQuick Fixes for Data Loader")
    print("=" * 50)
    
    print("1. Replace lattice embedding formula with simple concatenation:")
    print("   lattice_batch = torch.cat([premise_batch, hypothesis_batch], dim=1)")
    print("   Update model input_dim from 768 to 1536")
    
    print("\n2. Or try element-wise difference:")
    print("   lattice_batch = premise_batch - hypothesis_batch")
    print("   Keep model input_dim as 768")
    
    print("\n3. Or use normalized lattice formula:")
    print("   premise_norm = F.normalize(premise_batch, dim=1)")
    print("   hypothesis_norm = F.normalize(hypothesis_batch, dim=1)")
    print("   lattice_batch = (premise_norm * hypothesis_norm) / (")
    print("       torch.abs(premise_norm) + torch.abs(hypothesis_norm) + epsilon)")


def test_with_actual_sbert_data(data_path, max_samples=1000):
    """
    Test with your actual SBERT embeddings
    """
    print(f"Loading actual SBERT embeddings from {data_path}")
    print("=" * 50)
    
    # Load your actual data
    try:
        data = torch.load(data_path, weights_only=False)
        print(f"Loaded data keys: {list(data.keys())}")
        
        # Get a subset for analysis
        total_samples = len(data['labels'])
        if max_samples and max_samples < total_samples:
            indices = torch.randperm(total_samples)[:max_samples]
            premise_embeddings = torch.stack([data['premise_embeddings'][i] for i in indices])
            hypothesis_embeddings = torch.stack([data['hypothesis_embeddings'][i] for i in indices])
            labels = torch.tensor([0 if data['labels'][i] == 'entailment' 
                                 else 1 if data['labels'][i] == 'neutral' 
                                 else 2 for i in indices])
        else:
            premise_embeddings = data['premise_embeddings']
            hypothesis_embeddings = data['hypothesis_embeddings']
            labels = torch.tensor([0 if label == 'entailment' 
                                 else 1 if label == 'neutral' 
                                 else 2 for label in data['labels']])
        
        print(f"Analyzing {len(labels)} samples")
        print(f"Premise embeddings shape: {premise_embeddings.shape}")
        print(f"Hypothesis embeddings shape: {hypothesis_embeddings.shape}")
        
        # Check label distribution
        unique_labels, counts = torch.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            class_name = ['entailment', 'neutral', 'contradiction'][label]
            print(f"  {class_name}: {count} samples")
        
        # Analyze current lattice formula
        analyze_lattice_embeddings(premise_embeddings, hypothesis_embeddings, labels, 
                                 "Your Actual SBERT Embeddings")
        
        # Test alternative formulas
        test_alternative_embeddings(premise_embeddings, hypothesis_embeddings, labels)
        
        return premise_embeddings, hypothesis_embeddings, labels
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure the data path is correct and the file exists.")
        return None, None, None


def analyze_sbert_embedding_properties(premise_embeddings, hypothesis_embeddings):
    """
    Deep dive into SBERT embedding properties
    """
    print(f"\nDetailed SBERT Embedding Analysis")
    print("=" * 50)
    
    # Check if embeddings are normalized (typical for SBERT)
    premise_norms = torch.norm(premise_embeddings, dim=1)
    hypothesis_norms = torch.norm(hypothesis_embeddings, dim=1)
    
    print(f"Premise embedding norms:")
    print(f"  Mean: {premise_norms.mean().item():.4f}")
    print(f"  Std: {premise_norms.std().item():.4f}")
    print(f"  Range: [{premise_norms.min().item():.4f}, {premise_norms.max().item():.4f}]")
    
    print(f"\nHypothesis embedding norms:")
    print(f"  Mean: {hypothesis_norms.mean().item():.4f}")
    print(f"  Std: {hypothesis_norms.std().item():.4f}")
    print(f"  Range: [{hypothesis_norms.min().item():.4f}, {hypothesis_norms.max().item():.4f}]")
    
    # Check if they're unit normalized
    if torch.abs(premise_norms.mean() - 1.0) < 0.1:
        print("✅ Premise embeddings appear to be unit normalized")
    else:
        print("⚠️  Premise embeddings are NOT unit normalized")
    
    if torch.abs(hypothesis_norms.mean() - 1.0) < 0.1:
        print("✅ Hypothesis embeddings appear to be unit normalized")
    else:
        print("⚠️  Hypothesis embeddings are NOT unit normalized")
    
    # Check cosine similarities between premise and hypothesis
    cos_sims = torch.cosine_similarity(premise_embeddings, hypothesis_embeddings, dim=1)
    print(f"\nCosine similarities between premise-hypothesis pairs:")
    print(f"  Mean: {cos_sims.mean().item():.4f}")
    print(f"  Std: {cos_sims.std().item():.4f}")
    print(f"  Range: [{cos_sims.min().item():.4f}, {cos_sims.max().item():.4f}]")
    
    # Analyze what happens in lattice formula components
    print(f"\nLattice formula component analysis:")
    
    # Numerator: premise * hypothesis (element-wise)
    numerator = premise_embeddings * hypothesis_embeddings
    print(f"Numerator (p * h):")
    print(f"  Mean: {numerator.mean().item():.6f}")
    print(f"  Std: {numerator.std().item():.6f}")
    print(f"  Range: [{numerator.min().item():.6f}, {numerator.max().item():.6f}]")
    
    # Denominator: |p| + |h|
    denominator = torch.abs(premise_embeddings) + torch.abs(hypothesis_embeddings)
    print(f"\nDenominator (|p| + |h|):")
    print(f"  Mean: {denominator.mean().item():.6f}")
    print(f"  Std: {denominator.std().item():.6f}")
    print(f"  Range: [{denominator.min().item():.6f}, {denominator.max().item():.6f}]")
    
    # Final ratio
    epsilon = 1e-8
    ratio = numerator / (denominator + epsilon)
    print(f"\nFinal ratio (numerator/denominator):")
    print(f"  Mean: {ratio.mean().item():.6f}")
    print(f"  Std: {ratio.std().item():.6f}")
    print(f"  Range: [{ratio.min().item():.6f}, {ratio.max().item():.6f}]")


def quick_fix_data_loader():
    """
    Suggest quick fixes for the data loader
    """
    print(f"\nQuick Fixes for Data Loader")
    print("=" * 50)
    
    print("1. Replace lattice embedding formula with simple concatenation:")
    print("   lattice_batch = torch.cat([premise_batch, hypothesis_batch], dim=1)")
    print("   Update model input_dim from 768 to 1536")
    
    print("\n2. Or try element-wise difference:")
    print("   lattice_batch = premise_batch - hypothesis_batch")
    print("   Keep model input_dim as 768")
    
    print("\n3. Or use normalized lattice formula:")
    print("   premise_norm = F.normalize(premise_batch, dim=1)")
    print("   hypothesis_norm = F.normalize(hypothesis_batch, dim=1)")
    print("   lattice_batch = (premise_norm * hypothesis_norm) / (")
    print("       torch.abs(premise_norm) + torch.abs(hypothesis_norm) + epsilon)")


if __name__ == "__main__":
    # Test with your actual SBERT embeddings
    print("ANALYZING YOUR ACTUAL SBERT EMBEDDINGS")
    print("="*60)
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    
    data_path = find_data_file()
    
    if data_path:
        premise_emb, hypothesis_emb, labels = test_with_actual_sbert_data(data_path, max_samples=1000)
        
        if premise_emb is not None:
            # Deep dive into SBERT properties
            analyze_sbert_embedding_properties(premise_emb, hypothesis_emb)
        else:
            print("Failed to process actual data, running synthetic test...")
            test_with_synthetic_fallback()
    else:
        print("Data file not found, running synthetic test...")
        test_with_synthetic_fallback()
    
    # Show fixes regardless
    quick_fix_data_loader()