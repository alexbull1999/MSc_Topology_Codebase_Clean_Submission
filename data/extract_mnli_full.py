"""
MNLI dataset extraction script
Extracts MNLI data in the same format as extract_snli_full.py
Compatible with existing TDA entailment pipeline
"""

import pandas as pd
import json
import os
from datasets import Dataset, load_from_disk
import numpy as np

def extract_mnli_full_dataset(arrow_file_path, output_path, seed=42):
    """
    Extract the entire MNLI dataset from arrow file
    Converts to toy dataset format: [premise, hypothesis, label]
    
    Args:
        arrow_file_path: Path to MNLI arrow file (e.g., 'data/raw/mnli/train/data-00000-of-00001.arrow')
        output_path: Path to save processed JSON file
        seed: Random seed for shuffling
    """
    # Label mapping for conversion (same as SNLI)
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

    # Load dataset from arrow file
    dataset = Dataset.from_file(arrow_file_path)
    df = dataset.to_pandas()
    
    print(f"Original MNLI columns: {df.columns.tolist()}")
    print(f"Original shape: {df.shape}")
    
    # Filter out any invalid labels (sometimes datasets have -1 for invalid samples)
    df = df[df['label'].isin([0, 1, 2])]

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"Full MNLI dataset loaded:")
    print(f"Total samples: {len(df)}")
    print(f"Label distribution (original integers):")
    print(df['label'].value_counts().sort_index())

    # Save as JSON in toy dataset format: [premise, hypothesis, label]
    output_data = []
    for _, row in df.iterrows():
        output_data.append([
            row['premise'],  # First element: premise
            row['hypothesis'],  # Second element: hypothesis
            label_map[row['label']]  # Third element: converted label
        ])

    # Show converted label distribution
    converted_labels = [item[2] for item in output_data]  # Labels are at index 2
    print(f"Label distribution (converted to strings):")
    for label in ['entailment', 'neutral', 'contradiction']:
        count = converted_labels.count(label)
        print(f"  {label}: {count}")

    print(f"Sample format:")
    print(f"First example: {output_data[0]}")
    print(f"Expected format: [premise, hypothesis, label]")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved {len(output_data)} samples to {output_path}")
    return df


if __name__ == "__main__":
    # Configuration - matching your existing paths
    mnli_raw_path = "data/raw/mnli/validation_mismatched/data-00000-of-00001.arrow"
    
    # Output paths matching your existing structure
    full_output_path = "data/raw/mnli/validation_mismatched/mnli_full_validation_mismatched.json"
    
    print("MNLI Dataset Extraction")
    print("=" * 50)
    
    # Check if raw data exists
    if not os.path.exists(mnli_raw_path):
        print(f"Error: MNLI data not found at {mnli_raw_path}")
        print("Please ensure MNLI dataset is available in data/raw/mnli")
        print("If you need to download it:")
        print("from datasets import load_dataset")
        print("mnli = load_dataset('multi_nli')")
        print("mnli.save_to_disk('data/raw/mnli')")
        exit(1)
    
    # Extract full dataset
    print("\n1. Extracting full MNLI dataset...")
    df_full = extract_mnli_full_dataset(mnli_raw_path, full_output_path)
    
    print("\nExtraction complete!")
    print(f"Full dataset: {full_output_path}")
    print("You can now run your existing experiments using MNLI data instead of SNLI.")