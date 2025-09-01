
from datasets import Dataset
import json
import os
from pathlib import Path
import pandas as pd


def extract_snli_full_dataset(arrow_file_path, output_path, seed=42):
    """
    Extract the entire SNLI train dataset
    Converts to toy dataset format: [premise, hypothesis, label]
    """
    # Label mapping for conversion
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

    # Load dataset
    dataset = Dataset.from_file(arrow_file_path)
    df = dataset.to_pandas()
    
    # Filter out any invalid labels (sometimes datasets have -1 for invalid samples)
    df = df[df['label'].isin([0, 1, 2])]

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"Full SNLI dataset loaded:")
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

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved {len(output_data)} samples to {output_path}")
    return df


if __name__ == "__main__":
    arrow_file = 'data/raw/snli/validation/data-00000-of-00001.arrow'
    
    # Extract full dataset
    full_output_path = 'data/raw/snli/validation/snli_full_val.json'
    df = extract_snli_full_dataset(arrow_file, full_output_path)