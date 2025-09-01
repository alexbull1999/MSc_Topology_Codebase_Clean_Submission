from datasets import Dataset
import json
import os
from pathlib import Path
import pandas as pd


def extract_snli_balanced_subset(arrow_file_path, output_path, samples_per_class=3330, seed=42):
    """
    Extract a balanced subset with equal samples from each class
    Total samples = samples_per_class * 3 ≈ 5K
    Converts to toy dataset format: [premise, hypothesis, label]
    """
    # Label mapping for conversion
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

    # Load dataset
    dataset = Dataset.from_file(arrow_file_path)
    df = dataset.to_pandas()

    # Sample equally from each class
    balanced_dfs = []
    for label in [0, 1, 2]:  # entailment, neutral, contradiction
        label_df = df[df['label'] == label].sample(
            n=min(samples_per_class, len(df[df['label'] == label])),
            random_state=seed
        )
        balanced_dfs.append(label_df)

    # Combine and shuffle
    balanced_df = pd.concat(balanced_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"Balanced subset created:")
    print(f"Total samples: {len(balanced_df)}")
    print(f"Label distribution (original integers):")
    print(balanced_df['label'].value_counts())

    # Save as JSON in toy dataset format: [premise, hypothesis, label]
    output_data = []
    for _, row in balanced_df.iterrows():
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
    return balanced_df



if __name__=="__main__":
    arrow_file = 'data/raw/snli/test/data-00000-of-00001.arrow'
    output_path = 'data/raw/snli/test/TEST_snli_10k_subset_balanced.json'

    df = extract_snli_balanced_subset(arrow_file, output_path)