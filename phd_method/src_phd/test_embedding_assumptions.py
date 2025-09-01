import torch
import numpy as np
import os
from typing import Dict, List, Tuple


class EmbeddingAssumptionTester:

    def __init__(self, processed_data_path: str):
        print("Loading processed data...")
        self.processed_data = torch.load(processed_data_path, weights_only=False)

        self.concatenated_embeddings = self.processed_data['concatenated_embeddings'] #[n, 1536]
        self.labels = self.processed_data['labels'] #[n]
        self.premise_embeddings = self.processed_data['premise_embeddings'] #[n, 768]
        self.hypothesis_embeddings = self.processed_data['hypothesis_embeddings'] #[n, 768]
        
    def create_directional_embeddings(self) -> torch.Tensor:
        """Create directional embeddings that preserve logical direction - in case standard embeddings
        don't work well for k-NN class consistency"""
        print("Creating directional embeddings")

        premise_embs = self.premise_embeddings
        hypothesis_embs = self.hypothesis_embeddings

        #Captures direction
        difference_vector = premise_embs - hypothesis_embs #[n, 768]
        directional_embeddings = torch.cat([
            self.concatenated_embeddings,
            difference_vector
        ], dim=-1) 

        return directional_embeddings

    def create_alternative_directional_embeddings(self) -> torch.Tensor:
        premise_embs = self.premise_embeddings
        hypothesis_embs = self.hypothesis_embeddings

        #Captures direction
        difference_vector = premise_embs - hypothesis_embs #[n, 768]
        directional_embeddings = torch.cat([
            premise_embs,
            hypothesis_embs,
            difference_vector
        ], dim=-1) 

        return directional_embeddings

    def test_label_consistency_in_neighbourhood(self, embeddings: torch.Tensor, embedding_type: str, k_neighbours: int=1000, test_samples: int=1000) -> Dict:
        """Test how often nearest neighbours have same label; using k=200 as min value for compute_phd_dim"""
        print(f"Testing label consistency for {embedding_type} embeddings...")
        print(f"Using k={k_neighbours} neighbours on {test_samples} test samples")

        #Randomly sample test indicies
        test_indices = np.random.choice(len(self.labels), size=min(test_samples, len(self.labels)), replace=False)
        consistency_scores = []
        label_breakdown = {'entailment': [], 'neutral': [], 'contradiction': []}

        for i in test_indices:
            query_emb = embeddings[i]
            true_label = self.labels[i]

            distances = torch.cdist(query_emb.unsqueeze(0), embeddings)
            _, nearest_indices = torch.topk(distances, k=k_neighbours+1, largest=False)

            #Exclude self (first index)
            neighbour_indices = nearest_indices[0][1:]
            neighbour_labels = [self.labels[idx] for idx in neighbour_indices]

            #Calculate consistency (how many neighbours have same label)
            same_label_count = neighbour_labels.count(true_label)
            consistency = same_label_count / len(neighbour_labels)

            consistency_scores.append(consistency)
            label_breakdown[true_label].append(consistency)

        overall_consistency = np.mean(consistency_scores)
        consistency_std = np.std(consistency_scores)

        class_stats = {}
        for label, scores in label_breakdown.items():
            if scores:
                class_stats[label] = {
                    'mean_consistency': np.mean(scores),
                    'std_consistency': np.std(scores),
                    'n_samples': len(scores)
                }

        results = {
            'embedding_type': embedding_type,
            'overall_consistency': overall_consistency,
            'consistency_std': consistency_std,
            'class_stats': class_stats,
        }

        print(f"\n{embedding_type.upper()} RESULTS:")
        print(f"Overall label consistency: {overall_consistency:.3f} ± {consistency_std:.3f}")
        for label, stats in class_stats.items():
            print(f"  {label}: {stats['mean_consistency']:.3f} ± {stats['std_consistency']:.3f} (n={stats['n_samples']})")

        return results


    def compare_embedding_approaches(self):
        print("=" * 80)
        print("COMPARING EMBEDDING APPROACHES FOR PHD CLASSIFICATION")
        print("=" * 80)
        
        results = {}

        # Test 1: Standard concatenated embeddings
        results['standard'] = self.test_label_consistency_in_neighbourhood(
            self.concatenated_embeddings,
            "Standard Concatenated"
        )

        #Test 2: Concatenated directional
        directional_embeddings = self.create_directional_embeddings()
        results['directional'] = self.test_label_consistency_in_neighbourhood(
            directional_embeddings,
            "Concatenated Directional"
        )

        alternative_directional = self.create_alternative_directional_embeddings()
        results['alternative'] = self.test_label_consistency_in_neighbourhood(
            alternative_directional,
            "Non-Concatenated Directional"
        )

        standard_consistency = results['standard']['overall_consistency']
        directional_consistency = results['directional']['overall_consistency']
        alternative_consistency = results['alternative']['overall_consistency']
        
        print(f"Standard embeddings:    {standard_consistency:.3f}")
        print(f"Directional embeddings: {directional_consistency:.3f}")
        print(f"Alternative embeddings: {alternative_consistency:.3f}")

        return results


def run_tests():

    data_path = "phd_method/phd_data/processed/snli_10k_subset_balanced_phd_roberta.pt"        

    tester = EmbeddingAssumptionTester(data_path)

    results = tester.compare_embedding_approaches()

if __name__ == "__main__":
    results = run_tests()




        

