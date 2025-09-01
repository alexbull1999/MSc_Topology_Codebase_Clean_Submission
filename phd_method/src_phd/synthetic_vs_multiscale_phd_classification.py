import torch
import numpy as np
from typing import Dict, List, Tuple
from topology import calculate_ph_dim
from transformers import AutoTokenizer, AutoModel

class PHDMethodTester:

    def __init__(self, processed_data_path: str):
        print("Loading processed data...")
        self.processed_data = torch.load(processed_data_path, weights_only=False)
        self.concatenated_embeddings = self.processed_data['concatenated_embeddings']
        self.labels = self.processed_data['labels']
        self.premise_embeddings=self.processed_data['premise_embeddings']
        self.hypothesis_embeddings = self.processed_data['hypothesis_embeddings']
        self.premise_texts = self.processed_data['texts']['premises']
        self.hypothesis_texts = self.processed_data['texts']['hypotheses']

        #Class baselines from previous PHD computation
        self.class_baselines = {
            'entailment': 18.960741,
            'neutral': 20.814201,
            'contradiction': 22.791120
        }

        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.model=AutoModel.from_pretrained('roberta-base')
        self.model.eval()

        self.device = 'cuda' if self.concatenated_embeddings.is_cuda else 'cpu'
        self.model = self.model.to(self.device)


    def compute_synthetic_neighbourhood_phd(self, query_embedding: torch.Tensor, noise_scale: float=0.0001) -> float:
        synthetic_neighbourhood = []

        for _ in range(500):
            noise = torch.randn_like(query_embedding) * noise_scale
            perturbed = query_embedding + noise
            synthetic_neighbourhood.append(perturbed.cpu().numpy())

        synthetic_neighbourhood = np.array(synthetic_neighbourhood)

        return calculate_ph_dim(W=synthetic_neighbourhood, min_points=200, max_points=500, point_jump=50, h_dim=0, 
            print_error=True, metric="euclidean", alpha=1.0, seed=42)
 

    def get_word_embeddings(self, text:str) -> np.ndarray:
        """Extract word-level embeddings from text"""
        tokens = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}  # Move to device
        with torch.no_grad():
            outputs=self.model(**tokens)
            #Use token embeddings, skipping [CLS] and [SEP]
            word_embeddings = outputs.last_hidden_state[0][1:-1]
        return word_embeddings.cpu().numpy()

    def get_phrase_embeddings(self, text: str) -> np.ndarray:
        """Extract phrase-level embeddings using a sliding window"""
        words = text.split()
        if len(words) < 2:
            return np.empty((0, 768))

        phrase_embeddings = []
        #Create 2-3 word phrases
        for i in range(len(words) - 1):
            for phrase_len in [2,3]:
                if i + phrase_len <= len(words):
                    phrase = ' '.join(words[i:i+phrase_len])
                    tokens = self.tokenizer(phrase, return_tensors='pt', truncation=True)
                    tokens = {k: v.to(self.device) for k, v in tokens.items()}  # Move to device
                    with torch.no_grad():
                        outputs = self.model(**tokens)
                        # Use [CLS] as phrase representation
                        phrase_emb = outputs.last_hidden_state[0][0]
                        phrase_embeddings.append(phrase_emb.cpu().numpy())

        return np.array(phrase_embeddings) 




    def compute_multiscale_phd(self, premise_text: str, hypothesis_text: str, premise_emb: torch.Tensor, hypothesis_emb: torch.Tensor, concatenated_emb: torch.Tensor) -> float:

        #Scale 1: Word_level embeddings
        premise_words = self.get_word_embeddings(premise_text)
        hypothesis_words = self.get_word_embeddings(hypothesis_text)

        #Scale 2: Phrase-level embeddings
        premise_phrases = self.get_phrase_embeddings(premise_text)
        hypothesis_phrases = self.get_phrase_embeddings(hypothesis_text)
        
        # Scale 3: Sentence-level embeddings
        premise_sentence = premise_emb.cpu().numpy().reshape(1, -1)
        hypothesis_sentence = hypothesis_emb.cpu().numpy().reshape(1, -1)
        
        # Combine into unified point cloud
        point_cloud_parts = [premise_words, hypothesis_words, premise_sentence, hypothesis_sentence]

        if len(premise_phrases) > 0:
            point_cloud_parts.append(premise_phrases)
        if len(hypothesis_phrases) > 0:
            point_cloud_parts.append(hypothesis_phrases)

        point_cloud = np.vstack(point_cloud_parts)

         # If we don't have enough points, add noise to concatenated embedding to reach 200 points
        if len(point_cloud) < 200:
            points_needed = 200 - len(point_cloud)
            noise_scale = 0.0001
            
            for i in range(points_needed):
                if i % 2 == 0:
                    noise = np.random.randn(*premise_emb.shape) * noise_scale
                    perturbed = premise_emb.cpu().numpy() + noise
                else:
                    noise = np.random.randn(*hypothesis_emb.shape) * noise_scale
                    perturbed = hypothesis_emb.cpu().numpy() + noise
                point_cloud = np.vstack([point_cloud, perturbed.reshape(1, -1)])
        
        return calculate_ph_dim(W=point_cloud,
                               min_points=200,
                               max_points=min(500, len(point_cloud)),
                               point_jump=50,
                               h_dim=0,
                               print_error=True,
                               metric="euclidean",
                               alpha=1.0,
                               seed=42)


    def classify_sample(self, phd_score: float) -> str:
        """Classify based on PHD score using nearest class baseline"""
        if np.isnan(phd_score):
            print("ERROR - PHD SCORE FOR SAMPLE RETURNED AS NAN")
            raise 
            
        distances = {
            label: abs(phd_score - baseline)
            for label, baseline in self.class_baselines.items()
        }
        return min(distances.keys(), key=distances.get)



    def test_method(self, method_name: str, test_samples: int = 500) -> Dict:
        """Test a PHD method on random samples"""
        print(f"Testing {method_name} method...")

        test_indices = np.random.choice(len(self.labels), size=min(test_samples, len(self.labels)), replace=False)
        correct_predictions = 0
        predictions = []
        true_labels = []

        for i in test_indices:
            true_label = self.labels[i]

            if method_name == "Synthetic Neighbourhood":
                query_emb = self.concatenated_embeddings[i]
                phd_score = self.compute_synthetic_neighbourhood_phd(query_emb)
            elif method_name == "Multi-Scale Decomposition":
                premise_text = self.premise_texts[i]
                hypothesis_text = self.hypothesis_texts[i]
                premise_emb = self.premise_embeddings[i]
                hypothesis_emb = self.hypothesis_embeddings[i]
                concatenated_emb = self.concatenated_embeddings[i]
                phd_score = self.compute_multiscale_phd(premise_text, hypothesis_text, 
                                                      premise_emb, hypothesis_emb, concatenated_emb)
        
            predicted_label = self.classify_sample(phd_score)
            if predicted_label == true_label:
                correct_predictions += 1
                
            predictions.append(predicted_label)
            true_labels.append(true_label)
        
        accuracy = correct_predictions / len(test_indices)

        # Calculate per-class accuracy
        class_stats = {}
        for label in ['entailment', 'neutral', 'contradiction']:
            class_indices = [i for i, l in enumerate(true_labels) if l == label]
            if class_indices:
                class_correct = sum(1 for i in class_indices if predictions[i] == true_labels[i])
                class_stats[label] = {
                    'accuracy': class_correct / len(class_indices),
                    'count': len(class_indices)
                }
        
        results = {
            'method': method_name,
            'overall_accuracy': accuracy,
            'class_stats': class_stats,
            'total_samples': len(test_indices)
        }
        
        print(f"\n{method_name.upper()} RESULTS:")
        print(f"Overall accuracy: {accuracy:.3f}")
        for label, stats in class_stats.items():
            print(f"  {label}: {stats['accuracy']:.3f} (n={stats['count']})")
            
        return results

    def compare_methods(self):
        """Compare both PHD methods"""
        print("=" * 60)
        print("COMPARING PHD CLASSIFICATION METHODS")
        print("=" * 60)
        
        results = {}
        
        results['synthetic'] = self.test_method("Synthetic Neighbourhood")
        results['multiscale'] = self.test_method("Multi-Scale Decomposition")
        
        print("\nSUMMARY COMPARISON:")
        synthetic_acc = results['synthetic']['overall_accuracy']
        multiscale_acc = results['multiscale']['overall_accuracy']
        
        print(f"Synthetic Neighbourhood:    {synthetic_acc:.3f}")
        print(f"Multi-Scale Decomposition: {multiscale_acc:.3f}")

        return results

def run_phd_comparison():
    data_path = "phd_method/phd_data/processed/snli_10k_subset_balanced_phd_roberta.pt"
    tester = PHDMethodTester(data_path)
    results = tester.compare_methods()
    
    return results


if __name__ == "__main__":
    results = run_phd_comparison()






                    
