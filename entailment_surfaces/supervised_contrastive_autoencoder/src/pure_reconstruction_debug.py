"""
Debug script to investigate why k-NN predicts only neutral class
for the pure reconstruction autoencoder
"""

import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import sys
from attention_autoencoder_model import AttentionAutoencoder
from contrastive_autoencoder_model_global import ContrastiveAutoencoder
from data_loader_global import GlobalDataLoader


def debug_reconstruction_model():
    """Debug the pure reconstruction model's latent space and k-NN behavior"""
    
    print("🔍 DEBUGGING PURE RECONSTRUCTION MODEL")
    print("=" * 60)
    
    # Configuration
    MODEL_PATH = "entailment_surfaces/supervised_contrastive_autoencoder/experiments/moor_topo-contrastive_autoencoder_noattention_20250725_170549/checkpoints/checkpoint_epoch_50.pt"  # Update this
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    print(f"Model path: {MODEL_PATH}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model not found at {MODEL_PATH}")
        print("Please update MODEL_PATH to your actual model checkpoint")
        return
    
    try:
        # 1. Load the model
        print("\n1️⃣ Loading model...")
        model = ContrastiveAutoencoder(
            input_dim=1536,  # SBERT concat
            latent_dim=75,  # Your reconstruction model
            hidden_dims=[1024, 768, 512, 256, 128],
            dropout_rate=0.2
        )
        
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        print("Saved model keys:", list(checkpoint['model_state_dict'].keys()))
        print("Current model keys:", list(model.state_dict().keys()))
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully")
        
        # 2. Load data
        print("\n2️⃣ Loading data...")
        from data_loader_global import GlobalDataLoader
        
        loader = GlobalDataLoader(
            train_path="data/processed/snli_full_standard_SBERT.pt",
            val_path="data/processed/snli_full_standard_SBERT_validation.pt", 
            test_path="data/processed/snli_full_standard_SBERT_test.pt",
            embedding_type='concat',
            sample_size=None,
            batch_size=1020
        )
        
        train_dataset, val_dataset, test_dataset = loader.load_data()
        train_loader, val_loader, test_loader = loader.get_dataloaders(batch_size=1020)
        
        print("✅ Data loaded successfully")
        
        # 3. Extract latent representations for a subset
        print("\n3️⃣ Extracting latent representations...")
        
        def extract_subset_latents(dataloader, max_samples=5000):
            all_latents = []
            all_labels = []
            sample_count = 0
            
            with torch.no_grad():
                for batch in dataloader:
                    if sample_count >= max_samples:
                        break
                        
                    embeddings = batch['embeddings'].to(device)
                    labels = batch['labels']
                    
                    latent, _ = model(embeddings)
                    
                    all_latents.append(latent.cpu())
                    all_labels.append(labels)
                    
                    sample_count += len(labels)
            
            latents = torch.cat(all_latents, dim=0)[:max_samples]
            labels = torch.cat(all_labels, dim=0)[:max_samples]
            
            return latents.numpy(), labels.numpy()
        
        # Extract training and validation subsets
        train_latents, train_labels = extract_subset_latents(train_loader, max_samples=10000)
        val_latents, val_labels = extract_subset_latents(val_loader, max_samples=2000)
        
        print(f"Train samples: {len(train_latents)}, Val samples: {len(val_latents)}")
        
        # 4. Analyze class distributions
        print("\n4️⃣ Analyzing class distributions...")
        train_class_dist = np.bincount(train_labels)
        val_class_dist = np.bincount(val_labels)
        
        print(f"Training class distribution: {train_class_dist}")
        print(f"  Entailment: {train_class_dist[0]} ({train_class_dist[0]/len(train_labels)*100:.1f}%)")
        print(f"  Neutral: {train_class_dist[1]} ({train_class_dist[1]/len(train_labels)*100:.1f}%)")
        print(f"  Contradiction: {train_class_dist[2]} ({train_class_dist[2]/len(train_labels)*100:.1f}%)")
        
        print(f"\nValidation class distribution: {val_class_dist}")
        print(f"  Entailment: {val_class_dist[0]} ({val_class_dist[0]/len(val_labels)*100:.1f}%)")
        print(f"  Neutral: {val_class_dist[1]} ({val_class_dist[1]/len(val_labels)*100:.1f}%)")
        print(f"  Contradiction: {val_class_dist[2]} ({val_class_dist[2]/len(val_labels)*100:.1f}%)")
        
        # 5. Analyze latent space collapse
        print("\n5️⃣ Analyzing latent space structure...")
        
        print(f"Latent space shape: {train_latents.shape}")
        print(f"Latent space statistics:")
        print(f"  Overall mean: {train_latents.mean():.6f}")
        print(f"  Overall std: {train_latents.std():.6f}")
        print(f"  Min value: {train_latents.min():.6f}")
        print(f"  Max value: {train_latents.max():.6f}")
        
        # Check per-dimension statistics
        per_dim_std = train_latents.std(axis=0)
        print(f"  Dimensions with std > 0.01: {np.sum(per_dim_std > 0.01)}/{len(per_dim_std)}")
        print(f"  Dimensions with std > 0.001: {np.sum(per_dim_std > 0.001)}/{len(per_dim_std)}")
        print(f"  Max dimension std: {per_dim_std.max():.6f}")
        print(f"  Min dimension std: {per_dim_std.min():.6f}")
        
        # Check pairwise distances
        print("\n6️⃣ Checking pairwise distances...")
        sample_indices = np.random.choice(len(train_latents), min(1000, len(train_latents)), replace=False)
        sample_latents = train_latents[sample_indices]
        
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(sample_latents)
        
        # Remove diagonal (distance to self = 0)
        distances_no_diag = distances[np.triu_indices_from(distances, k=1)]
        
        print(f"Pairwise distances statistics:")
        print(f"  Mean distance: {distances_no_diag.mean():.6f}")
        print(f"  Std distance: {distances_no_diag.std():.6f}")
        print(f"  Min distance: {distances_no_diag.min():.6f}")
        print(f"  Max distance: {distances_no_diag.max():.6f}")
        print(f"  Distances < 0.001: {np.sum(distances_no_diag < 0.001)} / {len(distances_no_diag)}")
        
        # 7. Test k-NN behavior
        print("\n7️⃣ Testing k-NN classifier behavior...")
        
        # Train k-NN
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(train_latents, train_labels)
        
        # Make predictions
        predictions = knn.predict(val_latents)
        pred_dist = np.bincount(predictions, minlength=3)
        
        print(f"k-NN predictions distribution: {pred_dist}")
        print(f"  Predicts entailment: {pred_dist[0]} times ({pred_dist[0]/len(predictions)*100:.1f}%)")
        print(f"  Predicts neutral: {pred_dist[1]} times ({pred_dist[1]/len(predictions)*100:.1f}%)")
        print(f"  Predicts contradiction: {pred_dist[2]} times ({pred_dist[2]/len(predictions)*100:.1f}%)")
        
        # Check a few individual predictions
        print(f"\n8️⃣ Analyzing individual k-NN decisions...")
        for i in range(min(5, len(val_latents))):
            query_point = val_latents[i:i+1]
            distances, indices = knn.kneighbors(query_point)
            neighbor_labels = train_labels[indices[0]]
            
            print(f"Sample {i} (true label: {val_labels[i]}):")
            print(f"  Distances to 5 neighbors: {distances[0]}")
            print(f"  Neighbor labels: {neighbor_labels}")
            print(f"  Predicted: {predictions[i]}")
        
        # 9. Final diagnosis
        print("\n9️⃣ DIAGNOSIS:")
        print("=" * 40)
        
        if train_class_dist[1] > train_class_dist[0] + train_class_dist[2]:
            print("🔴 CLASS IMBALANCE: Neutral class dominates training data")
            print("   → k-NN biased toward predicting neutral")
        
        if pred_dist[1] / len(predictions) > 0.8:
            print("🔴 k-NN PREDICTION COLLAPSE: Almost always predicts neutral")
            print("   → Combination of latent collapse + class imbalance")
        
        print("\nThis explains why k-NN performance is essentially random!")

         # 10. Test actual reconstruction behavior
        print("\n🔟 Testing actual reconstruction behavior...")
        print("=" * 50)
        
        # Test with a few different inputs
        reconstructions = []
        inputs = []
        
        with torch.no_grad():
            for i in range(5):
                input_sample = val_dataset[i]['embeddings'].unsqueeze(0).to(device)
                latent, reconstructed = model(input_sample)
                
                inputs.append(input_sample[0].cpu())
                reconstructions.append(reconstructed[0].cpu())
                
                print(f"\nSample {i}:")
                print(f"  Input (first 5 dims): {input_sample[0][:5].cpu().numpy()}")
                print(f"  Latent (first 5 dims): {latent[0][:5].cpu().numpy()}")  
                print(f"  Reconstructed (first 5 dims): {reconstructed[0][:5].cpu().numpy()}")
                
                recon_mse = torch.mean((reconstructed - input_sample)**2).item()
                print(f"  Individual reconstruction MSE: {recon_mse:.6f}")
        
        # Check if all reconstructions are identical
        print(f"\nChecking if all reconstructions are identical:")
        for i in range(1, 5):
            recon_diff = torch.mean((reconstructions[0] - reconstructions[i])**2).item()
            print(f"  Reconstruction difference sample 0 vs {i}: {recon_diff:.6f}")
        
        # Check if all inputs are different
        print(f"\nChecking if inputs are actually different:")
        for i in range(1, 5):
            input_diff = torch.mean((inputs[0] - inputs[i])**2).item()
            print(f"  Input difference sample 0 vs {i}: {input_diff:.6f}")
        
        # Test what happens when we manually set latent to different values
        print(f"\n🧪 Testing manual latent manipulation...")
        test_input = val_dataset[0]['embeddings'].unsqueeze(0).to(device)
        
        # Get original reconstruction
        with torch.no_grad():
            original_latent, original_recon = model(test_input)
            
            # Manually create different latent vectors
            modified_latent1 = torch.zeros_like(original_latent)
            modified_latent2 = torch.ones_like(original_latent) * 0.1
            modified_latent3 = torch.randn_like(original_latent) * 0.1
            
            # Pass through decoder only
            decoder_recon1 = model.decoder(modified_latent1)
            decoder_recon2 = model.decoder(modified_latent2) 
            decoder_recon3 = model.decoder(modified_latent3)
            
            print(f"Original reconstruction (first 5): {original_recon[0][:5].cpu().numpy()}")
            print(f"Zero latent reconstruction (first 5): {decoder_recon1[0][:5].cpu().numpy()}")
            print(f"0.1 latent reconstruction (first 5): {decoder_recon2[0][:5].cpu().numpy()}")
            print(f"Random latent reconstruction (first 5): {decoder_recon3[0][:5].cpu().numpy()}")
            
            # Check differences
            diff1 = torch.mean((original_recon - decoder_recon1)**2).item()
            diff2 = torch.mean((original_recon - decoder_recon2)**2).item()
            diff3 = torch.mean((original_recon - decoder_recon3)**2).item()
            
            print(f"Difference from original: zero={diff1:.6f}, 0.1={diff2:.6f}, random={diff3:.6f}")
        
        # 11. Final comprehensive diagnosis
        print("\n1️⃣1️⃣ COMPREHENSIVE DIAGNOSIS:")
        print("=" * 50)
        
        if diff1 < 0.001 and diff2 < 0.001 and diff3 < 0.001:
            print("🔴 DECODER IGNORES LATENT: Decoder produces same output regardless of latent input!")
            print("   → This explains both latent collapse AND good reconstruction")
            print("   → The decoder learned to output some 'average' representation")
            print("   → Reconstruction loss is computed incorrectly or model has bypass connection")
        else:
            print("🟡 DECODER RESPONDS TO LATENT: Different latents produce different outputs")
            print("   → But all samples still collapse to same latent point")
            print("   → This suggests encoder problem or optimization failure")
        
        # Check if reconstructions are just outputting the mean
        all_inputs = torch.stack(inputs)
        mean_input = all_inputs.mean(dim=0)
        
        mean_diff = torch.mean((reconstructions[0] - mean_input)**2).item()
        print(f"\nIs reconstruction just the mean input? Difference: {mean_diff:.6f}")
        
        if mean_diff < 0.0001:
            print("🔴 RECONSTRUCTION IS MEAN: Model learned to output average of all inputs!")
            print("   → This minimizes MSE when all latents are identical")
            print("   → Classic degenerate autoencoder solution")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_reconstruction_model()