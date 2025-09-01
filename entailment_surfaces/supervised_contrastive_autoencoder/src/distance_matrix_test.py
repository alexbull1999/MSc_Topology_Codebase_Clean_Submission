import torch
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import sys
import os
import time
from pathlib import Path
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from gph.python import ripser_parallel
from sklearn.metrics.pairwise import pairwise_distances

# Add your project path so you can import your modules
sys.path.append('/homes/ahb24/MSc_Topology_Codebase/entailment_surfaces/supervised_contrastive_autoencoder/src')

# Import your data loading functions (adjust these imports to match your actual files)
try:
    from global_data_loader import GlobalDataLoader  # Replace with your actual data loader import
except ImportError:
    print("⚠️ Could not import data loaders. You'll need to adjust the imports.")
    print("Please check the exact names of your data loading functions.")


def sample_W(W, nSamples, isRandom=True):
    n = W.shape[0]
    if n <= nSamples:
        return W
    random_indices = np.random.choice(n, size=nSamples, replace=False)
    
    # FIXED: For distance matrices (square), sample both rows AND columns
    if W.shape[0] == W.shape[1]:
        return W[np.ix_(random_indices, random_indices)]
    else:
        return W[random_indices]

def calculate_ph_dim(W: np.ndarray,
                     min_points=200,
                     max_points=1000,
                     point_jump=50,
                     h_dim=0,
                     print_error=True,
                     metric=None,
                     alpha: float = 1.,
                     seed: int = 42) -> float:
    # sample_fn should output a [num_points, dim] array

    np.random.seed(seed)

    logger.info(f"Calculating PH dimension with points {min_points} to {max_points}, seed: {seed}")

    # sample our points
    test_n = range(min_points, max_points, point_jump)
    logger.debug(f"Number of test points for PH dimension computation: {len(test_n)}")
    lengths = []
    for n in tqdm(test_n):
        if metric is None:
            # diagrams = ripser(sample_W(W, n))['dgms']
            diagrams = ripser_parallel(sample_W(W, n), maxdim=h_dim, n_threads=-1)['dgms']
        else:
            # diagrams = ripser(sample_W(W, n), metric=metric)['dgms']
            diagrams = ripser_parallel(sample_W(W, n), metric=metric, maxdim=h_dim, n_threads=-1)['dgms']

        if len(diagrams) > h_dim:
            d = diagrams[h_dim]
            d = d[d[:, 1] < np.inf]
            lengths.append(np.power((d[:, 1] - d[:, 0]), alpha).sum())  # The fact that \alpha = 1 appears here
        else:
            lengths.append(0.0)
    lengths = np.array(lengths)

    # compute our ph dim by running a linear least squares
    x = np.log(np.array(list(test_n)))
    y = np.log(lengths)
    N = len(x)
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    b = y.mean() - m * x.mean()

    error = ((y - (m * x + b)) ** 2).mean()

    if print_error:
        logger.debug(f"Ph Dimension Calculation has an approximate error of: {error}.")

    return alpha / (1 - m)


def test_distance_matrix_embedding_preserves_intrinsic_dim(dataloader, device, calculate_ph_dimension_func):
    """
    Test if distance matrix embedding preserves intrinsic dimensions
    """
    print("🔍 Testing Distance Matrix Embedding vs Direct PH Dimension...")
    
    # Collect features by class
    class_features = {0: [], 1: [], 2: []}  # entailment, neutral, contradiction
    class_names = ['entailment', 'neutral', 'contradiction']
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            embeddings = batch['embeddings'].to(device)
            labels = batch['labels'].to(device)
            
            for class_idx in range(3):
                class_mask = (labels == class_idx)
                if class_mask.sum() > 0:
                    class_features[class_idx].append(embeddings[class_mask].cpu())
            
            if batch_idx >= 10:  # Test on first 10 batches
                break
    
    results = {}
    
    for class_idx, class_name in enumerate(class_names):
        if class_features[class_idx]:
            # Combine features for this class
            all_features = torch.cat(class_features[class_idx], dim=0)
            
            # Subsample to manageable size
            if len(all_features) > 1000:
                indices = torch.randperm(len(all_features))[:1000]
                all_features = all_features[indices]
            
            features_np = all_features.numpy()
            
            print(f"\n{class_name.upper()}:")
            print(f"  Features shape: {features_np.shape}")
            
            # Method 1: Direct PH dimension calculation (ground truth)
            print("  Computing direct PH dimension...")
            direct_ph_dim = calculate_ph_dimension_func(
                features_np, 
                min_points=200, 
                max_points=min(800, len(features_np)-50),
                metric='euclidean'
            )
            
            # Method 2: Distance matrix embedding then PH dimension
            print("  Computing distance matrix...")
            distance_matrix = pairwise_distances(features_np, metric='euclidean')
            
            # Test both normalized and unnormalized
            dm_unnormalized = distance_matrix.copy()
            dm_normalized = distance_matrix / distance_matrix.max()
            
            print("  Computing PH dimension on unnormalized distance matrix...")
            embedding_ph_dim_unnorm = calculate_ph_dimension_func(
                dm_unnormalized,
                min_points=200,
                max_points=min(800, len(dm_unnormalized)-50),
                metric='precomputed'  # Important: use precomputed for distance matrix
            )
            
            print("  Computing PH dimension on normalized distance matrix...")
            embedding_ph_dim_norm = calculate_ph_dimension_func(
                dm_normalized, 
                min_points=200,
                max_points=min(800, len(dm_normalized)-50),
                metric='precomputed'
            )
            
            print(f"  Direct PH dimension:           {direct_ph_dim:.4f}")
            print(f"  Distance embedding (unnorm):  {embedding_ph_dim_unnorm:.4f}")
            print(f"  Distance embedding (norm):    {embedding_ph_dim_norm:.4f}")
            
            # Calculate preservation ratios
            ratio_unnorm = embedding_ph_dim_unnorm / direct_ph_dim if direct_ph_dim != 0 else float('inf')
            ratio_norm = embedding_ph_dim_norm / direct_ph_dim if direct_ph_dim != 0 else float('inf')
            
            print(f"  Preservation ratio (unnorm):  {ratio_unnorm:.4f}")
            print(f"  Preservation ratio (norm):    {ratio_norm:.4f}")
            
            results[class_name] = {
                'direct_ph_dim': direct_ph_dim,
                'embedding_ph_dim_unnorm': embedding_ph_dim_unnorm,
                'embedding_ph_dim_norm': embedding_ph_dim_norm,
                'ratio_unnorm': ratio_unnorm,
                'ratio_norm': ratio_norm
            }
    
    print("\n" + "="*60)
    print("SUMMARY: Distance Matrix Embedding Preservation Test")
    print("="*60)
    
    for class_name, result in results.items():
        print(f"{class_name}:")
        print(f"  Direct: {result['direct_ph_dim']:.2f} → Embedding (unnorm): {result['embedding_ph_dim_unnorm']:.2f} (ratio: {result['ratio_unnorm']:.2f})")
        print(f"  Direct: {result['direct_ph_dim']:.2f} → Embedding (norm):   {result['embedding_ph_dim_norm']:.2f} (ratio: {result['ratio_norm']:.2f})")
    
    # Check if embedding preserves dimensions well
    avg_ratio_unnorm = np.mean([r['ratio_unnorm'] for r in results.values()])
    avg_ratio_norm = np.mean([r['ratio_norm'] for r in results.values()])
    
    print(f"\nAverage preservation ratios:")
    print(f"  Unnormalized: {avg_ratio_unnorm:.2f}")
    print(f"  Normalized:   {avg_ratio_norm:.2f}")
    
    if 0.8 <= avg_ratio_unnorm <= 1.2:
        print("✅ Unnormalized distance embedding preserves intrinsic dimension well!")
    else:
        print("❌ Unnormalized distance embedding does NOT preserve intrinsic dimension")
        
    if 0.8 <= avg_ratio_norm <= 1.2:
        print("✅ Normalized distance embedding preserves intrinsic dimension well!")
    else:
        print("❌ Normalized distance embedding does NOT preserve intrinsic dimension")
    
    return results

def main():
    """
    Main function to run the test
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Import your GlobalDataLoader
        from data_loader_global import GlobalDataLoader
        
        # Set up data paths (adjust these to your actual paths)
        data_config = {
            'train_path': 'data/processed/snli_full_standard_SBERT.pt',
            'val_path': 'data/processed/snli_full_standard_SBERT_validation.pt', 
            'test_path': 'data/processed/snli_full_standard_SBERT_test.pt',
            'embedding_type': 'concat',  # or 'concat', 'difference'
            'sample_size': 5000,  # Use subset for faster testing
            'batch_size': 1020
        }
        
        print("Creating GlobalDataLoader...")
        data_loader = GlobalDataLoader(
            train_path=data_config['train_path'],
            val_path=data_config['val_path'],
            test_path=data_config['test_path'],
            embedding_type=data_config['embedding_type'],
            sample_size=data_config['sample_size']
        )
        
        print("Loading datasets...")
        train_dataset, val_dataset, test_dataset = data_loader.load_data()
        
        print("Creating data loaders...")
        train_loader, val_loader, test_loader = data_loader.get_dataloaders(
            batch_size=data_config['batch_size'],
            balanced_sampling=True
        )
        
        print(f"Loaded data: {len(train_loader)} training batches")
        
        # Run the test
        results = test_distance_matrix_embedding_preserves_intrinsic_dim(
            train_loader, device, calculate_ph_dim
        )
        
    except Exception as e:
        print(f"❌ Could not load data automatically: {e}")
        print("\n📝 MANUAL SETUP NEEDED:")
        print("Please check:")
        print("1. Data file paths exist:")
        print("   - data/snli_train_embeddings.pt")
        print("   - data/snli_val_embeddings.pt") 
        print("   - data/snli_test_embeddings.pt")
        print("2. GlobalDataLoader import works")
        print("\nOr manually create your train_loader and call:")
        print("results = test_distance_matrix_embedding_preserves_intrinsic_dim(train_loader, device, calculate_ph_dim)")

if __name__ == "__main__":
    main()