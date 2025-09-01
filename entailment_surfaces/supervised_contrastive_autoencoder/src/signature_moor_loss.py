import torch
import torch.nn as nn
from torch_topological.nn import VietorisRipsComplex, SignatureLoss


class MoorSignatureLossWithLifting(nn.Module):
    """
    Moor et al. topological signature loss using lifted distance matrices
    This allows comparison between different dimensional spaces (1536D vs 75D)
    """
    
    def __init__(self, max_dimension=0, distance_metric='euclidean',
                 p=2, normalise=True, dimensions=0):
        super().__init__()
        self.max_dimension = max_dimension
        self.distance_metric = distance_metric
        
        # Initialize VietorisRips complex
        self.vr_complex = VietorisRipsComplex(
            dim=max_dimension,
            p=2,
            keep_infinite_features=False
        )
        
        # Initialize Moor's SignatureLoss from torch_topological
        self.signature_loss = SignatureLoss(
            p=p,
            normalise=normalise,
            dimensions=dimensions
        )
        
        print(f"MoorSignatureLossWithLifting initialized:")
        print(f"  p-norm: {p}")
        print(f"  Normalise: {normalise}")
        print(f"  Dimensions: {dimensions}")

    def distance_matrix_embedding(self, features):
        """
        Lift points to distance matrix embedding space
        This makes 1536D and 75D spaces comparable
        """
        if self.distance_metric == 'euclidean':
            distance_matrix = torch.cdist(features, features, p=2)
        elif self.distance_metric == 'cosine':
            features_norm = torch.nn.functional.normalize(features, p=2, dim=1)
            cosine_sim = torch.mm(features_norm, features_norm.t())
            distance_matrix = 1.0 - cosine_sim
        else:
            distance_matrix = torch.cdist(features, features, p=2)
        
        # Normalize distance matrix to [0, 1] range
        if distance_matrix.max() > 0:
            distance_matrix = distance_matrix / distance_matrix.max()
        
        return distance_matrix

    def compute_persistence_information(self, distance_matrix):
        """Compute persistence information for the lifted space"""
        try:
            persistence_info = self.vr_complex(distance_matrix, treat_as_distances=True)
            return persistence_info
                
        except Exception as e:
            print(f"Persistence computation failed: {e}")
            return []

    def forward(self, input_features, latent_features):
        """
        Forward pass using Moor's SignatureLoss on lifted distance matrices
        """
        batch_size = input_features.shape[0]
        
        if batch_size < 10:
            return torch.tensor(0.0, device=input_features.device, requires_grad=True)
        
        try:
            # Step 1: Lift both spaces to distance matrix embeddings
            # This makes them comparable despite different dimensions
            input_lifted = self.distance_matrix_embedding(input_features)
            latent_lifted = self.distance_matrix_embedding(latent_features)
            
            # Step 2: Compute persistence information for both lifted spaces
            input_persistence = self.compute_persistence_information(input_lifted)
            latent_persistence = self.compute_persistence_information(latent_lifted)
            
            # Step 3: Apply Moor's SignatureLoss
            # X = (point_cloud, persistence_info), Y = (point_cloud, persistence_info)
            input_tuple = (input_lifted, input_persistence)
            latent_tuple = (latent_lifted, latent_persistence)
            
            signature_loss = self.signature_loss(input_tuple, latent_tuple)
            
            return signature_loss
            
        except Exception as e:
            print(f"Error in signature loss computation: {e}")
            penalty = torch.tensor(0.001, device=input_features.device, requires_grad=True)
            return penalty


def test_moor_signature_loss():
    """Test the Moor signature loss with lifting"""
    print("Testing MoorSignatureLossWithLifting...")
    
    batch_size = 100
    input_dim = 1536
    latent_dim = 75
    
    input_features = torch.randn(batch_size, input_dim, requires_grad=True)
    latent_features = torch.randn(batch_size, latent_dim, requires_grad=True)
    
    if torch.cuda.is_available():
        input_features = input_features.cuda()
        latent_features = latent_features.cuda()
    
    loss_fn = MoorSignatureLossWithLifting(
        signature_weight=1.0,
        distance_weight=1.0,
        p=2,
        normalise=True,
        dimensions=(0, 1)  # Both H0 and H1
    )
    
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()
    
    total_loss, signature_loss, dist_loss = loss_fn(input_features, latent_features)
    
    print(f"Total loss: {total_loss.item():.6f}")
    print(f"Signature loss: {signature_loss.item():.6f}")
    print(f"Distance loss: {dist_loss.item():.6f}")
    
    # Test gradients
    total_loss.backward()
    print(f"Gradients OK: {input_features.grad is not None}")
    
    print("Test passed!")

if __name__ == "__main__":
    test_moor_signature_loss()