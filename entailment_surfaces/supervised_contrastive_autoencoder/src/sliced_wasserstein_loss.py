import torch
import torch.nn as nn
from torch_topological.nn import SlicedWassersteinDistance  # Changed from Kernel to Distance
from torch_topological.nn import VietorisRipsComplex


class SlicedWassersteinTopologicalLoss(nn.Module):
    """
    Topological loss using Sliced Wasserstein distance instead of Gromov-Wasserstein
    """
    
    def __init__(self, sw_weight=1.0, distance_weight=0.1, 
                 max_dimension=1, distance_metric='euclidean',
                 num_directions=10):  # Removed sigma parameter
        super().__init__()
        self.sw_weight = sw_weight
        self.distance_weight = distance_weight
        self.max_dimension = max_dimension
        self.distance_metric = distance_metric
        
        # Store device - will be set when first called
        self.device = None
        
        # Initialize VietorisRips complex (same as before)
        self.vr_complex = VietorisRipsComplex(
            dim=max_dimension,
            p=2,
            keep_infinite_features=False
        )
        
        # NEW: Sliced Wasserstein distance - will be moved to device later
        self.sw_distance = SlicedWassersteinDistance(
            num_directions=num_directions
        )
        
        print(f"SlicedWassersteinTopologicalLoss initialized:")
        print(f"  SW weight: {sw_weight}")
        print(f"  Distance weight: {distance_weight}")
        print(f"  Num directions: {num_directions}")

    def distance_matrix_embedding(self, features):
        """Same as before - this works perfectly"""
        if self.distance_metric == 'euclidean':
            distance_matrix = torch.cdist(features, features, p=2)
        elif self.distance_metric == 'cosine':
            features_norm = torch.nn.functional.normalize(features, p=2, dim=1)
            cosine_sim = torch.mm(features_norm, features_norm.t())
            distance_matrix = 1.0 - cosine_sim
        else:
            distance_matrix = torch.cdist(features, features, p=2)
        
        # Normalize distance matrix to [0, 1] range (this worked well)
        if distance_matrix.max() > 0:
            distance_matrix = distance_matrix / distance_matrix.max()
        
        return distance_matrix

    def compute_persistence_diagram(self, distance_matrix):
        """Same as before - returns PersistenceInformation objects"""
        try:
            # Don't try to modify the persistence objects - torch_topological handles device placement
            persistence_info = self.vr_complex(distance_matrix, treat_as_distances=True)
            return persistence_info
                
        except Exception as e:
            print(f"Persistence computation failed: {e}")
            # Return empty list
            return []

    def compute_sliced_wasserstein_distance(self, input_persistence, latent_persistence):
        """
        NEW: Use SlicedWassersteinDistance instead of GW distance
        """
        try:
            # Check if we have valid persistence information
            if len(input_persistence) == 0 or len(latent_persistence) == 0:
                print("    Warning: Empty persistence diagram")
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Debug: Check what's in the persistence objects
            # print(f"    Input persistence: {len(input_persistence)} dimensions")
            # print(f"    Latent persistence: {len(latent_persistence)} dimensions")
            
            # Check device of persistence diagrams
            # for i, info in enumerate(input_persistence):
            #     if hasattr(info, 'diagram') and len(info.diagram) > 0:
            #         print(f"    Input dim {i} diagram device: {info.diagram.device}, shape: {info.diagram.shape}")
                    
            # for i, info in enumerate(latent_persistence):
            #     if hasattr(info, 'diagram') and len(info.diagram) > 0:
            #         print(f"    Latent dim {i} diagram device: {info.diagram.device}, shape: {info.diagram.shape}")
            
            # Check if any diagrams actually have features
            input_has_features = any(len(info.diagram) > 0 for info in input_persistence if hasattr(info, 'diagram'))
            latent_has_features = any(len(info.diagram) > 0 for info in latent_persistence if hasattr(info, 'diagram'))
            
            if not input_has_features or not latent_has_features:
                print("    Warning: No topological features found")
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # print(f"    SW distance module device: {next(self.sw_distance.parameters()).device if list(self.sw_distance.parameters()) else 'no params'}")
            
            # The SlicedWassersteinDistance should handle device placement automatically
            sw_distance = self.sw_distance(input_persistence, latent_persistence)
            
            # print(f"    Sliced Wasserstein distance: {sw_distance.item():.6f}")
            
            return sw_distance
            
        except Exception as e:
            print(f"Sliced Wasserstein computation failed: {e}")
            import traceback
            traceback.print_exc()  # This will help debug the exact error
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def distance_preservation_loss(self, input_features, latent_features):
        """Same MDS-style loss as before"""
        input_distances = torch.cdist(input_features, input_features, p=2)
        latent_distances = torch.cdist(latent_features, latent_features, p=2)
        
        # Normalize to same scale
        input_distances = input_distances / (input_distances.max() + 1e-8)
        latent_distances = latent_distances / (latent_distances.max() + 1e-8)
        
        # Stress function
        numerator = torch.sum((input_distances - latent_distances) ** 2)
        denominator = torch.sum(input_distances ** 2)
        
        return numerator / (denominator + 1e-8)

    def forward(self, input_features, latent_features):
        """
        Main forward pass using Sliced Wasserstein - all topological computation on CPU
        """
        batch_size = input_features.shape[0]
        
        # Set device on first call - FORCE TO CPU for torch_topological compatibility
        if self.device is None:
            self.device = torch.device('cpu')
            self.vr_complex = self.vr_complex.to(self.device)
            self.sw_distance = self.sw_distance.to(self.device)
        
        # Skip if batch too small
        if batch_size < 10:
            return torch.tensor(0.0, device=input_features.device, requires_grad=True)
        
        try:
            # Step 1: Distance matrix embedding (move to CPU for persistence computation)
            input_dist_matrix = self.distance_matrix_embedding(input_features).cpu()
            latent_dist_matrix = self.distance_matrix_embedding(latent_features).cpu()
            
            # Step 2: Compute persistence diagrams (on CPU)
            input_persistence = self.compute_persistence_diagram(input_dist_matrix)
            latent_persistence = self.compute_persistence_diagram(latent_dist_matrix)
            
            # Step 3: Sliced Wasserstein distance (all on CPU)
            sw_loss = self.compute_sliced_wasserstein_distance(input_persistence, latent_persistence)
            
            # Step 4: Distance preservation loss (keep on original device)
            dist_loss = self.distance_preservation_loss(input_features, latent_features)
            
            # Step 5: Combined loss - move SW loss back to original device
            sw_loss = sw_loss.to(input_features.device)
            total_loss = self.sw_weight * sw_loss + self.distance_weight * dist_loss
            
            return total_loss, sw_loss, dist_loss
            
        except Exception as e:
            print(f"Error in SW loss computation: {e}")
            penalty = torch.tensor(0.001, device=input_features.device, requires_grad=True)
            return penalty, penalty, penalty


def test_sliced_wasserstein_loss():
    """Test the new loss function"""
    print("Testing SlicedWassersteinTopologicalLoss...")
    
    batch_size = 100
    input_dim = 1536
    latent_dim = 75
    
    input_features = torch.randn(batch_size, input_dim, requires_grad=True)
    latent_features = torch.randn(batch_size, latent_dim, requires_grad=True)
    
    # Test with different configurations
    loss_fn = SlicedWassersteinTopologicalLoss(
        sw_weight=1.0,
        distance_weight=1.0,
        num_directions=10,
        sigma=1.0
    )
    
    total_loss, sw_loss, dist_loss = loss_fn(input_features, latent_features)
    
    print(f"Total loss: {total_loss.item():.6f}")
    print(f"SW loss: {sw_loss.item():.6f}")
    print(f"Distance loss: {dist_loss.item():.6f}")
    
    # Test gradients
    total_loss.backward()
    print(f"Gradients OK: {input_features.grad is not None}")
    
    print("Test passed!")

if __name__ == "__main__":
    test_sliced_wasserstein_loss()