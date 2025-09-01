"""
Gromov-Wasserstein Loss Function for Topological Autoencoder
Replaces Moor topological loss with distance matrix embedding approach
"""

import torch
import torch.nn as nn
import numpy as np
from torch_topological.nn import VietorisRipsComplex
from distance_matrix_test import calculate_ph_dim


class GromovWassersteinTopologicalLoss(nn.Module):
    """
    Gromov-Wasserstein based topological loss function combining:
    1. Distance matrix embedding preprocessing 
    2. Persistence diagram computation
    3. Gromov-Wasserstein distance comparison
    4. Distance preservation loss (MDS-style)
    """
    
    def __init__(self, gw_weight=1.0, distance_weight=0.1, distance_type='stress', 
                 max_dimension=1, distance_metric='euclidean', min_persistence=0.2, significance_weight=10.0):
        super().__init__()
        self.gw_weight = gw_weight
        self.distance_weight = distance_weight
        self.distance_type = distance_type
        self.max_dimension = max_dimension
        self.distance_metric = distance_metric
        self.min_persistence = min_persistence      # NEW
        self.significance_weight = significance_weight  # NEW
        
        # Initialize VietorisRips complex for differentiable persistence computation
        self.vr_complex = VietorisRipsComplex(
            dim=max_dimension,
            p=2,
            keep_infinite_features=False
        )
        
        print(f"GromovWassersteinTopologicalLoss initialized:")
        print(f"  GW weight: {gw_weight}")
        print(f"  Distance weight: {distance_weight}")
        print(f"  Distance type: {distance_type}")
        print(f"  Max dimension: {max_dimension}")
        print(f"  Distance metric: {distance_metric}")

    
    def distance_matrix_embedding(self, features):
        """
        Lift points to distance matrix embedding space
        
        Args:
            features: [batch_size, feature_dim] tensor
            
        Returns:
            distance_matrix: [batch_size, batch_size] distance matrix
        """
        if self.distance_metric == 'euclidean':
            # Use PyTorch's differentiable cdist for Euclidean distance
            distance_matrix = torch.cdist(features, features, p=2)
        elif self.distance_metric == 'cosine':
            # Compute cosine distance using PyTorch operations
            features_norm = torch.nn.functional.normalize(features, p=2, dim=1)
            cosine_sim = torch.mm(features_norm, features_norm.t())
            distance_matrix = 1.0 - cosine_sim
        else:
            # For other metrics, fall back to cdist with appropriate p-norm
            if self.distance_metric == 'manhattan':
                distance_matrix = torch.cdist(features, features, p=1)
            else:
                # Default to Euclidean if unknown metric
                print(f"Warning: Unknown metric '{self.distance_metric}', using Euclidean")
                distance_matrix = torch.cdist(features, features, p=2)
        

        # if distance_matrix.max() > 0:
        #     distance_matrix = distance_matrix / distance_matrix.max()

        # print(f"    Distance matrix: min={distance_matrix.min():.3f}, max={distance_matrix.max():.3f}, mean={distance_matrix.mean():.3f}")

        return distance_matrix


    def compute_persistence_diagram(self, distance_matrix):
        """
        Compute persistence diagrams from distance matrix using torch_topological
        
        Args:
            distance_matrix: [n, n] distance matrix tensor
            
        Returns:
            persistence_diagrams: Dict with keys 'h0', 'h1' containing diagrams
        """
        try:
            # Use torch_topological for differentiable persistence computation
            persistence_info = self.vr_complex(distance_matrix, treat_as_distances=True)
            
            diagrams = {}
            
            # Extract H0 (connected components) and H1 (loops) diagrams
            for dim in range(min(len(persistence_info), self.max_dimension + 1)):
                if dim < len(persistence_info):
                    diagram = persistence_info[dim].diagram
                    
                    # Filter out infinite features (death time = inf)
                    finite_mask = torch.isfinite(diagram[:, 1])
                    diagram = diagram[finite_mask]
                    
                    # Store with appropriate key
                    if dim == 0:
                        diagrams['h0'] = diagram
                    elif dim == 1:
                        diagrams['h1'] = diagram
                else:
                    # No features found for this dimension
                    if dim == 0:
                        print("ERROR - No diagram found for H0")
                        raise
                    elif dim == 1:
                        print("ERROR - No diagram found for H1")
                        raise
            
            # Ensure both H0 and H1 are present
            if 'h0' not in diagrams:
                print("ERROR - No diagram found for H0")
                raise

            if 'h1' not in diagrams:
                print("ERROR - No diagram found for H1")
                raise

            # for dim_name, diagram in diagrams.items():
            #     if len(diagram) > 0:
            #         births = diagram[:, 0]
            #         deaths = diagram[:, 1]
            #         print(f"    {dim_name}: {len(diagram)} features, birth_range=[{births.min():.3f}, {births.max():.3f}], death_range=[{deaths.min():.3f}, {deaths.max():.3f}]")
                
            return diagrams
                
        except Exception as e:
            print(f"Warning: Persistence computation failed: {e}")
            raise


    def sinkhorn_approximation(self, diagram1, diagram2, reg=1.0, max_iter=100):
        """
        Sinkhorn approximation of Wasserstein distance between persistence diagrams
        
        Args:
            diagram1, diagram2: [n, 2] persistence diagrams
            reg: Regularization parameter
            max_iter: Maximum iterations
            
        Returns:
            distance: Scalar distance value
        """
        if len(diagram1) == 0 or len(diagram2) == 0:
            # Handle empty diagrams
            return torch.tensor(0.0, device=diagram1.device, requires_grad=True)
        
        # Pad diagrams to same size
        max_size = max(len(diagram1), len(diagram2))
        
        # Pad with diagonal points (birth=death=0)
        if len(diagram1) < max_size:
            padding = torch.zeros((max_size - len(diagram1), 2), 
                                device=diagram1.device, dtype=diagram1.dtype)
            diagram1 = torch.cat([diagram1, padding], dim=0)
        
        if len(diagram2) < max_size:
            padding = torch.zeros((max_size - len(diagram2), 2), 
                                device=diagram2.device, dtype=diagram2.dtype)
            diagram2 = torch.cat([diagram2, padding], dim=0)
        
        # Compute cost matrix (Euclidean distance between points)
        cost_matrix = torch.cdist(diagram1, diagram2, p=2)
        
        # Sinkhorn iterations
        n, m = cost_matrix.shape
        u = torch.ones(n, device=cost_matrix.device) / n
        v = torch.ones(m, device=cost_matrix.device) / m
        
        K = torch.exp(-cost_matrix / reg)
        
        for _ in range(max_iter):
            u = 1.0 / (K @ v + 1e-8)
            v = 1.0 / (K.T @ u + 1e-8)
        
        # Compute transport plan and distance
        transport_plan = u.unsqueeze(1) * K * v.unsqueeze(0)
        distance = torch.sum(transport_plan * cost_matrix)
        
        return distance

    def filter_meaningful_features(self, diagram, min_persistence=0.01):
        """Only keep features with substantial persistence"""
        if len(diagram) == 0:
            return diagram
    
        persistence = diagram[:, 1] - diagram[:, 0]
        # print(f"    Persistence range: [{persistence.min():.3f}, {persistence.max():.3f}], mean: {persistence.mean():.3f}")

        meaningful_mask = persistence >= min_persistence
        filtered = diagram[meaningful_mask]
    
        # print(f"    Filtered: {len(diagram)} → {len(filtered)} features (min_pers={min_persistence})")
        return filtered

    def persistence_significance_loss(self, latent_diagrams):
        """Reward creation of substantial persistence features"""
        total_significance = torch.tensor(0.0, device=list(latent_diagrams.values())[0].device, requires_grad=True)
    
        for dim_name, diagram in latent_diagrams.items():
            if len(diagram) == 0:
                continue
            
            persistence = diagram[:, 1] - diagram[:, 0]
        
            # Reward substantial persistence features
            if len(persistence) > 0:
                # Use negative loss because we want to maximize substantial features
                significance = -torch.mean(persistence) * 10 # Simple linear reward
                total_significance = total_significance + significance
            
                # print(f"    {dim_name} significance reward: {significance.item():.6f}")
    
        return total_significance
    
    def gromov_wasserstein_distance(self, diagrams1, diagrams2, min_persistence=0.05):
        """
        Compute combined Gromov-Wasserstein distance across all dimensions
        
        Args:
            diagrams1, diagrams2: Dict containing persistence diagrams for each dimension
            
        Returns:
            total_distance: Combined distance across H0 and H1
        """

        # Filter both diagrams to only include meaningful features
        filtered_diagrams1 = {}
        filtered_diagrams2 = {}
    
        for dim_key in ['h0', 'h1']:
            filtered_diagrams1[dim_key] = self.filter_meaningful_features(
                diagrams1[dim_key], min_persistence
            )
            filtered_diagrams2[dim_key] = self.filter_meaningful_features(
                diagrams2[dim_key], min_persistence
            )
    
        # Compute GW distance on filtered diagrams
        h0_dist = self.sinkhorn_approximation(filtered_diagrams1['h0'], filtered_diagrams2['h0'])
        h1_dist = self.sinkhorn_approximation(filtered_diagrams1['h1'], filtered_diagrams2['h1'])
    
        # Weight H1 more heavily
        total_distance = 0.5 * h0_dist + 0.5 * h1_dist
    
        # print(f"    Filtered GW: H0={h0_dist.item():.6f}, H1={h1_dist.item():.6f}")
    
        return total_distance
        

    def distance_preservation_loss(self, input_features, latent_features):
        """
        Distance preservation loss functions
        
        Args:
            input_features: [batch_size, input_dim] 
            latent_features: [batch_size, latent_dim]
            
        Returns:
            loss: Scalar loss value
        """
        # Compute distance matrices
        input_distances = torch.cdist(input_features, input_features, p=2)
        latent_distances = torch.cdist(latent_features, latent_features, p=2)
        
        if self.distance_type == 'direct':
            # Direct L2 loss between distance matrices
            return torch.mean((input_distances - latent_distances) ** 2)
        
        elif self.distance_type == 'stress':
            # Stress function (normalized MDS)
            numerator = torch.sum((input_distances - latent_distances) ** 2)
            denominator = torch.sum(input_distances ** 2)
            return numerator / (denominator + 1e-8)
        
        elif self.distance_type == 'sammon':
            # Sammon's mapping (weighted by distance)
            weights = 1.0 / (input_distances + 1e-8)
            numerator = torch.sum(weights * (input_distances - latent_distances) ** 2)
            denominator = torch.sum(weights * input_distances)
            return numerator / (denominator + 1e-8)
        
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")
    
    def forward(self, input_features, latent_features):
        """
        Compute combined Gromov-Wasserstein and distance preservation loss
        
        Args:
            input_features: [batch_size, input_dim] original space features
            latent_features: [batch_size, latent_dim] latent space features
            
        Returns:
            total_loss: Combined loss value
            gw_loss: Gromov-Wasserstein component  
            dist_loss: Distance preservation component
        """
        # Step 1: Distance matrix embedding
        input_dist_matrix = self.distance_matrix_embedding(input_features)
        latent_dist_matrix = self.distance_matrix_embedding(latent_features)

        # DEBUG
        input_ph_dim = calculate_ph_dim(input_dist_matrix.detach().cpu().numpy(), metric='precomputed')
        latent_ph_dim = calculate_ph_dim(latent_dist_matrix.detach().cpu().numpy(), metric='precomputed')
        # print(f"Input PH dim: {input_ph_dim:.2f}, Latent PH dim: {latent_ph_dim:.2f}")
        
        # Step 2: Compute persistence diagrams (both H0 and H1)
        input_persistence = self.compute_persistence_diagram(input_dist_matrix)
        latent_persistence = self.compute_persistence_diagram(latent_dist_matrix)
        
        # Step 3: Gromov-Wasserstein loss (topological) - comparing both H0 and H1
        gw_loss = self.gromov_wasserstein_distance(input_persistence, latent_persistence)
        
        # Step 4: Distance preservation loss (geometric)
        significance_reward = self.persistence_significance_loss(latent_persistence)

        dist_loss = self.distance_preservation_loss(input_features, latent_features)
        
        # Step 5: Combined loss
        total_loss = self.gw_weight * gw_loss + self.distance_weight * dist_loss + self.significance_weight * significance_reward

        # print(f"  Enhanced Loss Components:")
        # print(f"    Raw GW loss (filtered): {gw_loss.item():.6f}")
        # print(f"    Significance reward: {significance_reward.item():.6f}")
        # print(f"    Distance loss: {dist_loss.item():.6f}")
        # print(f"    Total loss: {total_loss.item():.6f}")
        
        return total_loss, gw_loss, dist_loss


def test_gromov_wasserstein_loss():
    """Test the Gromov-Wasserstein loss function"""
    print("Testing GromovWassersteinTopologicalLoss...")
    
    # Create test data
    batch_size = 20  # Smaller for testing
    input_dim = 100  # Smaller for testing
    latent_dim = 10
    
    # Generate synthetic data
    input_features = torch.randn(batch_size, input_dim, requires_grad=True)
    latent_features = torch.randn(batch_size, latent_dim, requires_grad=True)
    
    # Create loss function
    loss_fn = GromovWassersteinTopologicalLoss(
        gw_weight=10.0,
        distance_weight=0.01,
        distance_type='stress'
    )
    
    # Compute loss
    total_loss, gw_loss, dist_loss = loss_fn(input_features, latent_features)
    
    print(f"Total loss: {total_loss.item():.6f}")
    print(f"GW loss: {gw_loss.item():.6f}")
    print(f"Distance loss: {dist_loss.item():.6f}")
    
    # Test gradient flow
    total_loss.backward()
    print(f"Input gradients exist: {input_features.grad is not None}")
    print(f"Latent gradients exist: {latent_features.grad is not None}")
    
    print("Test passed!")


if __name__ == "__main__":
    test_gromov_wasserstein_loss()
            