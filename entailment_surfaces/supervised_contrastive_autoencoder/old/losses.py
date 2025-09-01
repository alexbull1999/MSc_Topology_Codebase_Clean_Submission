"""
Supervised Contrastive Loss Functions
Loss functions for training the contrastive autoencoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss implementation
    
    Pulls samples from the same class together while pushing samples 
    from different classes apart in the latent space.
    
    Args:
        temperature: Temperature parameter for scaling similarities
        base_temperature: Base temperature for normalization
    """
    
    def __init__(self, temperature=0.1, base_temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self, features, labels):
        """
        Compute supervised contrastive loss
        
        Args:
            features: Latent features [batch_size, feature_dim]
            labels: Class labels [batch_size]
            
        Returns:
            loss: Scalar loss value
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features to unit sphere
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create label masks
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(device)
        mask_negative = 1 - mask_positive
        
        # Remove diagonal (self-similarity)
        mask_positive = mask_positive - torch.eye(batch_size).to(device)
        
        # For numerical stability, subtract max
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log probabilities
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean log probability over positive pairs
        # Add a small epsilon to the denominator to prevent division by zero
        epsilon = 1e-8
        mean_log_prob_pos = (mask_positive * log_prob).sum(1) / (mask_positive.sum(1) + epsilon)
        
        # Loss is negative log probability
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        
        # Average over batch, excluding samples with no positive pairs
        valid_samples = mask_positive.sum(1) > 0
        loss = loss[valid_samples].mean()
        
        return loss


class TripletContrastiveLoss(nn.Module):
    """
    Triplet-based contrastive loss
    For each anchor, ensures d(anchor, positive) < d(anchor, negative) - margin
    Works directly on distances in latent space (no normalization)
    """
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, features, labels):
        """
        Triplet contrastive loss
        
        Args:
            features: Latent features [batch_size, feature_dim] - raw latent vectors
            labels: Class labels [batch_size]
        """
        batch_size = features.shape[0]
        device = features.device
        
        # For each sample, find positive and negative examples
        total_loss = 0.0
        num_triplets = 0
        
        for i in range(batch_size):
            anchor_label = labels[i]
            anchor_features = features[i:i+1]  # Keep batch dimension
            
            # Find positive samples (same class, different from anchor)
            pos_mask = (labels == anchor_label) & (torch.arange(batch_size, device=device) != i)
            if not pos_mask.any():
                continue
                
            # Find negative samples (different class)
            neg_mask = labels != anchor_label
            if not neg_mask.any():
                continue
            
            pos_features = features[pos_mask]
            neg_features = features[neg_mask]
            
            # Compute distances
            pos_distances = torch.norm(anchor_features - pos_features, dim=1)
            neg_distances = torch.norm(anchor_features - neg_features, dim=1)
            
            # For each positive, find hardest negative
            for pos_dist in pos_distances:
                # Hardest negative = closest negative to anchor
                hardest_neg_dist = neg_distances.min()
                
                # Triplet loss: want pos_dist + margin < neg_dist
                triplet_loss = torch.clamp(pos_dist - hardest_neg_dist + self.margin, min=0)
                total_loss += triplet_loss
                num_triplets += 1
        
        if num_triplets > 0:
            return total_loss / num_triplets
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


class ProperContrastiveLoss(nn.Module):
    """
    Proper contrastive loss: pull same-class together, push different-class apart
    Works directly on distances in latent space
    """
    
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin  # Minimum distance between different classes
    
    def forward(self, features, labels):
        """
        Pull same-class samples together, push different-class apart
        
        Args:
            features: Latent features [batch_size, feature_dim] - raw latent vectors
            labels: Class labels [batch_size]
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Compute pairwise distances (not similarities!)
        distances = torch.cdist(features, features, p=2)
        
        # Create masks
        labels_expanded = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels_expanded, labels_expanded.T).float().to(device)
        mask_no_diagonal = 1 - torch.eye(batch_size, device=device)
        mask_positive = mask_positive * mask_no_diagonal
        mask_negative = (1 - torch.eq(labels_expanded, labels_expanded.T).float().to(device)) * mask_no_diagonal
        
        # Get positive and negative distances
        pos_distances = distances[mask_positive.bool()]
        neg_distances = distances[mask_negative.bool()]
        
        if len(pos_distances) == 0 or len(neg_distances) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # CONTRASTIVE LOSS:
        # 1. Pull positive pairs together (minimize distance)
        pos_loss = pos_distances.mean()
        
        # 2. Push negative pairs apart (maximize distance, but with margin)
        neg_loss = torch.clamp(self.margin - neg_distances, min=0).mean()
        
        total_loss = pos_loss + neg_loss
        return total_loss



class FullDatasetContrastiveLoss(nn.Module):
    """
    Contrastive loss that uses the entire dataset for each update
    """
    
    def __init__(self, margin=2.0, update_frequency=10):
        super().__init__()
        self.margin = margin
        self.update_frequency = update_frequency
        self.global_features = None
        self.global_labels = None
        self.batch_count = 0
    
    def update_global_dataset(self, dataloader, model, device):
        """
        Extract features for the entire dataset
        """
        print("Extracting features for entire dataset...")
        model.eval()
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                embeddings = batch['embeddings'].to(device)
                labels = batch['labels'].to(device)
                
                # Get latent features
                latent, _ = model(embeddings)
                
                all_features.append(latent.cpu())
                all_labels.append(labels.cpu())
        
        self.global_features = torch.cat(all_features, dim=0).to(device)
        self.global_labels = torch.cat(all_labels, dim=0).to(device)
        
        print(f"Global dataset: {self.global_features.shape[0]} samples")
        
        # Analyze global separation
        self._analyze_global_separation()
        
        model.train()
    
    def _analyze_global_separation(self):
        """
        Analyze separation in the global dataset
        """
        if self.global_features is None:
            return
        
        distances = torch.cdist(self.global_features, self.global_features, p=2)
        
        labels_expanded = self.global_labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels_expanded, labels_expanded.T).float()
        mask_no_diagonal = 1 - torch.eye(len(self.global_labels), device=self.global_features.device)
        mask_positive = mask_positive * mask_no_diagonal
        mask_negative = (1 - torch.eq(labels_expanded, labels_expanded.T).float()) * mask_no_diagonal
        
        pos_distances = distances[mask_positive.bool()]
        neg_distances = distances[mask_negative.bool()]
        
        if len(pos_distances) > 0 and len(neg_distances) > 0:
            print(f"GLOBAL ANALYSIS:")
            print(f"  Pos distances: {pos_distances.mean().item():.3f} ± {pos_distances.std().item():.3f}")
            print(f"  Neg distances: {neg_distances.mean().item():.3f} ± {neg_distances.std().item():.3f}")
            print(f"  Separation ratio: {(neg_distances.mean() / pos_distances.mean()).item():.2f}x")
            print(f"  Gap: {(neg_distances.min() - pos_distances.max()).item():.3f}")
    
    def forward(self, current_features, current_labels):
        """
        Compute contrastive loss using global dataset
        """
        self.batch_count += 1
        
        if self.global_features is None:
            # Fallback to current batch if global not available
            return self._compute_contrastive_loss(current_features, current_labels)
        
        # Subsample from global dataset to make computation feasible
        global_size = self.global_features.shape[0]
        if global_size > 3000:  # Subsample if too large
            indices = torch.randperm(global_size)[:3000]
            global_features_subset = self.global_features[indices]
            global_labels_subset = self.global_labels[indices]
        else:
            global_features_subset = self.global_features
            global_labels_subset = self.global_labels
        
        # Combine current batch with global samples
        all_features = torch.cat([current_features, global_features_subset], dim=0)
        all_labels = torch.cat([current_labels, global_labels_subset], dim=0)
        
        print(f"Full dataset contrastive: {len(current_features)} current + {len(global_features_subset)} global = {len(all_features)} total")
        
        return self._compute_contrastive_loss(all_features, all_labels)
    
    def _compute_contrastive_loss(self, features, labels):
        """
        Compute contrastive loss on feature set
        """
        batch_size = features.shape[0]
        device = features.device
        
        distances = torch.cdist(features, features, p=2)
        
        labels_expanded = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels_expanded, labels_expanded.T).float().to(device)
        mask_no_diagonal = 1 - torch.eye(batch_size, device=device)
        mask_positive = mask_positive * mask_no_diagonal
        mask_negative = (1 - torch.eq(labels_expanded, labels_expanded.T).float().to(device)) * mask_no_diagonal
        
        pos_distances = distances[mask_positive.bool()]
        neg_distances = distances[mask_negative.bool()]
        
        if len(pos_distances) == 0 or len(neg_distances) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        pos_loss = pos_distances.mean()
        neg_loss = torch.clamp(self.margin - neg_distances, min=0).mean()
        
        total_loss = pos_loss + neg_loss
        
        # Debug current batch separation
        separation_ratio = neg_distances.mean() / pos_distances.mean()
        gap = neg_distances.min() - pos_distances.max()
        print(f"  Full dataset distances: Pos={pos_distances.mean():.3f}, Neg={neg_distances.mean():.3f}, Ratio={separation_ratio:.2f}x, Gap={gap:.3f}")
        
        return total_loss


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for autoencoder
    
    Args:
        loss_type: Type of reconstruction loss ('mse' or 'l1')
    """
    
    def __init__(self, loss_type='mse'):
        super(ReconstructionLoss, self).__init__()
        self.loss_type = loss_type
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, reconstructed, original):
        """
        Compute reconstruction loss
        
        Args:
            reconstructed: Reconstructed embeddings [batch_size, input_dim]
            original: Original embeddings [batch_size, input_dim]
            
        Returns:
            loss: Scalar loss value
        """
        return self.loss_fn(reconstructed, original)


class CombinedLoss(nn.Module):
    """
    Combined loss function for contrastive autoencoder
    
    Combines supervised contrastive loss with reconstruction loss
    
    Args:
        contrastive_weight: Weight for contrastive loss component
        reconstruction_weight: Weight for reconstruction loss component
        temperature: Temperature for contrastive loss
        reconstruction_type: Type of reconstruction loss
    """
    
    def __init__(self, contrastive_weight=1.0, reconstruction_weight=1.0, 
                 temperature=0.1, reconstruction_type='mse', margin=1.0):
        super(CombinedLoss, self).__init__()
        
        self.contrastive_weight = contrastive_weight
        self.reconstruction_weight = reconstruction_weight 
        
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=0.1)
        self.reconstruction_loss = ReconstructionLoss(loss_type=reconstruction_type)

    # @staticmethod
    # def get_contrastive_beta(epoch, warmup_epochs=10, max_beta=2.0, schedule_type='linear', total_epochs=50):
    #     """
    #     Beta scheduling for contrastive loss weight
        
    #     Args:
    #         epoch: Current epoch number
    #         warmup_epochs: Number of epochs with pure reconstruction (β=0)
    #         max_beta: Maximum contrastive weight
    #         schedule_type: 'linear', 'cosine', or 'exponential'
    #     """
    #     if epoch < warmup_epochs:
    #         return 0.0  # Pure reconstruction phase
    
    #     # Calculate progress after warmup - USE total_epochs parameter
    #     progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    #     progress = min(1.0, progress)  # Clamp to [0, 1]
        
    #     if schedule_type == 'linear':
    #         return progress * max_beta
    #     elif schedule_type == 'cosine':
    #         import math
    #         return 0.5 * max_beta * (1 + math.cos(math.pi * (1 - progress)))
    #     elif schedule_type == 'exponential':
    #         return max_beta * (progress ** 2)
    #     else:
    #         return progress * max_beta

    def forward(self, latent_features, labels, reconstructed, original, contrastive_weight=None):
        """
        Compute combined loss
        Args:
            latent_features: Latent representations [batch_size, latent_dim]
            labels: Class labels [batch_size]
            reconstructed: Reconstructed embeddings [batch_size, input_dim]
            original: Original embeddings [batch_size, input_dim]
            contrastive_weight: Beta parameter for scheduling
        Returns:
            total_loss: Combined loss value
            contrastive_loss: Contrastive loss component
            reconstruction_loss: Reconstruction loss component
        """
        if contrastive_weight is None:
            contrastive_weight = self.contrastive_weight
        # Use dynamic weight instead of fixed weight
        contrastive_loss = self.contrastive_loss(latent_features, labels)
        reconstruction_loss = self.reconstruction_loss(reconstructed, original)
        
        total_loss = (contrastive_weight * contrastive_loss + 
                      self.reconstruction_weight * reconstruction_loss)

        reconstruction_loss = reconstruction_loss * self.reconstruction_weight #DEBUG 
        
        return total_loss, contrastive_loss, reconstruction_loss

    def get_distance_stats(self, latent_features, labels):
        """
        Get distance statistics for debugging
        Returns dictionary with positive/negative distance stats
        """
        batch_size = latent_features.shape[0]
        device = latent_features.device
        
        # Compute pairwise distances
        distances = torch.cdist(latent_features, latent_features, p=2)
        
        # Create masks
        labels_expanded = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels_expanded, labels_expanded.T).float().to(device)
        mask_no_diagonal = 1 - torch.eye(batch_size, device=device)
        mask_positive = mask_positive * mask_no_diagonal
        mask_negative = (1 - torch.eq(labels_expanded, labels_expanded.T).float().to(device)) * mask_no_diagonal
        
        # Get distance statistics
        pos_distances = distances[mask_positive.bool()]
        neg_distances = distances[mask_negative.bool()]
        
        if len(pos_distances) == 0 or len(neg_distances) == 0:
            return {
                'pos_mean': 0.0, 'pos_std': 0.0, 'pos_min': 0.0, 'pos_max': 0.0,
                'neg_mean': 0.0, 'neg_std': 0.0, 'neg_min': 0.0, 'neg_max': 0.0,
                'separation_ratio': 0.0, 'gap': 0.0
            }
        
        stats = {
            'pos_mean': pos_distances.mean().item(),
            'pos_std': pos_distances.std().item(),
            'pos_min': pos_distances.min().item(),
            'pos_max': pos_distances.max().item(),
            'neg_mean': neg_distances.mean().item(),
            'neg_std': neg_distances.std().item(),
            'neg_min': neg_distances.min().item(),
            'neg_max': neg_distances.max().item(),
            'separation_ratio': (neg_distances.mean() / pos_distances.mean()).item(),
            'gap': (neg_distances.min() - pos_distances.max()).item()
        }
        
        return stats





# Modified CombinedLoss to use full dataset
class FullDatasetCombinedLoss(nn.Module):
    """
    Combined loss using full dataset contrastive loss
    """
    
    def __init__(self, contrastive_weight=1.0, reconstruction_weight=0.0, 
                 margin=2.0, update_frequency=5):
        super().__init__()
        
        self.contrastive_weight = contrastive_weight
        self.reconstruction_weight = reconstruction_weight
        
        self.contrastive_loss = FullDatasetContrastiveLoss(margin, update_frequency)
        
        if reconstruction_weight > 0:
            from losses import ReconstructionLoss
            self.reconstruction_loss = ReconstructionLoss()
        else:
            self.reconstruction_loss = None
    
    def forward(self, latent_features, labels, reconstructed=None, original=None, contrastive_weight=None):
        """
        Compute combined loss with full dataset contrastive
        """
        if contrastive_weight is None:
            contrastive_weight = self.contrastive_weight
        
        # Full dataset contrastive loss
        contrastive_loss = self.contrastive_loss(latent_features, labels)
        
        # Reconstruction loss
        if self.reconstruction_loss is not None and reconstructed is not None and original is not None:
            reconstruction_loss = self.reconstruction_loss(reconstructed, original)
        else:
            reconstruction_loss = torch.tensor(0.0, device=latent_features.device)
        
        # Combined loss
        total_loss = (contrastive_weight * contrastive_loss + 
                     self.reconstruction_weight * reconstruction_loss)
        
        return total_loss, contrastive_loss, reconstruction_loss
    

def test_losses():
    """Test loss function implementations"""
    print("Testing Loss Functions")
    print("=" * 40)
    
    # Create test data
    batch_size = 32
    latent_dim = 75
    input_dim = 768
    
    # Generate random features and labels
    latent_features = torch.randn(batch_size, latent_dim)
    labels = torch.randint(0, 3, (batch_size,))  # 3 classes: 0, 1, 2
    original_embeddings = torch.randn(batch_size, input_dim)
    reconstructed_embeddings = torch.randn(batch_size, input_dim)
    
    print(f"Latent features shape: {latent_features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Original embeddings shape: {original_embeddings.shape}")
    print(f"Reconstructed embeddings shape: {reconstructed_embeddings.shape}")
    print()
    
    # Test individual losses
    print("Testing Individual Loss Components:")
    print("-" * 40)
    
    # Contrastive loss
    contrastive_loss_fn = SupervisedContrastiveLoss(temperature=0.1)
    contrastive_loss = contrastive_loss_fn(latent_features, labels)
    print(f"Contrastive Loss: {contrastive_loss.item():.4f}")
    
    # Reconstruction loss
    reconstruction_loss_fn = ReconstructionLoss(loss_type='mse')
    reconstruction_loss = reconstruction_loss_fn(reconstructed_embeddings, original_embeddings)
    print(f"Reconstruction Loss: {reconstruction_loss.item():.4f}")
    
    # Combined loss
    print("\nTesting Combined Loss:")
    print("-" * 40)
    
    combined_loss_fn = CombinedLoss(
        contrastive_weight=1.0,
        reconstruction_weight=1.0,
        temperature=0.1
    )
    
    total_loss, cont_loss, recon_loss = combined_loss_fn(
        latent_features, labels, reconstructed_embeddings, original_embeddings
    )
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Contrastive Component: {cont_loss.item():.4f}")
    print(f"Reconstruction Component: {recon_loss.item():.4f}")
    
    # Test gradient flow
    print("\nTesting Gradient Flow:")
    print("-" * 40)
    
    # Create simple model for gradient test
    test_model = nn.Linear(latent_dim, latent_dim)
    test_features = test_model(latent_features)
    
    # Compute loss and backpropagate
    loss, _, _ = combined_loss_fn(test_features, labels, reconstructed_embeddings, original_embeddings)
    loss.backward()
    
    # Check if gradients exist
    has_gradients = any(p.grad is not None for p in test_model.parameters())
    print(f"Gradients computed: {has_gradients}")
    
    if has_gradients:
        grad_norms = [p.grad.norm().item() for p in test_model.parameters() if p.grad is not None]
        print(f"Gradient norms: {grad_norms}")
    
    print("\nTesting different class distributions:")
    print("-" * 40)
    
    # Test with different label distributions
    test_cases = [
        torch.zeros(batch_size, dtype=torch.long),  # All same class
        torch.randint(0, 3, (batch_size,)),         # Random distribution
        torch.cat([torch.zeros(10), torch.ones(10), torch.full((12,), 2)]).long()  # Balanced
    ]
    
    for i, test_labels in enumerate(test_cases):
        try:
            loss = contrastive_loss_fn(latent_features, test_labels)
            print(f"Test case {i+1}: Loss = {loss.item():.4f}")
        except Exception as e:
            print(f"Test case {i+1}: Error - {e}")
    
    print("\nLoss function tests completed!")


if __name__ == "__main__":
    test_losses()
