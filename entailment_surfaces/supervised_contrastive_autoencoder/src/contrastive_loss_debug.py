import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImprovedContrastiveLoss(nn.Module):
    """
    Improved contrastive loss with relative optimization
    Focus on ratios rather than absolute distances
    """
    
    def __init__(self, margin_ratio=2.0):
        super().__init__()
        self.margin_ratio = margin_ratio  # neg_dist should be margin_ratio * pos_dist
    
    def forward(self, features, labels):
        batch_size = features.shape[0]
        device = features.device
        
        # Compute pairwise distances
        distances = torch.cdist(features, features, p=2)
        
        # Create masks
        labels_expanded = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels_expanded, labels_expanded.T).float().to(device)
        mask_no_diagonal = 1 - torch.eye(batch_size, device=device)
        mask_positive = mask_positive * mask_no_diagonal
        mask_negative = (1 - torch.eq(labels_expanded, labels_expanded.T).float().to(device)) * mask_no_diagonal
        
        pos_distances = distances[mask_positive.bool()]
        neg_distances = distances[mask_negative.bool()]
        
        if len(pos_distances) == 0 or len(neg_distances) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # RELATIVE OPTIMIZATION: neg_distances should be margin_ratio times larger than pos_distances
        avg_pos = pos_distances.mean()
        avg_neg = neg_distances.mean()
        
        # Target: avg_neg = margin_ratio * avg_pos
        target_neg = self.margin_ratio * avg_pos
        
        # Loss components:
        # 1. Minimize positive distances (but don't force them to zero)
        pos_loss = avg_pos / (avg_pos.detach() + 1e-8)  # Normalize by current scale
        
        # 2. Maximize negative distances relative to positive
        neg_loss = torch.clamp(target_neg - avg_neg, min=0) / (target_neg.detach() + 1e-8)
        
        return pos_loss + neg_loss


class NormalizedContrastiveLoss(nn.Module):
    """
    Work on normalized features - this is what similarity-based methods do
    """
    
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        batch_size = features.shape[0]
        device = features.device
        
        # Normalize features to unit sphere
        features_norm = F.normalize(features, dim=1)
        
        # Compute cosine similarities
        similarities = torch.matmul(features_norm, features_norm.T) / self.temperature
        
        # Create masks
        labels_expanded = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels_expanded, labels_expanded.T).float().to(device)
        mask_no_diagonal = 1 - torch.eye(batch_size, device=device)
        mask_positive = mask_positive * mask_no_diagonal
        
        # Get positive similarities (want these to be high, close to 1)
        pos_similarities = similarities[mask_positive.bool()]
        
        if len(pos_similarities) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Simple loss: maximize positive similarities
        # Since similarities are in [-1, 1], we want them close to 1
        loss = 1.0 - pos_similarities.mean()  # Loss decreases as similarities approach 1
        
        return loss


class AdaptiveMarginLoss(nn.Module):
    """
    Adaptive margin based on current distance statistics
    """
    
    def __init__(self, margin_percentile=0.5):
        super().__init__()
        self.margin_percentile = margin_percentile
    
    def forward(self, features, labels):
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
        
        # Adaptive margin: use median of current negative distances
        adaptive_margin = torch.quantile(neg_distances, self.margin_percentile)
        
        # Loss components
        pos_loss = pos_distances.mean()
        neg_loss = torch.clamp(adaptive_margin - neg_distances, min=0).mean()
        
        return pos_loss + neg_loss


def test_improved_losses():
    """Test improved contrastive losses"""
    print("Testing Improved Contrastive Losses")
    print("=" * 50)
    
    # Create better synthetic data
    torch.manual_seed(42)
    
    # Create tighter clusters (reduce within-class variance)
    class_0 = torch.randn(10, 75) * 0.5 + torch.tensor([3.0] * 75)   # Tight cluster at +3
    class_1 = torch.randn(10, 75) * 0.5 + torch.tensor([0.0] * 75)   # Tight cluster at 0  
    class_2 = torch.randn(10, 75) * 0.5 + torch.tensor([-3.0] * 75)  # Tight cluster at -3
    
    features = torch.cat([class_0, class_1, class_2], dim=0)
    labels = torch.cat([torch.zeros(10), torch.ones(10), torch.full((10,), 2)]).long()
    
    # Analyze the synthetic data first
    distances = torch.cdist(features, features, p=2)
    labels_expanded = labels.contiguous().view(-1, 1)
    mask_positive = torch.eq(labels_expanded, labels_expanded.T).float()
    mask_no_diagonal = 1 - torch.eye(len(labels))
    mask_positive = mask_positive * mask_no_diagonal
    mask_negative = (1 - torch.eq(labels_expanded, labels_expanded.T).float()) * mask_no_diagonal
    
    pos_distances = distances[mask_positive.bool()]
    neg_distances = distances[mask_negative.bool()]
    
    print(f"Synthetic data analysis:")
    print(f"  Positive distances: {pos_distances.mean().item():.4f} ± {pos_distances.std().item():.4f}")
    print(f"  Negative distances: {neg_distances.mean().item():.4f} ± {neg_distances.std().item():.4f}")
    print(f"  Separation ratio: {(neg_distances.mean() / pos_distances.mean()).item():.2f}x")
    print()
    
    # Test improved losses
    losses = {
        'Improved (Relative)': ImprovedContrastiveLoss(margin_ratio=2.0),
        'Normalized (Cosine)': NormalizedContrastiveLoss(temperature=0.5),
        'Adaptive Margin': AdaptiveMarginLoss(margin_percentile=0.5),
    }
    
    for name, loss_fn in losses.items():
        try:
            loss_value = loss_fn(features, labels)
            print(f"{name}: {loss_value.item():.4f}")
            
        except Exception as e:
            print(f"{name}: ERROR - {e}")
        
        print()
    
    # Test what happens with perfect clusters
    print("Testing with PERFECT synthetic clusters:")
    print("-" * 40)
    
    # Create perfect clusters (very tight, well separated)
    perfect_class_0 = torch.randn(10, 75) * 0.01 + torch.tensor([10.0] * 75)
    perfect_class_1 = torch.randn(10, 75) * 0.01 + torch.tensor([0.0] * 75)
    perfect_class_2 = torch.randn(10, 75) * 0.01 + torch.tensor([-10.0] * 75)
    
    perfect_features = torch.cat([perfect_class_0, perfect_class_1, perfect_class_2], dim=0)
    
    for name, loss_fn in losses.items():
        try:
            perfect_loss = loss_fn(perfect_features, labels)
            print(f"{name} (perfect): {perfect_loss.item():.4f}")
            
        except Exception as e:
            print(f"{name} (perfect): ERROR - {e}")


def fix_original_triplet_loss():
    """Debug why triplet loss returned 0.0"""
    torch.manual_seed(42)
    
    # Same synthetic data
    class_0 = torch.randn(10, 75) + torch.tensor([2.0] * 75)
    class_1 = torch.randn(10, 75) + torch.tensor([0.0] * 75)
    class_2 = torch.randn(10, 75) + torch.tensor([-2.0] * 75)
    
    features = torch.cat([class_0, class_1, class_2], dim=0)
    labels = torch.cat([torch.zeros(10), torch.ones(10), torch.full((10,), 2)]).long()
    
    print("Debugging Triplet Loss:")
    print("=" * 30)
    
    # Manual triplet loss calculation
    total_loss = 0.0
    num_triplets = 0
    margin = 1.0
    
    for i in range(len(features)):
        anchor_label = labels[i]
        anchor_features = features[i:i+1]
        
        # Find positives and negatives
        pos_mask = (labels == anchor_label) & (torch.arange(len(features)) != i)
        neg_mask = labels != anchor_label
        
        if not pos_mask.any() or not neg_mask.any():
            continue
            
        pos_features = features[pos_mask]
        neg_features = features[neg_mask]
        
        # Compute distances
        pos_distances = torch.norm(anchor_features - pos_features, dim=1)
        neg_distances = torch.norm(anchor_features - neg_features, dim=1)
        
        print(f"Anchor {i} (class {anchor_label.item()}):")
        print(f"  Pos distances: {pos_distances.mean().item():.4f}")
        print(f"  Neg distances: {neg_distances.mean().item():.4f}")
        print(f"  Min neg distance: {neg_distances.min().item():.4f}")
        
        # Check triplet constraint
        for pos_dist in pos_distances:
            hardest_neg_dist = neg_distances.min()
            violation = pos_dist - hardest_neg_dist + margin
            print(f"    Triplet: pos={pos_dist.item():.4f}, neg={hardest_neg_dist.item():.4f}, violation={violation.item():.4f}")
            
            if violation > 0:
                total_loss += violation
                num_triplets += 1
        
        if i >= 2:  # Just show first few for debugging
            break
    
    print(f"\nFinal triplet loss: {total_loss/max(num_triplets, 1):.4f} (from {num_triplets} triplets)")


if __name__ == "__main__":
    test_improved_losses()
    print("\n" + "="*60 + "\n")
    fix_original_triplet_loss()