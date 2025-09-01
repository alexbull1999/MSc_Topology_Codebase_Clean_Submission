# FILE: losses.py
"""
InfoNCE + Order Embeddings + Topological loss functions
CORRECTED VERSION with proper Vendrov et al. order embedding loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss with global dataset updates
    """
    
    def __init__(self, temperature=0.07, max_global_samples=5000):
        super().__init__()
        self.temperature = temperature
        self.max_global_samples = max_global_samples
        
        # Global dataset storage
        self.global_features = None
        self.global_labels = None
        
        print(f"InfoNCE Loss initialized with global updates:")
        print(f"  Temperature: {temperature}")
        print(f"  Max global samples: {max_global_samples}")
    
    def update_global_dataset(self, dataloader, model, device):
        """
        Extract features for the entire dataset using the current model
        """
        print("🌍 Extracting global features for InfoNCE...")
        model.eval()
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                premise_embeddings = batch['premise_embedding'].to(device)
                hypothesis_embeddings = batch['hypothesis_embedding'].to(device)
                labels = batch['label'].to(device)
                
                # Concatenate premise and hypothesis for model input
                combined_embeddings = torch.cat([premise_embeddings, hypothesis_embeddings], dim=1)
                
                # Get latent features
                latent, _ = model(combined_embeddings)
                
                all_features.append(latent.cpu())
                all_labels.append(labels.cpu())
                
                if batch_idx % 50 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        # Combine all features
        full_features = torch.cat(all_features, dim=0).to(device)
        full_labels = torch.cat(all_labels, dim=0).to(device)
        
        # Subsample if too large
        if len(full_features) > self.max_global_samples:
            indices = torch.randperm(len(full_features))[:self.max_global_samples]
            self.global_features = full_features[indices]
            self.global_labels = full_labels[indices]
            print(f"  Subsampled to {self.max_global_samples} samples")
        else:
            self.global_features = full_features
            self.global_labels = full_labels
        
        print(f"  Global dataset updated: {self.global_features.shape[0]} samples")
        model.train()
    
    def forward(self, features, labels):
        """
        InfoNCE loss computation with global negatives
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Normalize features to unit sphere
        features = F.normalize(features, p=2, dim=1)
        
        # Use global features if available, otherwise fall back to batch
        if self.global_features is not None and len(self.global_features) > batch_size:
            # Combine batch features with global features
            global_features_norm = F.normalize(self.global_features, p=2, dim=1)
            
            # For each sample in batch, compute InfoNCE against global dataset
            losses = []
            for i in range(batch_size):
                anchor = features[i:i+1]  # [1, feature_dim]
                anchor_label = labels[i]
                
                # Find global positives (same class as anchor)
                pos_mask = (self.global_labels == anchor_label)
                if pos_mask.sum() == 0:
                    continue
                
                # Compute similarities: anchor vs all global samples
                similarities = torch.matmul(anchor, global_features_norm.T) / self.temperature  # [1, global_size]
                similarities = similarities.squeeze(0)  # [global_size]
                
                # Get positive and negative similarities
                pos_similarities = similarities[pos_mask]
                
                # InfoNCE: -log(sum(exp(pos)) / sum(exp(all)))
                pos_exp_sum = torch.sum(torch.exp(pos_similarities))
                all_exp_sum = torch.sum(torch.exp(similarities))
                
                loss_i = -torch.log(pos_exp_sum / (all_exp_sum + 1e-8) + 1e-8)
                losses.append(loss_i)
            
            if len(losses) > 0:
                return torch.stack(losses).mean()
            else:
                return torch.tensor(0.0, device=device, requires_grad=True)
        
        else:
            # Fall back to batch-only InfoNCE (original implementation)
            similarity_matrix = torch.matmul(features, features.T) / self.temperature
            
            labels_expanded = labels.contiguous().view(-1, 1)
            mask_positive = torch.eq(labels_expanded, labels_expanded.T).float().to(device)
            mask_no_diagonal = 1 - torch.eye(batch_size, device=device)
            mask_positive = mask_positive * mask_no_diagonal
            
            losses = []
            for i in range(batch_size):
                pos_similarities = similarity_matrix[i] * mask_positive[i]
                all_similarities = similarity_matrix[i] * mask_no_diagonal[i]
                
                if pos_similarities.sum() > 0:
                    pos_exp_sum = torch.sum(torch.exp(pos_similarities))
                    all_exp_sum = torch.sum(torch.exp(all_similarities))
                    
                    loss_i = -torch.log(pos_exp_sum / (all_exp_sum + 1e-8) + 1e-8)
                    losses.append(loss_i)
            
            if len(losses) > 0:
                return torch.stack(losses).mean()
            else:
                return torch.tensor(0.0, device=device, requires_grad=True)


class OrderEmbeddingLoss(nn.Module):
    """
    Proper Vendrov et al. order embedding loss with max-margin formulation
    E(u,v) = ||max(0, v-u)||²
    """
    
    def __init__(self, entailment_margin=0.3, neutral_margin=1.0, neutral_upper_bound=1.5, contradiction_margin=2.0):
        super().__init__()
        self.entailment_margin = entailment_margin
        self.neutral_margin = neutral_margin
        self.neutral_upper_bound = neutral_upper_bound
        self.contradiction_margin = contradiction_margin
        print(f"Order Embedding Loss initialized:")
        print(f"  Entailment target: ~{entailment_margin}")
        print(f"  Neutral range: [{neutral_margin}, {neutral_upper_bound}]")
        print(f"  Contradiction minimum: {contradiction_margin}")
    
    def order_violation_energy(self, premise_embeddings, hypothesis_embeddings):
        """
        Compute order violation energy: E(u,v) = ||max(0, u-v)||²
        For entailment: premise ⊑ hypothesis, so violations should be premise - hypothesis
        """
        # Correct order: premise should be ≤ hypothesis for entailment
        # So we penalize when premise > hypothesis
        violations = torch.clamp(premise_embeddings - hypothesis_embeddings, min=0)
        energy = torch.norm(violations, p=2, dim=1) ** 2
        
        # Normalize by dimension to keep energies reasonable
        energy = energy / premise_embeddings.shape[1]
        
        return energy
    
    def forward(self, premise_embeddings, hypothesis_embeddings, labels):
        """
        Separate margin loss with neutral sandwiching for better class separation
        """
        device = premise_embeddings.device
        
        # Compute order violation energy for all pairs
        energy = self.order_violation_energy(premise_embeddings, hypothesis_embeddings)
        
        # Debug: print energy statistics occasionally
        if hasattr(self, 'debug_counter'):
            self.debug_counter += 1
        else:
            self.debug_counter = 1
            
        if self.debug_counter % 1000 == 0:  # Print every 1000 calls
            print(f"Order Loss Debug - Energy stats: min={energy.min():.4f}, max={energy.max():.4f}, mean={energy.mean():.4f}")
            for label_val in [0, 1, 2]:
                mask = (labels == label_val)
                if mask.sum() > 0:
                    label_energies = energy[mask]
                    label_name = ['entailment', 'neutral', 'contradiction'][label_val]
                    print(f"  {label_name}: mean={label_energies.mean():.4f}, std={label_energies.std():.4f}")
        
        # Separate margin loss based on label
        losses = []
        for i, label in enumerate(labels):
            if label == 0:  # entailment
                # For entailment, we want energy to be close to entailment_margin
                # Penalize if energy is too high
                loss = torch.clamp(energy[i] - self.entailment_margin, min=0)
                losses.append(loss)
            elif label == 1:  # neutral - sandwich between margins
                # Keep energy in [neutral_margin, neutral_upper_bound]
                lower_violation = torch.clamp(self.neutral_margin - energy[i], min=0)
                upper_violation = torch.clamp(energy[i] - self.neutral_upper_bound, min=0)
                loss = lower_violation + upper_violation
                losses.append(loss)
            else:  # contradiction (label == 2)
                # Ensure energy > contradiction_margin
                loss = torch.clamp(self.contradiction_margin - energy[i], min=0)
                losses.append(loss)
        
        if len(losses) > 0:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


class MoorTopologicalLoss(nn.Module):
    """
    Moor topological loss from your existing implementation
    Copied from moor_topological_loss.py
    """
    def __init__(self):
        super().__init__()
        # Import torch_topological here to avoid import issues if not available
        try:
            from torch_topological.nn import VietorisRipsComplex
            self.vr_complex = VietorisRipsComplex(dim=0, keep_infinite_features=False)
            print("MoorTopologicalLoss Initialized: Using 0-dimensional persistence pairings (MST edges).")
        except ImportError:
            print("Warning: torch_topological not available. Moor loss will return zero.")
            self.vr_complex = None

    def _get_persistence_pairings(self, x: torch.Tensor):
        if self.vr_complex is None or x.shape[0] < 2:
            return None
        persistence_info_list = self.vr_complex(x)
        if persistence_info_list and persistence_info_list[0] is not None:
            return persistence_info_list[0].pairing
        return None

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Calculates the topological loss between an input point cloud x and its
        latent representation z.
        """
        batch_size = x.shape[0]
        if batch_size < 2 or self.vr_complex is None:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        dist_matrix_x = torch.cdist(x, x, p=2)
        dist_matrix_z = torch.cdist(z, z, p=2)

        pi_x_np = self._get_persistence_pairings(x)
        pi_z_np = self._get_persistence_pairings(z)

        if pi_x_np is None or pi_z_np is None:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        # Convert the numpy pairings from the library into PyTorch tensors
        pi_x = torch.from_numpy(pi_x_np).to(x.device)
        pi_z = torch.from_numpy(pi_z_np).to(z.device)

        # The pairings are (dim, birth_idx, death_idx). We only need the point indices.
        edges_x = pi_x[:, 1:].long()
        edges_z = pi_z[:, 1:].long()
        
        distances_x_from_pi_x = dist_matrix_x[edges_x[:, 0], edges_x[:, 1]]
        distances_z_from_pi_x = dist_matrix_z[edges_x[:, 0], edges_x[:, 1]]
        loss_x_z = torch.sum((distances_x_from_pi_x - distances_z_from_pi_x)**2)

        distances_x_from_pi_z = dist_matrix_x[edges_z[:, 0], edges_z[:, 1]]
        distances_z_from_pi_z = dist_matrix_z[edges_z[:, 0], edges_z[:, 1]]
        loss_z_x = torch.sum((distances_x_from_pi_z - distances_z_from_pi_z)**2)

        return (loss_x_z + loss_z_x) / batch_size


class CombinedLoss(nn.Module):
    """
    Combined InfoNCE + Order Embeddings + Moor Topological + Reconstruction loss
    """
    
    def __init__(self, 
                 infonce_weight=1.0,
                 order_weight=0.1,
                 topological_weight=0.05,
                 reconstruction_weight=0.3,
                 temperature=0.07,
                 max_global_samples=5000,
                 # Order embedding parameters
                 entailment_margin=0.3,
                 neutral_margin=1.0,
                 neutral_upper_bound=1.5,
                 contradiction_margin=2.0,
                 topo_frequency=1000):
        super().__init__()
        
        self.infonce_loss = InfoNCELoss(temperature=temperature, max_global_samples=max_global_samples)
        self.order_loss = OrderEmbeddingLoss(
            entailment_margin=entailment_margin,
            neutral_margin=neutral_margin,
            neutral_upper_bound=neutral_upper_bound,
            contradiction_margin=contradiction_margin
        )
        self.moor_loss = MoorTopologicalLoss()
        self.reconstruction_loss = nn.MSELoss()
        
        self.infonce_weight = infonce_weight
        self.order_weight = order_weight
        self.topological_weight = topological_weight
        self.reconstruction_weight = reconstruction_weight
        self.topo_frequency = topo_frequency
        
        # Track iterations for sparse topological application
        self.iteration_count = 0
        
        print(f"CombinedLoss initialized:")
        print(f"  InfoNCE weight: {infonce_weight}")
        print(f"  Order weight: {order_weight}")
        print(f"  Topological weight: {topological_weight}")
        print(f"  Reconstruction weight: {reconstruction_weight}")
        print(f"  Topological frequency: {topo_frequency} iterations")
    
    def update_global_dataset(self, dataloader, model, device):
        """Update global dataset for InfoNCE loss"""
        self.infonce_loss.update_global_dataset(dataloader, model, device)
    
    def forward(self, premise_latent, hypothesis_latent, 
                premise_reconstructed, hypothesis_reconstructed,
                premise_original, hypothesis_original, labels):
        """
        Combined loss computation with sparse topological regularization
        """
        device = premise_latent.device
        
        # Combine premise and hypothesis for InfoNCE
        combined_latent = torch.cat([premise_latent, hypothesis_latent], dim=0)
        combined_labels = torch.cat([labels, labels], dim=0)
        combined_original = torch.cat([premise_original, hypothesis_original], dim=0)
        
        # 1. InfoNCE Loss
        infonce_loss_value = self.infonce_loss(combined_latent, combined_labels)
        
        # 2. Order Embedding Loss (uses proper Vendrov formulation)
        order_loss_value = self.order_loss(premise_latent, hypothesis_latent, labels)
        
        # 3. Reconstruction Loss
        premise_recon_loss = self.reconstruction_loss(premise_reconstructed, premise_original)
        hypothesis_recon_loss = self.reconstruction_loss(hypothesis_reconstructed, hypothesis_original)
        reconstruction_loss_value = (premise_recon_loss + hypothesis_recon_loss) / 2
        
        # 4. Moor Topological Loss (sparse application)
        topological_loss_value = torch.tensor(0.0, device=device, requires_grad=True)
        topological_applied = False
        
        self.iteration_count += 1
        if (self.iteration_count % self.topo_frequency == 0 and 
            self.topological_weight > 0 and 
            combined_latent.shape[0] >= 4):  # Need minimum samples for topology
            
            # Apply Moor topological loss: preserve topology from input to latent
            try:
                topological_loss_value = self.moor_loss(combined_original, combined_latent)
                topological_applied = True
            except Exception as e:
                print(f"Warning: Topological loss computation failed: {e}")
                topological_loss_value = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Total loss
        total_loss = (self.infonce_weight * infonce_loss_value + 
                     self.order_weight * order_loss_value +
                     self.topological_weight * topological_loss_value +
                     self.reconstruction_weight * reconstruction_loss_value)
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'infonce_loss': infonce_loss_value.item(),
            'order_loss': order_loss_value.item(),
            'topological_loss': topological_loss_value.item(),
            'reconstruction_loss': reconstruction_loss_value.item(),
            'topological_applied': topological_applied
        }