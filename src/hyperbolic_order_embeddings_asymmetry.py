import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import geoopt
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple
import json
import logging
import sys

"""
if 3-way margin loss performs better; add that to this file 
"""

from .order_embeddings_asymmetry import EntailmentDataset

def flush_output():
    """Force output to appear immediately in SLURM"""
    sys.stdout.flush()
    sys.stderr.flush()


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


class PureHyperbolicLearnableOrderEmbeddingModel(nn.Module):
    """
    Pure Hyperbolic Order Embedding Model with learnable energy function parameters
    
    This model:
    1. Uses only hyperbolic operations throughout (pure hyperbolic)
    2. Learns optimal energy function parameters during training
    3. Uses same parameters for all classes (blind testing compatible)
    4. Should give maximum theoretical performance benefits
    """
    
    def __init__(self, bert_dim: int = 768, order_dim: int = 50, asymmetry_weight: float = 0.2):
        super().__init__()
        self.bert_dim = bert_dim
        self.order_dim = order_dim
        self.asymmetry_weight = asymmetry_weight
        
        # Numerical stability
        self.eps = 1e-8
        self.max_norm = 0.98  # Stay away from boundary of unit ball

        # Poincaré ball manifold
        self.ball = geoopt.PoincareBall()
        
        # PURE HYPERBOLIC NEURAL NETWORK LAYERS
        # All parameters live on the hyperbolic manifold
        
        # Initial projection dimension (we need to map from 768D BERT to smaller space first)
        intermediate_dim = min(order_dim * 4, 200)  # Reasonable intermediate size
        
        # Hyperbolic weight matrices - these are ManifoldParameters
        self.hyp_weight1 = geoopt.ManifoldParameter(
            self._init_hyperbolic_matrix(bert_dim, intermediate_dim),
            manifold=self.ball
        )
        self.hyp_bias1 = geoopt.ManifoldParameter(
            self._init_hyperbolic_vector(intermediate_dim),
            manifold=self.ball
        )
        
        self.hyp_weight2 = geoopt.ManifoldParameter(
            self._init_hyperbolic_matrix(intermediate_dim, order_dim),
            manifold=self.ball
        )
        self.hyp_bias2 = geoopt.ManifoldParameter(
            self._init_hyperbolic_vector(order_dim),
            manifold=self.ball
        )
        
        # Asymmetric projection weights (also hyperbolic)
        self.asym_weight = geoopt.ManifoldParameter(
            self._init_hyperbolic_matrix(order_dim, order_dim),
            manifold=self.ball
        )
        self.asym_bias = geoopt.ManifoldParameter(
            self._init_hyperbolic_vector(order_dim),
            manifold=self.ball
        )
        
        # LEARNABLE ENERGY FUNCTION PARAMETERS
        # These are standard parameters (not manifold parameters)
        # because they're scalars, not points in hyperbolic space
        
        self.specificity_scaling = nn.Parameter(
            torch.tensor(1.0),  # Initialize with reasonable guess
            requires_grad=True
        )
        self.base_tolerance = nn.Parameter(
            torch.tensor(0.5),  # Initialize with reasonable guess
            requires_grad=True
        )
        self.proximity_weight = nn.Parameter(
            torch.tensor(0.5),  # Weight for proximity vs specificity violation
            requires_grad=True
        )
        
        # Additional learnable parameters for energy function refinement
        self.specificity_power = nn.Parameter(
            torch.tensor(1.0),  # Power for specificity difference (allows non-linear scaling)
            requires_grad=True
        )
        self.distance_scaling = nn.Parameter(
            torch.tensor(2.0),  # Overall scaling for hyperbolic distances
            requires_grad=True
        )
        
        # FIXED: More reasonable parameter constraints
        self.register_buffer('min_scaling', torch.tensor(0.1))     # Increased from 0.05
        self.register_buffer('max_scaling', torch.tensor(5.0))     # Increased from 3.0
        self.register_buffer('min_tolerance', torch.tensor(0.05))  # Increased from 0.01
        self.register_buffer('max_tolerance', torch.tensor(2.0))   # Increased from 1.5
        self.register_buffer('min_weight', torch.tensor(0.05))     # Increased from 0.01
        self.register_buffer('max_weight', torch.tensor(3.0))      # Increased from 2.0
        self.register_buffer('min_power', torch.tensor(0.5))       # Same
        self.register_buffer('max_power', torch.tensor(2.0))       # Same
        


    def _init_hyperbolic_matrix(self, in_dim: int, out_dim: int) -> torch.Tensor:
        """Initialize hyperbolic weight matrix with small values"""
        # Start with small random values and ensure they're in the unit ball
        matrix = torch.randn(in_dim, out_dim) * 0.1
        # Ensure all points are inside unit ball
        norms = torch.norm(matrix, dim=-1, keepdim=True)
        matrix = torch.where(norms >= self.max_norm, matrix * (self.max_norm / (norms + self.eps)), matrix)
        return matrix

    def _init_hyperbolic_vector(self, dim: int) -> torch.Tensor:
        """Initialize hyperbolic bias vector"""
        vector = torch.randn(dim) * 0.1
        norm = torch.norm(vector)
        if norm >= self.max_norm:
            vector = vector * (self.max_norm / (norm + self.eps))
        return vector

    def get_constrained_parameters(self):
        """Get energy function parameters constrained to reasonable ranges"""
        specificity_scaling = torch.clamp(self.specificity_scaling, self.min_scaling, self.max_scaling)
        base_tolerance = torch.clamp(self.base_tolerance, self.min_tolerance, self.max_tolerance)
        proximity_weight = torch.clamp(self.proximity_weight, self.min_weight, self.max_weight)
        specificity_power = torch.clamp(self.specificity_power, self.min_power, self.max_power)
        distance_scaling = torch.clamp(self.distance_scaling, self.min_scaling, self.max_scaling)
        
        return specificity_scaling, base_tolerance, proximity_weight, specificity_power, distance_scaling

    def mobius_linear(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        NUMERICALLY STABLE Möbius linear transformation
        """
        batch_size = x.shape[0]
    
        try:
            # Ensure input is safe
            x = self._safe_clamp_to_ball(x)
        
            # Step 1: Map input to tangent space at origin
            x_tangent = self.ball.logmap0(x)
        
            # Check for NaN/inf after logmap
            if torch.isnan(x_tangent).any() or torch.isinf(x_tangent).any():
                print("Warning: NaN/inf in tangent space")
                flush_output()
                raise
        
            # Step 2: Apply transformation with careful dimension matching
            if weight.dim() == 2:
                # Better dimension handling
                input_dim = min(x_tangent.shape[-1], weight.shape[0])
                output_dim = weight.shape[1]
            
                # Take only the dimensions that match
                x_tangent_proj = x_tangent[:, :input_dim]
                weight_proj = weight[:input_dim, :]
            
                transformed = torch.matmul(x_tangent_proj, weight_proj)
            else:
                # Element-wise for bias vector
                min_dim = min(x_tangent.shape[-1], bias.shape[0])
                transformed = x_tangent[:, :min_dim] + bias[:min_dim]
        
            # Step 3: Map back to hyperbolic space with conservative scaling
            scale_factor = 0.9  
        
            # Clamp transformed values
            transformed = torch.clamp(transformed, min=-20.0, max=20.0)
        
            result = self.ball.expmap0(transformed * scale_factor)
        
            # Step 4: Ensure points stay inside unit ball
            result = self._safe_clamp_to_ball(result)
        
        except Exception as e:
            print(f"Error in mobius_linear: {e}")
            flush_output()
            raise
    
        return result

    def hyperbolic_nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hyperbolic nonlinearity function
        
        We map to tangent space, apply nonlinearity, then map back.
        Uses tanh which works well in hyperbolic space.
        """
        # Map to tangent space
        x_tangent = self.ball.logmap0(x)
        
        # Apply nonlinearity in tangent space
        # Tanh is preferred over ReLU in hyperbolic space
        activated = torch.tanh(x_tangent)
        
        # Map back with conservative scaling
        result = self.ball.expmap0(activated * 0.5)
        
        # Ensure inside unit ball
        norms = torch.norm(result, dim=-1, keepdim=True)
        result = torch.where(norms >= self.max_norm, result * (self.max_norm / (norms + self.eps)), result)
        
        return result

    def forward(self, bert_embeddings: torch.Tensor) -> torch.Tensor:
        """
        NUMERICALLY STABLE pure hyperbolic forward pass
        """
        batch_size = bert_embeddings.shape[0]
    
        # Check for NaN/inf in input
        if torch.isnan(bert_embeddings).any() or torch.isinf(bert_embeddings).any():
            print("Warning: NaN/inf detected in BERT embeddings")
            bert_embeddings = torch.where(torch.isnan(bert_embeddings) | torch.isinf(bert_embeddings), 
                                        torch.zeros_like(bert_embeddings), bert_embeddings)
    
        # MUCH more conservative initial scaling
        initial_scale = 0.8  # Even smaller than before
    
        # Take subset and normalize input
        input_subset = bert_embeddings[:, :self.hyp_weight1.shape[0]]
        # input_subset = input_subset / (torch.norm(input_subset, dim=-1, keepdim=True) + 1e-8)  # Normalize
    
        # Initial projection to hyperbolic space with safety
        try:
            x = self.ball.expmap0(input_subset * initial_scale)
            x = self._safe_clamp_to_ball(x)
            # DEBUG: Check embedding magnitudes
            if torch.norm(x[0]).item() < 1e-4:
                print(f"Warning: Very small embeddings after initial projection: {torch.norm(x[0]).item():.6f}")
                flush_output()

        except Exception as e:
            print(f"Error in initial projection: {e}")
            flush_output()
            raise
    
        # First pure hyperbolic layer with error handling
        try:
            x = self.mobius_linear(x, self.hyp_weight1, self.hyp_bias1)
            x = self._safe_clamp_to_ball(x)
            x = self.hyperbolic_nonlinearity(x)
            x = self._safe_clamp_to_ball(x)
        except Exception as e:
            print(f"Error in first hyperbolic layer: {e}")
            flush_output()
            raise
    
        # Second pure hyperbolic layer with error handling
        try:
            x = self.mobius_linear(x, self.hyp_weight2, self.hyp_bias2)
            x = self._safe_clamp_to_ball(x)
            x = self.hyperbolic_nonlinearity(x)
            x = self._safe_clamp_to_ball(x)
        except Exception as e:
            print(f"Error in second hyperbolic layer: {e}")
            flush_output()
            raise
        
    
        return x

    # FIX 2: Completely redesigned energy function that actually produces meaningful values:
    def learnable_hyperbolic_order_violation_energy(self, u_emb: torch.Tensor, v_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        REDESIGNED hyperbolic order violation energy that produces meaningful values
        """
        # Get constrained learnable parameters
        specificity_scaling, base_tolerance, proximity_weight, specificity_power, distance_scaling = self.get_constrained_parameters()
    
        try:
            # Clamp embeddings to stay well inside unit ball
            u_emb_safe = self._safe_clamp_to_ball(u_emb)
            v_emb_safe = self._safe_clamp_to_ball(v_emb)
        
            # Component 1: SIMPLE distance-based energy (this will not be zero!)
            pairwise_dist = self.ball.dist(u_emb_safe, v_emb_safe)
        
            # Component 2: Norm-based specificity
            u_norms = self.ball.dist0(u_emb_safe)
            v_norms = self.ball.dist0(v_emb_safe)

            # DEBUG: Print distance ranges
            # if pairwise_dist.numel() > 0:
            #     print(f"DEBUG: Distance range: {pairwise_dist.min().item():.6f} to {pairwise_dist.max().item():.6f}")
            #     print(f"DEBUG: Norm range: {u_norms.min().item():.6f} to {u_norms.max().item():.6f}")
                # flush_output()
        
            # SIMPLE energy function that will produce non-zero values
            # Base energy from distance
            base_energy = pairwise_dist * distance_scaling * 5.0
        
            # Add specificity component
            specificity_diff = torch.abs(u_norms - v_norms)
            specificity_energy = specificity_diff * specificity_scaling * 3.0
        
            # Combine with learnable weights
            total_violation = base_energy + specificity_energy + (base_tolerance * 2.0)
        
            # Ensure non-zero values
            total_violation = torch.clamp(total_violation, min=0.01, max=20.0)
        
        except Exception as e:
            print(f"Error in energy computation: {e}")
            raise
    
        return {
            'total_energy': total_violation,
            'specificity_violation': specificity_energy,
            'proximity_violation': base_energy,
            'expected_geodesic': base_tolerance.expand(total_violation.shape),
            'u_norms': u_norms,
            'v_norms': v_norms,
            'pairwise_dist': pairwise_dist,
            'learned_params': {
                'specificity_scaling': specificity_scaling.item(),
                'base_tolerance': base_tolerance.item(),
                'proximity_weight': proximity_weight.item(),
                'specificity_power': specificity_power.item(),
                'distance_scaling': distance_scaling.item()
            }
        }


    # FIX 2: Add safe clamping method to the model class:
    def _safe_clamp_to_ball(self, x: torch.Tensor) -> torch.Tensor:
        """Safely clamp points to stay well inside Poincaré ball"""
        norms = torch.norm(x, dim=-1, keepdim=True)
        safe_max = 0.9  
    
        # Replace any NaN or inf values
        x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        norms = torch.where(torch.isnan(norms) | torch.isinf(norms), torch.ones_like(norms), norms)
    
        # Clamp to safe region
        x_safe = torch.where(norms > safe_max, x * (safe_max / (norms + self.eps)), x)
    
        return x_safe

    def hyperbolic_asymmetric_features(self, hyperbolic_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Pure hyperbolic asymmetric feature transformation
        """
        return self.mobius_linear(hyperbolic_embeddings, self.asym_weight, self.asym_bias)

    def hyperbolic_asymmetric_energy(self, premise_emb: torch.Tensor, hypothesis_emb: torch.Tensor) -> torch.Tensor:
        """
        Pure hyperbolic asymmetric energy using only hyperbolic distances
        """
        premise_asym = self.hyperbolic_asymmetric_features(premise_emb)
        hypothesis_asym = self.hyperbolic_asymmetric_features(hypothesis_emb)
        
        # Use pure hyperbolic distance
        _, _, _, _, distance_scaling = self.get_constrained_parameters()
        return self.ball.dist(premise_asym, hypothesis_asym) * distance_scaling

    def compute_bidirectional_energies(self, premise_emb: torch.Tensor, hypothesis_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all energies using pure hyperbolic operations and learnable parameters
        """
        # Forward and backward order violation energies
        forward_result = self.learnable_hyperbolic_order_violation_energy(premise_emb, hypothesis_emb)
        backward_result = self.learnable_hyperbolic_order_violation_energy(hypothesis_emb, premise_emb)
        
        # Pure hyperbolic asymmetric energy
        asym_energy = self.hyperbolic_asymmetric_energy(premise_emb, hypothesis_emb)
        
        # Asymmetry measure
        forward_energy = forward_result['total_energy']
        backward_energy = backward_result['total_energy']
        asymmetry_measure = torch.abs(forward_energy - backward_energy)
        
        return {
            'forward_energy': forward_energy,
            'backward_energy': backward_energy,
            'asymmetric_energy': asym_energy,
            'asymmetry_measure': asymmetry_measure,
            'learned_params': forward_result['learned_params'],
            'detailed_forward': forward_result,
            'detailed_backward': backward_result
        }

    def get_learned_parameters_summary(self) -> Dict[str, float]:
        """
        Get a summary of the current learned parameter values
        Useful for monitoring training and understanding what the model learned
        """
        specificity_scaling, base_tolerance, proximity_weight, specificity_power, distance_scaling = self.get_constrained_parameters()
        
        return {
            'specificity_scaling': specificity_scaling.item(),
            'base_tolerance': base_tolerance.item(),
            'proximity_weight': proximity_weight.item(),
            'specificity_power': specificity_power.item(),
            'distance_scaling': distance_scaling.item()
        }


class PureHyperbolicOrderEmbeddingTrainer:
    """
    Enhanced trainer with 3-way margin loss and corrected asymmetric loss
    """
    
    def __init__(self, model: PureHyperbolicLearnableOrderEmbeddingModel, device='cuda' if torch.cuda.is_available() else 'cpu', lr: float=1e-3):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        
        # Use Riemannian optimizer for hyperbolic parameters
        self.optimizer = geoopt.optim.RiemannianAdam([
            {'params': [p for n, p in model.named_parameters() if 'hyp_' in n or 'asym_' in n], 'lr': lr/3}, #very small lr for hyperbolic
            {'params': [p for n, p in model.named_parameters() if 'hyp_' not in n and 'asym_' not in n], 'lr': lr}
        ], weight_decay=1e-5)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-8
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.energy_rankings = []
        self.parameter_history = []


    # # FIX 3: Much simpler loss function that will work with the energy values:
    # def compute_enhanced_hyperbolic_loss(self, premise_embs: torch.Tensor, hypothesis_embs: torch.Tensor, 
    #                                 labels: torch.Tensor, label_strs: List[str], 
    #                                 margin: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    #     """
    #     SIMPLIFIED but effective 3-way hyperbolic loss
    #     """
    #     # Get pure hyperbolic embeddings
    #     premise_hyp = self.model(premise_embs)
    #     hypothesis_hyp = self.model(hypothesis_embs)
    
    #     # Compute hyperbolic order violation energies
    #     energy_dict = self.model.compute_bidirectional_energies(premise_hyp, hypothesis_hyp)
    #     forward_energies = energy_dict['forward_energy']
    
    #     # Create masks for each class
    #     entailment_mask = (labels == 0)
    #     neutral_mask = (labels == 1)
    #     contradiction_mask = (labels == 2)
    
    #     # SIMPLE 3-CLASS LOSS with clear targets
    #     total_loss = 0.0
    
    #     # Target energies that make sense
    #     entailment_target = 0.0    # Low but not zero
    #     neutral_target = 5.0       # Medium
    #     contradiction_target = 10.0 # High
    
    #     # MSE loss to specific targets
    #     if entailment_mask.any():
    #         entailment_energies = forward_energies[entailment_mask]
    #         entailment_loss = torch.nn.functional.mse_loss(
    #             entailment_energies, 
    #             torch.full_like(entailment_energies, entailment_target)
    #         )
    #         total_loss += entailment_loss
    
    #     if neutral_mask.any():
    #         neutral_energies = forward_energies[neutral_mask]
    #         neutral_loss = torch.nn.functional.mse_loss(
    #             neutral_energies, 
    #             torch.full_like(neutral_energies, neutral_target)
    #         )
    #         total_loss += neutral_loss
    
    #     if contradiction_mask.any():
    #         contradiction_energies = forward_energies[contradiction_mask]
    #         contradiction_loss = torch.nn.functional.mse_loss(
    #             contradiction_energies, 
    #             torch.full_like(contradiction_energies, contradiction_target)
    #         )
    #         total_loss += contradiction_loss
    
    #     # Add ranking loss to ensure order
    #     ranking_loss = 0.0
    #     large_margin = 2.0
    #     if entailment_mask.any() and neutral_mask.any():
    #         ent_mean = forward_energies[entailment_mask].mean()
    #         neut_mean = forward_energies[neutral_mask].mean()
    #         ranking_loss += torch.clamp(ent_mean - neut_mean + large_margin, min=0)
    
    #     if neutral_mask.any() and contradiction_mask.any():
    #         neut_mean = forward_energies[neutral_mask].mean()
    #         cont_mean = forward_energies[contradiction_mask].mean()
    #         ranking_loss += torch.clamp(neut_mean - cont_mean + large_margin, min=0)
    
    #     # Combine losses
    #     total_loss = total_loss + 0.5 * ranking_loss
    
    #     # Simple asymmetric loss
    #     asymmetric_energies = energy_dict['asymmetric_energy']
    #     asymmetric_loss = torch.mean(asymmetric_energies)  # Just minimize asymmetric energy
    
    #     # Total combined loss
    #     final_loss = total_loss + self.model.asymmetry_weight * asymmetric_loss
    
    #     # Compute energy statistics for monitoring
    #     energy_stats = self._compute_energy_statistics(energy_dict, label_strs)
    
    #     return final_loss, forward_energies, energy_stats


    def compute_enhanced_hyperbolic_loss(self, premise_embs: torch.Tensor, hypothesis_embs: torch.Tensor, 
                                labels: torch.Tensor, label_strs: List[str], 
                                margin: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        ADAPTIVE hyperbolic loss with dynamic targets
        """
        # Get pure hyperbolic embeddings
        premise_hyp = self.model(premise_embs)
        hypothesis_hyp = self.model(hypothesis_embs)

        # Compute hyperbolic order violation energies
        energy_dict = self.model.compute_bidirectional_energies(premise_hyp, hypothesis_hyp)
        forward_energies = energy_dict['forward_energy']

        # ADAPTIVE TARGETS based on current energy distribution
        ent_mask = (labels == 0)
        neu_mask = (labels == 1)
        con_mask = (labels == 2)
    
        # Get current energy ranges for adaptive targeting
        if forward_energies.numel() > 0:
            energy_min = forward_energies.min().item()
            energy_max = forward_energies.max().item()
            energy_range = energy_max - energy_min
        
            # Adaptive targets: spread across current range
            ent_target = energy_min + 0.2 * energy_range  # Low end
            neu_target = energy_min + 0.5 * energy_range  # Middle
            con_target = energy_min + 0.8 * energy_range  # High end
        else:
            # Fallback if no energies
            ent_target, neu_target, con_target = 3.0, 5.0, 7.0

        print(f"DEBUG: Adaptive targets - Ent: {ent_target:.2f}, Neu: {neu_target:.2f}, Con: {con_target:.2f}")
    
        # MSE loss to adaptive targets
        total_loss = 0.0
    
        if ent_mask.any():
            ent_energies = forward_energies[ent_mask]
            ent_loss = torch.nn.functional.mse_loss(
                ent_energies, torch.full_like(ent_energies, ent_target)
            )
            total_loss += ent_loss

        if neu_mask.any():
            neu_energies = forward_energies[neu_mask]
            neu_loss = torch.nn.functional.mse_loss(
                neu_energies, torch.full_like(neu_energies, neu_target)
            )
            total_loss += neu_loss

        if con_mask.any():
            con_energies = forward_energies[con_mask]
            con_loss = torch.nn.functional.mse_loss(
                con_energies, torch.full_like(con_energies, con_target)
            )
            total_loss += con_loss

        # Keep existing ranking loss
        ranking_loss = 0.0
    
        if ent_mask.any() and neu_mask.any():
            ent_mean = forward_energies[ent_mask].mean()
            neu_mean = forward_energies[neu_mask].mean()
            ranking_loss += torch.clamp(ent_mean - neu_mean + margin, min=0)

        if neu_mask.any() and con_mask.any():
            neu_mean = forward_energies[neu_mask].mean()
            con_mean = forward_energies[con_mask].mean()
            ranking_loss += torch.clamp(neu_mean - con_mean + margin, min=0)

        # Combine losses
        total_loss = total_loss + 0.5 * ranking_loss

        # Keep existing asymmetric loss
        asymmetric_energies = energy_dict['asymmetric_energy']
        asymmetric_loss = torch.mean(asymmetric_energies)
    
        final_loss = total_loss + self.model.asymmetry_weight * asymmetric_loss

        # Compute energy statistics for monitoring
        energy_stats = self._compute_energy_statistics(energy_dict, label_strs)

        return final_loss, forward_energies, energy_stats
    

    def compute_corrected_asymmetric_loss(self, premise_hyp: torch.Tensor, hypothesis_hyp: torch.Tensor, 
                                        labels: torch.Tensor, label_strs: List[str]) -> torch.Tensor:
        """
        Corrected asymmetric loss: Both entailment and contradiction have HIGH asymmetry
        
        Key insight:
        - Entailment: HIGH asymmetry with specific pattern (low forward, high backward)
        - Neutral: LOW asymmetry (symmetric - both directions unrelated)
        - Contradiction: HIGH asymmetry with different pattern (high forward, variable backward)
        """
        energy_dict = self.model.compute_bidirectional_energies(premise_hyp, hypothesis_hyp)
        
        asymmetric_loss = 0.0
        for i, label_str in enumerate(label_strs):
            forward_e = energy_dict['forward_energy'][i]
            backward_e = energy_dict['backward_energy'][i]
            asymmetric_e = energy_dict['asymmetric_energy'][i]
            asymmetry_measure = torch.abs(forward_e - backward_e)
            
            if label_str == 'entailment':
                # SPECIFIC asymmetric pattern: low forward, high backward
                asymmetric_loss += torch.nn.functional.mse_loss(forward_e, torch.tensor(0.1, device=self.device))
                asymmetric_loss += torch.nn.functional.mse_loss(backward_e, torch.tensor(0.8, device=self.device))
                # Encourage HIGH asymmetry magnitude
                asymmetric_loss += torch.nn.functional.mse_loss(asymmetry_measure, torch.tensor(0.7, device=self.device))
                
            elif label_str == 'neutral':
                # LOW asymmetry (symmetric)
                target_energy = torch.tensor(0.6, device=self.device)
                asymmetric_loss += torch.nn.functional.mse_loss(forward_e, target_energy)
                asymmetric_loss += torch.nn.functional.mse_loss(backward_e, target_energy)
                # Encourage LOW asymmetry magnitude - key insight!
                asymmetric_loss += torch.nn.functional.mse_loss(asymmetry_measure, torch.tensor(0.0, device=self.device))
                
            elif label_str == 'contradiction':
                # DIFFERENT asymmetric pattern: high forward, variable backward
                asymmetric_loss += torch.nn.functional.mse_loss(forward_e, torch.tensor(0.9, device=self.device))
                # Allow variable backward energy (contradiction can have different patterns)
                
                # Encourage HIGH asymmetry magnitude (similar to entailment)
                asymmetric_loss += torch.nn.functional.mse_loss(asymmetry_measure, torch.tensor(0.6, device=self.device))
                
                # Also encourage high asymmetric energy
                asymmetric_loss += torch.nn.functional.mse_loss(asymmetric_e, torch.tensor(0.8, device=self.device))
        
        return asymmetric_loss / len(label_strs)


    def _compute_energy_statistics(self, energy_dict: Dict[str, torch.Tensor], 
                                 label_strs: List[str]) -> Dict[str, Dict[str, float]]:
        """Compute energy statistics for monitoring"""
        stats = {'entailment': [], 'neutral': [], 'contradiction': []}
        
        forward_energies = energy_dict['forward_energy']
        backward_energies = energy_dict['backward_energy']
        asymmetry_measures = energy_dict['asymmetry_measure']
        asymmetric_energies = energy_dict['asymmetric_energy']

        # DEBUG: Print some sample values
        # if len(forward_energies) > 0:
        #     print(f"DEBUG: Sample forward energies: {forward_energies[:3].detach().cpu().numpy()}")
        #     print(f"DEBUG: Energy range: {forward_energies.min().item():.4f} to {forward_energies.max().item():.4f}")
            # flush_output()
        
        for i, label_str in enumerate(label_strs):
            if label_str in stats:
                stats[label_str].append({
                    'forward_energy': forward_energies[i].item(),
                    'backward_energy': backward_energies[i].item(),
                    'asymmetry_measure': asymmetry_measures[i].item(),
                    'asymmetric_energy': asymmetric_energies[i].item()
                })
        
        return stats

    def train_epoch(self, dataloader, margin: float=1.0) -> float:
        """Train one epoch with enhanced 3-way loss and corrected asymmetric patterns"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Track parameter values during training
        epoch_params = []
        
        for batch in dataloader:
            premise_embs = batch['premise_emb'].to(self.device)
            hypothesis_embs = batch['hypothesis_emb'].to(self.device)
            labels = batch['label'].to(self.device)
            label_strs = batch['label_str']
            
            self.optimizer.zero_grad()
            
            # Compute enhanced 3-way hyperbolic loss
            loss, forward_energies, energy_stats = self.compute_enhanced_hyperbolic_loss(
                premise_embs, hypothesis_embs, labels, label_strs, margin
            )
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Track learned parameters
            if num_batches % 10 == 0:  # Every 10 batches
                params = self.model.get_learned_parameters_summary()
                epoch_params.append(params)
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        # Store parameter evolution
        if epoch_params:
            avg_params = {k: np.mean([p[k] for p in epoch_params]) for k in epoch_params[0].keys()}
            self.parameter_history.append(avg_params)
        
        return avg_loss
    
    def evaluate(self, dataloader, margin: float=1.0) -> Tuple[float, Dict]:
        """Evaluate model with enhanced 3-way loss and energy analysis - FIXED VERSION"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
    
        # Collect individual energy records by label for detailed analysis
        all_energy_records = {'entailment': [], 'neutral': [], 'contradiction': []}
    
        with torch.no_grad():
            for batch in dataloader:
                premise_embs = batch['premise_emb'].to(self.device)
                hypothesis_embs = batch['hypothesis_emb'].to(self.device)
                labels = batch['label'].to(self.device)
                label_strs = batch['label_str']
            
                loss, forward_energies, energy_stats = self.compute_enhanced_hyperbolic_loss(
                    premise_embs, hypothesis_embs, labels, label_strs, margin
                )
            
                total_loss += loss.item()
                num_batches += 1
            
                # Accumulate individual energy records (not summary stats)
                for label in all_energy_records:
                    if label in energy_stats:
                        # energy_stats[label] contains individual records from _compute_energy_statistics
                        # We need to extract the individual records, not the summary
                        all_energy_records[label].extend(energy_stats[label])
    
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
    
        # Compute final summary statistics from all collected records
        summary_stats = {}
        for label, energy_list in all_energy_records.items():
            if energy_list:
                # Now energy_list contains individual energy dictionaries
                forward_energies = [e['forward_energy'] for e in energy_list]
                backward_energies = [e['backward_energy'] for e in energy_list]
                asymmetries = [e['asymmetry_measure'] for e in energy_list]
                asym_energies = [e['asymmetric_energy'] for e in energy_list]
            
                summary_stats[label] = {
                    'count': len(energy_list),
                    'forward_energy': {
                        'mean': np.mean(forward_energies),
                        'std': np.std(forward_energies)
                    },
                    'backward_energy': {
                        'mean': np.mean(backward_energies),
                        'std': np.std(backward_energies)
                    },
                    'asymmetry_measure': {
                        'mean': np.mean(asymmetries),
                        'std': np.std(asymmetries)
                    },
                    'asymmetric_energy': {
                        'mean': np.mean(asym_energies),
                        'std': np.std(asym_energies)
                    }
                }
    
        self.energy_rankings.append(summary_stats)
    
        return avg_loss, summary_stats



def train_pure_hyperbolic_order_embeddings(processed_data_path: str, output_dir: str = "models/",
                                         epochs: int = 50, batch_size: int = 32, order_dim: int = 50,
                                         asymmetry_weight: float = 0.2, lr: float=1e-3, margin: float=1.0, random_seed: int = 42):
    """
    Train pure hyperbolic order embedding model with learnable energy parameters
    
    Args:
        processed_data_path: Path to processed BERT embeddings
        output_dir: Directory to save model
        epochs: Number of training epochs
        batch_size: Training batch size
        order_dim: Hyperbolic embedding dimension
        asymmetry_weight: Weight for asymmetric loss component
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (trained_model, trainer)
    """
    
    generator = set_random_seed(random_seed)
    
    print(f"Loading data from {processed_data_path}")
    processed_data = torch.load(processed_data_path, weights_only=False)
    
    # Create dataset and dataloaders
    dataset = EntailmentDataset(processed_data)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training on {train_size} samples, validating on {val_size} samples")
    
    # Initialize pure hyperbolic model and trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PureHyperbolicLearnableOrderEmbeddingModel(
        bert_dim=768, 
        order_dim=order_dim, 
        asymmetry_weight=asymmetry_weight
    )
    trainer = PureHyperbolicOrderEmbeddingTrainer(model, device, lr)
    
    print(f"Training pure hyperbolic model on {device}")
    print(f"Initial learned parameters: {model.get_learned_parameters_summary()}")
    
    # Training loop
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader, margin)
        
        # Validate
        val_loss, energy_stats = trainer.evaluate(val_loader, margin)
        trainer.scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Print current learned parameters
            current_params = model.get_learned_parameters_summary()
            print(f"Learned parameters: {current_params}")
            
            if energy_stats:
                print("  Energy Rankings:")
                for label, stats in energy_stats.items():
                    if 'forward_energy' in stats:
                        print(f"    {label}: Forward={stats['forward_energy']['mean']:.4f}±{stats['forward_energy']['std']:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, f"best_pure_hyperbolic_order_embedding_model.pt")
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'bert_dim': 768,
                    'order_dim': order_dim,
                    'asymmetry_weight': asymmetry_weight
                },
                'best_val_loss': best_val_loss,
                'epoch': epoch,
                'learned_parameters': model.get_learned_parameters_summary(),
                'training_history': {
                    'train_losses': trainer.train_losses,
                    'val_losses': trainer.val_losses,
                    'parameter_history': trainer.parameter_history
                }
            }, model_path)
            
            print(f"Saved best model to {model_path}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break
    
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    print(f"Final learned parameters: {model.get_learned_parameters_summary()}")
    
    return model, trainer

def validate_hyperbolic_energy_rankings(trainer: PureHyperbolicOrderEmbeddingTrainer) -> bool:
    """
    Validate that hyperbolic energy rankings follow expected hierarchy
    
    Args:
        trainer: Trained pure hyperbolic trainer
        
    Returns:
        bool: True if energy rankings are correct (entailment < neutral < contradiction)
    """
    
    if not trainer.energy_rankings:
        print("No energy rankings available for validation")
        return False
    
    # Get final energy rankings
    final_rankings = trainer.energy_rankings[-1]
    
    print("\n=== HYPERBOLIC ENERGY RANKING VALIDATION ===")
    
    # Check if all labels are present
    required_labels = ['entailment', 'neutral', 'contradiction']
    missing_labels = [label for label in required_labels if label not in final_rankings]
    
    if missing_labels:
        print(f"Missing labels in final rankings: {missing_labels}")
        return False
    
    # Extract mean forward energies for comparison
    try:
        entailment_energy = final_rankings['entailment']['forward_energy']['mean']
        neutral_energy = final_rankings['neutral']['forward_energy']['mean']
        contradiction_energy = final_rankings['contradiction']['forward_energy']['mean']
        
        print(f"Pure Hyperbolic Energy Rankings:")
        print(f"   Entailment:    {entailment_energy:.4f}")
        print(f"   Neutral:       {neutral_energy:.4f}")
        print(f"   Contradiction: {contradiction_energy:.4f}")
        
        # Check expected hierarchy: entailment < neutral < contradiction
        ranking_correct = (entailment_energy < neutral_energy < contradiction_energy)
        
        if ranking_correct:
            print("SUCCESS: Hyperbolic energy hierarchy is CORRECT!")
            print("Core hypothesis validated: entailment < neutral < contradiction")
            
            # Additional analysis of learned parameters
            if hasattr(trainer.model, 'get_learned_parameters_summary'):
                learned_params = trainer.model.get_learned_parameters_summary()
                print(f"\n📋 Final Learned Parameters:")
                for param_name, value in learned_params.items():
                    print(f"   {param_name}: {value:.4f}")
        else:
            print("Energy rankings are INCORRECT")
            print("Expected: entailment < neutral < contradiction")
            print("This suggests the pure hyperbolic approach may need parameter tuning")
            
            # Show what went wrong
            if entailment_energy >= neutral_energy:
                print(f"   Problem: Entailment energy ({entailment_energy:.4f}) >= Neutral energy ({neutral_energy:.4f})")
            if neutral_energy >= contradiction_energy:
                print(f"   Problem: Neutral energy ({neutral_energy:.4f}) >= Contradiction energy ({contradiction_energy:.4f})")
        
        return ranking_correct
        
    except KeyError as e:
        print(f"Error accessing energy rankings: {e}")
        print("Available keys in final rankings:")
        for label, stats in final_rankings.items():
            print(f"  {label}: {list(stats.keys())}")
        return False

def plot_pure_hyperbolic_training_progress(trainer: PureHyperbolicOrderEmbeddingTrainer, save_path: str = "plots/"):
    """
    Plot training progress for pure hyperbolic order embeddings including learned parameters
    
    Args:
        trainer: Trained pure hyperbolic trainer
        save_path: Directory to save plots
    """
    
    os.makedirs(save_path, exist_ok=True)
    
    # Create comprehensive figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Training/Validation Loss
    epochs = range(1, len(trainer.train_losses) + 1)
    ax1.plot(epochs, trainer.train_losses, 'b-', label='Training Loss', linewidth=2)
    if trainer.val_losses:
        ax1.plot(epochs, trainer.val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Pure Hyperbolic Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Energy Rankings Over Time
    if trainer.energy_rankings:
        entail_means = []
        neutral_means = []
        contra_means = []
        
        for ranking in trainer.energy_rankings:
            entail_means.append(ranking.get('entailment', {}).get('forward_energy', {}).get('mean', 0))
            neutral_means.append(ranking.get('neutral', {}).get('forward_energy', {}).get('mean', 0))
            contra_means.append(ranking.get('contradiction', {}).get('forward_energy', {}).get('mean', 0))
        
        epochs_val = range(1, len(entail_means) + 1)
        ax2.plot(epochs_val, entail_means, 'g-', label='Entailment', linewidth=3)
        ax2.plot(epochs_val, neutral_means, 'b-', label='Neutral', linewidth=3)
        ax2.plot(epochs_val, contra_means, 'r-', label='Contradiction', linewidth=3)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Hyperbolic Order Energy')
        ax2.set_title('Hyperbolic Energy Rankings Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learned Parameters Evolution
    if trainer.parameter_history:
        param_names = list(trainer.parameter_history[0].keys())
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, param_name in enumerate(param_names):
            param_values = [epoch_params[param_name] for epoch_params in trainer.parameter_history]
            epochs_params = range(1, len(param_values) + 1)
            ax3.plot(epochs_params, param_values, color=colors[i % len(colors)], 
                    label=param_name.replace('_', ' ').title(), linewidth=2)
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Parameter Value')
        ax3.set_title('Learned Energy Function Parameters')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final Energy Distribution Comparison
    if trainer.energy_rankings:
        final_rankings = trainer.energy_rankings[-1]
        labels = []
        forward_energies = []
        asymmetries = []
        
        for label, stats in final_rankings.items():
            if 'forward_energy' in stats:
                labels.append(label.capitalize())
                forward_energies.append(stats['forward_energy']['mean'])
                asymmetries.append(stats.get('asymmetry_measure', {}).get('mean', 0))
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax4.bar(x - width/2, forward_energies, width, label='Forward Energy', alpha=0.8, color='skyblue')
        ax4.bar(x + width/2, asymmetries, width, label='Asymmetry Measure', alpha=0.8, color='lightcoral')
        
        ax4.set_xlabel('Relationship Type')
        ax4.set_ylabel('Energy')
        ax4.set_title('Final Pure Hyperbolic Energy Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'pure_hyperbolic_order_embedding_training.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Pure hyperbolic training plots saved to {save_path}")


def test_pure_hyperbolic_order_embeddings():
    """
    Test pure hyperbolic order embeddings with learnable parameters
    """
    print("Testing Pure Hyperbolic Order Embeddings...")
    
    # Check if processed data exists
    processed_data_path = "data/processed/snli_full_standard_BERT.pt"
    if not os.path.exists(processed_data_path):
        print(f"Processed data not found at {processed_data_path}")
        print("Please run text_processing.py first!")
        return
    
    # Train model with optimized hyperparameters
    model, trainer = train_pure_hyperbolic_order_embeddings(
        processed_data_path=processed_data_path,
        epochs=80,
        batch_size=32,
        order_dim=50,  # Keep same as original for comparison
        asymmetry_weight=0.3,  # Slightly higher for better asymmetry learning
        random_seed=42
    )
    
    # Validate rankings
    ranking_correct = validate_hyperbolic_energy_rankings(trainer)
    
    # Plot progress
    plot_pure_hyperbolic_training_progress(trainer)
    
    # Final summary
    if ranking_correct:
        print("\nSUCCESS: Pure hyperbolic order embeddings working correctly!")
        print("Ready for advanced hyperbolic entailment analysis!")
        
        # Show final learned parameters
        final_params = model.get_learned_parameters_summary()
        print(f"\nFinal Optimized Parameters:")
        for param_name, value in final_params.items():
            print(f"   {param_name}: {value:.4f}")
    else:
        print("\nWARNING: Pure hyperbolic energy rankings need adjustment")
        print("Consider tuning hyperparameters or increasing training epochs")
    
    return model, trainer


def test_energy_with_large_variation():
    """Test with expectation of larger embedding values"""
    model = PureHyperbolicLearnableOrderEmbeddingModel(order_dim=10, asymmetry_weight=0.1)
    
    # Create more different inputs
    premise1 = torch.randn(2, 768) * 2.0   # Larger variation
    premise2 = torch.randn(2, 768) * 2.0 + 3.0  
    
    hypothesis1 = torch.randn(2, 768) * 2.0  
    hypothesis2 = torch.randn(2, 768) * 2.0 - 3.0  
    
    print("=== TESTING LARGE HYPERBOLIC EMBEDDING VARIATION ===")
    
    premise_hyp1 = model(premise1)
    premise_hyp2 = model(premise2)
    hypothesis_hyp1 = model(hypothesis1)
    hypothesis_hyp2 = model(hypothesis2)
    
    print(f"Premise embeddings 1: {premise_hyp1[0, :3]}")
    print(f"Premise embeddings 2: {premise_hyp2[0, :3]}")
    print(f"Hypothesis embeddings 1: {hypothesis_hyp1[0, :3]}")
    print(f"Hypothesis embeddings 2: {hypothesis_hyp2[0, :3]}")
    
    # Check embedding magnitudes
    p1_norm = torch.norm(premise_hyp1[0]).item()
    p2_norm = torch.norm(premise_hyp2[0]).item()
    h1_norm = torch.norm(hypothesis_hyp1[0]).item()
    h2_norm = torch.norm(hypothesis_hyp2[0]).item()
    
    print(f"\nEmbedding norms:")
    print(f"Premise 1 norm: {p1_norm:.6f}")
    print(f"Premise 2 norm: {p2_norm:.6f}")
    print(f"Hypothesis 1 norm: {h1_norm:.6f}")
    print(f"Hypothesis 2 norm: {h2_norm:.6f}")
    
    # Check differences
    premise_diff = torch.norm(premise_hyp1[0] - premise_hyp2[0]).item()
    hypothesis_diff = torch.norm(hypothesis_hyp1[0] - hypothesis_hyp2[0]).item()
    
    print(f"\nEmbedding differences:")
    print(f"Premise difference: {premise_diff:.6f}")
    print(f"Hypothesis difference: {hypothesis_diff:.6f}")
    
    # Test energies
    energy_dict1 = model.compute_bidirectional_energies(premise_hyp1, hypothesis_hyp1)
    energy_dict2 = model.compute_bidirectional_energies(premise_hyp2, hypothesis_hyp2)
    
    print(f"\nEnergy comparison:")
    print(f"Energy set 1: {energy_dict1['forward_energy']}")
    print(f"Energy set 2: {energy_dict2['forward_energy']}")
    
    energy_diff = torch.norm(energy_dict1['forward_energy'] - energy_dict2['forward_energy']).item()
    print(f"Energy difference: {energy_diff:.6f}")
    
    # Success criteria: embeddings should be > 1e-3, energy differences > 0.1
    embedding_ok = min(p1_norm, p2_norm, h1_norm, h2_norm) > 1e-3
    difference_ok = energy_diff > 0.1
    
    if embedding_ok and difference_ok:
        print("✅ SUCCESS: Large embeddings and meaningful energy differences!")
        return True
    else:
        print("❌ Still need larger embeddings or energy differences")
        print(f"   Min embedding norm: {min(p1_norm, p2_norm, h1_norm, h2_norm):.6f} (need > 1e-3)")
        print(f"   Energy difference: {energy_diff:.6f} (need > 0.1)")
        return False


if __name__ == "__main__":
    test_energy_with_large_variation()
    
