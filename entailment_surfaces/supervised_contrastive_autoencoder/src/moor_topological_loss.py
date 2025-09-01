# FILE: moor_topological_loss.py

import torch
import torch.nn as nn
import numpy as np
from torch_topological.nn import VietorisRipsComplex

class MoorTopologicalLoss(nn.Module):
    """
    Implements the topological loss from the "Topological Autoencoders" paper
    by Moor et al. (2020), focusing on 0-dimensional persistence pairings.
    """
    def __init__(self):
        super().__init__()
        self.vr_complex = VietorisRipsComplex(dim=0, keep_infinite_features=False)
        print("MoorTopologicalLoss Initialized: Using 0-dimensional persistence pairings (MST edges).")

    def _get_persistence_pairings(self, x: torch.Tensor):
        if x.shape[0] < 2: return None
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
        if batch_size < 2: return torch.tensor(0.0, device=x.device)

        dist_matrix_x = torch.cdist(x, x, p=2)
        dist_matrix_z = torch.cdist(z, z, p=2)

        pi_x_np = self._get_persistence_pairings(x)
        pi_z_np = self._get_persistence_pairings(z)

        if pi_x_np is None or pi_z_np is None:
            return torch.tensor(0.0, device=x.device)

        # >>> FIX: Convert the numpy pairings from the library into PyTorch tensors
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