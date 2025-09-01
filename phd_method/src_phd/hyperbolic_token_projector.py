import torch
import torch.nn as nn
import geoopt
import numpy as np
import os
from typing import Dict, List, Tuple
import random
import pickle
from pathlib import Path
from tqdm import tqdm # For progress bars
from independent_order_model import OrderEmbeddingModel
from independent_asymmetry_model import AsymmetryTransformModel

def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TokenLevelHyperbolicProjector(nn.Module):
    """
    Projects 768D token-level order embeddings to hyperbolic space (Poincaré ball)
    """

    def __init__(self, order_dim: int = 768, hyperbolic_dim: int = 768, compression_ratio: float = 1.0):
        super().__init__()
        self.order_dim = order_dim
        self.hyperbolic_dim = int(hyperbolic_dim * compression_ratio)
        self.compression_ratio = compression_ratio

        # Poincaré ball manifold
        self.ball = geoopt.PoincareBall()

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(order_dim, self.hyperbolic_dim),
            nn.Tanh()
        )

        # Scaling factor to ensure points stay inside unit ball
        self.scale_factor = 0.9

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for hyperbolic stability"""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, order_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project token-level order embeddings to Poincaré ball
        Args:
            order_embeddings: [batch_size, num_tokens, 768] Token-level order embeddings
        Returns:
            hyperbolic_embeddings: [batch_size, num_tokens, hyperbolic_dim] points in Poincaré ball
        """
        projected = self.projection(order_embeddings)
        scaled = projected * self.scale_factor
        hyperbolic_embeddings = self.ball.expmap0(scaled)

        return hyperbolic_embeddings

    def hyperbolic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute hyperbolic distances between points"""
        return self.ball.dist(x, y)


class HyperbolicProjectorTrainer:
    """
    Train the hyperbolic projector using SNLI train data
    """
    
    def __init__(self, 
                 order_model_path: str,
                 asymmetry_model_path: str,
                 hyperbolic_dim: int = 768,
                 compression_ratio: float = 1.0):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.order_model = self._load_order_model(order_model_path)
        self.asymmetry_model = self._load_asymmetry_model(asymmetry_model_path)
        
        self.hyperbolic_projector = TokenLevelHyperbolicProjector(
            order_dim=768,
            hyperbolic_dim=hyperbolic_dim,
            compression_ratio=compression_ratio
        ).to(self.device)
        
        print(f"Hyperbolic trainer initialized on {self.device}")
        
    def _load_order_model(self, model_path: str):
        if not os.path.exists(model_path):
            print(f"Warning: Order model not found at {model_path}. Using a dummy model.")
            return OrderEmbeddingModel(hidden_size=768).to(self.device).eval()
        
        print(f"Loading order model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model = OrderEmbeddingModel(hidden_size=768)
        
        state_dict_key = next((k for k in ['order_model_state_dict', 'model_state_dict', 'state_dict'] if k in checkpoint), None)
        if state_dict_key:
            model.load_state_dict(checkpoint[state_dict_key])
        else:
            raise KeyError(f"No recognizable state dict key found in checkpoint. Available keys: {list(checkpoint.keys())}")
        
        model.to(self.device)
        model.eval()
        print(f"✅ Order model loaded (768D→768D)")
        return model
        
    def _load_asymmetry_model(self, model_path: str):
        if not os.path.exists(model_path):
            print(f"Warning: Asymmetry model not found at {model_path}. Using a dummy model.")
            return AsymmetryTransformModel(hidden_size=768).to(self.device).eval()

        print(f"Loading asymmetry model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model = AsymmetryTransformModel(hidden_size=768)

        state_dict_key = next((k for k in ['asymmetry_model_state_dict', 'model_state_dict', 'state_dict'] if k in checkpoint), None)
        if state_dict_key:
            model.load_state_dict(checkpoint[state_dict_key])
        else:
            raise KeyError(f"No recognizable state dict key found in checkpoint. Available keys: {list(checkpoint.keys())}")

        model.to(self.device)
        model.eval()
        print(f"✅ Asymmetry model loaded (768D→768D)")
        return model

    def hyperbolic_entailment_loss(self, 
                                 premise_hyp: torch.Tensor, 
                                 hypothesis_hyp: torch.Tensor, 
                                 premise_mask: torch.Tensor,
                                 hypothesis_mask: torch.Tensor,
                                 labels: torch.Tensor) -> torch.Tensor:
        """
        Efficient loss function for entailment cone structure in hyperbolic space.
        """
        # Expand dims for broadcasting
        p_expanded = premise_hyp.unsqueeze(2)  # [B, Np, 1, D]
        h_expanded = hypothesis_hyp.unsqueeze(1)  # [B, 1, Nh, D]
        
        # Compute pairwise distances
        dist_matrix = self.hyperbolic_projector.hyperbolic_distance(p_expanded, h_expanded) # [B, Np, Nh]
        
        # Create a mask for valid token pairs
        mask = premise_mask.unsqueeze(2) * hypothesis_mask.unsqueeze(1) # [B, Np, Nh]
        
        # Apply mask and compute mean distance per sample
        masked_dist = dist_matrix * mask
        sum_dist = masked_dist.sum(dim=[1, 2])
        num_pairs = mask.sum(dim=[1, 2])
        avg_distance = sum_dist / (num_pairs + 1e-8)
        
        # Target distances for entailment cone structure
        target_distances = torch.tensor([0.1, 0.5, 0.9], device=self.device)[labels]
        
        loss = nn.functional.mse_loss(avg_distance, target_distances)
        return loss

    def _convert_label_to_int(self, label):
        label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        return label_map.get(str(label).lower(), 0)
    
    def train(self,
              train_data_path: str,
              save_dir: str,
              val_split: float = 0.2,
              num_epochs: int = 50,
              batch_size: int = 32, # Reduced default batch size
              learning_rate: float = 0.001,
              patience: int = 10):
        
        print(f"Training hyperbolic projector for {num_epochs} epochs...")
        
        try:
            with open(train_data_path, 'rb') as f:
                all_data = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: Training data not found at {train_data_path}")
            return
            
        indices = np.random.permutation(len(all_data['labels']))
        val_size = int(len(indices) * val_split)
        train_indices, val_indices = indices[val_size:], indices[:val_size]
        
        print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
        
        optimizer = geoopt.optim.RiemannianAdam(self.hyperbolic_projector.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.hyperbolic_projector.train()
            train_losses = []
            
            train_batches = [train_indices[i:i+batch_size] for i in range(0, len(train_indices), batch_size)]
            
            for batch_indices in tqdm(train_batches, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
                
                premise_tensors = [all_data['premise_tokens'][i] for i in batch_indices]
                hypothesis_tensors = [all_data['hypothesis_tokens'][i] for i in batch_indices]
                labels = torch.tensor([self._convert_label_to_int(all_data['labels'][i]) for i in batch_indices], device=self.device)
                
                premise_padded = nn.utils.rnn.pad_sequence(premise_tensors, batch_first=True).to(self.device)
                hypothesis_padded = nn.utils.rnn.pad_sequence(hypothesis_tensors, batch_first=True).to(self.device)

                premise_mask = (premise_padded.sum(dim=-1) != 0).float()
                hypothesis_mask = (hypothesis_padded.sum(dim=-1) != 0).float()

                with torch.no_grad():
                    premise_order = self.order_model(premise_padded)
                    hypothesis_order = self.order_model(hypothesis_padded)
                    premise_asym = self.asymmetry_model(premise_order)
                    hypothesis_asym = self.asymmetry_model(hypothesis_order)
                
                premise_hyp = self.hyperbolic_projector(premise_asym)
                hypothesis_hyp = self.hyperbolic_projector(hypothesis_asym)

                loss = self.hyperbolic_entailment_loss(premise_hyp, hypothesis_hyp, premise_mask, hypothesis_mask, labels)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.hyperbolic_projector.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(loss.item())

            self.hyperbolic_projector.eval()
            val_losses = []
            val_batches = [val_indices[i:i+batch_size] for i in range(0, len(val_indices), batch_size)]
            
            with torch.no_grad():
                for batch_indices in tqdm(val_batches, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                    premise_tensors = [all_data['premise_tokens'][i] for i in batch_indices]
                    hypothesis_tensors = [all_data['hypothesis_tokens'][i] for i in batch_indices]
                    labels = torch.tensor([self._convert_label_to_int(all_data['labels'][i]) for i in batch_indices], device=self.device)

                    premise_padded = nn.utils.rnn.pad_sequence(premise_tensors, batch_first=True).to(self.device)
                    hypothesis_padded = nn.utils.rnn.pad_sequence(hypothesis_tensors, batch_first=True).to(self.device)
                    
                    premise_mask = (premise_padded.sum(dim=-1) != 0).float()
                    hypothesis_mask = (hypothesis_padded.sum(dim=-1) != 0).float()

                    premise_order = self.order_model(premise_padded)
                    hypothesis_order = self.order_model(hypothesis_padded)
                    premise_asym = self.asymmetry_model(premise_order)
                    hypothesis_asym = self.asymmetry_model(hypothesis_order)
                    
                    premise_hyp = self.hyperbolic_projector(premise_asym)
                    hypothesis_hyp = self.hyperbolic_projector(hypothesis_asym)
                    
                    val_loss = self.hyperbolic_entailment_loss(premise_hyp, hypothesis_hyp, premise_mask, hypothesis_mask, labels)
                    val_losses.append(val_loss.item())
            
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_model(save_dir, 'mnli_best_hyperbolic_projector.pt', epoch+1, avg_val_loss)
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
            scheduler.step(avg_val_loss)
            
        print(f"✅ Training completed! Best validation loss: {best_val_loss:.4f}")

    def save_model(self, save_dir: str, filename: str, epoch: int, val_loss: float):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'projector_state_dict': self.hyperbolic_projector.state_dict(),
            'val_loss': val_loss,
            'model_config': {
                'order_dim': self.hyperbolic_projector.order_dim,
                'hyperbolic_dim': self.hyperbolic_projector.hyperbolic_dim,
                'compression_ratio': self.hyperbolic_projector.compression_ratio
            }
        }
        torch.save(checkpoint, save_path)
        print(f"✅ Model saved to {save_path} (Val Loss: {val_loss:.4f})")

def main():
    set_random_seed(42)
    print("Training Token-Level Hyperbolic Projector")
    print("=" * 50)
    
    config = {
        'order_model_path': "MSc_Topology_Codebase/phd_method/models/separate_models/mnli_order_embedding_model_separate_margins.pt",
        'asymmetry_model_path': "MSc_Topology_Codebase/phd_method/models/separate_models/mnli_asymmetry_transform_model_(match_SNLI_v2).pt",
        'train_data_path': "/vol/bitbucket/ahb24/tda_entailment_new/mnli_train_sbert_tokens.pkl",
        'save_dir': "MSc_Topology_Codebase/phd_method/models/separate_models/",
        'hyperbolic_dim': 768,
        'compression_ratio': 1.0,
        'val_split': 0.2,
        'num_epochs': 50,
        'batch_size': 32, # Start with a smaller batch size
        'learning_rate': 0.001,
        'patience': 10
    }
    
    trainer = HyperbolicProjectorTrainer(
        order_model_path=config['order_model_path'],
        asymmetry_model_path=config['asymmetry_model_path'],
        hyperbolic_dim=config['hyperbolic_dim'],
        compression_ratio=config['compression_ratio']
    )
    
    trainer.train(
        train_data_path=config['train_data_path'],
        save_dir=config['save_dir'],
        val_split=config['val_split'],
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        patience=config['patience']
    )

if __name__ == "__main__":
    main()