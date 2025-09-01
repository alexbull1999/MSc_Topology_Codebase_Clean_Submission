"""
Autoencoder model for InfoNCE + Order Embeddings approach
"""

import torch
import torch.nn as nn


class InfoNCEOrderAutoencoder(nn.Module):
    """
    Simple autoencoder for InfoNCE + Order Embeddings training
    """
    
    def __init__(self, input_dim=1536, latent_dim=75, hidden_dims=[512, 256], dropout_rate=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final encoder layer
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        self.decoder = nn.Linear(latent_dim, input_dim)

        
        print(f"InfoNCEOrderAutoencoder initialized:")
        print(f"  Input dim: {input_dim}")
        print(f"  Latent dim: {latent_dim}")
        print(f"  Hidden dims: {hidden_dims}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation to input space"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass: encode then decode"""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed
