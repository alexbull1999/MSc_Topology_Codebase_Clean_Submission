"""
Attention-based Contrastive Autoencoder Model
Adds a self-attention layer at the bottleneck to learn global relationships.
"""

import torch
import torch.nn as nn

class AttentionAutoencoder(nn.Module):
    """
    Supervised contrastive autoencoder with a self-attention layer
    at the bottleneck to better capture topological structure.
    """
    
    def __init__(self, input_dim=768, latent_dim=50, hidden_dims=None, dropout_rate=0.2, n_heads=5):
        """
        Initialize the attention-based autoencoder
        
        Args:
            input_dim (int): Input embedding dimension.
            latent_dim (int): Latent space dimension.
            hidden_dims (list): List of hidden layer dimensions.
            dropout_rate (float): Dropout rate for regularization.
            n_heads (int): Number of attention heads. Must be a divisor of latent_dim.
        """
        super(AttentionAutoencoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        # --- Encoder (Projects input to initial latent space) ---
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # --- Self-Attention Bottleneck (Refines latent space) ---
        self.attention_layer_norm = nn.LayerNorm(latent_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=n_heads,
            dropout=dropout_rate,
            batch_first=False
        )

        # # --- Decoder (Reconstructs from refined latent space) --- <--- MIRROR WAS CAUSING DECODER DEGENERACY???
        # decoder_layers = []
        # prev_dim = latent_dim
        # for hidden_dim in reversed(hidden_dims):
        #     decoder_layers.extend([
        #         nn.Linear(prev_dim, hidden_dim),
        #         nn.ReLU(),
        #         nn.Dropout(dropout_rate)
        #     ])
        #     prev_dim = hidden_dim
        # decoder_layers.append(nn.Linear(prev_dim, input_dim))
        # self.decoder = nn.Sequential(*decoder_layers)

        self.decoder = nn.Linear(latent_dim, input_dim)

        print(f"AttentionAutoencoder initialized:")
        print(f"  Input dim: {input_dim}")
        print(f"  Latent dim: {latent_dim}")
        print(f"  Hidden dims: {hidden_dims}")
        print(f"  Attention Heads: {n_heads}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def encode(self, x):
        """
        Encodes the input into the refined latent space.
        This method now includes the self-attention step.
        """
        # 1. Pass through the standard encoder
        latent_initial = self.encoder(x)
        
        # 2. Reshape for attention and process
        # (batch, features) -> (seq_len, batch, features)
        latent_reshaped = latent_initial.unsqueeze(0)
        
        attention_output, _ = self.attention(
            query=latent_reshaped,
            key=latent_reshaped,
            value=latent_reshaped
        )
        
        # 3. Reshape back and apply residual connection + normalization
        attention_output_reshaped = attention_output.squeeze(0)
        latent_final = self.attention_layer_norm(latent_initial + attention_output_reshaped)
        
        return latent_final

    def decode(self, z):
        """
        Decodes a latent vector into the reconstructed input space.
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Full forward pass through the autoencoder.
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed
    
    def get_latent_representations(self, x):
        """
        Gets only the final latent representations (for evaluation).
        """
        with torch.no_grad():
            return self.encode(x)