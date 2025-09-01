"""
Contrastive Autoencoder Model
Clean implementation for global dataset training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveAutoencoder(nn.Module):
    """
    Supervised contrastive autoencoder for learning entailment manifold structure
    """
    
    def __init__(self, input_dim=768, latent_dim=75, hidden_dims=None, dropout_rate=0.2):
        """
        Initialize the contrastive autoencoder
        
        Args:
            input_dim: Input embedding dimension
            latent_dim: Latent space dimension
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super(ContrastiveAutoencoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final encoder layer to latent space
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (reverse of encoder). <--- THIS LED TO DECODER DEGENERACY 
        # decoder_layers = []
        # prev_dim = latent_dim
        
        # for hidden_dim in reversed(hidden_dims):
        #     decoder_layers.extend([
        #         nn.Linear(prev_dim, hidden_dim),
        #         nn.ReLU(),
        #         nn.Dropout(dropout_rate)
        #     ])
        #     prev_dim = hidden_dim
        
        # # Final decoder layer to reconstruct input
        # decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        # self.decoder = nn.Sequential(*decoder_layers)

        self.decoder = nn.Linear(latent_dim, input_dim) 
        
        print(f"ContrastiveAutoencoder initialized:")
        print(f"  Input dim: {input_dim}")
        print(f"  Latent dim: {latent_dim}")
        print(f"  Hidden dims: {hidden_dims}")
        print(f"  Dropout rate: {dropout_rate}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def encode(self, x):
        """
        Encode input to latent space
        
        Args:
            x: Input embeddings [batch_size, input_dim]
            
        Returns:
            latent: Latent representations [batch_size, latent_dim]
        """
        return self.encoder(x)
    
    def decode(self, z):
        """
        Decode latent representations to input space
        
        Args:
            z: Latent representations [batch_size, latent_dim]
            
        Returns:
            reconstructed: Reconstructed embeddings [batch_size, input_dim]
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass through autoencoder
        
        Args:
            x: Input embeddings [batch_size, input_dim]
            
        Returns:
            latent: Latent representations [batch_size, latent_dim]
            reconstructed: Reconstructed embeddings [batch_size, input_dim]
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed
    
    def get_latent_representations(self, x):
        """
        Get only latent representations (for evaluation)
        
        Args:
            x: Input embeddings [batch_size, input_dim]
            
        Returns:
            latent: Latent representations [batch_size, latent_dim]
        """
        with torch.no_grad():
            return self.encode(x)


def test_model():
    """Test the model with synthetic data"""
    print("Testing ContrastiveAutoencoder...")
    
    # Test parameters
    batch_size = 32
    input_dim = 768
    latent_dim = 75
    
    # Create model
    model = ContrastiveAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=[512, 256],
        dropout_rate=0.2
    )
    
    # Test data
    x = torch.randn(batch_size, input_dim)
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    print(f"Input shape: {x.shape}")
    
    latent, reconstructed = model(x)
    
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Test individual methods
    latent_only = model.get_latent_representations(x)
    print(f"Latent only shape: {latent_only.shape}")
    
    # Test reconstruction quality
    mse_loss = F.mse_loss(reconstructed, x)
    print(f"Reconstruction MSE: {mse_loss.item():.6f}")
    
    print("✅ Model test completed successfully!")


if __name__ == "__main__":
    test_model()