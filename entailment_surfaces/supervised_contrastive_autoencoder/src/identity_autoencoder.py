import torch
import torch.nn as nn


class IdentityAutoencoder(nn.Module):
    """
    Identity autoencoder that does absolutely nothing to the input.
    Perfect baseline - keeps 1536D SBERT embeddings completely unchanged.
    """
    
    def __init__(self, input_dim=1536):
        """
        Initialize identity autoencoder
        
        Args:
            input_dim (int): Input dimension (should match your SBERT embeddings)
        """
        super(IdentityAutoencoder, self).__init__()
        self.input_dim = input_dim
        
        # No actual layers needed - just store the dimension
        print(f"IdentityAutoencoder initialized with input_dim={input_dim}")
    
    def forward(self, x):
        """
        Forward pass - return input unchanged
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            latent: Same as input (no compression)
            reconstructed: Same as input (perfect reconstruction)
        """
        # Both latent and reconstructed are identical to input
        return x, x
    
    def encode(self, x):
        """
        Encode function - return input unchanged
        """
        return x
    
    def decode(self, latent):
        """
        Decode function - return input unchanged
        """
        return latent


def test_identity_autoencoder():
    """Test the identity autoencoder with sample data"""
    print("Testing Identity Autoencoder...")
    
    # Create model
    model = IdentityAutoencoder(input_dim=1536)
    
    # Create sample SBERT embeddings
    batch_size = 32
    sample_embeddings = torch.randn(batch_size, 1536)
    
    print(f"Input shape: {sample_embeddings.shape}")
    
    # Forward pass
    latent, reconstructed = model(sample_embeddings)
    
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Verify they're identical
    assert torch.allclose(sample_embeddings, latent), "Latent should be identical to input"
    assert torch.allclose(sample_embeddings, reconstructed), "Reconstructed should be identical to input"
    assert torch.allclose(latent, reconstructed), "Latent and reconstructed should be identical"
    
    # Check reconstruction error (should be exactly 0)
    mse_loss = nn.MSELoss()
    reconstruction_error = mse_loss(reconstructed, sample_embeddings)
    print(f"Reconstruction error: {reconstruction_error.item()}")
    
    assert reconstruction_error.item() == 0.0, "Reconstruction error should be exactly 0"
    
    print("✅ Identity autoencoder test passed!")
    print("This model will give you perfect baseline metrics:")
    print("  - Reconstruction MSE: 0.0")
    print("  - Latent representations: Identical to original SBERT embeddings")
    print("  - No compression: 1536D → 1536D → 1536D")


if __name__ == "__main__":
    test_identity_autoencoder()