"""
Supervised Contrastive Autoencoder Model
Core model architecture for learning entailment manifold structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveAutoencoder(nn.Module):
    """
    Autoencoder with encoder-decoder architecture for learning entailment manifold
    
    Args:
        input_dim: Dimension of input embeddings (768 for lattice containment)
        latent_dim: Dimension of latent space (50 by default)
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout probability
    """
    
    def __init__(self, input_dim=768, latent_dim=75, hidden_dims=[512, 256], dropout_rate=0.2):
        super(ContrastiveAutoencoder, self).__init__()
        
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
        
        # Build decoder (reverse of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final decoder layer back to input space
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self.apply(self._init_weights)


    def _init_weights(self, module):
        """Initialize weights using Xavier uniform initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation back to input space"""
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
    
    def get_latent_representation(self, x):
        """Get latent representation without reconstruction"""
        with torch.no_grad():
            return self.encode(x)
    
    def reconstruct(self, x):
        """Reconstruct input through encode-decode cycle"""
        with torch.no_grad():
            latent = self.encode(x)
            return self.decode(latent)
    
    def interpolate(self, z1, z2, alpha):
        """
        Interpolate between two latent representations
        
        Args:
            z1: First latent representation [latent_dim]
            z2: Second latent representation [latent_dim]
            alpha: Interpolation factor (0 = z1, 1 = z2)
            
        Returns:
            Interpolated latent representation [latent_dim]
        """
        return (1 - alpha) * z1 + alpha * z2
    
    def generate_interpolation_path(self, z1, z2, num_steps=10):
        """
        Generate interpolation path between two latent points
        
        Args:
            z1: Start latent representation [latent_dim]
            z2: End latent representation [latent_dim]
            num_steps: Number of interpolation steps
            
        Returns:
            List of interpolated latent representations
        """
        alphas = torch.linspace(0, 1, num_steps)
        path = []
        
        for alpha in alphas:
            interpolated = self.interpolate(z1, z2, alpha)
            path.append(interpolated)
        
        return path
    
    def print_architecture(self):
        """Print model architecture summary"""
        print("Contrastive Autoencoder Architecture")
        print("=" * 50)
        print(f"Input Dimension: {self.input_dim}")
        print(f"Latent Dimension: {self.latent_dim}")
        print(f"Hidden Dimensions: {self.hidden_dims}")
        print(f"Dropout Rate: {self.dropout_rate}")
        print()
        
        print("Encoder:")
        for i, layer in enumerate(self.encoder):
            print(f"  {i}: {layer}")
        print()
        
        print("Decoder:")
        for i, layer in enumerate(self.decoder):
            print(f"  {i}: {layer}")
        print()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")


def test_model():
    """Test model instantiation and forward pass"""
    print("Testing ContrastiveAutoencoder Model")
    print("=" * 40)
    
    # Create model
    model = ContrastiveAutoencoder(
        input_dim=768,
        latent_dim=75,
        hidden_dims=[512, 256],
        dropout_rate=0.2
    )
    
    # Print architecture
    model.print_architecture()
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 768)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    latent, reconstructed = model(x)
    
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Test individual components
    latent_only = model.get_latent_representation(x)
    reconstructed_only = model.reconstruct(x)
    
    print(f"Latent only shape: {latent_only.shape}")
    print(f"Reconstructed only shape: {reconstructed_only.shape}")
    
    # Test interpolation
    z1 = latent[0]
    z2 = latent[1]
    
    interpolated = model.interpolate(z1, z2, 0.5)
    print(f"Interpolated shape: {interpolated.shape}")
    
    path = model.generate_interpolation_path(z1, z2, num_steps=5)
    print(f"Interpolation path length: {len(path)}")
    print(f"Each path point shape: {path[0].shape}")
    
    print("\nModel test completed successfully!")


if __name__ == "__main__":
    test_model()