"""
Projection Layer Module
Bridges vision and language embedding spaces - this is the key trainable component!
"""

import torch
import torch.nn as nn
from typing import Optional


class ProjectionLayer(nn.Module):
    """
    Multi-layer perceptron to project vision features to language space
    This is the KEY component we'll train to align vision and language
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 2048,
        output_dim: int = 768,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Args:
            input_dim: Vision encoder output dimension
            hidden_dim: Hidden layer dimension
            output_dim: Language model input dimension
            num_layers: Number of MLP layers
            dropout: Dropout probability
            activation: Activation function (gelu, relu, silu)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Choose activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build MLP layers
        layers = []
        
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # First layer
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout),
            ])
            
            # Hidden layers
            for _ in range(num_layers - 2):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    self.activation,
                    nn.Dropout(dropout),
                ])
            
            # Output layer
            layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.projection = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features to language space
        
        Args:
            vision_features: [B, input_dim] or [B, seq_len, input_dim]
            
        Returns:
            Projected features: [B, output_dim] or [B, seq_len, output_dim]
        """
        projected = self.projection(vision_features)
        projected = self.layer_norm(projected)
        return projected
    
    def get_num_params(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing Projection Layer...")
    
    proj = ProjectionLayer(
        input_dim=768,
        hidden_dim=2048,
        output_dim=768,
        num_layers=2
    )
    
    print(f"Parameters: {proj.get_num_params():,}")
    
    batch_size = 4
    vision_features = torch.randn(batch_size, 768)
    
    with torch.no_grad():
        output = proj(vision_features)
    
    print(f"Input shape: {vision_features.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\n✓ Projection layer test passed!")