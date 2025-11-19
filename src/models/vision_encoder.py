"""
Vision Encoder Module
Uses pre-trained CLIP or ViT models to extract image features
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from typing import Optional, Dict


class VisionEncoder(nn.Module):
    """
    Vision encoder using pre-trained CLIP ViT model
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        freeze: bool = True,
        image_size: int = 224
    ):
        """
        Args:
            model_name: HuggingFace model name
            freeze: Whether to freeze the encoder weights
            image_size: Input image size
        """
        super().__init__()
        
        self.model_name = model_name
        self.image_size = image_size
        
        # Load pre-trained CLIP vision model
        self.encoder = CLIPVisionModel.from_pretrained(model_name)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        
        # Get hidden dimension
        self.hidden_size = self.encoder.config.hidden_size
        
        # Freeze if specified
        if freeze:
            self.freeze()
    
    def freeze(self):
        """Freeze all parameters in the vision encoder"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print(f"✓ Vision encoder frozen")
    
    def unfreeze(self):
        """Unfreeze all parameters in the vision encoder"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print(f"✓ Vision encoder unfrozen")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through vision encoder
        
        Args:
            pixel_values: Preprocessed images [B, C, H, W]
            return_dict: Whether to return dict or tensor
            
        Returns:
            Image features [B, hidden_size] or dict with pooled and sequence outputs
        """
        outputs = self.encoder(pixel_values=pixel_values)
        
        if return_dict:
            return {
                'pooled_output': outputs.pooler_output,
                'last_hidden_state': outputs.last_hidden_state,
            }
        
        return outputs.pooler_output
    
    def get_num_params(self) -> Dict[str, int]:
        """Get parameter count"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }


if __name__ == "__main__":
    print("Testing Vision Encoder...")
    
    encoder = VisionEncoder()
    
    print(f"\nModel: {encoder.model_name}")
    print(f"Hidden size: {encoder.hidden_size}")
    
    params = encoder.get_num_params()
    print(f"\nParameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")
    
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        output = encoder(dummy_images, return_dict=True)
    
    print(f"\nOutput shapes:")
    print(f"  Pooled: {output['pooled_output'].shape}")
    print(f"  Sequence: {output['last_hidden_state'].shape}")
    
    print("\n✓ Vision encoder test passed!")
    