"""
Inference script for VLM - Generate captions from images
"""

import torch
import yaml
from PIL import Image
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.vlm import VisionLanguageModel


def load_model(checkpoint_path: str, config_path: str, device: str = 'cuda'):
    """Load trained VLM model from checkpoint"""
    
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract model config
    model_config = config['model']
    
    print("Initializing model...")
    model = VisionLanguageModel(
        vision_model_name=model_config['vision_encoder']['name'],
        language_model_name=model_config['language_model']['name'],
        projection_hidden_dim=model_config['projection']['hidden_dim'],
        projection_num_layers=model_config['projection']['num_layers'],
        freeze_vision=model_config['vision_encoder']['freeze'],
        freeze_language_layers=model_config['language_model']['freeze_layers'],
        dropout=model_config['projection']['dropout'],
    )
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully!\n")
    return model


def generate_caption(
    model: VisionLanguageModel,
    image_path: str,
    device: str = 'cuda',
    max_length: int = 50,
    temperature: float = 0.7
):
    """Generate caption for a single image"""
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    # Use CLIP's processor to preprocess the image
    inputs = model.vision_encoder.processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    # Generate caption
    print(f"Generating caption for: {image_path}")
    caption = model.generate(
        pixel_values=pixel_values,
        prompt="",
        max_length=max_length,
        temperature=temperature
    )
    
    return caption


def main():
    """Main inference function"""
    
    # Paths
    checkpoint_path = "./checkpoints/best_model.pt"
    config_path = "./configs/base_config.yaml"
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load model
    model = load_model(checkpoint_path, config_path, device)
    
    # Example: Generate caption for a COCO validation image
    test_image = "./data/coco/images/val2017/000000000139.jpg"
    
    if Path(test_image).exists():
        caption = generate_caption(model, test_image, device)
        print(f"\n{'='*60}")
        print(f"Image: {test_image}")
        print(f"Generated Caption: {caption}")
        print(f"{'='*60}\n")
    else:
        print(f"Test image not found: {test_image}")
        print("Please provide a valid image path")


if __name__ == "__main__":
    main()
