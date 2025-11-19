"""
Vision-Language Model (VLM)
Combines vision encoder, projection layer, and language model
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Optional, Dict, Tuple

from .vision_encoder import VisionEncoder
from .projection import ProjectionLayer


class VisionLanguageModel(nn.Module):
    """
    Complete Vision-Language Model for image captioning and conversation
    
    Architecture:
        Image → Vision Encoder (frozen CLIP) → Projection (trainable) → LM (GPT-2) → Text
    """
    
    def __init__(
        self,
        vision_model_name: str = "openai/clip-vit-base-patch16",
        language_model_name: str = "gpt2",
        projection_hidden_dim: int = 2048,
        projection_num_layers: int = 2,
        freeze_vision: bool = True,
        freeze_language_layers: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        print("Initializing Vision-Language Model...")
        
        # 1. Vision Encoder (frozen)
        print(f"Loading vision encoder: {vision_model_name}")
        self.vision_encoder = VisionEncoder(
            model_name=vision_model_name,
            freeze=freeze_vision
        )
        vision_dim = self.vision_encoder.hidden_size
        
        # 2. Language Model
        print(f"Loading language model: {language_model_name}")
        self.language_model = GPT2LMHeadModel.from_pretrained(language_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(language_model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.language_model.config.pad_token_id = self.tokenizer.eos_token_id
        
        language_dim = self.language_model.config.hidden_size
        
        # 3. Projection Layer (trainable)
        print(f"Creating projection layer: {vision_dim} → {language_dim}")
        self.projection = ProjectionLayer(
            input_dim=vision_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=language_dim,
            num_layers=projection_num_layers,
            dropout=dropout
        )
        
        if freeze_language_layers > 0:
            self._freeze_language_layers(freeze_language_layers)
        
        self.image_token_id = self._add_special_token("<image>")
        
        print("✓ Model initialized successfully!")
        self._print_model_info()
    
    def _add_special_token(self, token: str) -> int:
        """Add special token to tokenizer"""
        num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': [token]})
        if num_added > 0:
            self.language_model.resize_token_embeddings(len(self.tokenizer))
        return self.tokenizer.convert_tokens_to_ids(token)
    
    def _freeze_language_layers(self, num_layers: int):
        """Freeze first N layers of language model"""
        layers_to_freeze = self.language_model.transformer.h[:num_layers]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        print(f"✓ Froze first {num_layers} layers of language model")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            pixel_values: Images [B, 3, H, W]
            input_ids: Text token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            labels: Labels for language modeling [B, seq_len]
            
        Returns:
            Dict with 'loss', 'logits', and 'vision_features'
        """
        batch_size = pixel_values.size(0)
        
        # Extract vision features
        vision_output = self.vision_encoder(pixel_values, return_dict=True)
        vision_features = vision_output['pooled_output']
        
        # Project to language space
        vision_embeddings = self.projection(vision_features)
        
        # Get text embeddings
        text_embeddings = self.language_model.transformer.wte(input_ids)
        
        # Combine: prepend vision embeddings to text
        vision_embeddings = vision_embeddings.unsqueeze(1)
        combined_embeddings = torch.cat([vision_embeddings, text_embeddings], dim=1)
        
        # Update attention mask
        if attention_mask is not None:
            vision_attention = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([vision_attention, attention_mask], dim=1)
        
        # Update labels
        if labels is not None:
            vision_labels = torch.full((batch_size, 1), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([vision_labels, labels], dim=1)
        
        # Forward through language model
        outputs = self.language_model(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        if return_dict:
            return {
                'loss': outputs.loss if labels is not None else None,
                'logits': outputs.logits,
                'vision_features': vision_features,
                'projected_features': vision_embeddings.squeeze(1)
            }
        
        return outputs.loss, outputs.logits
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        prompt: str = "",
        max_length: int = 50,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate caption for an image"""
        self.eval()
        device = pixel_values.device
        
        # Get vision features
        vision_output = self.vision_encoder(pixel_values, return_dict=True)
        vision_features = vision_output['pooled_output']
        vision_embeddings = self.projection(vision_features).unsqueeze(1)
        
        # Tokenize prompt
        if prompt:
            prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
            prompt_embeddings = self.language_model.transformer.wte(prompt_ids)
            combined_embeddings = torch.cat([vision_embeddings, prompt_embeddings], dim=1)
        else:
            combined_embeddings = vision_embeddings
        
        # Generate
        generated_ids = []
        current_embeddings = combined_embeddings
        
        for _ in range(max_length):
            outputs = self.language_model(inputs_embeds=current_embeddings)
            logits = outputs.logits[:, -1, :]
            
            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            generated_ids.append(next_token.item())
            next_embedding = self.language_model.transformer.wte(next_token)
            current_embeddings = torch.cat([current_embeddings, next_embedding], dim=1)
        
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    def get_num_params(self) -> Dict[str, int]:
        """Get parameter counts"""
        vision_params = self.vision_encoder.get_num_params()
        projection_params = sum(p.numel() for p in self.projection.parameters())
        lm_params_total = sum(p.numel() for p in self.language_model.parameters())
        lm_params_trainable = sum(p.numel() for p in self.language_model.parameters() if p.requires_grad)
        
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'vision_total': vision_params['total'],
            'vision_trainable': vision_params['trainable'],
            'projection': projection_params,
            'language_total': lm_params_total,
            'language_trainable': lm_params_trainable,
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }
    
    def _print_model_info(self):
        """Print model architecture information"""
        params = self.get_num_params()
        
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE")
        print("="*60)
        print(f"Vision Encoder: {params['vision_total']:,} params ({params['vision_trainable']:,} trainable)")
        print(f"Projection Layer: {params['projection']:,} params (trainable)")
        print(f"Language Model: {params['language_total']:,} params ({params['language_trainable']:,} trainable)")
        print("-"*60)
        print(f"TOTAL: {params['total']:,} params")
        print(f"TRAINABLE: {params['trainable']:,} params ({100*params['trainable']/params['total']:.1f}%)")
        print(f"FROZEN: {params['frozen']:,} params ({100*params['frozen']/params['total']:.1f}%)")
        print("="*60 + "\n")


if __name__ == "__main__":
    print("Testing Vision-Language Model...")
    
    model = VisionLanguageModel()
    
    print("\nTesting forward pass...")
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_text = model.tokenizer(
        ["A photo of a cat", "A beautiful sunset"],
        return_tensors='pt',
        padding=True,
        truncation=True
    )
    
    with torch.no_grad():
        outputs = model(
            pixel_values=dummy_images,
            input_ids=dummy_text['input_ids'],
            attention_mask=dummy_text['attention_mask'],
            labels=dummy_text['input_ids']
        )
    
    print(f"Loss: {outputs['loss']}")
    print(f"Logits shape: {outputs['logits'].shape}")
    
    print("\n✓ VLM test passed!")