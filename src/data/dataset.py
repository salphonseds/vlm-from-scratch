"""
COCO Dataset for Image Captioning
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pycocotools.coco import COCO


class COCOCaptionDataset(Dataset):
    """
    COCO Dataset for image captioning
    Each image has 5 captions
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train2017",
        vision_processor=None,
        tokenizer=None,
        max_caption_length: int = 40,
        return_all_captions: bool = False
    ):
        """
        Args:
            root_dir: Root directory containing COCO data
            split: train2017 or val2017
            vision_processor: Image processor from vision encoder
            tokenizer: Tokenizer from language model
            max_caption_length: Maximum caption length
            return_all_captions: If True, return all 5 captions
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.vision_processor = vision_processor
        self.tokenizer = tokenizer
        self.max_caption_length = max_caption_length
        self.return_all_captions = return_all_captions
        
        # Paths
        self.images_dir = self.root_dir / "images" / split
        self.annotations_file = self.root_dir / "annotations" / f"captions_{split}.json"
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")
        
        # Load COCO annotations
        print(f"Loading COCO {split} annotations...")
        self.coco = COCO(str(self.annotations_file))
        self.image_ids = list(self.coco.imgs.keys())
        
        print(f"✓ Loaded {len(self.image_ids)} images with captions")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with 'pixel_values', 'input_ids', 'attention_mask', 'labels'
        """
        image_id = self.image_ids[idx]
        
        # Load image
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = self.images_dir / image_info['file_name']
        image = Image.open(image_path).convert('RGB')
        
        # Get captions
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        captions = [ann['caption'] for ann in anns]
        
        # Select caption
        if self.return_all_captions:
            caption = captions
        else:
            caption = np.random.choice(captions)
        
        # Process image
        if self.vision_processor is not None:
            pixel_values = self.vision_processor(images=image, return_tensors='pt')['pixel_values'][0]
        else:
            pixel_values = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Process caption
        if self.tokenizer is not None and not self.return_all_captions:
            encoding = self.tokenizer(
                caption,
                max_length=self.max_caption_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'][0]
            attention_mask = encoding['attention_mask'][0]
            
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            return {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'caption': caption,
                'image_id': image_id
            }
        else:
            return {
                'pixel_values': pixel_values,
                'captions': captions,
                'image_id': image_id,
                'image_path': str(image_path)
            }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    captions = [item['caption'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'captions': captions,
        'image_ids': image_ids
    }


if __name__ == "__main__":
    print("Testing COCO Dataset...")
    
    dataset = COCOCaptionDataset(
        root_dir="./data/coco",
        split="train2017"
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Pixel values shape: {sample['pixel_values'].shape}")
    print(f"Number of captions: {len(sample['captions'])}")
    print(f"First caption: {sample['captions'][0]}")
    
    print("\n✓ Dataset test passed!")