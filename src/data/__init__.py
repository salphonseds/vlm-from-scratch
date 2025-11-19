"""
Data module
"""

from .dataset import COCOCaptionDataset, collate_fn

__all__ = [
    'COCOCaptionDataset',
    'collate_fn',
]