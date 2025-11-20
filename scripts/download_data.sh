#!/bin/bash

# Script to download COCO Captions dataset

set -e

echo "======================================"
echo "Downloading COCO Captions Dataset"
echo "======================================"

# Create data directory
DATA_DIR="./data/coco"
mkdir -p $DATA_DIR/images
mkdir -p $DATA_DIR/annotations

cd $DATA_DIR

# Download train images (13GB)
echo ""
echo "Downloading train2017 images (13GB)..."
if [ ! -f "train2017.zip" ]; then
    wget http://images.cocodataset.org/zips/train2017.zip
    echo "✓ Downloaded train2017.zip"
else
    echo "✓ train2017.zip already exists"
fi

# Download val images (1GB)
echo ""
echo "Downloading val2017 images (1GB)..."
if [ ! -f "val2017.zip" ]; then
    wget http://images.cocodataset.org/zips/val2017.zip
    echo "✓ Downloaded val2017.zip"
else
    echo "✓ val2017.zip already exists"
fi

# Download annotations (241MB)
echo ""
echo "Downloading annotations..."
if [ ! -f "annotations_trainval2017.zip" ]; then
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    echo "✓ Downloaded annotations_trainval2017.zip"
else
    echo "✓ annotations_trainval2017.zip already exists"
fi

# Extract train images
echo ""
echo "Extracting train2017 images..."
if [ ! -d "images/train2017" ]; then
    unzip -q train2017.zip -d images/
    echo "✓ Extracted train2017"
else
    echo "✓ train2017 already extracted"
fi

# Extract val images
echo ""
echo "Extracting val2017 images..."
if [ ! -d "images/val2017" ]; then
    unzip -q val2017.zip -d images/
    echo "✓ Extracted val2017"
else
    echo "✓ val2017 already extracted"
fi

# Extract annotations
echo ""
echo "Extracting annotations..."
if [ ! -d "annotations" ]; then
    unzip -q annotations_trainval2017.zip
    echo "✓ Extracted annotations"
else
    echo "✓ Annotations already extracted"
fi

echo ""
echo "======================================"
echo "✓ COCO Dataset Download Complete!"
echo "======================================"
echo ""
echo "Ready to train! 🚀"
