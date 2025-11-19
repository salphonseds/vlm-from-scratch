# Vision-Language Model (VLM) from Scratch

A learning project to build a Vision-Language Model with conversational capabilities using pre-trained encoders and custom alignment layers.

## 🎯 Project Goals

- Understand multi-modal AI architecture deeply
- Build an image captioning model
- Add conversational AI capabilities
- Train on AWS SageMaker (g5.12xlarge)

## 🏗️ Architecture

```
Image → Vision Encoder (CLIP/ViT) → Projection Layer → Language Model (GPT-2) → Response
                                          ↓
                                   Conversation Context
```

**Components:**
- **Vision Encoder**: Pre-trained CLIP ViT-B/16 (87M params, frozen)
- **Projection Layer**: MLP to align vision & language embeddings (6M params, trainable)
- **Language Model**: Pre-trained GPT-2 (124M params, fine-tuned)

**Total**: 217M parameters, 60% trainable

## 📁 Project Structure

```
vlm-from-scratch/
├── notebooks/
│   └── 01_data_exploration.ipynb      # Explore COCO dataset
├── src/
│   ├── data/
│   │   ├── dataset.py                 # PyTorch Dataset classes
│   │   └── __init__.py
│   ├── models/
│   │   ├── vision_encoder.py          # CLIP/ViT encoder
│   │   ├── projection.py              # Alignment layer
│   │   ├── vlm.py                     # Complete VLM model
│   │   └── __init__.py
│   ├── train.py                       # Main training script
│   └── __init__.py
├── configs/
│   ├── base_config.yaml               # Base configuration
│   └── training_config.yaml           # Training parameters
├── scripts/
│   └── download_data.sh               # Download datasets
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore rules
└── README.md                          # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
chmod +x scripts/download_data.sh
bash scripts/download_data.sh
```

### 3. Explore Data

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 4. Train Model

```bash
# Single GPU
python src/train.py --config configs/training_config.yaml

# Multi-GPU (4x A10G on g5.12xlarge)
torchrun --nproc_per_node=4 src/train.py --config configs/training_config.yaml
```

## 📊 Datasets

### Phase 1: COCO Captions
- **Size**: 330K images with 5 captions each
- **Purpose**: Learn basic image-to-text alignment
- **Download**: Automatic via scripts

### Phase 2: LLaVA-Instruct-150K (Future)
- **Size**: 150K instruction-following examples
- **Purpose**: Add conversational capabilities

## 🖥️ Hardware Requirements

**Minimum:**
- GPU: 16GB VRAM (e.g., V100, A10G)
- RAM: 32GB
- Storage: 50GB

**Recommended (Used in this project):**
- **AWS g5.12xlarge**: 4x A10G GPUs (24GB each)
- RAM: 192GB
- Storage: 100GB

## 📈 Expected Results

| Epoch | Train Loss | Val Loss | Time | Captions |
|-------|------------|----------|------|----------|
| 0 | 8.5 | 8.2 | - | Random |
| 5 | 3.0 | 2.8 | ~3h | Basic phrases |
| 10 | 2.2 | 2.3 | ~6h | Good sentences ✅ |

## 🎓 Learning Path

1. **Week 1-2**: Data preparation & model architecture
2. **Week 3-4**: Training & optimization
3. **Week 5-6**: Conversational capabilities
4. **Week 7**: Deployment & demo

## 📝 Key Learnings

- Vision-language alignment techniques
- Multi-modal attention mechanisms
- Efficient fine-tuning strategies
- Multi-GPU distributed training
- Transfer learning with frozen encoders

## 🤝 Contributing

This is a learning project, but feedback and suggestions are welcome!

## 📄 License

MIT License - Feel free to use this for learning!

## 🙏 Acknowledgments

- COCO Dataset team
- Hugging Face for pre-trained models
- LLaVA project for instruction datasets
- AWS SageMaker for compute resources

---

**Status**: 🚧 Active Development | **Last Updated**: November 2025