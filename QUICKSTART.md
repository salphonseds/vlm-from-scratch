# Quick Start Guide 🚀

Get your VLM training running in minutes!

## ✅ Step 1: Verify Files

```bash
cd ~/projects02/vlm-from-scratch
ls -la
```

You should see:
- README.md
- requirements.txt
- configs/
- src/
- scripts/
- notebooks/

## ✅ Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will take 5-10 minutes.

## ✅ Step 3: Test Model

```bash
# Test vision encoder
python src/models/vision_encoder.py

# Test projection layer
python src/models/projection.py

# Test complete VLM
python src/models/vlm.py
```

All should print "✓ test passed!"

## ✅ Step 4: Download COCO Dataset

```bash
# Make script executable
chmod +x scripts/download_data.sh

# Download data (takes 30-60 minutes)
bash scripts/download_data.sh
```

**Downloads**: ~14GB total
- Train images: 13GB (118K images)
- Val images: 1GB (5K images)
- Annotations: 241MB

## ✅ Step 5: Start Training

### Single GPU (testing):
```bash
python src/train.py --config configs/training_config.yaml
```

### Multi-GPU (production):
```bash
torchrun --nproc_per_node=4 src/train.py --config configs/training_config.yaml
```

## 📊 Monitor Training

```bash
# Watch training logs
tail -f logs/training.log

# Monitor GPUs
watch -n 1 nvidia-smi

# Check checkpoints
ls -lh checkpoints/
```

## 🎯 Expected Results

| Time | Epoch | Loss | Quality |
|------|-------|------|---------|
| 0h | 0 | 8.5 | Random |
| 3h | 5 | 3.0 | Basic words |
| 6h | 10 | 2.3 | Good captions! ✅ |

## 🎉 That's It!

You're now training a Vision-Language Model from scratch!

**Next Steps**:
- Monitor training progress
- Test on custom images
- Experiment with hyperparameters
- Add conversational capabilities (Phase 2)

**Questions?** Check README.md for detailed documentation!