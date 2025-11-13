# Debug Mode Guide - Quick Testing with Dataset Subsets

This guide explains how to use debug mode to quickly test your ideas on a small subset of the data before running full training.

---

## Why Use Debug Mode?

When developing and testing new features or ideas:

✅ **Fast iteration** - Train on 100 samples in minutes instead of hours
✅ **Quick validation** - Verify your code works before full training
✅ **Save resources** - Don't waste GPU hours on buggy code
✅ **Rapid prototyping** - Test multiple approaches quickly

**Use debug mode for:**
- Testing new features
- Debugging code
- Verifying configurations
- Quick sanity checks

**Use full training for:**
- Final experiments
- Benchmark comparisons
- Paper results

---

## Quick Start

### 1. Use the Debug Config

A debug config is already created for you:

```bash
python main.py \
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr_text_debug.yaml \
    --exp_name debug_test \
    --gpu 0
```

This will:
- Train on only **100 samples** (instead of ~30,000)
- Validate on only **50 samples** (instead of ~8,000)
- Run for **10 epochs** (instead of 600)
- Take ~5-10 minutes instead of days

### 2. Expected Output

```
[DEBUG] Limiting train dataset from 28974 to 100 samples
[DEBUG] Limiting test dataset from 7908 to 50 samples
[GRADIENT-ACCUMULATION] Effective batch size: 8
[Training] EPOCH: 1 EpochTime = 0.5 (s) ...
```

---

## Configuration

### Debug Config Parameters

Edit `cfgs/Projected_ShapeNet34_models/AdaPoinTr_text_debug.yaml`:

```yaml
max_epoch: 10  # Reduced from 600

# Enable debug mode
debug_mode: true
debug_train_samples: 100  # Use only 100 training samples
debug_val_samples: 50     # Use only 50 validation samples
```

### Adjust Subset Size

**Very quick test (1-2 minutes):**
```yaml
debug_mode: true
debug_train_samples: 20   # Tiny subset
debug_val_samples: 10
max_epoch: 3
```

**Medium test (5-10 minutes):**
```yaml
debug_mode: true
debug_train_samples: 100  # Default
debug_val_samples: 50
max_epoch: 10
```

**Larger test (30-60 minutes):**
```yaml
debug_mode: true
debug_train_samples: 500
debug_val_samples: 200
max_epoch: 50
```

**Full training:**
```yaml
debug_mode: false  # Disable debug mode
# Or simply don't set debug_mode
max_epoch: 600
```

---

## Use Cases

### Use Case 1: Test New Feature

**Scenario:** You added ULIP alignment loss and want to verify it works.

```bash
# Quick test with debug mode
python main.py \
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr_text_debug.yaml \
    --exp_name test_ulip_loss \
    --gpu 0

# Check logs for:
# - No errors
# - ULIP loss is computed
# - Training completes
```

**Time: 5-10 minutes** ✅

### Use Case 2: Compare Approaches

**Scenario:** Testing different text encoder models.

**Test A - ViT-bigG-14:**
```yaml
# AdaPoinTr_text_debug.yaml
debug_mode: true
debug_train_samples: 100
model:
  text_encoder_name: 'ViT-bigG-14'
```

```bash
python main.py --config cfgs/.../AdaPoinTr_text_debug.yaml --exp_name debug_vit_bigg
```

**Test B - CLIP-Large:**
```yaml
# AdaPoinTr_text_debug_clip.yaml
debug_mode: true
debug_train_samples: 100
model:
  text_encoder_name: 'openai/clip-vit-large-patch14'
  use_ulip_loss: false  # Incompatible with CLIP-Large
```

```bash
python main.py --config cfgs/.../AdaPoinTr_text_debug_clip.yaml --exp_name debug_clip
```

**Compare results in TensorBoard:**
```bash
tensorboard --logdir experiments/
```

**Time per test: 5-10 minutes** ✅

### Use Case 3: Hyperparameter Search

**Scenario:** Test different learning rates quickly.

```bash
# Test LR=0.0001
python main.py --config cfgs/.../AdaPoinTr_text_debug.yaml --exp_name debug_lr_1e4

# Edit config, change lr to 0.0002
# Test LR=0.0002
python main.py --config cfgs/.../AdaPoinTr_text_debug.yaml --exp_name debug_lr_2e4

# Edit config, change lr to 0.00005
# Test LR=0.00005
python main.py --config cfgs/.../AdaPoinTr_text_debug.yaml --exp_name debug_lr_5e5
```

**Total time: 15-30 minutes for 3 tests** ✅

### Use Case 4: Gradient Accumulation Testing

**Scenario:** Test if gradient accumulation works correctly.

```yaml
# AdaPoinTr_text_debug.yaml
debug_mode: true
debug_train_samples: 100

dataset:
  train:
    others: {bs: 4}  # Small batch

gradient_accumulation_steps: 2  # Test accumulation
```

```bash
python main.py --config cfgs/.../AdaPoinTr_text_debug.yaml --exp_name debug_grad_accum
```

**Time: 5-10 minutes** ✅

---

## Creating Custom Debug Configs

### Option 1: Copy and Modify

```bash
# Copy existing debug config
cp cfgs/Projected_ShapeNet34_models/AdaPoinTr_text_debug.yaml \
   cfgs/Projected_ShapeNet34_models/AdaPoinTr_text_debug_custom.yaml

# Edit as needed
nano cfgs/Projected_ShapeNet34_models/AdaPoinTr_text_debug_custom.yaml
```

### Option 2: Add Debug Parameters to Any Config

Add these lines to any existing config:

```yaml
# At the end of your config file
max_epoch: 10  # Reduce from 600

# Debug mode configuration
debug_mode: true
debug_train_samples: 100
debug_val_samples: 50
```

---

## Best Practices

### 1. Always Test in Debug Mode First

```bash
# ✅ Good workflow
# Step 1: Debug mode test
python main.py --config .../AdaPoinTr_text_debug.yaml --exp_name debug_test

# Step 2: If it works, run full training
python main.py --config .../AdaPoinTr_text.yaml --exp_name full_training

# ❌ Bad workflow
# Don't start full training without testing first!
python main.py --config .../AdaPoinTr_text.yaml --exp_name full_training
# (discovers bug 10 hours later...)
```

### 2. Use Appropriate Subset Sizes

| Test Type | Train Samples | Val Samples | Time |
|-----------|---------------|-------------|------|
| Quick sanity check | 20 | 10 | 1-2 min |
| Feature testing | 100 | 50 | 5-10 min |
| Hyperparameter search | 200-500 | 100 | 15-30 min |
| Pre-full-training check | 1000 | 500 | 1-2 hours |

### 3. Check Key Metrics

During debug runs, verify:
- ✅ Training completes without errors
- ✅ Losses decrease (even on small dataset)
- ✅ Model saves checkpoints
- ✅ Validation runs successfully
- ✅ Memory usage is reasonable

### 4. Interpret Debug Results Carefully

⚠️ **Important:** Debug mode results are NOT representative of full training!

**What debug mode tells you:**
- ✅ Code works
- ✅ No crashes or errors
- ✅ Model can overfit small data (good sign!)
- ✅ Configuration is valid

**What debug mode DOESN'T tell you:**
- ❌ Final performance
- ❌ Generalization ability
- ❌ Overfitting to full dataset
- ❌ Convergence behavior

### 5. Disable Debug Mode for Production

Before final experiments:

```yaml
# Make sure to disable debug mode!
debug_mode: false
max_epoch: 600
```

Or use the non-debug config:
```bash
python main.py --config cfgs/.../AdaPoinTr_text.yaml  # Not _debug.yaml
```

---

## Advanced Usage

### Separate Train/Val Subset Sizes

```yaml
debug_mode: true
debug_train_samples: 200  # More training samples
debug_val_samples: 50     # Fewer validation samples
```

### Use with Gradient Accumulation

```yaml
debug_mode: true
debug_train_samples: 100

dataset:
  train:
    others: {bs: 8}

gradient_accumulation_steps: 6
# Effective batch size = 8 × 6 = 48
# But only processes 100 samples total
```

### Debug Mode with Pretrained Weights

```yaml
debug_mode: true
debug_train_samples: 100

pretrained_adapointr_path: /path/to/pretrained.pth
# Test that pretrained loading works correctly
```

### Debug Multi-GPU Setup

```bash
# Test distributed training on small dataset
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    main.py \
    --config cfgs/.../AdaPoinTr_text_debug.yaml \
    --exp_name debug_multi_gpu \
    --distributed
```

---

## Troubleshooting

### Issue 1: Debug mode not activating

**Symptom:** Still trains on full dataset despite `debug_mode: true`

**Solution:** Make sure you're using the debug config file:
```bash
# Wrong - using regular config
python main.py --config cfgs/.../AdaPoinTr_text.yaml

# Correct - using debug config
python main.py --config cfgs/.../AdaPoinTr_text_debug.yaml
```

### Issue 2: "Not enough samples" error

**Symptom:** Error about batch size being larger than dataset

**Solution:** Reduce batch size or increase debug_train_samples:
```yaml
dataset:
  train:
    others: {bs: 4}  # Reduce from 8

debug_train_samples: 20  # Must be >= batch_size
```

### Issue 3: Still takes too long

**Symptom:** Debug run takes 30+ minutes

**Solution:** Reduce samples and epochs more aggressively:
```yaml
debug_train_samples: 20   # Very small
debug_val_samples: 10
max_epoch: 3              # Just a few epochs
```

### Issue 4: Can't reproduce results

**Symptom:** Different results each debug run

**Cause:** Small dataset has high variance

**Solution:** This is normal for debug mode. For reproducibility:
- Use larger debug_train_samples (500+)
- Set random seed (already done in code)
- Run multiple times and average

---

## Performance Comparison

### Training Time Examples

**Full training:**
```yaml
debug_mode: false
# ~28,000 training samples
# 600 epochs
# Time: 2-3 days on single GPU
```

**Debug mode (default):**
```yaml
debug_mode: true
debug_train_samples: 100
# 100 training samples
# 10 epochs
# Time: 5-10 minutes on single GPU
```

**Speed-up: ~300-500x faster** 🚀

### Memory Usage

Debug mode and full training use similar memory (per batch), so it's not primarily for saving memory. The main benefit is **time savings**.

---

## Workflow Example

### Complete Development Workflow

```bash
# 1. Initial test - verify code works
python main.py \
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr_text_debug.yaml \
    --exp_name debug_initial \
    --gpu 0
# Time: 5-10 min

# 2. Test different hyperparameters
# Edit config, test LR, ULIP weight, etc.
python main.py --config .../AdaPoinTr_text_debug.yaml --exp_name debug_hp1
python main.py --config .../AdaPoinTr_text_debug.yaml --exp_name debug_hp2
python main.py --config .../AdaPoinTr_text_debug.yaml --exp_name debug_hp3
# Time: 5-10 min each = 15-30 min total

# 3. Medium-scale test with best config
# Edit debug config: debug_train_samples: 1000, max_epoch: 50
python main.py --config .../AdaPoinTr_text_debug.yaml --exp_name debug_medium
# Time: 1-2 hours

# 4. Full training with finalized config
python main.py \
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml \
    --exp_name final_training \
    --gpu 0
# Time: 2-3 days

# Total time saved: ~2-3 failed full training runs avoided!
```

---

## Summary

**Debug Mode Quick Reference:**

| Parameter | Debug Value | Full Training Value |
|-----------|-------------|---------------------|
| `debug_mode` | `true` | `false` or omitted |
| `debug_train_samples` | 100 | N/A |
| `debug_val_samples` | 50 | N/A |
| `max_epoch` | 10 | 600 |
| **Training time** | **5-10 min** | **2-3 days** |

**Commands:**

```bash
# Quick test
python main.py --config cfgs/.../AdaPoinTr_text_debug.yaml --exp_name debug_test

# Full training
python main.py --config cfgs/.../AdaPoinTr_text.yaml --exp_name full_train
```

**Pro tip:** Always test in debug mode before starting expensive full training runs!

---

For more details on training configurations, see:
- [TEXT_TRAINING_AND_EVALUATION_GUIDE.md](TEXT_TRAINING_AND_EVALUATION_GUIDE.md)
- [GRADIENT_ACCUMULATION_GUIDE.md](GRADIENT_ACCUMULATION_GUIDE.md)
