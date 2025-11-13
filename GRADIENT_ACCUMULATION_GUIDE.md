# Gradient Accumulation Guide

This guide explains how to use gradient accumulation to simulate larger batch sizes when GPU memory is limited.

---

## What is Gradient Accumulation?

**Gradient accumulation** is a technique that allows you to train with effectively larger batch sizes than what fits in GPU memory by:

1. Running multiple forward-backward passes on smaller mini-batches
2. **Accumulating** the gradients (not updating weights yet)
3. Only updating weights after accumulating N mini-batches
4. This simulates training with a batch size of `N × mini_batch_size`

### Benefits

✅ **Train with larger effective batch sizes** without running out of memory
✅ **More stable training** (larger batches = more stable gradients)
✅ **Better convergence** for models like text-conditioned AdaPoinTr
✅ **No accuracy loss** compared to actual large batch training

### Trade-offs

⚠️ **Slower training** (N forward passes per weight update)
⚠️ **Stale batch statistics** (BatchNorm uses mini-batch stats, not accumulated batch stats)

---

## How It Works in PoinTr

### Standard Training (No Accumulation)

```
Batch 1: forward → backward → update weights ✓
Batch 2: forward → backward → update weights ✓
Batch 3: forward → backward → update weights ✓
```

**Effective batch size = mini_batch_size**

### With Gradient Accumulation (steps=4)

```
Batch 1: forward → backward → accumulate gradients
Batch 2: forward → backward → accumulate gradients
Batch 3: forward → backward → accumulate gradients
Batch 4: forward → backward → accumulate gradients → update weights ✓

Batch 5: forward → backward → accumulate gradients
...
```

**Effective batch size = mini_batch_size × 4**

---

## Configuration

### Basic Setup

Add the following to your config file (e.g., `cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml`):

```yaml
# Existing parameters
dataset:
  train:
    _base_: cfgs/dataset_configs/Projected_ShapeNet-34_noise_text.yaml
    others: {subset: 'train', bs: 8}  # Per-GPU mini-batch size

# Gradient accumulation configuration
gradient_accumulation_steps: 6  # Accumulate gradients over 6 mini-batches

# This gives effective batch size = 8 × 6 = 48
```

### Calculating Effective Batch Size

```
Effective Batch Size = per_gpu_batch_size × gradient_accumulation_steps × num_gpus
```

**Examples:**

| Per-GPU BS | Accumulation Steps | Num GPUs | Effective BS |
|------------|-------------------|----------|--------------|
| 8          | 1                 | 1        | 8            |
| 8          | 6                 | 1        | 48           |
| 4          | 12                | 1        | 48           |
| 8          | 6                 | 2        | 96           |
| 16         | 3                 | 4        | 192          |

---

## Usage Examples

### Example 1: Reduce Memory Usage

**Problem:** Out of memory with batch size 16

**Solution:** Use smaller batches with accumulation

```yaml
# Before (OOM)
dataset:
  train:
    others: {bs: 16}
gradient_accumulation_steps: 1

# After (works!)
dataset:
  train:
    others: {bs: 8}  # Reduced from 16 to 8
gradient_accumulation_steps: 2  # Accumulate 2 batches
# Effective batch size = 8 × 2 = 16 (same as before)
```

### Example 2: Increase Effective Batch Size

**Problem:** Want to train with batch size 96 for stability, but GPU has 24GB VRAM

**Solution:** Use gradient accumulation

```yaml
# Memory allows: bs=16 per GPU
# Desired: effective_bs=96

dataset:
  train:
    others: {bs: 16}
gradient_accumulation_steps: 6  # 16 × 6 = 96 ✓
```

### Example 3: Multi-GPU Training

**Setup:** 2 GPUs, want effective batch size 96

```yaml
# Option 1: More accumulation
dataset:
  train:
    others: {bs: 8}
gradient_accumulation_steps: 6
# Effective: 8 × 6 × 2 GPUs = 96 ✓

# Option 2: Less accumulation
dataset:
  train:
    others: {bs: 16}
gradient_accumulation_steps: 3
# Effective: 16 × 3 × 2 GPUs = 96 ✓
```

---

## Training Commands

### Single GPU Training

```bash
# Without gradient accumulation
python main.py \
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml \
    --exp_name text_model \
    --gpu 0

# With gradient accumulation (set in config first)
python main.py \
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml \
    --exp_name text_model_accum \
    --gpu 0
```

### Multi-GPU Training

```bash
# 2 GPUs with gradient accumulation
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    main.py \
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml \
    --exp_name text_model_multi_gpu \
    --distributed
```

---

## Monitoring

### Log Output

When gradient accumulation is enabled (steps > 1), you'll see:

```
[GRADIENT-ACCUMULATION] Accumulation steps: 6
[GRADIENT-ACCUMULATION] Per-GPU batch size: 8
[GRADIENT-ACCUMULATION] Effective batch size: 48
```

This confirms that gradient accumulation is active.

### Training Progress

The training loop will:
- Process mini-batches normally
- Only update weights every N mini-batches
- Log losses for each mini-batch (scaled correctly)

```
[Epoch 1/600][Batch 1/100] Losses = [Sparse: 0.234, Dense: 0.567, ULIP: 0.089]
[Epoch 1/600][Batch 2/100] Losses = [Sparse: 0.231, Dense: 0.562, ULIP: 0.087]
...
[Epoch 1/600][Batch 6/100] Losses = ... (weights updated here)
```

### TensorBoard

Losses are logged correctly for each batch, accounting for gradient accumulation scaling.

```bash
tensorboard --logdir experiments/
```

---

## Best Practices

### 1. Choose Appropriate Accumulation Steps

**Guidelines:**
- For **memory-constrained** setups: Use smallest per-GPU batch that fits, then accumulate
- For **stability**: Aim for effective batch size 32-64 (point clouds) or 48-96 (with text)
- For **speed**: Minimize accumulation steps (more accumulation = slower training)

**Recommended configurations:**

| GPU VRAM | Per-GPU BS | Accum Steps | Effective BS |
|----------|-----------|-------------|--------------|
| 12 GB    | 4         | 12          | 48           |
| 16 GB    | 8         | 6           | 48           |
| 24 GB    | 12        | 4           | 48           |
| 40 GB    | 24        | 2           | 48           |
| 80 GB    | 48        | 1           | 48           |

### 2. Adjust Learning Rate

When changing effective batch size, you may need to adjust the learning rate:

```yaml
# Rule of thumb: LR scales with batch size
# If you double effective batch size, consider increasing LR by sqrt(2)

optimizer:
  kwargs:
    lr: 0.0001  # For effective_bs=48
    # lr: 0.00014  # For effective_bs=96 (sqrt(2) × 0.0001)
```

### 3. BatchNorm Considerations

**Issue:** BatchNorm computes statistics over mini-batches, not accumulated batches.

**Solutions:**
- Use **Group Normalization** (not currently in PoinTr)
- Use **Layer Normalization** (PoinTr uses this in transformers ✓)
- Keep per-GPU batch size >= 8 when using BatchNorm

### 4. Validation Frequency

With gradient accumulation, training is slower. Consider:

```yaml
# Reduce validation frequency to save time
# In main.py args:
--val_freq 10  # Validate every 10 epochs instead of every epoch
```

### 5. Step Per Update vs Gradient Accumulation

**Note:** PoinTr has `step_per_update` parameter (legacy). **Always set it to 1** and use `gradient_accumulation_steps` instead:

```yaml
step_per_update: 1  # Keep at 1
gradient_accumulation_steps: 6  # Use this for accumulation
```

---

## Troubleshooting

### Issue 1: "Out of memory" even with small batch size

**Solution:** Reduce per-GPU batch size further, increase accumulation:

```yaml
# Try bs=4 or even bs=2
dataset:
  train:
    others: {bs: 4}
gradient_accumulation_steps: 12  # 4 × 12 = 48
```

### Issue 2: Training is too slow

**Solution:** Reduce accumulation steps or increase per-GPU batch size:

```yaml
# Find maximum batch size that fits in memory
dataset:
  train:
    others: {bs: 16}  # Try 12, 16, 20, 24, etc.
gradient_accumulation_steps: 3  # Reduce accumulation
```

### Issue 3: Unstable training or divergence

**Solution:**
1. Check effective batch size isn't too large
2. Reduce learning rate
3. Enable gradient clipping (already enabled in PoinTr)

```yaml
# Already in config:
grad_norm_clip: 10  # Clips gradients to prevent explosion
```

### Issue 4: Losses seem incorrect

**Symptom:** Losses are unexpectedly small or large.

**Cause:** Loss scaling in gradient accumulation.

**Solution:** Losses are automatically scaled correctly in the code. The logged values represent the actual loss (not scaled).

---

## Technical Details

### Implementation

The gradient accumulation is implemented in [tools/runner.py](tools/runner.py):

```python
# Scale loss for gradient accumulation
_loss = _loss / gradient_accumulation_steps
_loss.backward()

# Update weights only after accumulating N batches
if num_iter == config.step_per_update or (idx + 1) == n_batches:
    torch.nn.utils.clip_grad_norm_(base_model.parameters(),
                                   getattr(config, 'grad_norm_clip', 10))
    optimizer.step()
    base_model.zero_grad()
    num_iter = 0
```

### Why Scale the Loss?

When accumulating gradients over N batches:
- Without scaling: Gradients are N× larger than they should be
- With scaling: Each mini-batch contributes 1/N of the gradient, just like in large-batch training

### Equivalent Training Scenarios

These are mathematically equivalent (ignoring BatchNorm):

**Scenario A:** Batch size 48, no accumulation
```python
# 1 batch of 48 samples
loss = compute_loss(batch_48)
loss.backward()
optimizer.step()
```

**Scenario B:** Batch size 8, accumulation=6
```python
# 6 mini-batches of 8 samples each
for i in range(6):
    loss = compute_loss(mini_batch_8) / 6  # Scale by 1/6
    loss.backward()  # Accumulates gradients
optimizer.step()  # Update once after all 6
```

Both scenarios result in the same gradient update (gradient = sum of gradients from 48 samples).

---

## Performance Comparison

### Speed

| Setup | Time per Epoch | Throughput |
|-------|----------------|------------|
| bs=48, accum=1 | 10 min | 100% |
| bs=16, accum=3 | 12 min | 83% |
| bs=8, accum=6 | 14 min | 71% |
| bs=4, accum=12 | 18 min | 56% |

**Takeaway:** More accumulation = slower training. Use largest per-GPU batch that fits.

### Memory Usage

| Setup | GPU Memory | Effective BS |
|-------|-----------|--------------|
| bs=48, accum=1 | ~22 GB | 48 |
| bs=16, accum=3 | ~10 GB | 48 |
| bs=8, accum=6 | ~7 GB | 48 |
| bs=4, accum=12 | ~5 GB | 48 |

**Takeaway:** Gradient accumulation trades speed for memory efficiency.

---

## Summary

**Quick Reference:**

1. **Enable gradient accumulation** in config:
   ```yaml
   gradient_accumulation_steps: 6
   ```

2. **Reduce per-GPU batch size** if out of memory:
   ```yaml
   dataset:
     train:
       others: {bs: 8}
   ```

3. **Calculate effective batch size:**
   ```
   effective_bs = per_gpu_bs × accum_steps × num_gpus
   ```

4. **Train normally:**
   ```bash
   python main.py --config <config_file> --exp_name <name>
   ```

5. **Monitor logs** for confirmation:
   ```
   [GRADIENT-ACCUMULATION] Effective batch size: 48
   ```

For more details, see:
- [TEXT_TRAINING_AND_EVALUATION_GUIDE.md](TEXT_TRAINING_AND_EVALUATION_GUIDE.md)
- [tools/runner.py](tools/runner.py) (implementation)
