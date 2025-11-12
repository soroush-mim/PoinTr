# Text-Conditioned AdaPoinTr Training and Evaluation Guide

This guide covers the enhanced training and evaluation capabilities for text-conditioned AdaPoinTr, including:
1. Loading pretrained weights from non-text models
2. Evaluating on both seen and unseen categories during training
3. Standalone evaluation script for comprehensive seen/unseen testing

---

## Table of Contents

1. [Loading Pretrained Weights](#loading-pretrained-weights)
2. [Training with Dual Evaluation](#training-with-dual-evaluation)
3. [Standalone Evaluation Script](#standalone-evaluation-script)
4. [Configuration Files](#configuration-files)
5. [Examples](#examples)

---

## Loading Pretrained Weights

### Overview

You can now initialize your text-conditioned AdaPoinTr model with weights from a pretrained non-text AdaPoinTr checkpoint. This enables **transfer learning** from models trained without text conditioning.

### How It Works

The loading mechanism:
- ✅ Loads all shared geometric processing components (encoder, decoder, grouper, etc.)
- ✅ Skips text-specific components (text encoder, mlp_query_text, ULIP modules)
- ✅ Randomly initializes text-specific parameters
- ✅ Provides detailed logging of loaded/skipped/missing parameters

### Configuration

Add the following to your config file:

```yaml
# Optional: Load pretrained weights from non-text AdaPoinTr checkpoint
pretrained_adapointr_path: /path/to/pretrained/adapointr/ckpt-best.pth
```

Set to `null` or omit to train from scratch.

### Example

```yaml
# cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml

pretrained_adapointr_path: /path/to/experiments/AdaPoinTr_baseline/ckpt-best.pth

model:
  NAME: AdaPoinTr
  use_text_conditioning: true
  # ... rest of config
```

### Training Command

```bash
python main.py \\
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml \\
    --exp_name adapointr_text_pretrained \\
    --gpu 0
```

### Expected Output

```
[PRETRAINED] Loading pretrained AdaPoinTr weights from /path/to/ckpt-best.pth...
[PRETRAINED] Skipping mlp_query.0.weight (replaced by mlp_query_text)
[PRETRAINED] ========== Loading Summary ==========
[PRETRAINED] Total pretrained parameters: 245
[PRETRAINED] Successfully loaded: 237
[PRETRAINED] Skipped (text-specific or mismatched): 8
[PRETRAINED] Missing in pretrained (will be randomly initialized): 15
[PRETRAINED] Missing keys (new text components - expected):
[PRETRAINED]   - text_encoder.model.text_model.embeddings.position_embedding.weight
[PRETRAINED]   - mlp_query_text.0.weight
[PRETRAINED]   - mlp_query_text.0.bias
[PRETRAINED]   ... and 12 more
[PRETRAINED] Checkpoint from epoch 299 (performance = {'CDL1': 0.023, 'CDL2': 0.056})
[PRETRAINED] =====================================
```

---

## Training with Dual Evaluation

### Overview

During training, you can now evaluate your model on **both seen and unseen categories** at each validation interval. This helps track:
- Performance on training categories (seen)
- Generalization to novel categories (unseen)
- Overfitting to seen categories

### Configuration

Add a `val_unseen` dataset to your config:

```yaml
dataset:
  train:
    _base_: cfgs/dataset_configs/Projected_ShapeNet-34_noise_text.yaml
    others: {subset: 'train'}

  val:
    _base_: cfgs/dataset_configs/Projected_ShapeNet-34_noise_text.yaml
    others: {subset: 'test'}

  # Uncomment to enable unseen evaluation during training
  val_unseen:
    _base_: cfgs/dataset_configs/Projected_ShapeNet-Unseen21_noise_text.yaml
    others: {subset: 'test'}

  test:
    _base_: cfgs/dataset_configs/Projected_ShapeNet-34_noise_text.yaml
    others: {subset: 'test'}
```

### Training Command

```bash
python main.py \\
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml \\
    --exp_name adapointr_text_dual_eval \\
    --gpu 0
```

### Expected Output

During validation epochs, you'll see:

```
============================================================
[VALIDATION] Evaluating on SEEN categories
============================================================
[Seen] [VALIDATION] Start validating epoch 50
[Seen] Test[100/1000] Taxonomy = 02691156 Sample = 1a04e3eab45ca15dd86060f189eb133
...
[Validation] EPOCH: 50  Metrics = ['0.0234', '0.0567', ...]
============================ TEST RESULTS ============================
Taxonomy    #Sample    CDL1    CDL2    ...
02691156    100        0.023   0.056   ...
...
Overall                0.025   0.061   ...

============================================================
[VALIDATION] Evaluating on UNSEEN categories
============================================================
[Unseen] [VALIDATION] Start validating epoch 50
[Unseen] Test[100/500] Taxonomy = 04379243 Sample = 1b3d7f3e5c8a9d2f7e1a2b3c4d5e6f7
...
[Validation] EPOCH: 50  Metrics = ['0.0456', '0.0892', ...]
============================ TEST RESULTS ============================
Taxonomy    #Sample    CDL1    CDL2    ...
04379243    50         0.045   0.089   ...
...
Overall                0.048   0.095   ...

[VALIDATION] Seen CDL1: 0.0250 | Unseen CDL1: 0.0480
```

### TensorBoard Monitoring

Metrics are logged separately for seen and unseen:

**Seen Categories:**
- `Seen/Loss/Epoch/Sparse`
- `Seen/Loss/Epoch/Dense`
- `Seen/Metric/CDL1`
- `Seen/Metric/CDL2`
- `Seen/Metric/F1`

**Unseen Categories:**
- `Unseen/Loss/Epoch/Sparse`
- `Unseen/Loss/Epoch/Dense`
- `Unseen/Metric/CDL1`
- `Unseen/Metric/CDL2`
- `Unseen/Metric/F1`

View with:
```bash
tensorboard --logdir experiments/
```

---

## Standalone Evaluation Script

### Overview

Use `evaluate_text_seen_unseen.py` to comprehensively evaluate a trained text-conditioned model on both seen and unseen categories in a single run.

### Features

- ✅ Evaluates on seen categories (Projected_ShapeNet-34)
- ✅ Evaluates on unseen categories (Projected_ShapeNet-Unseen21)
- ✅ Provides detailed per-category metrics
- ✅ Computes overall metrics for both splits
- ✅ Calculates performance gap (seen vs unseen)
- ✅ Saves results to JSON file
- ✅ Uses text captions for both evaluations

### Usage

```bash
python evaluate_text_seen_unseen.py \\
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml \\
    --ckpts /path/to/experiments/adapointr_text/ckpt-best.pth \\
    --exp_name my_model_eval \\
    --output_dir eval_results
```

### Arguments

- `--config`: Path to model config file (should be the text-conditioned config)
- `--ckpts`: Path to trained checkpoint
- `--exp_name`: Name for this evaluation (used in output filename)
- `--output_dir`: Directory to save results (default: `eval_results`)

### Output

#### Console Output

```
================================================================================
Text-Conditioned AdaPoinTr Evaluation on Seen and Unseen Categories
================================================================================
Checkpoint: /path/to/ckpt-best.pth
Config: cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml
================================================================================
[DATASET] Building seen categories dataset...
[DATASET] Seen dataset size: 1234
[DATASET] Building unseen categories dataset...
[DATASET] Unseen dataset size: 567
[MODEL] Building model...
[MODEL] Loading checkpoint...
[MODEL] Model loaded successfully
================================================================================
[Seen] Starting evaluation on Seen categories
================================================================================
[Seen] Test [100/1234] Taxonomy = 02691156 Sample = abc123
...
================================================================================
[Seen] DETAILED RESULTS
================================================================================
Taxonomy    #Sample    CDL1    CDL2    F1      #ModelName
02691156    100        0.023   0.056   0.912   airplane
02828884    80         0.019   0.048   0.925   bench
...
Overall                0.025   0.061   0.908

================================================================================
[Seen] Summary:
[Seen]   CDL1: 0.0250
[Seen]   CDL2: 0.0610
[Seen]   F1: 0.9080
================================================================================

================================================================================
[Unseen] Starting evaluation on Unseen categories
================================================================================
[Unseen] Test [100/567] Taxonomy = 04379243 Sample = xyz789
...
================================================================================
[Unseen] DETAILED RESULTS
================================================================================
Taxonomy    #Sample    CDL1    CDL2    F1      #ModelName
04379243    50         0.045   0.089   0.856   table
04530566    45         0.052   0.098   0.834   vessel
...
Overall                0.048   0.095   0.845

================================================================================
[Unseen] Summary:
[Unseen]   CDL1: 0.0480
[Unseen]   CDL2: 0.0950
[Unseen]   F1: 0.8450
================================================================================

Results saved to eval_results/my_model_eval_results.json

================================================================================
COMPARISON SUMMARY
================================================================================
Seen CDL1:   0.0250
Unseen CDL1: 0.0480
Difference:  0.0230

Seen CDL2:   0.0610
Unseen CDL2: 0.0950
Difference:  0.0340
================================================================================
Evaluation complete!
```

#### JSON Output

File: `eval_results/my_model_eval_results.json`

```json
{
  "checkpoint": "/path/to/ckpt-best.pth",
  "config": "cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml",
  "seen": {
    "split": "Seen",
    "overall": {
      "CDL1": 0.025,
      "CDL2": 0.061,
      "F1": 0.908
    },
    "per_category": {
      "02691156": {
        "name": "airplane",
        "count": 100,
        "metrics": {
          "CDL1": 0.023,
          "CDL2": 0.056,
          "F1": 0.912
        }
      },
      ...
    }
  },
  "unseen": {
    "split": "Unseen",
    "overall": {
      "CDL1": 0.048,
      "CDL2": 0.095,
      "F1": 0.845
    },
    "per_category": {
      "04379243": {
        "name": "table",
        "count": 50,
        "metrics": {
          "CDL1": 0.045,
          "CDL2": 0.089,
          "F1": 0.856
        }
      },
      ...
    }
  },
  "comparison": {
    "CDL1_seen": 0.025,
    "CDL1_unseen": 0.048,
    "CDL1_diff": 0.023,
    "CDL2_seen": 0.061,
    "CDL2_unseen": 0.095,
    "CDL2_diff": 0.034
  }
}
```

---

## Configuration Files

### Available Configs

#### 1. Training on Seen Categories with Text (Standard)

**File:** `cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml`

```yaml
dataset:
  train:
    _base_: cfgs/dataset_configs/Projected_ShapeNet-34_noise_text.yaml
    others: {subset: 'train'}
  val:
    _base_: cfgs/dataset_configs/Projected_ShapeNet-34_noise_text.yaml
    others: {subset: 'test'}

pretrained_adapointr_path: null  # Set to path for pretrained weights

model:
  NAME: AdaPoinTr
  use_text_conditioning: true
  use_ulip_loss: true
  # ... other params
```

#### 2. Training on Seen, Testing on Unseen

**File:** `cfgs/Projected_ShapeNetUnseen21_models/AdaPoinTr_text.yaml`

```yaml
dataset:
  train:
    _base_: cfgs/dataset_configs/Projected_ShapeNet-34_noise_text.yaml
    others: {subset: 'train'}
  val:
    _base_: cfgs/dataset_configs/Projected_ShapeNet-34_noise_text.yaml
    others: {subset: 'test'}
  test:
    _base_: cfgs/dataset_configs/Projected_ShapeNet-Unseen21_noise_text.yaml
    others: {subset: 'test'}

pretrained_adapointr_path: null

model:
  NAME: AdaPoinTr
  use_text_conditioning: true
  use_ulip_loss: true
  # ... other params
```

#### 3. Dataset Configs

**Seen (34 categories):**
- With text: `cfgs/dataset_configs/Projected_ShapeNet-34_noise_text.yaml`
- Without text: `cfgs/dataset_configs/Projected_ShapeNet-34_noise.yaml`

**Unseen (21 categories):**
- With text: `cfgs/dataset_configs/Projected_ShapeNet-Unseen21_noise_text.yaml`
- Without text: `cfgs/dataset_configs/Projected_ShapeNet-Unseen21_noise.yaml`

---

## Examples

### Example 1: Train from Scratch with Text

```bash
python main.py \\
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml \\
    --exp_name text_from_scratch \\
    --gpu 0
```

### Example 2: Train with Pretrained Weights

1. Edit config:
```yaml
pretrained_adapointr_path: experiments/AdaPoinTr_baseline/ckpt-best.pth
```

2. Train:
```bash
python main.py \\
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml \\
    --exp_name text_pretrained \\
    --gpu 0
```

### Example 3: Train with Dual Evaluation (Seen + Unseen)

1. Edit config:
```yaml
dataset:
  train:
    _base_: cfgs/dataset_configs/Projected_ShapeNet-34_noise_text.yaml
    others: {subset: 'train'}
  val:
    _base_: cfgs/dataset_configs/Projected_ShapeNet-34_noise_text.yaml
    others: {subset: 'test'}
  val_unseen:  # Add this!
    _base_: cfgs/dataset_configs/Projected_ShapeNet-Unseen21_noise_text.yaml
    others: {subset: 'test'}
```

2. Train:
```bash
python main.py \\
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml \\
    --exp_name text_dual_eval \\
    --gpu 0
```

### Example 4: Evaluate Trained Model on Seen + Unseen

```bash
python evaluate_text_seen_unseen.py \\
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml \\
    --ckpts experiments/text_pretrained/ckpt-best.pth \\
    --exp_name final_eval \\
    --output_dir eval_results
```

### Example 5: Complete Pipeline

```bash
# 1. Train baseline model without text
python main.py \\
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr.yaml \\
    --exp_name baseline \\
    --gpu 0

# 2. Train text-conditioned model with pretrained weights
# Edit AdaPoinTr_text.yaml:
#   pretrained_adapointr_path: experiments/baseline/ckpt-best.pth
#   val_unseen: <uncomment>

python main.py \\
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml \\
    --exp_name text_model \\
    --gpu 0

# 3. Evaluate on both seen and unseen
python evaluate_text_seen_unseen.py \\
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml \\
    --ckpts experiments/text_model/ckpt-best.pth \\
    --exp_name text_model_eval \\
    --output_dir eval_results

# 4. View results
cat eval_results/text_model_eval_results.json
tensorboard --logdir experiments/
```

---

## Key Files

### Implementation Files

- `tools/builder.py`: Contains `load_pretrained_adapointr_for_text()` function
- `tools/runner.py`: Training loop with dual evaluation support
- `evaluate_text_seen_unseen.py`: Standalone evaluation script

### Configuration Files

- `cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml`: Main training config (seen)
- `cfgs/Projected_ShapeNetUnseen21_models/AdaPoinTr_text.yaml`: Training config for unseen testing
- `cfgs/dataset_configs/Projected_ShapeNet-34_noise_text.yaml`: Seen dataset with captions
- `cfgs/dataset_configs/Projected_ShapeNet-Unseen21_noise_text.yaml`: Unseen dataset with captions

---

## Tips and Best Practices

### 1. Pretrained Weight Loading

- ✅ **DO** use pretrained weights when you have a well-trained baseline model
- ✅ **DO** check the loading summary to ensure parameters loaded correctly
- ⚠️ **DON'T** use pretrained weights from models with different architectures
- ⚠️ **DON'T** forget that text components will be randomly initialized

### 2. Dual Evaluation

- ✅ **DO** enable dual evaluation to monitor generalization
- ✅ **DO** watch for increasing gap between seen/unseen performance (indicates overfitting)
- ⚠️ **DON'T** evaluate too frequently (adds significant overhead)
- ⚠️ **DON'T** forget to uncomment `val_unseen` in config

### 3. Evaluation

- ✅ **DO** use the standalone script for final comprehensive evaluation
- ✅ **DO** save and analyze the JSON results for detailed insights
- ✅ **DO** compare multiple checkpoints to find the best generalization
- ⚠️ **DON'T** rely solely on validation metrics during training

### 4. Interpreting Results

**Good generalization:**
- Seen CDL1: 0.025, Unseen CDL1: 0.035 (Difference: 0.010) ✅

**Poor generalization:**
- Seen CDL1: 0.020, Unseen CDL1: 0.060 (Difference: 0.040) ❌

**Target:** Keep the seen/unseen gap as small as possible while maintaining good overall performance.

---

## Troubleshooting

### Issue: Pretrained weights not loading

**Solution:** Check the path in config and verify the checkpoint file exists:
```bash
ls -lh /path/to/pretrained/ckpt-best.pth
```

### Issue: val_unseen not working

**Solution:** Make sure it's uncommented and properly indented in YAML:
```yaml
dataset:
  val_unseen:  # Must be at same level as 'val'
    _base_: cfgs/dataset_configs/Projected_ShapeNet-Unseen21_noise_text.yaml
```

### Issue: Evaluation script fails

**Solution:** Ensure caption file exists and paths are correct:
```bash
ls -lh /home/soroushm/data/Cap3D_automated_ShapeNet.csv
ls -lh data/ShapeNet55-34/Projected_ShapeNet-Unseen21_noise
```

### Issue: Out of memory during dual evaluation

**Solution:** Reduce batch size or evaluate less frequently:
```yaml
dataset:
  val:
    others: {subset: 'test', bs: 1}  # Reduce from default
```

Or change validation frequency in training command:
```bash
python main.py --config ... --val_freq 10  # Evaluate every 10 epochs instead of every epoch
```

---

## Summary

You now have three powerful new capabilities:

1. **Pretrained Weight Loading**: Bootstrap text-conditioned training from non-text models
2. **Dual Evaluation**: Monitor both seen and unseen performance during training
3. **Comprehensive Evaluation**: Standalone script for detailed seen/unseen analysis

These features enable:
- ✅ Faster convergence via transfer learning
- ✅ Better understanding of model generalization
- ✅ Comprehensive performance analysis
- ✅ Early detection of overfitting to seen categories

For more details on text conditioning and ULIP loss, see:
- `TEXT_CONDITIONING_IMPLEMENTATION.md`
- `ULIP_ALIGNMENT_LOSS_README.md`
