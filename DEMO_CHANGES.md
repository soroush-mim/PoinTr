# Demo Script Changes - Dataset Loading Fix

## Problem

The original demo script used `build_dataset_from_cfg()` directly, which didn't work properly with the PCN dataset configuration.

## Solution

Updated the demo script to use the same dataset loading approach as `eval_pcn_with_refinement.py`:

### Changes Made

1. **Updated `load_pcn_dataset()` function:**
   - Now uses `builder.dataset_builder(args, dataset_config)`
   - Returns both `(dataset, dataloader)` tuple
   - Properly handles caption CSV configuration
   - Automatically detects test/val/TEST config variants

2. **Added required command-line arguments:**
   - `--distributed` (default: False)
   - `--local_rank` (default: 0)
   - `--num_workers` (default: 4)
   - `--batch_size` (default: 1)

3. **Removed `--dataset_split` argument:**
   - Dataset split is now determined by the config file
   - Config specifies whether to use test or val split

4. **Improved caption handling:**
   - Caption priority: user-provided > dataset > CSV dict > default
   - Properly handles datasets with and without captions
   - Supports datasets that return 3 or 4 items

## Usage

### Before (didn't work):
```bash
python scripts/demo_refinement.py \
    --adapointr_config cfgs/PCN_models/AdaPoinTr.yaml \
    --adapointr_ckpt checkpoints/adapointr_pcn.pth \
    --ulip_ckpt checkpoints/ulip2_pointbert.pt \
    --dataset_split test \  # This didn't work properly
    --output_dir demo/
```

### After (works correctly):
```bash
python scripts/demo_refinement.py \
    --adapointr_config cfgs/PCN_models/AdaPoinTr.yaml \
    --adapointr_ckpt checkpoints/adapointr_pcn.pth \
    --ulip_ckpt checkpoints/ulip2_pointbert.pt \
    --caption_csv data/captions.csv \  # Optional
    --output_dir demo/
```

## Technical Details

### Dataset Loading Flow

```python
# Load config
config = cfg_from_yaml_file(config_path)

# Set caption config if provided
if caption_csv_path:
    config.dataset.val._base_.RETURN_CAPTIONS = True
    config.dataset.val._base_.CAPTION_CSV_PATH = caption_csv_path

# Build dataset using builder (same as eval script)
dataset, dataloader = builder.dataset_builder(args, config.dataset.val)

# Access samples
dataset_item = dataset[sample_idx]

# Handle caption variants
if len(dataset_item) == 4:
    taxonomy_id, model_id, data, caption = dataset_item
else:
    taxonomy_id, model_id, data = dataset_item
    caption = None
```

### Why This Works

1. **`builder.dataset_builder()`** is the official dataset loading method used throughout PoinTr
2. It properly initializes the dataset with all required parameters
3. It handles distributed training setup (even if not used)
4. It creates both dataset and dataloader with correct configurations
5. It respects the config file's dataset settings

### Config File Structure

Your config file (`cfgs/PCN_models/AdaPoinTr.yaml`) should have:

```yaml
dataset:
  val:  # or 'test' or 'TEST'
    _base_:
      NAME: PCN
      DATA_PATH: /path/to/pcn/data
      N_POINTS: 2048
      subset: test  # or 'val'
      RETURN_CAPTIONS: False  # Will be overridden if --caption_csv provided
      CAPTION_CSV_PATH: null
```

## Caption Priority

The demo now uses a clear priority system for captions:

1. **User-provided** via `--caption` (highest priority)
2. **From dataset** (loaded via config + CSV)
3. **From CSV dict** (fallback loading)
4. **Default**: "a 3d point cloud of an object" (lowest priority)

## Testing

To verify the fix works:

```bash
# Test 1: Basic loading (no captions)
python scripts/demo_refinement.py \
    --adapointr_config cfgs/PCN_models/AdaPoinTr.yaml \
    --adapointr_ckpt checkpoints/adapointr_pcn.pth \
    --ulip_ckpt checkpoints/ulip2_pointbert.pt \
    --output_dir test1/

# Test 2: With CSV captions
python scripts/demo_refinement.py \
    --adapointr_config cfgs/PCN_models/AdaPoinTr.yaml \
    --adapointr_ckpt checkpoints/adapointr_pcn.pth \
    --ulip_ckpt checkpoints/ulip2_pointbert.pt \
    --caption_csv data/captions.csv \
    --output_dir test2/

# Test 3: Override caption
python scripts/demo_refinement.py \
    --adapointr_config cfgs/PCN_models/AdaPoinTr.yaml \
    --adapointr_ckpt checkpoints/adapointr_pcn.pth \
    --ulip_ckpt checkpoints/ulip2_pointbert.pt \
    --caption "a custom caption" \
    --output_dir test3/
```

## Summary

✅ **Fixed**: Dataset loading now works correctly using `builder.dataset_builder()`
✅ **Fixed**: Caption loading from CSV via config
✅ **Fixed**: Proper handling of test/val splits from config
✅ **Added**: Support for datasets with and without captions
✅ **Improved**: Clear caption priority system

The demo script now works the same way as the evaluation script, ensuring consistency and reliability.
