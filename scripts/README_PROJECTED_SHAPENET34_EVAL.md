# Projected-ShapeNet-34 Evaluation with ULIP-2 Refinement

This script evaluates AdaPoinTr on the Projected-ShapeNet-34 dataset and reproduces **Table 4** from the AdaPoinTr paper with and without ULIP-2 refinement.

## Prerequisites

1. **Download the Projected-ShapeNet-34 dataset** following instructions in `DATASET.md`:
   - Download from [BaiduCloud](https://pan.baidu.com/s/14ei-HClbLr_5-xAG-00BHg?pwd=dycc) (code: dycc) or [HuggingFace](https://huggingface.co/datasets/zixiangAi/Projected-ShapeNet55-34)
   - Extract under `data/ShapeNet55-34/`
   - Run: `cat project_shapenet_pcd.tar* | tar xvf`

2. **Prepare checkpoints**:
   - AdaPoinTr checkpoint (trained on Projected-ShapeNet-34)
   - ULIP-2 checkpoint for refinement

## Dataset Structure

Expected directory structure:
```
PoinTr/
├── data/
│   ├── ShapeNet55-34/
│   │   ├── projected_partial_noise/
│   │   ├── shapenet_pc/
│   │   ├── Projected_ShapeNet-34_noise/
│   │   │   ├── train.txt
│   │   │   └── test.txt
│   │   └── Projected_ShapeNet-Unseen21_noise/
│   │       └── test.txt
```

## Basic Usage

### Evaluate on Both Seen and Unseen Categories (with and without refinement)

```bash
python scripts/eval_projected_shapenet34_with_refinement.py \
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr.yaml \
    --checkpoint /home/soroushm/data/AdaPoinTr_ps34.pth \
    --ulip_checkpoint /home/soroushm/data/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt \
    --output_dir results/projected_shapenet34_table4 \
    --caption_csv ~/data/Cap3D_automated_ShapeNet.csv
```

### Baseline Only (No Refinement)

```bash
python scripts/eval_projected_shapenet34_with_refinement.py \
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr.yaml \
    --checkpoint /path/to/adapointr_checkpoint.pth \
    --ulip_checkpoint /path/to/ulip2_checkpoint.pth \
    --no_refinement \
    --output_dir results/projected_shapenet34_baseline
```

### Refinement Only (Skip Baseline)

```bash
python scripts/eval_projected_shapenet34_with_refinement.py \
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr.yaml \
    --checkpoint /path/to/adapointr_checkpoint.pth \
    --ulip_checkpoint /path/to/ulip2_checkpoint.pth \
    --refinement_only \
    --output_dir results/projected_shapenet34_refined
```

### Evaluate Only Seen Categories

```bash
python scripts/eval_projected_shapenet34_with_refinement.py \
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr.yaml \
    --checkpoint ~/data/AdaPoinTr_ps34.pth \
    --ulip_checkpoint ~/data/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt \
    --eval_seen \
    --output_dir results/projected_shapenet34_seen
```

### Evaluate Only Unseen Categories

```bash
python scripts/eval_projected_shapenet34_with_refinement.py \
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr.yaml \
    --checkpoint /path/to/adapointr_checkpoint.pth \
    --ulip_checkpoint /path/to/ulip2_checkpoint.pth \
    --eval_unseen \
    --output_dir results/projected_shapenet34_unseen
```

## Optional Arguments

### Refinement Parameters (same as PCN evaluation)

```bash
--steps 15              # Number of refinement steps (default: 15)
--lr 0.05               # Refinement learning rate (default: 0.05)
--lambda_text 0.5       # Text alignment loss weight (default: 0.5)
--lambda_stick 2.0      # Sticking loss (CD) weight (default: 2.0)
--lambda_smooth 0.1     # Smoothness loss weight (default: 0.1)
--k_neighbors 8         # Number of neighbors for smoothness (default: 8)
```

### Captions

```bash
--caption_csv /path/to/captions.csv   # Optional: CSV file with captions
```

Format: `{taxonomy_id}_{model_id},caption text`

Example:
```
02691156_1a04e3eab45ca15dd86060f189eb133,a small airplane with wings
02958343_5a6ad7a24bb89733f412783097373bdc,a red sports car
```

### Other Options

```bash
--verbose               # Print detailed refinement progress
--save_outputs          # Save point clouds to disk
--device cuda           # Device (cuda or cpu)
--batch_size 1          # Batch size for evaluation
--num_workers 8         # Number of data loading workers
```

## Output

The script generates:

1. **Console output** with:
   - Per-category results in Table 4 format
   - Overall metrics for seen and unseen categories
   - Comparison tables (baseline vs refined)
   - Improvement metrics

2. **JSON file** (`evaluation_results.json`) containing:
   - All metrics for seen and unseen categories
   - Per-category breakdowns
   - Configuration parameters used

3. **Point clouds** (if `--save_outputs` is specified)

## Example Output

```
================================================================================
Table 4 Format Results - 34 Seen Categories
================================================================================
Category             Baseline CD-L1  Baseline F-Score   Refined CD-L1   Refined F-Score
--------------------------------------------------------------------------------------
02691156             12.50           0.6234             10.20           0.7012
02933112             15.30           0.5821             12.80           0.6543
...
================================================================================

================================================================================
Comparison Table - 34 Seen Categories
================================================================================
Method                         CD-L1 (×1000)        F-Score@1%
----------------------------------------------------------------------
Baseline (AdaPoinTr)           13.45                0.6123
With ULIP-2 Refinement         11.20                0.7021
----------------------------------------------------------------------
Improvement                    2.25                 0.0898
================================================================================
```

## Reproducing Table 4

To exactly reproduce **Table 4** from the AdaPoinTr paper:

1. Train AdaPoinTr on Projected-ShapeNet-34:
```bash
python main.py \
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr.yaml \
    --exp_name adapointr_projected_shapenet34
```

2. Run evaluation on both seen and unseen categories:
```bash
python scripts/eval_projected_shapenet34_with_refinement.py \
    --config cfgs/Projected_ShapeNet34_models/AdaPoinTr.yaml \
    --checkpoint experiments/adapointr_projected_shapenet34/ckpt-best.pth \
    --ulip_checkpoint /path/to/ulip2_checkpoint.pth \
    --output_dir results/table4_reproduction
```

The script will output results in the exact format of Table 4, showing:
- **34 seen categories**: Per-category and overall CD-ℓ1 and F-Score@1%
- **21 unseen categories**: Per-category and overall CD-ℓ1 and F-Score@1%
- **Comparison**: Baseline vs ULIP-2 refinement for both splits

## Notes

- The script uses **Point-BERT** as the 3D backbone (as specified)
- Input: **2048 points** (downsampled from partial point clouds)
- Output: **8192 points** (complete point clouds)
- Metrics: **CD-ℓ1 (×1000)** and **F-Score@1%** (as in Table 4)
- Default refinement parameters match the PCN evaluation settings

## Troubleshooting

### Dataset not found
Ensure the dataset is extracted to `data/ShapeNet55-34/` and the paths in the config match.

### CUDA out of memory
Reduce batch size: `--batch_size 1`

### Missing ULIP checkpoint
Download or specify the correct path to your ULIP-2 Point-BERT checkpoint.

### No captions provided
The script will use generic captions ("a 3d point cloud") if no caption CSV is provided. This may slightly reduce refinement performance but is acceptable for baseline comparisons.
