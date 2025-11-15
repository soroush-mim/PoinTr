# New Benchmarks for AdaPoinTr

This document describes two new benchmarks added to the AdaPoinTr repository for evaluating generalization to unseen categories.

## Overview

Two benchmarks have been implemented:

1. **Benchmark 1: PCN Unseen Categories** - Evaluates on 8 ShapeNet categories NOT in the original PCN dataset
2. **Benchmark 2: Hold-out Class** - Holds out one PCN class during training and evaluates on it as unseen

---

## Benchmark 1: PCN Unseen Categories

### Description

Evaluates point cloud completion on 8 ShapeNet categories that are NOT in the original PCN benchmark. This tests the model's ability to generalize to completely new object categories.

### Unseen Categories

The following 8 ShapeNet categories are used (NOT in PCN's 8 categories):

| Category | Taxonomy ID | Description |
|----------|-------------|-------------|
| Bus | 02924116 | Public transit vehicles |
| Bed | 02818832 | Furniture for sleeping |
| Bookshelf | 02871439 | Storage for books |
| Bench | 02828884 | Seating furniture |
| Guitar | 03467517 | String musical instrument |
| Motorbike | 03790512 | Two-wheeled motor vehicle |
| Skateboard | 04225987 | Sports equipment |
| Pistol | 03948459 | Handheld firearm |

### Data Requirements

- **Location**: `data/ShapeNet55-34/projected_partial_noise/`
- **Format**: Same as Projected_ShapeNet
  - Partial point clouds: `projected_partial_noise/{taxonomy_id}/{model_id}/models/{rendering_id}.pcd`
  - Complete point clouds: `ShapeNet55-34/{taxonomy_id}/{model_id}/models/model_normalized.pcd`
- **Preprocessing**: 2048 input points, 8192 output points (same as PCN)

### Usage

#### Testing Only (Using Pretrained PCN Model)

```bash
# Test on unseen categories using a PCN-trained checkpoint
bash scripts/test_pcn_unseen.sh experiments/AdaPoinTr_PCN/ckpt-best.pth 0
```

This will:
1. Run standard PCN test evaluation (8 seen categories)
2. Automatically run unseen category evaluation (8 unseen categories)
3. Report CD-ℓ1 metrics per category and average

#### Expected Output

```
============================ TEST RESULTS ============================
STANDARD PCN TEST EVALUATION
Taxonomy    #Sample    CDL1    CDL2    F-Score@1%    ...
02691156    100        2.45    ...
...

================================================================================
UNSEEN CATEGORY EVALUATION (8 ShapeNet categories not in PCN)
================================================================================
Taxonomy    #Sample    CDL1    CDL2    F-Score@1%    ...
02924116    50         3.12    ...    (bus)
02818832    50         2.87    ...    (bed)
02871439    50         3.45    ...    (bookshelf)
...
Overall                3.05    ...
```

### Files Modified/Created

- **Dataset**: `datasets/PCNDatasetUnseen.py` - New dataset class for unseen categories
- **Config**: `cfgs/dataset_configs/PCNUnseen.yaml` - Dataset configuration
- **Model Config**: Modified `cfgs/PCN_models/AdaPoinTr.yaml` to include `test_unseen`
- **Evaluation**: Modified `tools/runner.py` to run unseen evaluation after standard test
- **Scripts**: `scripts/test_pcn_unseen.sh` - Convenience test script

---

## Benchmark 2: Hold-out Class

### Description

Holds out one of the 8 PCN classes during training and evaluates on it as an unseen category. This tests the model's ability to generalize within the same dataset domain.

### PCN Categories

| Category | Taxonomy ID | Usage |
|----------|-------------|-------|
| Airplane | 02691156 | Can be held out |
| Cabinet | 02933112 | Can be held out |
| Car | 02958343 | Can be held out |
| Chair | 03001627 | Can be held out |
| Lamp | 03636649 | Can be held out |
| Sofa | 04256520 | Can be held out |
| Table | 04379243 | Can be held out |
| Watercraft | 04530566 | Can be held out |

### Data Requirements

- **Location**: Same as standard PCN (`/data/soroush/adapointr/PCN/ShapeNetCompletion/`)
- **Format**: PCN format (partial and complete point clouds)
- **Preprocessing**: 2048 input points, 8192 output points

### Usage

#### Training

Train while holding out one category (e.g., airplane):

```bash
# Hold out airplane during training
bash scripts/train_pcn_holdout.sh 02691156 0
```

This will:
1. Train on 7 PCN categories (excluding airplane)
2. Validate on all 8 categories
3. Save checkpoints in `experiments/AdaPoinTr_PCN_holdout_airplane/`

#### Testing

Test the hold-out model:

```bash
# Test on held-out category
bash scripts/test_pcn_holdout.sh experiments/AdaPoinTr_PCN_holdout_airplane/ckpt-best.pth 02691156 0
```

This will:
1. Run standard PCN test evaluation (all 8 categories)
2. Run hold-out evaluation (only the held-out category)
3. Report CD-ℓ1 metrics

#### Example: Hold out different categories

```bash
# Hold out chair
bash scripts/train_pcn_holdout.sh 03001627 0
bash scripts/test_pcn_holdout.sh experiments/AdaPoinTr_PCN_holdout_chair/ckpt-best.pth 03001627 0

# Hold out car
bash scripts/train_pcn_holdout.sh 02958343 0
bash scripts/test_pcn_holdout.sh experiments/AdaPoinTr_PCN_holdout_car/ckpt-best.pth 02958343 0
```

### Expected Output

```
============================ TEST RESULTS ============================
STANDARD PCN TEST EVALUATION
Taxonomy    #Sample    CDL1    CDL2    F-Score@1%    ...
02691156    100        X.XX    ...    (airplane - held out, but included in test)
02933112    100        2.35    ...    (cabinet)
...

================================================================================
HOLD-OUT CLASS EVALUATION (Held-out PCN category)
================================================================================
Taxonomy    #Sample    CDL1    CDL2    F-Score@1%    ...
02691156    100        X.XX    ...    (airplane)
Overall                X.XX    ...
```

### Files Modified/Created

- **Dataset**: `datasets/PCNDatasetHoldout.py` - New dataset class with hold-out support
- **Config**:
  - `cfgs/dataset_configs/PCNHoldout_train.yaml` - Training config (excludes hold-out class)
  - `cfgs/dataset_configs/PCNHoldout_eval.yaml` - Evaluation config (only hold-out class)
- **Model Config**: `cfgs/PCN_models/AdaPoinTr_holdout.yaml` - Full training/test config
- **Evaluation**: Modified `tools/runner.py` to run hold-out evaluation
- **Scripts**:
  - `scripts/train_pcn_holdout.sh` - Training script
  - `scripts/test_pcn_holdout.sh` - Testing script

---

## Implementation Details

### Code Structure

```
PoinTr/
├── datasets/
│   ├── PCNDatasetUnseen.py        # Benchmark 1: Unseen categories dataset
│   └── PCNDatasetHoldout.py       # Benchmark 2: Hold-out class dataset
├── cfgs/
│   ├── dataset_configs/
│   │   ├── PCNUnseen.yaml         # Unseen categories config
│   │   ├── PCNHoldout_train.yaml  # Hold-out training config
│   │   └── PCNHoldout_eval.yaml   # Hold-out evaluation config
│   └── PCN_models/
│       ├── AdaPoinTr.yaml         # Modified: added test_unseen
│       └── AdaPoinTr_holdout.yaml # New: hold-out benchmark config
├── tools/
│   └── runner.py                  # Modified: added unseen & holdout evaluation
└── scripts/
    ├── test_pcn_unseen.sh         # Benchmark 1 test script
    ├── test_pcn_holdout.sh        # Benchmark 2 test script
    └── train_pcn_holdout.sh       # Benchmark 2 training script
```

### Key Features

1. **Automatic Evaluation**: Both benchmarks run automatically after standard PCN test
2. **Per-category Metrics**: CD-ℓ1 reported for each category and overall average
3. **Backward Compatible**: All existing functionality preserved
4. **Flexible Configuration**: Easy to change held-out class via config files

### Metrics Reported

For both benchmarks, the following metrics are reported:

- **CD-ℓ1** (Chamfer Distance L1) - Primary metric
- **CD-ℓ2** (Chamfer Distance L2)
- **F-Score@1%** - Precision/recall metric
- Per-category breakdown
- Overall average

---

## Experimental Protocol

### Recommended Experiments

1. **Baseline**: Train on standard PCN, test on:
   - Standard PCN test set (8 seen categories)
   - Unseen categories (8 categories from Benchmark 1)

2. **Hold-out Analysis**: For each of the 8 PCN categories:
   - Train with that category held out
   - Evaluate on held-out category
   - Compare to full PCN training

3. **Cross-benchmark Comparison**:
   - Compare unseen category performance (Benchmark 1) vs hold-out performance (Benchmark 2)
   - Analyze which types of categories generalize better

### Example Experiment

```bash
# 1. Train standard PCN model
python main.py --config cfgs/PCN_models/AdaPoinTr.yaml --exp_name AdaPoinTr_PCN

# 2. Test on PCN + unseen categories
bash scripts/test_pcn_unseen.sh experiments/AdaPoinTr_PCN/ckpt-best.pth 0

# 3. Train hold-out models for each category
for class_id in 02691156 02933112 02958343 03001627 03636649 04256520 04379243 04530566; do
    bash scripts/train_pcn_holdout.sh $class_id 0
done

# 4. Test each hold-out model
for class_id in 02691156 02933112 02958343 03001627 03636649 04256520 04379243 04530566; do
    # Get category name
    case $class_id in
        02691156) name="airplane" ;;
        02933112) name="cabinet" ;;
        02958343) name="car" ;;
        03001627) name="chair" ;;
        03636649) name="lamp" ;;
        04256520) name="sofa" ;;
        04379243) name="table" ;;
        04530566) name="watercraft" ;;
    esac

    bash scripts/test_pcn_holdout.sh experiments/AdaPoinTr_PCN_holdout_${name}/ckpt-best.pth $class_id 0
done
```

---

## Data Preparation

### Benchmark 1: Unseen Categories

Ensure the following data structure exists:

```
data/ShapeNet55-34/
├── projected_partial_noise/
│   ├── 02924116/    # bus
│   ├── 02818832/    # bed
│   ├── 02871439/    # bookshelf
│   ├── 02828884/    # bench
│   ├── 03467517/    # guitar
│   ├── 03790512/    # motorbike
│   ├── 04225987/    # skateboard
│   └── 03948459/    # pistol
├── 02924116/        # Complete point clouds
├── 02818832/
├── ...
└── Projected_ShapeNet-Unseen21_noise/
    └── test.txt     # List of test instances
```

### Benchmark 2: Hold-out Class

Uses existing PCN data:

```
/data/soroush/adapointr/PCN/ShapeNetCompletion/
├── train/
│   ├── partial/
│   └── complete/
└── test/
    ├── partial/
    └── complete/
```

---

## Troubleshooting

### Issue: No unseen category data

**Error**: `FileNotFoundError` when loading unseen categories

**Solution**:
1. Check that `data/ShapeNet55-34/Projected_ShapeNet-Unseen21_noise/test.txt` exists
2. Verify that partial point clouds exist in `projected_partial_noise/`
3. Run data preparation script if needed

### Issue: Hold-out class not being excluded

**Error**: Hold-out class appears in training data

**Solution**:
1. Check `HOLDOUT_CLASS` in `cfgs/dataset_configs/PCNHoldout_train.yaml`
2. Verify `HOLDOUT_ONLY: FALSE` for training config
3. Check dataset loading logs for confirmation

### Issue: Scripts not executable

**Error**: Permission denied when running `.sh` scripts

**Solution**:
```bash
chmod +x scripts/test_pcn_unseen.sh
chmod +x scripts/test_pcn_holdout.sh
chmod +x scripts/train_pcn_holdout.sh
```

---

## Citation

If you use these benchmarks in your research, please cite:

```bibtex
@article{yu2021pointr,
  title={PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers},
  author={Yu, Xumin and Rao, Yongming and Wang, Ziyi and Liu, Zuyan and Lu, Jiwen and Zhou, Jie},
  journal={ICCV},
  year={2021}
}

@article{yu2023adapointr,
  title={AdaPoinTr: Diverse Point Cloud Completion with Adaptive Geometry-Aware Transformers},
  author={Yu, Xumin and Rao, Yongming and Wang, Ziyi and Lu, Jiwen and Zhou, Jie},
  journal={TPAMI},
  year={2023}
}
```

---

## Contact

For questions or issues with these benchmarks, please open an issue on the GitHub repository.
