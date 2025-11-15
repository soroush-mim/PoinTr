# Quick Start Guide for New Benchmarks

## TL;DR

Two new benchmarks have been added to evaluate generalization:

1. **Unseen Categories**: Test on 8 ShapeNet categories NOT in PCN
2. **Hold-out Class**: Train without one PCN class, test on it

---

## Benchmark 1: Unseen Categories (Quick Start)

### Test on unseen categories with existing PCN checkpoint:

```bash
bash scripts/test_pcn_unseen.sh experiments/AdaPoinTr_PCN/ckpt-best.pth 0
```

**What it does:**
- Evaluates on 8 ShapeNet categories: bus, bed, bookshelf, bench, guitar, motorbike, skateboard, pistol
- Reports CD-ℓ1 per category and average
- Runs automatically after standard PCN test

**Output location:** `experiments/AdaPoinTr_PCN_Unseen/`

---

## Benchmark 2: Hold-out Class (Quick Start)

### Train while holding out airplane:

```bash
bash scripts/train_pcn_holdout.sh 02691156 0
```

### Test the hold-out model:

```bash
bash scripts/test_pcn_holdout.sh experiments/AdaPoinTr_PCN_holdout_airplane/ckpt-best.pth 02691156 0
```

**PCN Category IDs:**
```
02691156 = airplane      03636649 = lamp
02933112 = cabinet       04256520 = sofa
02958343 = car           04379243 = table
03001627 = chair         04530566 = watercraft
```

**What it does:**
- Trains on 7/8 PCN categories (excludes chosen class)
- Evaluates on held-out class
- Tests generalization within PCN domain

**Output location:** `experiments/AdaPoinTr_PCN_holdout_<category>/`

---

## Files Created

### Datasets
- `datasets/PCNDatasetUnseen.py` - Unseen categories dataset
- `datasets/PCNDatasetHoldout.py` - Hold-out class dataset

### Configs
- `cfgs/dataset_configs/PCNUnseen.yaml`
- `cfgs/dataset_configs/PCNHoldout_train.yaml`
- `cfgs/dataset_configs/PCNHoldout_eval.yaml`
- `cfgs/PCN_models/AdaPoinTr_holdout.yaml`

### Scripts
- `scripts/test_pcn_unseen.sh`
- `scripts/train_pcn_holdout.sh`
- `scripts/test_pcn_holdout.sh`

### Modified Files
- `datasets/__init__.py` - Added new dataset imports
- `cfgs/PCN_models/AdaPoinTr.yaml` - Added `test_unseen` dataset config
- `tools/runner.py` - Added automatic unseen & holdout evaluation

---

## Data Requirements

### Benchmark 1 (Unseen Categories)
Requires ShapeNet data for 8 unseen categories:
```
data/ShapeNet55-34/
├── projected_partial_noise/{taxonomy_id}/{model_id}/models/{idx}.pcd
├── {taxonomy_id}/{model_id}/models/model_normalized.pcd
└── Projected_ShapeNet-Unseen21_noise/test.txt
```

### Benchmark 2 (Hold-out Class)
Uses existing PCN data - no additional data needed.

---

## Verification

Check that everything is working:

```bash
# 1. Verify datasets are registered
python -c "from datasets import PCNDatasetUnseen, PCNDatasetHoldout; print('Datasets OK')"

# 2. Check config files exist
ls cfgs/dataset_configs/PCN*.yaml
ls cfgs/PCN_models/AdaPoinTr*.yaml

# 3. Verify scripts are executable
ls -l scripts/*pcn*.sh
```

---

## Full Documentation

See [BENCHMARKS_README.md](BENCHMARKS_README.md) for complete documentation including:
- Detailed descriptions
- Data preparation instructions
- Expected outputs
- Experimental protocols
- Troubleshooting

---

## Example Workflow

### Complete experiment for both benchmarks:

```bash
# 1. Train standard PCN model (if not already done)
python main.py --config cfgs/PCN_models/AdaPoinTr.yaml --exp_name AdaPoinTr_PCN

# 2. Test on unseen categories
bash scripts/test_pcn_unseen.sh experiments/AdaPoinTr_PCN/ckpt-best.pth 0

# 3. Train hold-out model (airplane)
bash scripts/train_pcn_holdout.sh 02691156 0

# 4. Test hold-out model
bash scripts/test_pcn_holdout.sh experiments/AdaPoinTr_PCN_holdout_airplane/ckpt-best.pth 02691156 0
```

Results will show:
- Standard PCN performance (8 seen categories)
- Unseen category performance (8 new categories)
- Hold-out performance (1 held-out PCN category)

This allows comparison of generalization across different scenarios.
