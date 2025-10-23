# ULIP-based Test-Time Refinement for Point Cloud Completion

This module implements plug-and-play test-time refinement for point cloud completion using frozen ULIP-2 encoders and text-to-3D alignment.

## Overview

The refinement system takes completed point clouds from AdaPoinTr and refines them using:
- **Frozen ULIP-2 3D encoder** (Point-BERT/PointNeXt)
- **Frozen ULIP-2 text encoder** (OpenCLIP ViT-G/14)
- **Multi-objective optimization** with three loss terms:
  - Text alignment loss: Maximizes similarity between 3D and text embeddings
  - Sticking loss: Keeps refined points close to initial completion (Chamfer Distance)
  - Smoothness loss: Encourages local smoothness

**Key Feature**: Only point positions are optimized - all model weights remain frozen.

## Architecture

```
AdaPoinTr Output (B, N, 3)
         ↓
    [Test-Time Refinement]
    ├─ ULIP-2 3D Encoder (frozen) → z_3D
    ├─ ULIP-2 Text Encoder (frozen) → z_T
    └─ Loss = λ_text·L_text + λ_stick·L_stick + λ_smooth·L_smooth
         ↓
   Refined Output (B, N, 3)
```

## Installation

The refinement module is self-contained in the `refinement/` folder:

```
refinement/
├── __init__.py              # Module exports
├── ulip_loader.py           # ULIP-2 encoder loading
├── ulip_refinement.py       # Main refinement class
├── losses.py                # Loss functions
└── README.md                # This file
```

### Dependencies

- PyTorch >= 1.8
- NumPy
- AdaPoinTr (existing installation)
- Optional: open_clip_torch (for OpenCLIP text encoder)

## Usage

### 1. Basic Usage

```python
from refinement import load_ulip_encoders, ULIPRefinement

# Load frozen ULIP-2 encoders
encoder_3d, encoder_text = load_ulip_encoders(
    checkpoint_path='path/to/ulip2_checkpoint.pth',
    device='cuda'
)

# Initialize refinement module
refiner = ULIPRefinement(
    encoder_3d=encoder_3d,
    encoder_text=encoder_text,
    lambda_text=0.5,    # Text alignment weight
    lambda_stick=2.0,   # Sticking loss weight
    lambda_smooth=0.1,  # Smoothness weight
    device='cuda'
)

# Refine point cloud
P_initial = adapointr_output  # (B, N, 3)
text_caption = "a 3d point cloud of a chair"

P_refined = refiner.refine(
    P_initial,
    text_caption,
    steps=15,
    lr=0.05,
    verbose=True
)
```

### 2. Batch Refinement

```python
# Refine multiple point clouds with different captions
P_batch = torch.randn(4, 2048, 3)  # Batch of 4 point clouds
captions = [
    "a 3d point cloud of a chair",
    "a 3d point cloud of a table",
    "a 3d point cloud of a lamp",
    "a 3d point cloud of a sofa"
]

P_refined_batch = refiner.refine_batch(
    P_batch,
    captions,
    steps=15,
    lr=0.05
)
```

### 3. Evaluating Text Alignment

```python
# Compare text alignment before and after refinement
comparison = refiner.compare_before_after(
    P_initial,
    P_refined,
    text_caption
)

print(f"Similarity before: {comparison['before']}")
print(f"Similarity after: {comparison['after']}")
print(f"Improvement: {comparison['improvement']}")
```

## Evaluation on PCN Dataset

### Prepare Captions CSV

Create a CSV file with captions for PCN dataset:

```csv
02691156_1a04e3eab45ca15dd86060f189eb133,a wooden chair with curved back
02691156_1a6ad7a24bb89733f412783097373bdc,an office chair with wheels
...
```

Format: `{taxonomy_id}_{model_id}, caption`

If a caption is missing for an instance, the system will use a fallback:
`"a 3d point cloud of a {class_name}"`

### Run Evaluation

```bash
# Evaluate with refinement
python scripts/eval_pcn_with_refinement.py \
    --config cfgs/PCN_models/AdaPoinTr.yaml \
    --checkpoint checkpoints/adapointr_pcn.pth \
    --ulip_checkpoint checkpoints/ulip2.pth \
    --caption_csv data/pcn_captions.csv \
    --output_dir results/pcn_refinement \
    --steps 15 \
    --lr 0.05 \
    --lambda_text 0.5 \
    --lambda_stick 2.0 \
    --lambda_smooth 0.1 \
    --verbose

# Evaluate baseline only (no refinement)
python scripts/eval_pcn_with_refinement.py \
    --config cfgs/PCN_models/AdaPoinTr.yaml \
    --checkpoint checkpoints/adapointr_pcn.pth \
    --ulip_checkpoint checkpoints/ulip2.pth \
    --no_refinement \
    --output_dir results/pcn_baseline
```

### Evaluation Results

The script saves results to `{output_dir}/evaluation_results.json`:

```json
{
  "args": { ... },
  "baseline": [
    {
      "taxonomy_id": "02691156",
      "model_id": "1a04e3eab45ca15dd86060f189eb133",
      "metrics": [0.0123, 0.0456, 0.789],
      "caption": "a wooden chair with curved back"
    },
    ...
  ],
  "refined": [
    {
      "taxonomy_id": "02691156",
      "model_id": "1a04e3eab45ca15dd86060f189eb133",
      "metrics_before": [0.0123, 0.0456, 0.789],
      "metrics_after": [0.0098, 0.0321, 0.856],
      "improvement": [0.0025, 0.0135, 0.067],
      "caption": "a wooden chair with curved back"
    },
    ...
  ],
  "improvement": [avg_cd_l1_improve, avg_cd_l2_improve, avg_fscore_improve]
}
```

## Hyperparameter Tuning

### Default Configuration

```python
config = RefinementConfig.default()
# steps=15, lr=0.05
# λ_text=0.5, λ_stick=2.0, λ_smooth=0.1
```

### Aggressive Refinement

Use when you want stronger text alignment:

```python
config = RefinementConfig.aggressive()
# steps=30, lr=0.08
# λ_text=1.0, λ_stick=1.5, λ_smooth=0.05
```

### Conservative Refinement

Use when you want to stay closer to AdaPoinTr output:

```python
config = RefinementConfig.conservative()
# steps=10, lr=0.03
# λ_text=0.3, λ_stick=3.0, λ_smooth=0.2
```

### Custom Configuration

```python
refiner = ULIPRefinement(
    encoder_3d, encoder_text,
    lambda_text=0.7,     # Increase for stronger text alignment
    lambda_stick=1.5,    # Decrease to allow more deviation from initial
    lambda_smooth=0.05,  # Decrease for less smoothness constraint
    k_neighbors=16       # Increase for larger neighborhood smoothness
)
```

## Loss Functions

### 1. Text Similarity Loss

```python
L_text = -cos_sim(E_3D(P), E_T(caption))
```

Maximizes alignment between 3D embedding and text embedding.

### 2. Sticking Loss (Chamfer Distance)

```python
L_stick = CD(P, P_0)
```

Keeps refined points close to initial completion from AdaPoinTr.

### 3. Smoothness Loss

```python
L_smooth = mean((P_i - mean(neighbors(P_i)))^2)
```

Encourages local smoothness by penalizing deviation from local neighborhood mean.

## Modifying PCN Dataset for Captions

The PCN dataset has been modified to support caption loading:

### In Config File

```yaml
dataset:
  val:
    _base_:
      RETURN_CAPTIONS: True  # Enable caption return
      CAPTION_CSV_PATH: data/pcn_captions.csv  # Path to CSV file
```

### In Code

```python
# Dataset will return (taxonomy_id, model_id, (partial, gt), caption)
for taxonomy_id, model_id, data, caption in dataloader:
    partial, gt = data
    print(f"Caption: {caption}")
```

## Integration with AdaPoinTr

The refinement module is designed to be **plug-and-play** with zero modifications to AdaPoinTr:

1. Run AdaPoinTr inference as usual
2. Take the output point cloud
3. Apply refinement
4. Use the refined output

```python
# Standard AdaPoinTr pipeline
partial = data['partial'].cuda()
ret = adapointr_model(partial)
dense_points = ret[-1]  # AdaPoinTr output

# Add refinement (optional)
if use_refinement:
    dense_points = refiner.refine(
        dense_points,
        caption,
        steps=15,
        lr=0.05
    )

# Continue with evaluation/visualization
metrics = compute_metrics(dense_points, gt)
```

## Troubleshooting

### ULIP-2 Checkpoint Not Found

If you don't have ULIP-2 checkpoints, the system will use dummy encoders for testing:

```python
# This will work but won't provide meaningful refinement
encoder_3d, encoder_text = load_ulip_encoders('dummy_checkpoint.pth', 'cuda')
```

To get real ULIP-2 checkpoints, download from the official ULIP-2 repository.

### Caption CSV Format

Ensure your CSV file has the correct format:
- First column: `{taxonomy_id}_{model_id}` (no spaces)
- Second column: caption text
- No header row

Example:
```csv
02691156_1a04e3eab45ca15dd86060f189eb133,a wooden chair
02933112_1a32f10b20170883663e90eaf6b4ca52,a cabinet with doors
```

### Memory Issues

If you run out of memory during refinement:
1. Reduce batch size to 1
2. Reduce number of refinement steps
3. Use fewer neighbors for smoothness (k_neighbors=4)

## Future Extensions

This module is designed to support future text-conditional generation:

```python
# Future: Text-conditional completion
P_completed = adapointr_model(
    partial=partial_pc,
    text_condition=caption  # To be implemented
)

# Current: Test-time refinement
P_refined = refiner.refine(
    adapointr_output,
    caption
)
```

The modular design allows easy extension to:
- Text-conditioned training
- Multi-modal completion
- Interactive refinement with user feedback

## Citation

If you use this refinement module, please cite:

```bibtex
@article{adapointr,
  title={AdaPoinTr: Diverse Point Cloud Completion with Adaptive Geometry-Aware Transformers},
  author={...},
  journal={...},
  year={2023}
}

@article{ulip2,
  title={ULIP-2: Towards Scalable Multimodal Pre-training for 3D Understanding},
  author={...},
  journal={...},
  year={2023}
}
```

## License

This refinement module follows the same license as the AdaPoinTr repository.
