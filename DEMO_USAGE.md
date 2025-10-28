# Demo Usage Guide: ULIP-based Refinement

This guide shows how to run the demo script that visualizes ULIP-based test-time refinement on PCN dataset samples.

## Quick Start

### Basic Usage

```bash
python scripts/demo_refinement.py \
    --adapointr_config cfgs/PCN_models/AdaPoinTr.yaml \
    --adapointr_ckpt checkpoints/adapointr_pcn.pth \
    --ulip_ckpt /path/to/your/ulip2_checkpoint.pt \
    --output_dir demo_results/ \
    --caption "a 3d point cloud of a chair"
```

### What the Demo Does

1. **Loads a random PCN sample** (partial point cloud + ground truth)
2. **Runs AdaPoinTr** to complete the partial point cloud
3. **Runs ULIP refinement** to align the completion with the text caption
4. **Saves all outputs**:
   - `partial.txt` - Input partial point cloud
   - `gt.txt` - Ground truth complete point cloud
   - `adapointr_output.txt` - AdaPoinTr completion
   - `refined_output.txt` - ULIP-refined output
   - `metadata.txt` - Information about the sample and results

## Key Arguments

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--adapointr_config` | Path to AdaPoinTr config file (e.g., `cfgs/PCN_models/AdaPoinTr.yaml`) |
| `--adapointr_ckpt` | Path to AdaPoinTr checkpoint (e.g., `checkpoints/adapointr_pcn.pth`) |
| `--ulip_ckpt` | Path to ULIP-2 checkpoint file |

### Optional Arguments

#### Dataset Selection
- `--dataset_split` - Dataset split: `train`, `test`, or `val` (default: `test`)
- `--sample_idx` - Specific sample index, or random if not specified

#### Refinement Settings
- `--caption` - Text description (default: `"a 3d point cloud of an object"`)
- `--refinement_steps` - Number of refinement iterations (default: 15)
- `--refinement_lr` - Learning rate for refinement (default: 0.05)
- `--lambda_text` - Text alignment loss weight (default: 0.5)
- `--lambda_stick` - Sticking loss weight (default: 2.0)
- `--lambda_smooth` - Smoothness loss weight (default: 0.1)

#### Output Settings
- `--output_dir` - Output directory (default: `demo_results`)
- `--save_format` - Format: `txt`, `npy`, or `ply` (default: `txt`)
- `--no_refinement` - Skip refinement (only run AdaPoinTr)

## Examples

### Example 1: Random Chair Sample

```bash
python scripts/demo_refinement.py \
    --adapointr_config cfgs/PCN_models/AdaPoinTr.yaml \
    --adapointr_ckpt checkpoints/adapointr_pcn.pth \
    --ulip_ckpt checkpoints/ulip2_pointbert.pt \
    --caption "a 3d point cloud of a chair" \
    --output_dir results/chair_demo
```

### Example 2: Specific Sample with Aggressive Refinement

```bash
python scripts/demo_refinement.py \
    --adapointr_config cfgs/PCN_models/AdaPoinTr.yaml \
    --adapointr_ckpt checkpoints/adapointr_pcn.pth \
    --ulip_ckpt checkpoints/ulip2_pointbert.pt \
    --sample_idx 42 \
    --caption "a 3d point cloud of an airplane" \
    --refinement_steps 30 \
    --refinement_lr 0.08 \
    --lambda_text 1.0 \
    --output_dir results/airplane_demo
```

### Example 3: Only Run AdaPoinTr (No Refinement)

```bash
python scripts/demo_refinement.py \
    --adapointr_config cfgs/PCN_models/AdaPoinTr.yaml \
    --adapointr_ckpt checkpoints/adapointr_pcn.pth \
    --ulip_ckpt dummy.pt \
    --no_refinement \
    --output_dir results/adapointr_only
```

### Example 4: Save as PLY Format

```bash
python scripts/demo_refinement.py \
    --adapointr_config cfgs/PCN_models/AdaPoinTr.yaml \
    --adapointr_ckpt checkpoints/adapointr_pcn.pth \
    --ulip_ckpt checkpoints/ulip2_pointbert.pt \
    --save_format ply \
    --output_dir results/ply_output
```

## Understanding the Output

After running the demo, you'll see output like:

```
================================================================================
ULIP Refinement Demo
================================================================================
Device: cuda

Loading PCN dataset (split=test)...
✓ Dataset loaded: 1200 samples

Selected sample index: 123
Taxonomy ID: 02691156
Model ID: 1a6f615e8b1b5ae4dbbc9440457e303e
Partial shape: torch.Size([2048, 3])
GT shape: torch.Size([16384, 3])

Loading AdaPoinTr model...
✓ AdaPoinTr model loaded successfully
✓ AdaPoinTr output shape: torch.Size([16384, 3])

Running refinement...
Caption: "a 3d point cloud of a chair"
Steps: 15

Step   0/15: Total=2.3456, Text=-0.1234, Stick=1.2345, Smooth=0.0678
Step   5/15: Total=1.8234, Text=-0.2345, Stick=0.9876, Smooth=0.0432
...
Step  14/15: Total=1.2345, Text=-0.4567, Stick=0.6789, Smooth=0.0234

✓ Refined output shape: torch.Size([16384, 3])

Comparing text alignment...
Before refinement: 0.1234
After refinement:  0.4567
Improvement:       0.3333

================================================================================
Saving results to: demo_results
================================================================================
Saved: demo_results/partial.txt
Saved: demo_results/gt.txt
Saved: demo_results/adapointr_output.txt
Saved: demo_results/refined_output.txt
Saved: demo_results/metadata.txt

✓ Demo completed successfully!
```

## Visualizing Results

### Using Open3D (if installed)

```bash
# Visualize partial input
python -c "import open3d as o3d; import numpy as np; pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(np.loadtxt('demo_results/partial.txt')); o3d.visualization.draw_geometries([pc])"

# Visualize refined output
python -c "import open3d as o3d; import numpy as np; pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(np.loadtxt('demo_results/refined_output.txt')); o3d.visualization.draw_geometries([pc])"
```

### Using MeshLab

If saved as PLY format, open directly in MeshLab:
```bash
meshlab demo_results/refined_output.ply
```

### Custom Visualization Script

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load point clouds
partial = np.loadtxt('demo_results/partial.txt')
gt = np.loadtxt('demo_results/gt.txt')
adapointr = np.loadtxt('demo_results/adapointr_output.txt')
refined = np.loadtxt('demo_results/refined_output.txt')

# Plot
fig = plt.figure(figsize=(16, 4))

ax1 = fig.add_subplot(141, projection='3d')
ax1.scatter(partial[:, 0], partial[:, 1], partial[:, 2], s=1, c='blue')
ax1.set_title('Partial Input')

ax2 = fig.add_subplot(142, projection='3d')
ax2.scatter(gt[:, 0], gt[:, 1], gt[:, 2], s=1, c='green')
ax2.set_title('Ground Truth')

ax3 = fig.add_subplot(143, projection='3d')
ax3.scatter(adapointr[:, 0], adapointr[:, 1], adapointr[:, 2], s=1, c='orange')
ax3.set_title('AdaPoinTr Output')

ax4 = fig.add_subplot(144, projection='3d')
ax4.scatter(refined[:, 0], refined[:, 1], refined[:, 2], s=1, c='red')
ax4.set_title('Refined Output')

plt.tight_layout()
plt.savefig('demo_results/comparison.png', dpi=150)
plt.show()
```

## Handling RGB Padding

**Important:** ULIP's 3D encoder expects input with shape **(B, N, 6)** where the 6 channels are:
- Channels 0-2: x, y, z coordinates
- Channels 3-5: r, g, b colors

Since AdaPoinTr outputs only **(B, N, 3)** (xyz coordinates), the ULIP encoder automatically pads with **RGB = 0.4** (neutral gray) following ULIP's standard preprocessing. This happens transparently in the `ULIP3DEncoder.forward()` method.

### How It Works

```python
# In ULIP3DEncoder.forward():
B, N, C = xyz.shape

if C == 3:
    # Add RGB channels with neutral gray (0.4) as per ULIP standard
    rgb = torch.ones(B, N, 3, device=xyz.device) * 0.4
    xyz = torch.cat([xyz, rgb], dim=-1)  # (B, N, 6)

# Now pass to PointBERT encoder
pc_feat = self.point_encoder(xyz)
```

This matches ULIP's training setup where color information was not available for ModelNet40 samples.

## Troubleshooting

### Issue: "ULIP models not found"

Make sure you've copied ULIP models:
```bash
cd /root/soroush/PoinTr
bash scripts/setup_ulip_models.sh  # If you have this script
# Or manually follow ULIP_SETUP_INSTRUCTIONS.md
```

### Issue: "Checkpoint not found"

Verify paths:
```bash
ls -l checkpoints/adapointr_pcn.pth
ls -l /path/to/your/ulip2_checkpoint.pt
```

### Issue: CUDA out of memory

Reduce point cloud size or use CPU:
```bash
python scripts/demo_refinement.py \
    --device cpu \
    ...
```

### Issue: Import errors

Make sure dependencies are installed:
```bash
pip install open_clip_torch timm easydict
```

## Next Steps

After running the demo:
1. Compare the visual quality of AdaPoinTr vs. refined outputs
2. Experiment with different captions and hyperparameters
3. Run the full evaluation script: `scripts/eval_pcn_with_refinement.py`
4. Try other datasets (ShapeNet-55, ShapeNet-34)

## Citation

If you use this refinement approach, please cite:
- **ULIP**: [ULIP: Learning Unified Representation of Language, Image and Point Cloud for 3D Understanding](https://arxiv.org/abs/2212.05171)
- **PoinTr**: [PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers](https://arxiv.org/abs/2108.08839)
- **AdaPoinTr**: [AdaPoinTr: Diverse Point Cloud Completion with Adaptive Geometry-Aware Transformers](https://arxiv.org/abs/2301.04545)
