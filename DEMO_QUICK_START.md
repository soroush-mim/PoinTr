# Quick Start: Demo Refinement Script

## Overview

The demo script now:
✅ **Reads captions from CSV file** (format: `taxonomy_id_model_id, caption`)
✅ **Saves outputs as PCD files** by default (viewable in CloudCompare, Open3D, etc.)
✅ **Prints the caption** used for refinement

## Usage

### Basic Usage with CSV Captions

```bash
python scripts/demo_refinement.py \
    --adapointr_config cfgs/PCN_models/AdaPoinTr.yaml \
    --adapointr_ckpt checkpoints/adapointr_pcn.pth \
    --ulip_ckpt /path/to/ulip2_checkpoint.pt \
    --caption_csv data/captions.csv \
    --output_dir demo_results/
```

### Caption Priority

The script uses captions in this order:

1. **User-provided caption** (via `--caption`) - highest priority
2. **Caption from CSV file** (via `--caption_csv`)
3. **Default fallback**: `"a 3d point cloud of an object"`

### Example: Random Sample with CSV Captions

```bash
python scripts/demo_refinement.py \
    --adapointr_config cfgs/PCN_models/AdaPoinTr.yaml \
    --adapointr_ckpt checkpoints/adapointr_pcn.pth \
    --ulip_ckpt checkpoints/ulip2_pointbert.pt \
    --caption_csv data/pcn_captions.csv \
    --output_dir demo_random/
```

**Note:** The dataset split (test/val) is determined by the config file, not a command-line argument.

**Output:**
```
Selected sample index: 123
Taxonomy ID: 02691156
Model ID: 1a6f615e8b1b5ae4dbbc9440457e303e
Caption from CSV: "a 3d point cloud of an airplane with swept wings"

Running refinement...
Caption: "a 3d point cloud of an airplane with swept wings"
...
```

### Example: Specific Sample

```bash
python scripts/demo_refinement.py \
    --adapointr_config cfgs/PCN_models/AdaPoinTr.yaml \
    --adapointr_ckpt checkpoints/adapointr_pcn.pth \
    --ulip_ckpt checkpoints/ulip2_pointbert.pt \
    --caption_csv data/pcn_captions.csv \
    --sample_idx 42 \
    --output_dir demo_sample_42/
```

### Example: Override Caption from CSV

```bash
python scripts/demo_refinement.py \
    --adapointr_config cfgs/PCN_models/AdaPoinTr.yaml \
    --adapointr_ckpt checkpoints/adapointr_pcn.pth \
    --ulip_ckpt checkpoints/ulip2_pointbert.pt \
    --caption_csv data/pcn_captions.csv \
    --caption "a modern chair with curved legs" \
    --output_dir demo_custom_caption/
```

**Output:**
```
Using user-provided caption: "a modern chair with curved legs"
```

### Example: Different Save Formats

```bash
# Save as PCD (default)
python scripts/demo_refinement.py ... --save_format pcd

# Save as PLY
python scripts/demo_refinement.py ... --save_format ply

# Save as text
python scripts/demo_refinement.py ... --save_format txt

# Save as numpy
python scripts/demo_refinement.py ... --save_format npy
```

## CSV Format

Your caption CSV file should have this format:

```csv
02691156_1a6f615e8b1b5ae4dbbc9440457e303e, a 3d point cloud of an airplane with swept wings
02691156_1a04e3eab45ca15dd86060f189eb133, a 3d point cloud of a commercial airliner
03001627_1a6ad7a24bb89733f412783097373bdc, a 3d point cloud of an office chair with wheels
03001627_1a32f10b20170883663e90eaf6b4ca52, a 3d point cloud of a wooden dining chair
...
```

**Format:**
- **Column 1:** `{taxonomy_id}_{model_id}`
- **Column 2:** Caption text

## Output Files

After running, you'll get:

```
demo_results/
├── partial.pcd              # Input partial point cloud
├── gt.pcd                   # Ground truth complete point cloud
├── adapointr_output.pcd     # AdaPoinTr completion
├── refined_output.pcd       # ULIP-refined output
└── metadata.txt             # Sample info, caption, metrics
```

### Example metadata.txt:

```
PCN Completion + ULIP Refinement Demo Results
================================================================================

sample_idx: 123
taxonomy_id: 02691156
model_id: 1a6f615e8b1b5ae4dbbc9440457e303e
instance_key: 02691156_1a6f615e8b1b5ae4dbbc9440457e303e
caption: a 3d point cloud of an airplane with swept wings
partial_points: 2048
gt_points: 16384
adapointr_points: 16384
refinement_steps: 15
refined_points: 16384
text_alignment_before: 0.1234
text_alignment_after: 0.4567
text_alignment_improvement: 0.3333
```

## Visualizing PCD Files

### Using Open3D (Python)

```python
import open3d as o3d

# Load and visualize
pc = o3d.io.read_point_cloud('demo_results/refined_output.pcd')
o3d.visualization.draw_geometries([pc])

# Or compare multiple point clouds
partial = o3d.io.read_point_cloud('demo_results/partial.pcd')
gt = o3d.io.read_point_cloud('demo_results/gt.pcd')
adapointr = o3d.io.read_point_cloud('demo_results/adapointr_output.pcd')
refined = o3d.io.read_point_cloud('demo_results/refined_output.pcd')

# Color them differently
partial.paint_uniform_color([1, 0, 0])      # Red
gt.paint_uniform_color([0, 1, 0])           # Green
adapointr.paint_uniform_color([0, 0, 1])    # Blue
refined.paint_uniform_color([1, 1, 0])      # Yellow

# Visualize side by side
o3d.visualization.draw_geometries([partial, refined])
```

### Using CloudCompare

```bash
cloudcompare -o demo_results/partial.pcd \
             -o demo_results/refined_output.pcd
```

### Using Command Line (One-liner)

```bash
python -c "import open3d as o3d; pc = o3d.io.read_point_cloud('demo_results/refined_output.pcd'); o3d.visualization.draw_geometries([pc])"
```

## Complete Example Workflow

```bash
# Step 1: Prepare your caption CSV file
cat > data/my_captions.csv << EOF
02691156_1a6f615e8b1b5ae4dbbc9440457e303e, a 3d point cloud of a fighter jet
03001627_1a32f10b20170883663e90eaf6b4ca52, a 3d point cloud of an ergonomic office chair
EOF

# Step 2: Run demo with captions
python scripts/demo_refinement.py \
    --adapointr_config cfgs/PCN_models/AdaPoinTr.yaml \
    --adapointr_ckpt checkpoints/adapointr_pcn.pth \
    --ulip_ckpt checkpoints/ulip2_pointbert.pt \
    --caption_csv data/my_captions.csv \
    --refinement_steps 20 \
    --refinement_lr 0.05 \
    --output_dir results/demo_$(date +%Y%m%d_%H%M%S)/

# Step 3: Visualize results
cd results/demo_*/
python -c "import open3d as o3d; \
    partial = o3d.io.read_point_cloud('partial.pcd'); \
    refined = o3d.io.read_point_cloud('refined_output.pcd'); \
    partial.paint_uniform_color([1, 0, 0]); \
    refined.paint_uniform_color([0, 1, 0]); \
    o3d.visualization.draw_geometries([partial, refined])"
```

## Batch Processing Multiple Samples

```bash
#!/bin/bash
# Process multiple samples with different indices

for idx in 0 10 20 50 100; do
    echo "Processing sample $idx..."
    python scripts/demo_refinement.py \
        --adapointr_config cfgs/PCN_models/AdaPoinTr.yaml \
        --adapointr_ckpt checkpoints/adapointr_pcn.pth \
        --ulip_ckpt checkpoints/ulip2_pointbert.pt \
        --caption_csv data/captions.csv \
        --sample_idx $idx \
        --output_dir results/sample_${idx}/ \
        --save_format pcd
done

echo "Done! Results in results/sample_*/"
```

## Common Issues

### Issue: "Caption CSV not found"

Make sure the path is correct:
```bash
ls -l data/captions.csv
```

### Issue: "No caption found for sample"

Check if your CSV contains the instance key:
```bash
grep "02691156_1a6f615e8b1b5ae4dbbc9440457e303e" data/captions.csv
```

### Issue: PCD file won't open

Verify the file was created:
```bash
ls -lh demo_results/*.pcd
head -20 demo_results/partial.pcd
```

## Next Steps

1. ✅ Run demo on a random sample
2. ✅ Check the printed caption and metadata
3. ✅ Visualize PCD files
4. ✅ Compare AdaPoinTr vs. refined outputs
5. ⬜ Run full evaluation: `scripts/eval_pcn_with_refinement.py`

## Advanced: Creating Your Own Caption CSV

```python
import csv

# Example: Create captions from taxonomy names
captions = []
for taxonomy_id, model_id in dataset_samples:
    taxonomy_name = get_taxonomy_name(taxonomy_id)  # e.g., "airplane"
    caption = f"a 3d point cloud of a {taxonomy_name}"
    captions.append([f"{taxonomy_id}_{model_id}", caption])

# Save to CSV
with open('data/my_captions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(captions)
```

Or use GPT to generate detailed captions:
```python
import csv
import openai

captions = []
for taxonomy_id, model_id in dataset_samples:
    # Generate detailed caption with GPT
    prompt = f"Generate a detailed caption for a 3D point cloud of category {taxonomy_name}"
    caption = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )["choices"][0]["message"]["content"]

    captions.append([f"{taxonomy_id}_{model_id}", caption])

# Save
with open('data/detailed_captions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(captions)
```
