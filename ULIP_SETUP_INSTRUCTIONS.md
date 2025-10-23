# ULIP Models Setup Instructions

This guide explains which files to copy from the ULIP repository to enable ULIP-based test-time refinement.

## Summary

You need to copy ULIP model files from `../ULIP/models/` to `PoinTr/ulip_models/` to load ULIP-2 encoders.

## Step-by-Step Instructions

### 1. Create the Target Directory

```bash
cd /root/soroush/PoinTr
mkdir -p ulip_models/utils
```

### 2. Copy Core ULIP Files

```bash
# Copy main model definitions
cp ../ULIP/models/ULIP_models.py ulip_models/

# Copy losses (required by ULIP_models)
cp ../ULIP/models/losses.py ulip_models/
```

### 3. Copy PointBERT Encoder (for ULIP-2)

```bash
# Copy entire pointbert directory
cp -r ../ULIP/models/pointbert ulip_models/
```

### 4. Copy Utilities

```bash
# Copy config utilities needed for YAML parsing
cp ../ULIP/utils/config.py ulip_models/utils/
cp ../ULIP/utils/__init__.py ulip_models/utils/
```

### 5. Create __init__.py Files

```bash
# Make it a Python package
touch ulip_models/__init__.py
```

### 6. Optional: Copy Additional Encoders

If you plan to use PointNeXt or PointNet2:

```bash
# For PointNeXt support
cp -r ../ULIP/models/pointnext ulip_models/

# For PointNet2 support
cp -r ../ULIP/models/pointnet2 ulip_models/
```

## Final Directory Structure

After copying, your structure should look like:

```
PoinTr/
├── ulip_models/                          # NEW
│   ├── __init__.py                       # Created in step 5
│   ├── ULIP_models.py                    # From step 2
│   ├── losses.py                         # From step 2
│   │
│   ├── pointbert/                        # From step 3
│   │   ├── point_encoder.py
│   │   ├── dvae.py
│   │   ├── misc.py
│   │   ├── checkpoint.py
│   │   ├── logger.py
│   │   ├── PointTransformer_8192point.yaml
│   │   └── ULIP_2_PointBERT_10k_colored_pointclouds.yaml
│   │
│   ├── pointnext/                        # Optional, from step 6
│   │   ├── pointnext.py
│   │   ├── pointnext-s.yaml
│   │   └── PointNeXt/                    # Large subdirectory
│   │
│   ├── pointnet2/                        # Optional, from step 6
│   │   ├── pointnet2.py
│   │   └── pointnet2_utils.py
│   │
│   └── utils/                            # From step 4
│       ├── __init__.py
│       └── config.py
│
└── refinement/
    └── ulip_loader.py                    # Already updated
```

## All-in-One Command

Execute all steps at once:

```bash
cd /root/soroush/PoinTr

# Create directories
mkdir -p ulip_models/utils

# Copy main files
cp ../ULIP/models/ULIP_models.py ulip_models/
cp ../ULIP/models/losses.py ulip_models/

# Copy PointBERT
cp -r ../ULIP/models/pointbert ulip_models/

# Copy utilities
cp ../ULIP/utils/config.py ulip_models/utils/
cp ../ULIP/utils/__init__.py ulip_models/utils/

# Create __init__ files
touch ulip_models/__init__.py

# Optional: Copy additional encoders
# cp -r ../ULIP/models/pointnext ulip_models/
# cp -r ../ULIP/models/pointnet2 ulip_models/

echo "✓ ULIP models copied successfully!"
```

## Verification

Check that files were copied correctly:

```bash
# List main files
ls -la ulip_models/

# Check PointBERT
ls -la ulip_models/pointbert/

# Check utils
ls -la ulip_models/utils/
```

You should see:
- `ULIP_models.py` and `losses.py` in `ulip_models/`
- Point encoder files in `ulip_models/pointbert/`
- `config.py` in `ulip_models/utils/`
- `__init__.py` files in both directories

## Usage After Setup

Once files are copied, you can load ULIP encoders:

```python
from refinement import load_ulip_encoders

# Load ULIP-2 with PointBERT (most common)
encoder_3d, encoder_text = load_ulip_encoders(
    checkpoint_path='/path/to/your/ulip2_checkpoint.pt',
    device='cuda',
    model_type='ULIP2_PointBERT'
)

# Use in refinement
from refinement import ULIPRefinement

refiner = ULIPRefinement(encoder_3d, encoder_text, device='cuda')
output_refined = refiner.refine(
    adapointr_output,
    "a 3d point cloud of a chair",
    steps=15,
    lr=0.05
)
```

## Model Types

The `load_ulip_encoders()` function supports three model types:

1. **`ULIP2_PointBERT`** (default, recommended)
   - ULIP-2 with PointBERT 3D encoder
   - Uses OpenCLIP ViT-G/14 for text
   - Requires: `pointbert/`

2. **`ULIP_PointBERT`** (legacy)
   - ULIP-1 with PointBERT 3D encoder
   - Uses custom CLIP-style text encoder
   - Requires: `pointbert/`

3. **`ULIP_PN_NEXT`** (legacy)
   - ULIP-1 with PointNeXt 3D encoder
   - Uses custom CLIP-style text encoder
   - Requires: `pointnext/`

## Troubleshooting

### If you see "ERROR: ULIP models not found"

The loader will print instructions:
```
ERROR: ULIP models not found at /root/soroush/PoinTr/ulip_models
Please copy ULIP models to PoinTr/ulip_models/ directory

Instructions:
  1. mkdir -p ulip_models
  2. cp -r ../ULIP/models/ULIP_models.py ulip_models/
  3. cp -r ../ULIP/models/losses.py ulip_models/
  4. cp -r ../ULIP/models/pointbert ulip_models/
  5. mkdir -p ulip_models/utils
  6. cp ../ULIP/utils/config.py ulip_models/utils/
  7. touch ulip_models/__init__.py ulip_models/utils/__init__.py
```

Follow these instructions and try again.

### If you see import errors

Make sure you have the required dependencies:
```bash
pip install open_clip_torch timm easydict
```

### Dummy Encoders

If the loader cannot find ULIP models or checkpoints, it will fall back to dummy encoders for testing. These work but won't provide meaningful refinement.

## Dependencies

Required Python packages:
- `torch` (already installed with AdaPoinTr)
- `open_clip_torch` (for ULIP-2 text encoder)
- `timm` (for vision models)
- `easydict` (for config parsing)

Install with:
```bash
pip install open_clip_torch timm easydict
```

## Notes

1. **File Size**: The `pointbert/` directory is small (~100KB). The `pointnext/PointNeXt/` directory is large (~10MB) and only needed for PointNeXt models.

2. **No Core Changes**: This setup does NOT modify any existing AdaPoinTr code. All ULIP files are in a separate `ulip_models/` directory.

3. **Checkpoint Format**: Your ULIP-2 checkpoint should be a `.pt` file with either:
   - `checkpoint['state_dict']` (standard format)
   - Direct state dict (alternative format)

4. **Model Weights**: The checkpoint should contain trained ULIP-2 weights. The loader will automatically handle the `module.` prefix if present (from DataParallel training).

## Example: Complete Workflow

```bash
# 1. Copy ULIP models (one-time setup)
cd /root/soroush/PoinTr
bash << 'EOF'
mkdir -p ulip_models/utils
cp ../ULIP/models/ULIP_models.py ulip_models/
cp ../ULIP/models/losses.py ulip_models/
cp -r ../ULIP/models/pointbert ulip_models/
cp ../ULIP/utils/config.py ulip_models/utils/
cp ../ULIP/utils/__init__.py ulip_models/utils/
touch ulip_models/__init__.py
EOF

# 2. Test the loader
python refinement/ulip_loader.py

# 3. Run evaluation with your checkpoint
python scripts/eval_pcn_with_refinement.py \
    --config cfgs/PCN_models/AdaPoinTr.yaml \
    --checkpoint checkpoints/adapointr_pcn.pth \
    --ulip_checkpoint /path/to/your/ulip2_checkpoint.pt \
    --caption_csv data/captions.csv \
    --output_dir results/
```

## Support

If you encounter issues:
1. Check that all files are copied correctly (see Verification section)
2. Verify your ULIP-2 checkpoint path
3. Check dependencies are installed
4. Review error messages - the loader provides detailed feedback

The loader has been updated in `refinement/ulip_loader.py` and will guide you through the setup process.
