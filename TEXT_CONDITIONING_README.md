# Text-Conditioned Point Cloud Completion with AdaPoinTr

## Overview

This implementation adds **text-conditioned query generation** to AdaPoinTr, allowing the model to leverage semantic information from text captions to guide point cloud completion. The text conditioning uses a frozen CLIP encoder to extract text features that are integrated into the adaptive query generation process.

## 🎯 Key Features

- ✅ **CLIP Text Encoder Integration**: Frozen CLIP-ViT-Large-Patch14 for extracting 768-dim semantic features
- ✅ **Text-Conditioned Query Generation**: Queries are generated based on both geometric and semantic features
- ✅ **Backward Compatible**: Model works with or without captions (captions=None uses original behavior)
- ✅ **Comprehensive Testing**: Full test suite for validation
- ✅ **Easy Configuration**: Simple YAML config to enable/disable text conditioning

## 📁 New Files

### Core Components
- `models/text_encoder.py` - CLIP text encoder wrapper (frozen parameters)
- `models/text_query_generator.py` - Text-conditioned query generators
- `test_text_conditioning.py` - Comprehensive test suite

### Configuration
- `cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml` - Model config with text conditioning
- `cfgs/dataset_configs/Projected_ShapeNet-34_noise_text.yaml` - Dataset config with captions

### Data
- `data/shapenet/captions_example.csv` - Example caption file (70+ samples)

### Documentation
- `TEXT_CONDITIONING_IMPLEMENTATION.md` - Detailed implementation documentation
- `TEXT_CONDITIONING_README.md` - This file

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install transformers  # For CLIP text encoder
```

### 2. Run Tests

Verify the implementation works:

```bash
python test_text_conditioning.py
```

Expected output: All 7 tests should pass ✅

### 3. Prepare Captions

Either use the example file or create your own:

```csv
{category_id}_{instance_id},caption text
02958343_abc123,a red sports car with sleek design
03001627_def456,a wooden chair with four legs
```

Update the path in `cfgs/dataset_configs/Projected_ShapeNet-34_noise_text.yaml`:

```yaml
CAPTION_FILE: 'data/shapenet/captions.csv'  # Update this path
```

### 4. Train

**With text conditioning:**
```bash
python main.py --config cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml \
    --exp_name adapointr_text_conditioned
```

**Without text conditioning (original model):**
```bash
python main.py --config cfgs/Projected_ShapeNet34_models/AdaPoinTr.yaml \
    --exp_name adapointr_baseline
```

## 🔧 Configuration

### Enable/Disable Text Conditioning

In your model config YAML:

```yaml
model:
  NAME: AdaPoinTr
  use_text_conditioning: true  # Set to false to disable
  text_encoder_name: 'openai/clip-vit-large-patch14'  # CLIP model to use
  # ... rest of config
```

### Dataset Configuration

In your dataset config YAML:

```yaml
USE_CAPTIONS: true  # Enable caption loading
CAPTION_FILE: 'data/shapenet/captions.csv'  # Path to caption CSV
```

## 📊 Caption CSV Format

```csv
{category_id}_{instance_id},caption text
```

Example:
```csv
02958343_1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d,a sports car with aerodynamic design
03001627_1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e,an office chair with adjustable height
04256520_1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f,a three-seater sofa with cushioned armrests
```

- **First column**: Unique identifier `{category_id}_{instance_id}`
- **Second column**: Natural language description of the object
- **No header row**

## 🧪 Testing

The test suite (`test_text_conditioning.py`) includes:

1. **Text Encoder Test**: Verifies CLIP loads correctly and parameters are frozen
2. **Text Feature Test**: Checks output shapes and gradient properties
3. **Query Generator Test**: Validates query generation with text features
4. **Forward Pass Test**: Tests model with and without captions
5. **Gradient Flow Test**: Ensures text encoder stays frozen during training
6. **Memory Usage Test**: Checks GPU memory consumption is reasonable
7. **Backward Compatibility Test**: Confirms model works without text conditioning

## 📝 Training Log Messages

Watch for these during training:

```
[TEXT_ENCODER] Loading CLIP text encoder: openai/clip-vit-large-patch14
[TEXT_ENCODER] CLIP text encoder loaded successfully
[TEXT_ENCODER] Text feature dimension: 768
[TEXT_ENCODER] All parameters frozen: True
[TEXT-CONDITIONING] Enabled with encoder: openai/clip-vit-large-patch14
[DATASET] Loading captions from data/shapenet/captions.csv
[DATASET] Loaded 1234 captions from CSV
[TEXT-CONDITIONING] Training with text captions enabled
```

## 🏗️ Architecture

```
Input: Partial Point Cloud + Text Caption
          |
          v
    [DGCNN Grouper] -----> Point Proxies
          |
          v
  [Transformer Encoder] --> Encoder Features
          |                       |
          |                       v
          |              [CLIP Text Encoder (frozen)]
          |                       |
          |                       v
          |                  Text Features [B, 768]
          |                       |
          +-------+-------+-------+
                  |
                  v
        [Text-Conditioned Query Generator]
                  |
                  v
              Queries [B, 512, 384]
                  |
                  v
      [Transformer Decoder]
                  |
                  v
         Decoded Features
                  |
                  v
          [Rebuild Head]
                  |
                  v
      Complete Point Cloud [B, 8192, 3]
```

## ⚙️ Implementation Details

### Text Encoder (CLIP)
- **Model**: openai/clip-vit-large-patch14
- **Output Dimension**: 768
- **Parameters**: Frozen (no gradient updates)
- **Mode**: Always in eval mode

### Query Generator
- **Input**: Encoder features [B, N, 384] + Text features [B, 768]
- **Architecture**: MLP with ReLU and 0.1 dropout
- **Output**: Queries [B, 512, 384]

### Integration Point
Text features are concatenated with geometric features during query generation:
```python
if use_text_conditioning and captions is not None:
    q = mlp_query_text(concat([encoder_features, text_features, coordinates]))
else:
    q = mlp_query(concat([encoder_features, coordinates]))
```

## 🔍 Troubleshooting

### "CLIP model not found"
```bash
pip install transformers
```

### "Caption file not found"
Check the path in your dataset config YAML matches your actual caption file location.

### "Memory error"
- Reduce batch size in training config
- Use gradient checkpointing (if implemented)
- Ensure CLIP encoder is not accumulating gradients

### Model not using captions
- Verify `use_text_conditioning: true` in model config
- Verify `USE_CAPTIONS: true` in dataset config
- Check training logs for `[TEXT-CONDITIONING]` messages

## 📚 References

- **AdaPoinTr Paper**: https://arxiv.org/abs/2301.04545
- **CLIP Paper**: https://arxiv.org/abs/2103.00020
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers

## 🔮 Future Enhancements

### ULIP-Based Alignment Loss (Optional)
For even better text-geometry alignment, consider implementing:
- Frozen ULIP 3D encoder (PointBERT/PointNeXt)
- Contrastive loss between ULIP 3D embeddings and CLIP text embeddings
- Weighted auxiliary loss term

This would provide explicit supervision for matching completed point clouds to their textual descriptions.

## 📄 License

Same as PoinTr (MIT License)

## 🤝 Contributing

For issues or improvements:
1. Test your changes with `test_text_conditioning.py`
2. Update documentation if adding features
3. Ensure backward compatibility (model works without captions)

## 📧 Contact

For questions about this implementation, refer to:
- `TEXT_CONDITIONING_IMPLEMENTATION.md` for technical details
- `test_text_conditioning.py` for usage examples
- Original PoinTr/AdaPoinTr repository for base model questions
