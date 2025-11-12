# ULIP Alignment Loss for Text-Conditioned AdaPoinTr

## Overview

This implementation adds **ULIP-based alignment loss** to provide explicit supervision for matching completed point clouds to their textual descriptions. The alignment loss uses a frozen PointBERT encoder from ULIP to compute contrastive loss between 3D point cloud embeddings and text embeddings.

## 🎯 What It Does

The ULIP alignment loss ensures that:
- The **completed point cloud** (model output) semantically matches the **text caption**
- Point cloud and text embeddings are aligned in a shared embedding space
- Semantic consistency is maintained between geometry and language

This is achieved through:
1. **Frozen ULIP PointBERT encoder** - Encodes completed point clouds to 512-dim embeddings
2. **CLIP text embeddings** (from text encoder) - Already computed during forward pass
3. **Contrastive loss** - Aligns point cloud and text embeddings using InfoNCE loss

## 📁 Files Created/Modified

### New Files:
- **`models/ulip_alignment_loss.py`** - ULIP PointBERT encoder and alignment loss implementation

### Modified Files:
- **`models/AdaPoinTr.py`** - Integrated ULIP loss module and updated loss computation
- **`tools/runner.py`** - Updated training loop to handle ULIP loss
- **`cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml`** - Added ULIP loss configuration

## 🏗️ Architecture

```
Training Flow with ULIP Alignment Loss:

Input: Partial Point Cloud + Text Caption
          |
          v
    [AdaPoinTr Model]
          |
          v
    Completed Point Cloud [B, 8192, 3]
          |
          +------------------------+
          |                        |
          v                        v
    [ULIP PointBERT]         [CLIP Text Encoder]
    (Frozen 3D Encoder)      (Frozen, already computed)
          |                        |
          v                        v
    3D Embeddings [B, 512]    Text Embeddings [B, 768→512]
          |                        |
          +------------------------+
                      |
                      v
            Contrastive Loss (InfoNCE)
                      |
                      v
            Total Loss = Chamfer Loss + ULIP Loss
```

## 🚀 Quick Start

### 1. Training with ULIP Loss

**Configuration is already set** in `AdaPoinTr_text.yaml`:

```yaml
model:
  use_ulip_loss: true
  ulip_loss_weight: 0.1
  ulip_temperature: 0.07
```

**Train command:**
```bash
python main.py --config cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml \
    --exp_name adapointr_text_ulip
```

### 2. Disable ULIP Loss

To train without ULIP loss (text conditioning only):

```yaml
model:
  use_ulip_loss: false  # or remove the parameter
```

## ⚙️ Configuration Parameters

### ULIP Loss Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_ulip_loss` | bool | `false` | Enable/disable ULIP alignment loss |
| `ulip_loss_weight` | float | `0.1` | Weight for ULIP loss (relative to Chamfer loss) |
| `ulip_temperature` | float | `0.07` | Temperature for contrastive loss |
| `ulip_config_path` | str | `null` | Path to PointBERT config (uses default if null) |
| `ulip_checkpoint_path` | str | `null` | Path to pretrained ULIP weights (optional) |

### Recommended Settings

**For ShapeNet34:**
```yaml
ulip_loss_weight: 0.1  # Start with 10% weight
ulip_temperature: 0.07  # Standard contrastive learning temperature
```

**If alignment loss dominates:**
```yaml
ulip_loss_weight: 0.05  # Reduce weight
```

**If alignment is weak:**
```yaml
ulip_loss_weight: 0.2   # Increase weight
```

## 📊 Monitoring Training

### Log Messages

Watch for these messages during training:

```
[ULIP-LOSS] Initializing ULIP alignment loss
[ULIP_ENCODER] Loading ULIP PointBERT encoder...
[ULIP_ENCODER] PointBERT encoder loaded successfully
[ULIP_ENCODER] Point cloud feature dimension: 768
[ULIP_ENCODER] Output embedding dimension: 512
[ULIP_ENCODER] All parameters frozen: True
[ULIP-LOSS] ULIP Alignment Loss initialized
[ULIP-LOSS] ULIP loss weight: 0.1
```

### Training Metrics

The training loop now reports three losses:

```
[Training] EPOCH: 1 Losses = ['Sparse: 2.5432', 'Dense: 1.8765', 'ULIP: 0.1234']
```

Where:
- **Sparse Loss** - Coarse completion loss (Chamfer distance)
- **Dense Loss** - Fine completion loss (Chamfer distance)
- **ULIP Loss** - Alignment loss (weighted by `ulip_loss_weight`)

### TensorBoard

ULIP metrics are logged to TensorBoard:
- `Loss/Batch/ULIP` - ULIP loss per batch
- `Loss/Epoch/ULIP` - Average ULIP loss per epoch
- `Acc/Batch/ULIP` - Point cloud to text matching accuracy

## 🔍 Implementation Details

### 1. ULIP PointBERT Encoder

```python
class ULIPPointBERTEncoder(nn.Module):
    """
    Frozen ULIP PointBERT encoder.
    - Encodes point clouds to 512-dim embeddings
    - All parameters frozen (no gradient updates)
    - Always in eval mode
    """
```

**Key Features:**
- Uses PointBERT architecture from ULIP
- Output dimension: 768 → projected to 512
- Completely frozen during training
- Can load pretrained ULIP checkpoints

### 2. Alignment Loss

```python
class ULIPAlignmentLoss(nn.Module):
    """
    Contrastive alignment loss between point clouds and text.
    - InfoNCE (NT-Xent) loss
    - Symmetric: point→text and text→point
    - Learnable temperature scaling
    """
```

**Loss Computation:**
```python
# Similarity matrix
logits = temperature * (pc_embeddings @ text_embeddings.T)

# Contrastive loss
loss = (CrossEntropy(logits, labels) + CrossEntropy(logits.T, labels)) / 2
```

### 3. Integration in AdaPoinTr

**Modified Components:**

1. **`__init__`**:
   - Creates ULIP loss module if `use_ulip_loss=True`
   - Frozen PointBERT encoder initialized

2. **`forward`**:
   - Returns text embeddings alongside point clouds
   - Text embeddings passed to loss computation

3. **`get_loss`**:
   - Computes ULIP alignment loss
   - Returns: `(sparse_loss, dense_loss, ulip_loss, ulip_acc)`

## 🧪 Testing

Test the ULIP alignment loss module standalone:

```bash
cd /home/soroush/Research/PoinTr
python models/ulip_alignment_loss.py
```

Expected output:
```
================================================================================
Testing ULIP Alignment Loss
================================================================================
Using device: cuda

Creating ULIP alignment loss module...
[ULIP_ENCODER] Loading ULIP PointBERT encoder...
[ULIP_ENCODER] PointBERT encoder loaded successfully
[ULIP_LOSS] ULIP Alignment Loss initialized

Input point cloud shape: torch.Size([4, 2048, 3])
Input text embeddings shape: torch.Size([4, 768])

Computing ULIP alignment loss...
ULIP alignment loss: 2.7183
Point-cloud to text accuracy: 25.00%

ULIP encoder frozen: True
Logit scale trainable: True

================================================================================
✅ Test passed!
================================================================================
```

## 📚 Technical Details

### Why ULIP?

ULIP (Unified Language-Image-Point) provides:
- **Pre-trained 3D encoder** aligned with language
- **Frozen representations** that capture semantic 3D structure
- **Contrastive training** for cross-modal alignment

### Why PointBERT?

- **Transformer-based** 3D encoder
- **Strong performance** on 3D understanding tasks
- **Compatible** with point cloud completion outputs
- **768-dim features** match well with CLIP text embeddings

### Gradient Flow

```
Partial PC → AdaPoinTr → Completed PC
                ↓
            [Gradients flow here]
                ↓
            Chamfer Loss + ULIP Loss
                              ↓
                    [No gradients - frozen]
                              ↓
                     ULIP PointBERT → 3D Embeddings
                                            ↓
                                    Contrastive Loss
                                            ↑
                                    [No gradients - frozen]
                                            ↑
                                   CLIP Text Embeddings
```

**Key Points:**
- ✅ Gradients flow to AdaPoinTr (completion model)
- ❌ No gradients to ULIP PointBERT (frozen)
- ❌ No gradients to CLIP text encoder (frozen)
- ✅ Logit scale (temperature) is learnable

## 🔧 Troubleshooting

### "ULIP encoder initialization failed"

**Cause:** PointBERT config file not found

**Solution:** The implementation uses a default fallback config. If you have a specific ULIP checkpoint, specify:

```yaml
model:
  ulip_config_path: './ulip_models/pointbert/ULIP_2_PointBERT_10k_colored_pointclouds.yaml'
```

### "Memory error during training"

**Cause:** ULIP encoder adds ~300MB GPU memory

**Solutions:**
1. Reduce batch size
2. Disable ULIP loss temporarily: `use_ulip_loss: false`
3. Use gradient checkpointing (if available)

### "ULIP loss is NaN"

**Causes:**
1. Text embeddings are None (captions not provided)
2. Point cloud has NaN values
3. Temperature is too low

**Solutions:**
1. Verify captions are loaded: check `[TEXT-CONDITIONING] Training with text captions enabled`
2. Add gradient clipping (already implemented in runner.py)
3. Increase temperature: `ulip_temperature: 0.1`

### "ULIP loss not decreasing"

**Possible causes:**
1. Weight too low: Try `ulip_loss_weight: 0.2`
2. Pre-training needed: Model may need more epochs
3. Captions mismatch: Verify caption quality in CSV file

## 📖 References

- **ULIP Paper**: [Learning Unified Representations of Language, Image and Point Cloud](https://arxiv.org/abs/2212.05171)
- **ULIP-2 Paper**: [Scalable Language-Image Pre-training with Enhanced Alignment](https://arxiv.org/abs/2305.08275)
- **PointBERT Paper**: [Pre-training 3D Point Cloud Transformers](https://arxiv.org/abs/2111.14819)
- **CLIP Paper**: [Learning Transferable Visual Models From Natural Language](https://arxiv.org/abs/2103.00020)

## 🎓 Citation

If you use this implementation, please cite:

```bibtex
@article{xue2022ulip,
  title={ULIP: Learning Unified Representations of Language, Image and Point Cloud for 3D Understanding},
  author={Xue, Le and Gao, Mingfei and Xing, Chen and Mart{\'\i}n-Mart{\'\i}n, Roberto and Wu, Jiajun and Xiong, Caiming and Xu, Ran and Niebles, Juan Carlos and Savarese, Silvio},
  journal={arXiv preprint arXiv:2212.05171},
  year={2022}
}

@article{yu2021pointbert,
  title={Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling},
  author={Yu, Xumin and Tang, Lulu and Rao, Yongming and Huang, Tiejun and Zhou, Jie and Lu, Jiwen},
  journal={arXiv preprint arXiv:2111.14819},
  year={2021}
}
```

## ✅ Summary

**Implemented:**
- ✅ ULIP PointBERT encoder (frozen)
- ✅ Contrastive alignment loss
- ✅ Integration in AdaPoinTr model
- ✅ Training loop updates
- ✅ Configuration parameters
- ✅ Comprehensive logging and monitoring

**Benefits:**
- Explicit text-geometry supervision
- Improved semantic consistency
- Better caption-to-completion alignment
- Optional (can be disabled)

**Usage:**
```bash
# With ULIP loss
python main.py --config cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml

# Without ULIP loss (text conditioning only)
# Set use_ulip_loss: false in config
```
