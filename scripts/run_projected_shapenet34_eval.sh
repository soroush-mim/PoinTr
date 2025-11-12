#!/bin/bash

# Example script to run Projected-ShapeNet-34 evaluation with ULIP-2 refinement
# This reproduces Table 4 from the AdaPoinTr paper

# =============================================================================
# Configuration - UPDATE THESE PATHS
# =============================================================================

# Path to your AdaPoinTr checkpoint
ADAPOINTR_CHECKPOINT="/home/soroushm/data/AdaPoinTr_ps34.pth"

# Path to your ULIP-2 checkpoint
ULIP2_CHECKPOINT="/home/soroushm/data/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"

# Output directory for results
OUTPUT_DIR="results/projected_shapenet34_table4/steps15"

# Optional: Path to caption CSV file (leave empty if not available)
CAPTION_CSV="~/data/Cap3D_automated_ShapeNet.csv"

# =============================================================================
# Evaluation Settings (matching PCN eval defaults)
# =============================================================================

CONFIG="cfgs/Projected_ShapeNet34_models/AdaPoinTr.yaml"
DEVICE="cuda"
BATCH_SIZE=1
NUM_WORKERS=8

# Refinement parameters (same as PCN eval)
STEPS=15
LR=0.001
LAMBDA_TEXT=1.5
LAMBDA_STICK=3.5
LAMBDA_SMOOTH=0.01
K_NEIGHBORS=8

# =============================================================================
# Run Evaluation
# =============================================================================

echo "=========================================="
echo "Projected-ShapeNet-34 Evaluation"
echo "Reproducing Table 4 from AdaPoinTr Paper"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Config:            $CONFIG"
echo "  AdaPoinTr ckpt:    $ADAPOINTR_CHECKPOINT"
echo "  ULIP-2 ckpt:       $ULIP2_CHECKPOINT"
echo "  Output dir:        $OUTPUT_DIR"
echo "  Device:            $DEVICE"
echo "  Batch size:        $BATCH_SIZE"
echo ""
echo "Refinement Settings:"
echo "  Steps:             $STEPS"
echo "  Learning rate:     $LR"
echo "  Lambda text:       $LAMBDA_TEXT"
echo "  Lambda stick:      $LAMBDA_STICK"
echo "  Lambda smooth:     $LAMBDA_SMOOTH"
echo "  K neighbors:       $K_NEIGHBORS"
echo ""
echo "=========================================="
echo ""

# Build the command
CMD="python scripts/eval_projected_shapenet34_with_refinement.py \
    --config $CONFIG \
    --checkpoint $ADAPOINTR_CHECKPOINT \
    --ulip_checkpoint $ULIP2_CHECKPOINT \
    --output_dir $OUTPUT_DIR \
    --device $DEVICE \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --steps $STEPS \
    --lr $LR \
    --lambda_text $LAMBDA_TEXT \
    --lambda_stick $LAMBDA_STICK \
    --lambda_smooth $LAMBDA_SMOOTH \
    --k_neighbors $K_NEIGHBORS \
    --verbose \
    --refinement_only
    # --eval_seen \
    # --eval_unseen"

# Add caption CSV if provided
if [ ! -z "$CAPTION_CSV" ]; then
    CMD="$CMD --caption_csv $CAPTION_CSV"
fi

# Run the evaluation
echo "Running evaluation..."
echo ""
eval $CMD

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
