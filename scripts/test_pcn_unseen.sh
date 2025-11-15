#!/bin/bash

# Test script for Benchmark 1: PCN Unseen Categories
# Evaluates AdaPoinTr on 8 ShapeNet categories NOT in PCN:
# bus, bed, bookshelf, bench, guitar, motorbike, skateboard, pistol
#
# Usage:
#   bash scripts/test_pcn_unseen.sh <checkpoint_path> [gpu_id]
#
# Example:
#   bash scripts/test_pcn_unseen.sh experiments/AdaPoinTr_PCN/ckpt-best.pth 0

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/test_pcn_unseen.sh <checkpoint_path> [gpu_id]"
    echo "Example: bash scripts/test_pcn_unseen.sh experiments/AdaPoinTr_PCN/ckpt-best.pth 0"
    exit 1
fi

CHECKPOINT=$1
GPU=${2:-0}

echo "=========================================="
echo "PCN Unseen Categories Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "GPU: $GPU"
echo ""
echo "Evaluating on 8 unseen ShapeNet categories:"
echo "  - bus (02924116)"
echo "  - bed (02818832)"
echo "  - bookshelf (02871439)"
echo "  - bench (02828884)"
echo "  - guitar (03467517)"
echo "  - motorbike (03790512)"
echo "  - skateboard (04225987)"
echo "  - pistol (03948459)"
echo "=========================================="

# Run evaluation
CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --test \
    --config cfgs/PCN_models/AdaPoinTr.yaml \
    --ckpts $CHECKPOINT \
    --exp_name AdaPoinTr_PCN_Unseen

echo ""
echo "Evaluation complete! Check experiments/AdaPoinTr_PCN_Unseen/ for results."
