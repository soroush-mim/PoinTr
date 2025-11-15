#!/bin/bash

# Test script for Benchmark 2: PCN Hold-out Class
# Evaluates AdaPoinTr on a held-out PCN class
#
# PCN categories and their taxonomy IDs:
#   - airplane (02691156)
#   - cabinet (02933112)
#   - car (02958343)
#   - chair (03001627)
#   - lamp (03636649)
#   - sofa (04256520)
#   - table (04379243)
#   - watercraft (04530566)
#
# Usage:
#   bash scripts/test_pcn_holdout.sh <checkpoint_path> <holdout_class> [gpu_id]
#
# Example:
#   bash scripts/test_pcn_holdout.sh experiments/AdaPoinTr_PCN_holdout_airplane/ckpt-best.pth 02691156 0

if [ $# -lt 2 ]; then
    echo "Usage: bash scripts/test_pcn_holdout.sh <checkpoint_path> <holdout_class> [gpu_id]"
    echo ""
    echo "PCN categories and their taxonomy IDs:"
    echo "  - airplane (02691156)"
    echo "  - cabinet (02933112)"
    echo "  - car (02958343)"
    echo "  - chair (03001627)"
    echo "  - lamp (03636649)"
    echo "  - sofa (04256520)"
    echo "  - table (04379243)"
    echo "  - watercraft (04530566)"
    echo ""
    echo "Example:"
    echo "  bash scripts/test_pcn_holdout.sh experiments/AdaPoinTr_PCN_holdout_airplane/ckpt-best.pth 02691156 0"
    exit 1
fi

CHECKPOINT=$1
HOLDOUT_CLASS=$2
GPU=${3:-0}

# Get category name from taxonomy ID
declare -A CATEGORY_NAMES=(
    ["02691156"]="airplane"
    ["02933112"]="cabinet"
    ["02958343"]="car"
    ["03001627"]="chair"
    ["03636649"]="lamp"
    ["04256520"]="sofa"
    ["04379243"]="table"
    ["04530566"]="watercraft"
)

CATEGORY_NAME=${CATEGORY_NAMES[$HOLDOUT_CLASS]:-"unknown"}

echo "=========================================="
echo "PCN Hold-out Class Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Held-out class: $CATEGORY_NAME ($HOLDOUT_CLASS)"
echo "GPU: $GPU"
echo "=========================================="

# Update the holdout class in the config files
sed -i "s/HOLDOUT_CLASS: '.*'/HOLDOUT_CLASS: '$HOLDOUT_CLASS'/" cfgs/dataset_configs/PCNHoldout_train.yaml
sed -i "s/HOLDOUT_CLASS: '.*'/HOLDOUT_CLASS: '$HOLDOUT_CLASS'/" cfgs/dataset_configs/PCNHoldout_eval.yaml

# Run evaluation
CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --test \
    --config cfgs/PCN_models/AdaPoinTr_holdout.yaml \
    --ckpts $CHECKPOINT \
    --exp_name AdaPoinTr_PCN_holdout_${CATEGORY_NAME}

echo ""
echo "Evaluation complete! Check experiments/AdaPoinTr_PCN_holdout_${CATEGORY_NAME}/ for results."
