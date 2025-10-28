"""
Demo script for ULIP-based refinement on PCN dataset.

Loads a random sample from PCN, runs AdaPoinTr completion, applies ULIP refinement,
and saves all intermediate results.

Usage:
    python scripts/demo_refinement.py \
        --adapointr_config cfgs/PCN_models/AdaPoinTr.yaml \
        --adapointr_ckpt checkpoints/adapointr_pcn.pth \
        --ulip_ckpt /path/to/ulip2_checkpoint.pt \
        --output_dir demo_results/ \
        --caption "a 3d point cloud of a chair"
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import random
import csv

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools import builder
from utils.config import cfg_from_yaml_file
from datasets import build_dataset_from_cfg
from refinement.ulip_loader import load_ulip_encoders
from refinement.ulip_refinement import ULIPRefinement


def save_point_cloud(points, filepath, format='pcd'):
    """
    Save point cloud to file.

    Args:
        points: (N, 3) numpy array
        filepath: Output file path
        format: 'txt', 'npy', 'ply', or 'pcd'
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == 'txt':
        np.savetxt(filepath, points, fmt='%.6f')
    elif format == 'npy':
        np.save(filepath, points)
    elif format == 'ply':
        # Simple PLY format
        with open(filepath, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(points)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('end_header\n')
            for p in points:
                f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
    elif format == 'pcd':
        # PCD format (Point Cloud Data)
        with open(filepath, 'w') as f:
            f.write('# .PCD v0.7 - Point Cloud Data file format\n')
            f.write('VERSION 0.7\n')
            f.write('FIELDS x y z\n')
            f.write('SIZE 4 4 4\n')
            f.write('TYPE F F F\n')
            f.write('COUNT 1 1 1\n')
            f.write(f'WIDTH {len(points)}\n')
            f.write('HEIGHT 1\n')
            f.write('VIEWPOINT 0 0 0 1 0 0 0\n')
            f.write(f'POINTS {len(points)}\n')
            f.write('DATA ascii\n')
            for p in points:
                f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Saved: {filepath}")


def load_adapointr_model(config_path, checkpoint_path, device='cuda'):
    """Load AdaPoinTr model."""
    print(f"\nLoading AdaPoinTr model...")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load config
    config = cfg_from_yaml_file(config_path)

    # Build model
    model = builder.model_builder(config.model)
    model.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'base_model' in checkpoint:
        state_dict = checkpoint['base_model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    print(f"✓ AdaPoinTr model loaded successfully")
    return model


def load_captions_from_csv(csv_path):
    """Load captions from CSV file."""
    captions = {}
    if csv_path and os.path.exists(csv_path):
        print(f"Loading captions from: {csv_path}")
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    # First column: {taxonomy_id}_{model_id}
                    # Second column: caption
                    instance_key = row[0].strip()
                    caption = row[1].strip()
                    captions[instance_key] = caption
        print(f"✓ Loaded {len(captions)} captions")
    return captions


def load_pcn_dataset(config_path, args, caption_csv_path=None):
    """Load PCN dataset using builder (same as eval script)."""
    print(f"\nLoading PCN dataset...")

    # Load config
    config = cfg_from_yaml_file(config_path)

    # Set caption path if provided
    if caption_csv_path:
        if hasattr(config.dataset, 'val') and hasattr(config.dataset.val, '_base_'):
            config.dataset.val._base_.RETURN_CAPTIONS = True
            config.dataset.val._base_.CAPTION_CSV_PATH = caption_csv_path
        # Try TEST config as well
        if hasattr(config.dataset, 'TEST') and hasattr(config.dataset.TEST, '_base_'):
            config.dataset.TEST._base_.RETURN_CAPTIONS = True
            config.dataset.TEST._base_.CAPTION_CSV_PATH = caption_csv_path
        # Try test config (lowercase)
        if hasattr(config.dataset, 'test'):
            config.dataset.test.RETURN_CAPTIONS = True
            config.dataset.test.CAPTION_CSV_PATH = caption_csv_path

    # Build dataset and dataloader using builder
    # Try val first, then test, then TEST
    dataset_config = None
    if hasattr(config.dataset, 'val'):
        dataset_config = config.dataset.val
    elif hasattr(config.dataset, 'test'):
        dataset_config = config.dataset.test
    elif hasattr(config.dataset, 'TEST'):
        dataset_config = config.dataset.TEST
    else:
        raise ValueError("No test/val dataset config found")

    # Build dataset and dataloader
    print(dataset_config)
    sampler, dataloader = builder.dataset_builder(args, dataset_config)

    # Get dataset from dataloader
    dataset = dataloader.dataset

    print(f"✓ Dataset loaded: {len(dataset)} samples")
    return dataset, dataloader


def main():
    parser = argparse.ArgumentParser(description='Demo ULIP refinement on PCN dataset')

    # Model paths
    parser.add_argument('--adapointr_config', type=str, required=True,
                        help='Path to AdaPoinTr config file')
    parser.add_argument('--adapointr_ckpt', type=str, required=True,
                        help='Path to AdaPoinTr checkpoint')
    parser.add_argument('--ulip_ckpt', type=str, required=True,
                        help='Path to ULIP-2 checkpoint')

    # Dataset
    parser.add_argument('--sample_idx', type=int, default=None,
                        help='Specific sample index (if None, choose random)')
    parser.add_argument('--caption_csv', type=str, default=None,
                        help='Path to CSV file with captions (format: taxonomy_id_model_id, caption)')

    # Refinement
    parser.add_argument('--caption', type=str, default=None,
                        help='Text caption for refinement (overrides caption from CSV if provided)')
    parser.add_argument('--refinement_steps', type=int, default=30,
                        help='Number of refinement steps')
    parser.add_argument('--refinement_lr', type=float, default=0.001,
                        help='Refinement learning rate')
    parser.add_argument('--lambda_text', type=float, default=1,
                        help='Text alignment loss weight')
    parser.add_argument('--lambda_stick', type=float, default=1.5,
                        help='Sticking loss weight')
    parser.add_argument('--lambda_smooth', type=float, default=0.01,
                        help='Smoothness loss weight')

    # Output
    parser.add_argument('--output_dir', type=str, default='demo_results',
                        help='Output directory')
    parser.add_argument('--save_format', type=str, default='pcd',
                        choices=['txt', 'npy', 'ply', 'pcd'],
                        help='Point cloud save format')
    parser.add_argument('--no_refinement', action='store_true',
                        help='Skip refinement step (only run AdaPoinTr)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    # Additional args needed for dataset builder
    parser.add_argument('--distributed', action='store_true', default=False,
                        help='Use distributed training')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (keep at 1 for demo)')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"ULIP Refinement Demo")
    print(f"{'='*80}")
    print(f"Device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # 1. Load dataset and captions
    # ========================================================================
    dataset, dataloader = load_pcn_dataset(args.adapointr_config, args, args.caption_csv)

    # Load captions if CSV provided (for fallback)
    captions_dict = {}
    if args.caption_csv:
        captions_dict = load_captions_from_csv(args.caption_csv)

    # Select random sample
    if args.sample_idx is None:
        sample_idx = random.randint(0, len(dataset) - 1)
    else:
        sample_idx = args.sample_idx

    print(f"\nSelected sample index: {sample_idx}")

    # Get data from dataset
    dataset_item = dataset[sample_idx]

    # Handle both with and without captions
    if len(dataset_item) == 4:
        taxonomy_id, model_id, data, dataset_caption = dataset_item
    else:
        taxonomy_id, model_id, data = dataset_item
        dataset_caption = None

    partial = data[0].to(device)  # (N_partial, 3)
    gt = data[1].to(device)       # (N_gt, 3)

    print(f"Taxonomy ID: {taxonomy_id}")
    print(f"Model ID: {model_id}")
    print(f"Partial shape: {partial.shape}")
    print(f"GT shape: {gt.shape}")

    # Get caption for this sample (priority: user > dataset > csv dict > default)
    instance_key = f"{taxonomy_id}_{model_id}"
    if args.caption:
        # User provided caption overrides everything
        caption = args.caption
        print(f"Using user-provided caption: \"{caption}\"")
    elif dataset_caption is not None:
        # Use caption from dataset (loaded from CSV via config)
        caption = dataset_caption
        print(f"Caption from dataset: \"{caption}\"")
    elif instance_key in captions_dict:
        # Use caption from CSV dict (fallback)
        caption = captions_dict[instance_key]
        print(f"Caption from CSV: \"{caption}\"")
    else:
        # Fallback to default
        caption = f"a 3d point cloud of an object"
        print(f"No caption found, using default: \"{caption}\"")

    # ========================================================================
    # 2. Run AdaPoinTr
    # ========================================================================
    model = load_adapointr_model(args.adapointr_config, args.adapointr_ckpt, device)

    print(f"\nRunning AdaPoinTr inference...")
    with torch.no_grad():
        # Add batch dimension
        partial_batch = partial.unsqueeze(0)  # (1, N, 3)

        # Forward pass
        ret = model(partial_batch)

        # Extract coarse and dense predictions
        if isinstance(ret, tuple) or isinstance(ret, list):
            coarse_output = ret[0]  # (1, M, 3)
            dense_output = ret[1]   # (1, N_out, 3)
        else:
            dense_output = ret
            coarse_output = None

        # Use dense output as final completion
        adapointr_output = dense_output.squeeze(0)  # (N_out, 3)

    print(f"✓ AdaPoinTr output shape: {adapointr_output.shape}")

    # ========================================================================
    # 3. Run ULIP Refinement (optional)
    # ========================================================================
    if not args.no_refinement:
        print(f"\nLoading ULIP encoders...")
        encoder_3d, encoder_text = load_ulip_encoders(
            checkpoint_path=args.ulip_ckpt,
            device=device,
            model_type='ULIP2_PointBERT'
        )

        print(f"\nInitializing refinement module...")
        refiner = ULIPRefinement(
            encoder_3d=encoder_3d,
            encoder_text=encoder_text,
            lambda_text=args.lambda_text,
            lambda_stick=args.lambda_stick,
            lambda_smooth=args.lambda_smooth,
            device=device
        )

        print(f"\nRunning refinement...")
        print(f"Caption: \"{caption}\"")
        print(f"Steps: {args.refinement_steps}")
        print(f"Learning rate: {args.refinement_lr}")

        # Add batch dimension for refinement
        adapointr_batch = adapointr_output.unsqueeze(0)  # (1, N, 3)

        refined_output = refiner.refine(
            P0=adapointr_batch,
            text_caption=caption,
            steps=args.refinement_steps,
            lr=args.refinement_lr,
            verbose=True
        )

        refined_output = refined_output.squeeze(0)  # (N, 3)
        print(f"\n✓ Refined output shape: {refined_output.shape}")

        # Compare text alignment
        print(f"\nComparing text alignment...")
        comparison = refiner.compare_before_after(
            adapointr_batch,
            refined_output.unsqueeze(0),
            caption
        )
        print(f"Before refinement: {comparison['before'].item():.4f}")
        print(f"After refinement:  {comparison['after'].item():.4f}")
        print(f"Improvement:       {comparison['improvement'].item():.4f}")
    else:
        refined_output = None

    # ========================================================================
    # 4. Save results
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"Saving results to: {output_dir}")
    print(f"{'='*80}")

    # Convert to numpy and move to CPU
    partial_np = partial.cpu().numpy()
    gt_np = gt.cpu().numpy()
    adapointr_np = adapointr_output.cpu().numpy()

    # Save point clouds
    save_point_cloud(partial_np, output_dir / f'partial.{args.save_format}', args.save_format)
    save_point_cloud(gt_np, output_dir / f'gt.{args.save_format}', args.save_format)
    save_point_cloud(adapointr_np, output_dir / f'adapointr_output.{args.save_format}', args.save_format)

    if refined_output is not None:
        refined_np = refined_output.cpu().numpy()
        save_point_cloud(refined_np, output_dir / f'refined_output.{args.save_format}', args.save_format)

    # Save metadata
    metadata = {
        'sample_idx': sample_idx,
        'taxonomy_id': taxonomy_id,
        'model_id': model_id,
        'instance_key': instance_key,
        'caption': caption,
        'partial_points': partial_np.shape[0],
        'gt_points': gt_np.shape[0],
        'adapointr_points': adapointr_np.shape[0],
        'refinement_steps': args.refinement_steps if not args.no_refinement else 0,
    }

    if refined_output is not None:
        metadata['refined_points'] = refined_np.shape[0]
        metadata['text_alignment_before'] = comparison['before'].item()
        metadata['text_alignment_after'] = comparison['after'].item()
        metadata['text_alignment_improvement'] = comparison['improvement'].item()

    # Save as text
    with open(output_dir / 'metadata.txt', 'w') as f:
        f.write('PCN Completion + ULIP Refinement Demo Results\n')
        f.write('='*80 + '\n\n')
        for key, value in metadata.items():
            f.write(f'{key}: {value}\n')

    print(f"\nSaved metadata: {output_dir / 'metadata.txt'}")

    print(f"\n{'='*80}")
    print(f"✓ Demo completed successfully!")
    print(f"{'='*80}")
    print(f"\nOutput files:")
    print(f"  - partial.{args.save_format}")
    print(f"  - gt.{args.save_format}")
    print(f"  - adapointr_output.{args.save_format}")
    if refined_output is not None:
        print(f"  - refined_output.{args.save_format}")
    print(f"  - metadata.txt")
    print(f"\nVisualize with:")
    if args.save_format == 'pcd':
        print(f"  # Using Open3D:")
        print(f"  python -c \"import open3d as o3d; pc = o3d.io.read_point_cloud('{output_dir}/partial.pcd'); o3d.visualization.draw_geometries([pc])\"")
        print(f"  # Or use CloudCompare, MeshLab, etc.")
    else:
        print(f"  python -m open3d.visualization.draw {output_dir}/partial.{args.save_format}")


if __name__ == '__main__':
    main()
