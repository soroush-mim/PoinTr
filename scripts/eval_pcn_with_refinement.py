"""
Evaluation script for PCN dataset with ULIP-based test-time refinement.

This script evaluates AdaPoinTr on the PCN dataset with optional ULIP refinement.
It computes metrics both before and after refinement to measure the improvement.

Usage:
    python scripts/eval_pcn_with_refinement.py \
        --config path/to/adapointr_config.yaml \
        --checkpoint path/to/adapointr_checkpoint.pth \
        --ulip_checkpoint path/to/ulip2_checkpoint.pth \
        --caption_csv path/to/captions.csv \
        --output_dir results/
"""

import argparse
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from tools import builder
from utils.config import cfg_from_yaml_file
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

# Import refinement modules
from refinement import load_ulip_encoders, ULIPRefinement


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate AdaPoinTr with ULIP refinement on PCN')

    # Model arguments
    parser.add_argument('--config', type=str, required=True,
                        help='Path to AdaPoinTr config YAML file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to AdaPoinTr checkpoint')
    parser.add_argument('--ulip_checkpoint', type=str, required=True,
                        help='Path to ULIP-2 checkpoint')
    parser.add_argument('--model_cache_dir', type=str,
                        default='/home/soroushm/data')

    # Dataset arguments
    parser.add_argument('--caption_csv', type=str, default=None,
                        help='Path to caption CSV file (format: {taxonomy_id}_{model_id}, caption)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override data root in config')

    # Refinement arguments
    parser.add_argument('--no_refinement', action='store_true',
                        help='Skip refinement and only evaluate baseline')
    parser.add_argument('--refinement_only', action='store_true',
                        help='Only evaluate with refinement (skip baseline)')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of refinement steps')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Refinement learning rate')
    parser.add_argument('--lambda_text', type=float, default=1.0,
                        help='Text alignment loss weight')
    parser.add_argument('--lambda_stick', type=float, default=1.5,
                        help='Sticking loss (CD) weight')
    parser.add_argument('--lambda_smooth', type=float, default=0.01,
                        help='Smoothness loss weight')
    parser.add_argument('--k_neighbors', type=int, default=8,
                        help='Number of neighbors for smoothness')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results/pcn_refinement',
                        help='Output directory for results')
    parser.add_argument('--save_outputs', action='store_true',
                        help='Save point clouds to disk')
    parser.add_argument('--verbose', action='store_true',
                        help='Print refinement progress')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()
    return args


def evaluate_baseline(model, test_dataloader, metrics_fn, device, logger=None):
    """Evaluate baseline AdaPoinTr without refinement."""
    print_log('Evaluating baseline (without refinement)...', logger=logger)

    model.eval()
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()

    all_results = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dataloader, desc='Baseline Eval')):
            # Handle both with and without captions
            if len(batch) == 4:
                taxonomy_ids, model_ids, data, captions = batch
            else:
                taxonomy_ids, model_ids, data = batch
                captions = None

            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            partial = data[0].to(device)
            gt = data[1].to(device)

            # Run AdaPoinTr
            ret = model(partial)
            dense_points = ret[-1]

            # Compute metrics
            _metrics = Metrics.get(dense_points, gt, require_emd=True)

            # Update category metrics
            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            # Store results
            all_results.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'metrics': _metrics,
                'caption': captions[0] if captions is not None else None
            })

            if (idx + 1) % 100 == 0:
                print_log(f'Baseline [{idx+1}/{len(test_dataloader)}] '
                         f'Taxonomy={taxonomy_id} Sample={model_id} '
                         f'Metrics={["%.4f" % m for m in _metrics]}', logger=logger)

    # Compute overall metrics
    for _, v in category_metrics.items():
        test_metrics.update(v.avg())

    print_log(f'[BASELINE] Overall Metrics = {["%.4f" % m for m in test_metrics.avg()]}', logger=logger)

    return test_metrics, category_metrics, all_results


def evaluate_with_refinement(model, refiner, test_dataloader, metrics_fn,
                             refinement_config, device, logger=None):
    """Evaluate with ULIP-based test-time refinement."""
    print_log('Evaluating with ULIP refinement...', logger=logger)

    model.eval()
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()

    all_results = []

    # Metrics before and after refinement
    improvement_metrics = AverageMeter(['CD-L1_improve', 'CD-L2_improve', 'F-Score_improve'])

    for idx, batch in enumerate(tqdm(test_dataloader, desc='Refinement Eval')):
        # Handle both with and without captions
        if len(batch) == 4:
            taxonomy_ids, model_ids, data, captions = batch
        else:
            taxonomy_ids, model_ids, data = batch
            captions = None

        taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
        model_id = model_ids[0]

        partial = data[0].to(device)
        gt = data[1].to(device)

        # Run AdaPoinTr to get initial completion
        with torch.no_grad():
            ret = model(partial)
            dense_points_initial = ret[-1]

        # Metrics before refinement
        metrics_before = Metrics.get(dense_points_initial, gt, require_emd=True)

        # Get caption
        if captions is not None:
            caption = captions[0]
        else:
            caption = f"a 3d point cloud"  # fallback

        # Refine point cloud
        dense_points_refined = refiner.refine(
            dense_points_initial,
            caption,
            steps=refinement_config['steps'],
            lr=refinement_config['lr'],
            verbose=refinement_config['verbose']
        )

        # Metrics after refinement
        with torch.no_grad():
            metrics_after = Metrics.get(dense_points_refined, gt, require_emd=True)

        # Compute improvement
        improvement = [
            (metrics_before[0] - metrics_after[0]) * 1000,  # CD-L1 improvement
            (metrics_before[1] - metrics_after[1]) * 1000,  # CD-L2 improvement
            (metrics_after[2] - metrics_before[2])  # F-Score improvement (higher is better)
        ]
        improvement_metrics.update(improvement)

        # Update category metrics (with refined results)
        if taxonomy_id not in category_metrics:
            category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
        category_metrics[taxonomy_id].update(metrics_after)

        # Store results
        all_results.append({
            'taxonomy_id': taxonomy_id,
            'model_id': model_id,
            'metrics_before': metrics_before,
            'metrics_after': metrics_after,
            'improvement': improvement,
            'caption': caption
        })

        if (idx + 1) % 100 == 0:
            print_log(f'Refinement [{idx+1}/{len(test_dataloader)}] '
                     f'Taxonomy={taxonomy_id} Sample={model_id}\n'
                     f'  Before: {["%.4f" % m for m in metrics_before]}\n'
                     f'  After:  {["%.4f" % m for m in metrics_after]}\n'
                     f'  Improve: {["%.4f" % i for i in improvement]}',
                     logger=logger)

    # Compute overall metrics
    for _, v in category_metrics.items():
        test_metrics.update(v.avg())

    print_log(f'[REFINED] Overall Metrics = {["%.4f" % m for m in test_metrics.avg()]}', logger=logger)
    print_log(f'[IMPROVEMENT] Avg Improvement = {["%.4f" % m for m in improvement_metrics.avg()]}', logger=logger)

    return test_metrics, category_metrics, all_results, improvement_metrics


def save_results(output_dir, baseline_results=None, refined_results=None,
                improvement_metrics=None, args=None):
    """Save evaluation results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'args': vars(args) if args is not None else {},
        'baseline': baseline_results,
        'refined': refined_results,
        'improvement': improvement_metrics
    }

    output_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(output_file, 'w') as f:
        # json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        json.dump(results, f, indent=2, default=str)


    print(f'Results saved to {output_file}')


def main():
    args = get_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logger
    logger = get_logger('PCN_Refinement_Eval')
    print_log('=' * 80, logger=logger)
    print_log('PCN Evaluation with ULIP-based Test-Time Refinement', logger=logger)
    print_log('=' * 80, logger=logger)

    # Load config
    print_log(f'Loading config from {args.config}', logger=logger)
    config = cfg_from_yaml_file(args.config)

    # Override caption path if provided
    if args.caption_csv is not None:
        config.dataset.val._base_.RETURN_CAPTIONS = True
        config.dataset.val._base_.CAPTION_CSV_PATH = args.caption_csv
        print_log(f'Using captions from {args.caption_csv}', logger=logger)

    # Build dataset
    print_log('Building dataset...', logger=logger)
    test_dataloader = builder.dataset_builder(args, config.dataset.val)[1]
    print_log(f'Dataset size: {len(test_dataloader)}', logger=logger)

    # Build AdaPoinTr model
    print_log(f'Loading AdaPoinTr model from {args.checkpoint}', logger=logger)
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.checkpoint)
    base_model.to(args.device)
    base_model.eval()

    # Initialize metrics
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    # Evaluate baseline
    baseline_metrics = None
    baseline_category_metrics = None
    baseline_results = None

    if not args.refinement_only:
        baseline_metrics, baseline_category_metrics, baseline_results = evaluate_baseline(
            base_model, test_dataloader, Metrics, args.device, logger
        )

    # Evaluate with refinement
    refined_metrics = None
    refined_category_metrics = None
    refined_results = None
    improvement_metrics = None

    if not args.no_refinement:
        # Load ULIP-2 encoders
        print_log(f'Loading ULIP-2 encoders from {args.ulip_checkpoint}', logger=logger)
        encoder_3d, encoder_text = load_ulip_encoders(args.ulip_checkpoint, args.device, model_cache_dir=args.model_cache_dir)

        # Initialize refinement module
        print_log('Initializing ULIP refinement module...', logger=logger)
        refiner = ULIPRefinement(
            encoder_3d,
            encoder_text,
            lambda_text=args.lambda_text,
            lambda_stick=args.lambda_stick,
            lambda_smooth=args.lambda_smooth,
            k_neighbors=args.k_neighbors,
            device=args.device
        )

        refinement_config = {
            'steps': args.steps,
            'lr': args.lr,
            'verbose': args.verbose
        }

        print_log(f'Refinement config: {refinement_config}', logger=logger)
        print_log(f'Loss weights: text={args.lambda_text}, stick={args.lambda_stick}, smooth={args.lambda_smooth}',
                 logger=logger)

        refined_metrics, refined_category_metrics, refined_results, improvement_metrics = evaluate_with_refinement(
            base_model, refiner, test_dataloader, Metrics,
            refinement_config, args.device, logger
        )

    # Print summary
    print_log('=' * 80, logger=logger)
    print_log('EVALUATION SUMMARY', logger=logger)
    print_log('=' * 80, logger=logger)

    if baseline_metrics is not None:
        print_log(f'Baseline Metrics: {["%.4f" % m for m in baseline_metrics.avg()]}', logger=logger)

    if refined_metrics is not None:
        print_log(f'Refined Metrics:  {["%.4f" % m for m in refined_metrics.avg()]}', logger=logger)

    if improvement_metrics is not None:
        print_log(f'Improvement:      {["%.4f" % m for m in improvement_metrics.avg()]}', logger=logger)

    print_log('=' * 80, logger=logger)

    # Save results
    save_results(
        args.output_dir,
        baseline_results=baseline_results,
        refined_results=refined_results,
        improvement_metrics=(sum(improvement_metrics) / len(improvement_metrics)) if improvement_metrics is not None else None,
        args=args
    )

    print_log(f'Evaluation complete. Results saved to {args.output_dir}', logger=logger)


if __name__ == '__main__':
    main()
