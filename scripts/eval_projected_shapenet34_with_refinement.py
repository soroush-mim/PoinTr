"""
Evaluation script for Projected-ShapeNet-34 dataset with ULIP-based test-time refinement.

This script reproduces Table 4 from the AdaPoinTr paper on the Projected-ShapeNet-34 dataset.
It evaluates both 34 seen categories and 21 unseen categories, with and without ULIP-2 refinement.

Usage:
    python scripts/eval_projected_shapenet34_with_refinement.py \
        --config cfgs/Projected_ShapeNet34_models/AdaPoinTr.yaml \
        --checkpoint path/to/adapointr_checkpoint.pth \
        --ulip_checkpoint path/to/ulip2_checkpoint.pth \
        --output_dir results/projected_shapenet34_refinement
"""

import argparse
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from collections import defaultdict

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


# ShapeNet-34 category taxonomy IDs (34 seen categories)
SHAPENET_34_SEEN_CATEGORIES = [
    '02691156',  # airplane
    '02933112',  # cabinet
    '03001627',  # chair
    '03636649',  # lamp
    '04090263',  # rifle
    '04379243',  # table
    '02958343',  # car
    '03211117',  # display
    '04401088',  # telephone
    '02828884',  # bench
    '03691459',  # loudspeaker
    '04530566',  # watercraft
    '02871439',  # bookshelf
    '03337140',  # filecabinet
    '04554684',  # washer
    '02773838',  # bag
    '02924116',  # bus
    '02808440',  # bathtub
    '02818832',  # bed
    '02843684',  # birdhouse
    '02876657',  # bottle
    '02880940',  # bowl
    '02954340',  # cap
    '02992529',  # cellphone
    '03046257',  # clock
    '03207941',  # dishwasher
    '03261776',  # earphone
    '03513137',  # helmet
    '03624134',  # knife
    '03642806',  # laptop
    '03797390',  # mug
    '03928116',  # piano
    '03948459',  # pillow
    '04099429',  # rocket
]

# ShapeNet-Unseen21 category taxonomy IDs (21 unseen categories)
SHAPENET_UNSEEN21_CATEGORIES = [
    '02801938',  # basket
    '02747177',  # trashbin
    '02942699',  # camera
    '02946921',  # can
    '03085013',  # keyboard
    '03467517',  # guitar
    '03790512',  # motorbike
    '03948459',  # pistol
    '03991062',  # flowerpot
    '04004475',  # printer
    '04074963',  # remote
    '04225987',  # skateboard
    '04460130',  # tower
    '03325088',  # faucet
    '03761084',  # microwave
    '03938244',  # pillow
    '04256520',  # sofa
    '04330267',  # stove
    '03710193',  # mailbox
    '03759954',  # microphone
    '04468005',  # train
]


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate AdaPoinTr with ULIP refinement on Projected-ShapeNet-34')

    # Model arguments
    parser.add_argument('--config', type=str, default='cfgs/Projected_ShapeNet34_models/AdaPoinTr.yaml',
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

    # Evaluation mode
    parser.add_argument('--eval_seen', action='store_true', default=True,
                        help='Evaluate on 34 seen categories')
    parser.add_argument('--eval_unseen', action='store_true', default=True,
                        help='Evaluate on 21 unseen categories')

    # Refinement arguments (same as PCN eval)
    parser.add_argument('--no_refinement', action='store_true',
                        help='Skip refinement and only evaluate baseline')
    parser.add_argument('--refinement_only', action='store_true',
                        help='Only evaluate with refinement (skip baseline)')
    parser.add_argument('--steps', type=int, default=15,
                        help='Number of refinement steps')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Refinement learning rate')
    parser.add_argument('--lambda_text', type=float, default=0.5,
                        help='Text alignment loss weight')
    parser.add_argument('--lambda_stick', type=float, default=2.0,
                        help='Sticking loss (CD) weight')
    parser.add_argument('--lambda_smooth', type=float, default=0.1,
                        help='Smoothness loss weight')
    parser.add_argument('--k_neighbors', type=int, default=8,
                        help='Number of neighbors for smoothness')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results/projected_shapenet34_refinement',
                        help='Output directory for results')
    parser.add_argument('--save_outputs', action='store_true',
                        help='Save point clouds to disk')
    parser.add_argument('--verbose', action='store_true',
                        help='Print refinement progress')

    # Device (same as PCN eval)
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()
    return args


def load_captions(caption_csv_path):
    """Load captions from CSV file."""
    if caption_csv_path is None or not os.path.exists(caption_csv_path):
        return None

    captions_dict = {}
    with open(caption_csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                key, caption = parts
                captions_dict[key] = caption
    return captions_dict


def evaluate_baseline(model, test_dataloader, device, category_filter=None, logger=None):
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

            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else str(taxonomy_ids[0])
            model_id = model_ids[0]

            # Filter by category if specified
            if category_filter is not None and taxonomy_id not in category_filter:
                continue

            partial = data[0].to(device)
            gt = data[1].to(device)

            # Run AdaPoinTr
            ret = model(partial)
            dense_points = ret[-1]
            coarse_points = ret[0]

            # Compute metrics (CD-L1 and F-Score only)
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
                         f'CD-L1={_metrics[0]*1000:.2f} F-Score={_metrics[2]:.4f}', logger=logger)

    # Compute overall metrics
    for _, v in category_metrics.items():
        test_metrics.update(v.avg())

    print_log(f'[BASELINE] Overall Metrics: CD-L1={test_metrics.avg()[0]*1000:.2f}, F-Score={test_metrics.avg()[2]:.4f}',
              logger=logger)

    return test_metrics, category_metrics, all_results


def evaluate_with_refinement(model, refiner, test_dataloader, refinement_config,
                             device, category_filter=None, captions_dict=None, logger=None):
    """Evaluate with ULIP-based test-time refinement."""
    print_log('Evaluating with ULIP refinement...', logger=logger)

    model.eval()
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    all_results = []

    # Metrics before and after refinement
    improvement_metrics = AverageMeter(['CD-L1_improve', 'F-Score_improve'])

    for idx, batch in enumerate(tqdm(test_dataloader, desc='Refinement Eval')):
        # Handle both with and without captions
        if len(batch) == 4:
            taxonomy_ids, model_ids, data, captions = batch
        else:
            taxonomy_ids, model_ids, data = batch
            captions = None

        taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else str(taxonomy_ids[0])
        model_id = model_ids[0]

        # Filter by category if specified
        if category_filter is not None and taxonomy_id not in category_filter:
            continue

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
        elif captions_dict is not None:
            key = f"{taxonomy_id}_{model_id}"
            caption = captions_dict.get(key, "a 3d point cloud")
        else:
            caption = "a 3d point cloud"  # fallback

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
            (metrics_before[0] - metrics_after[0]) * 1000,  # CD-L1 improvement (×1000)
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
                     f'  Before: CD-L1={metrics_before[0]*1000:.2f}, F-Score={metrics_before[2]:.4f}\n'
                     f'  After:  CD-L1={metrics_after[0]*1000:.2f}, F-Score={metrics_after[2]:.4f}\n'
                     f'  Improve: CD-L1={improvement[0]:.2f}, F-Score={improvement[1]:.4f}',
                     logger=logger)

    # Compute overall metrics
    for _, v in category_metrics.items():
        test_metrics.update(v.avg())

    print_log(f'[REFINED] Overall Metrics: CD-L1={test_metrics.avg()[0]*1000:.2f}, F-Score={test_metrics.avg()[2]:.4f}',
              logger=logger)
    print_log(f'[IMPROVEMENT] Avg Improvement: CD-L1={improvement_metrics.avg()[0]:.2f}, F-Score={improvement_metrics.avg()[1]:.4f}',
              logger=logger)

    return test_metrics, category_metrics, all_results, improvement_metrics


def print_table_4_format(baseline_metrics, refined_metrics, category_list, split_name, logger=None):
    """Print results in Table 4 format from AdaPoinTr paper."""
    print_log(f'\n{"="*80}', logger=logger)
    print_log(f'Table 4 Format Results - {split_name}', logger=logger)
    print_log(f'{"="*80}', logger=logger)

    # Print header
    header = f"{'Category':<20} {'Baseline CD-L1':<15} {'Baseline F-Score':<18} {'Refined CD-L1':<15} {'Refined F-Score':<18}"
    print_log(header, logger=logger)
    print_log('-' * 86, logger=logger)

    # Print per-category results
    for cat_id in sorted(category_list):
        if cat_id in baseline_metrics:
            baseline_cd = baseline_metrics[cat_id].avg()[0] * 1000
            baseline_fscore = baseline_metrics[cat_id].avg()[2]
        else:
            baseline_cd = 0.0
            baseline_fscore = 0.0

        if refined_metrics and cat_id in refined_metrics:
            refined_cd = refined_metrics[cat_id].avg()[0] * 1000
            refined_fscore = refined_metrics[cat_id].avg()[2]
        else:
            refined_cd = 0.0
            refined_fscore = 0.0

        row = f"{cat_id:<20} {baseline_cd:<15.2f} {baseline_fscore:<18.4f} {refined_cd:<15.2f} {refined_fscore:<18.4f}"
        print_log(row, logger=logger)

    print_log(f'{"="*80}\n', logger=logger)


def generate_comparison_table(baseline_results, refined_results, split_name, logger=None):
    """Generate comparison table for baseline vs ULIP-2 refinement."""
    print_log(f'\n{"="*100}', logger=logger)
    print_log(f'Comparison Table - {split_name}', logger=logger)
    print_log(f'{"="*100}', logger=logger)

    # Compute overall metrics
    baseline_cd = baseline_results['overall'][0] * 1000 if baseline_results else 0.0
    baseline_fscore = baseline_results['overall'][2] if baseline_results else 0.0

    refined_cd = refined_results['overall'][0] * 1000 if refined_results else 0.0
    refined_fscore = refined_results['overall'][2] if refined_results else 0.0

    improvement_cd = baseline_cd - refined_cd
    improvement_fscore = refined_fscore - baseline_fscore

    # Print table
    print_log(f"{'Method':<30} {'CD-L1 (×1000)':<20} {'F-Score@1%':<20}", logger=logger)
    print_log('-' * 70, logger=logger)
    print_log(f"{'Baseline (AdaPoinTr)':<30} {baseline_cd:<20.2f} {baseline_fscore:<20.4f}", logger=logger)
    print_log(f"{'With ULIP-2 Refinement':<30} {refined_cd:<20.2f} {refined_fscore:<20.4f}", logger=logger)
    print_log('-' * 70, logger=logger)
    print_log(f"{'Improvement':<30} {improvement_cd:<20.2f} {improvement_fscore:<20.4f}", logger=logger)
    print_log(f'{"="*100}\n', logger=logger)


def save_results(output_dir, baseline_results_seen=None, refined_results_seen=None,
                baseline_results_unseen=None, refined_results_unseen=None, args=None):
    """Save evaluation results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'args': vars(args) if args is not None else {},
        'seen_categories': {
            'baseline': baseline_results_seen,
            'refined': refined_results_seen,
        },
        'unseen_categories': {
            'baseline': baseline_results_unseen,
            'refined': refined_results_unseen,
        }
    }

    output_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f'Results saved to {output_file}')


def main():
    args = get_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logger
    logger = get_logger('Projected_ShapeNet34_Refinement_Eval')
    print_log('=' * 80, logger=logger)
    print_log('Projected-ShapeNet-34 Evaluation with ULIP-based Test-Time Refinement', logger=logger)
    print_log('Reproducing Table 4 from AdaPoinTr Paper', logger=logger)
    print_log('=' * 80, logger=logger)

    # Load config
    print_log(f'Loading config from {args.config}', logger=logger)
    config = cfg_from_yaml_file(args.config)
    dataset_name = config.dataset.test._base_.NAME
    print('dataset: ', dataset_name)

    # Override data root if provided
    if args.data_root is not None:
        config.dataset.val._base_.DATA_PATH = args.data_root
        print_log(f'Overriding data root to {args.data_root}', logger=logger)

    # Load captions if provided
    captions_dict = load_captions(args.caption_csv)
    if captions_dict:
        print_log(f'Loaded {len(captions_dict)} captions from {args.caption_csv}', logger=logger)
    else:
        print_log('No captions provided, using generic descriptions', logger=logger)

    # Build AdaPoinTr model
    print_log(f'Loading AdaPoinTr model from {args.checkpoint}', logger=logger)
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.checkpoint)
    base_model.to(args.device)
    base_model.eval()

    # Initialize results storage
    all_results = {
        'seen': {'baseline': None, 'refined': None},
        'unseen': {'baseline': None, 'refined': None}
    }

    # Evaluate on 34 seen categories
    if args.eval_seen:
        print_log('\n' + '='*80, logger=logger)
        print_log('Evaluating on 34 Seen Categories', logger=logger)
        print_log('='*80, logger=logger)

        # Build dataset (test split contains seen categories)
        print_log('Building dataset for seen categories...', logger=logger)
        test_dataloader = builder.dataset_builder(args, config.dataset.val)[1]
        print_log(f'Dataset size: {len(test_dataloader)}', logger=logger)

        # Evaluate baseline
        baseline_metrics_seen = None
        baseline_category_metrics_seen = None
        baseline_results_seen = None

        if not args.refinement_only:
            baseline_metrics_seen, baseline_category_metrics_seen, baseline_results_seen = evaluate_baseline(
                base_model, test_dataloader, args.device,
                category_filter=SHAPENET_34_SEEN_CATEGORIES, logger=logger
            )
            all_results['seen']['baseline'] = {
                'overall': baseline_metrics_seen.avg(),
                'per_category': {k: v.avg() for k, v in baseline_category_metrics_seen.items()}
            }

        # Evaluate with refinement
        refined_metrics_seen = None
        refined_category_metrics_seen = None
        refined_results_seen = None
        improvement_metrics_seen = None

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

            refined_metrics_seen, refined_category_metrics_seen, refined_results_seen, improvement_metrics_seen = \
                evaluate_with_refinement(
                    base_model, refiner, test_dataloader, refinement_config,
                    args.device, category_filter=SHAPENET_34_SEEN_CATEGORIES,
                    captions_dict=captions_dict, logger=logger
                )

            all_results['seen']['refined'] = {
                'overall': refined_metrics_seen.avg(),
                'per_category': {k: v.avg() for k, v in refined_category_metrics_seen.items()}
            }

        # Print results in Table 4 format
        if baseline_category_metrics_seen or refined_category_metrics_seen:
            print_table_4_format(baseline_category_metrics_seen, refined_category_metrics_seen,
                               SHAPENET_34_SEEN_CATEGORIES, '34 Seen Categories', logger=logger)

        # Generate comparison table
        if baseline_metrics_seen or refined_metrics_seen:
            generate_comparison_table(all_results['seen']['baseline'], all_results['seen']['refined'],
                                    '34 Seen Categories', logger=logger)

    # Evaluate on 21 unseen categories
    if args.eval_unseen:
        print_log('\n' + '='*80, logger=logger)
        print_log('Evaluating on 21 Unseen Categories', logger=logger)
        print_log('='*80, logger=logger)

        # Need to load unseen category dataset
        # Modify config to use unseen category test file
        unseen_config = config.copy()
        unseen_config.dataset.val._base_.DATA_PATH = unseen_config.dataset.val._base_.DATA_PATH.replace(
            'Projected_ShapeNet-34_noise', 'Projected_ShapeNet-Unseen21_noise'
        )

        print_log('Building dataset for unseen categories...', logger=logger)
        test_dataloader_unseen = builder.dataset_builder(args, unseen_config.dataset.val)[1]
        print_log(f'Dataset size: {len(test_dataloader_unseen)}', logger=logger)

        # Evaluate baseline
        baseline_metrics_unseen = None
        baseline_category_metrics_unseen = None
        baseline_results_unseen = None

        if not args.refinement_only:
            baseline_metrics_unseen, baseline_category_metrics_unseen, baseline_results_unseen = evaluate_baseline(
                base_model, test_dataloader_unseen, args.device,
                category_filter=SHAPENET_UNSEEN21_CATEGORIES, logger=logger
            )
            all_results['unseen']['baseline'] = {
                'overall': baseline_metrics_unseen.avg(),
                'per_category': {k: v.avg() for k, v in baseline_category_metrics_unseen.items()}
            }

        # Evaluate with refinement
        refined_metrics_unseen = None
        refined_category_metrics_unseen = None
        refined_results_unseen = None
        improvement_metrics_unseen = None

        if not args.no_refinement:
            # Reuse or reload ULIP-2 encoders
            if 'refiner' not in locals():
                print_log(f'Loading ULIP-2 encoders from {args.ulip_checkpoint}', logger=logger)
                encoder_3d, encoder_text = load_ulip_encoders(args.ulip_checkpoint, args.device, model_cache_dir=args.model_cache_dir)

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

            refined_metrics_unseen, refined_category_metrics_unseen, refined_results_unseen, improvement_metrics_unseen = \
                evaluate_with_refinement(
                    base_model, refiner, test_dataloader_unseen, refinement_config,
                    args.device, category_filter=SHAPENET_UNSEEN21_CATEGORIES,
                    captions_dict=captions_dict, logger=logger
                )

            all_results['unseen']['refined'] = {
                'overall': refined_metrics_unseen.avg(),
                'per_category': {k: v.avg() for k, v in refined_category_metrics_unseen.items()}
            }

        # Print results in Table 4 format
        if baseline_category_metrics_unseen or refined_category_metrics_unseen:
            print_table_4_format(baseline_category_metrics_unseen, refined_category_metrics_unseen,
                               SHAPENET_UNSEEN21_CATEGORIES, '21 Unseen Categories', logger=logger)

        # Generate comparison table
        if baseline_metrics_unseen or refined_metrics_unseen:
            generate_comparison_table(all_results['unseen']['baseline'], all_results['unseen']['refined'],
                                    '21 Unseen Categories', logger=logger)

    # Print final summary
    print_log('\n' + '='*80, logger=logger)
    print_log('FINAL SUMMARY - Table 4 Reproduction', logger=logger)
    print_log('='*80, logger=logger)

    if args.eval_seen and all_results['seen']['baseline']:
        print_log('\n34 Seen Categories:', logger=logger)
        baseline_cd = all_results['seen']['baseline']['overall'][0] * 1000
        baseline_fs = all_results['seen']['baseline']['overall'][2]
        print_log(f"  Baseline:    CD-L1={baseline_cd:.2f}, F-Score={baseline_fs:.4f}", logger=logger)

        if all_results['seen']['refined']:
            refined_cd = all_results['seen']['refined']['overall'][0] * 1000
            refined_fs = all_results['seen']['refined']['overall'][2]
            print_log(f"  Refined:     CD-L1={refined_cd:.2f}, F-Score={refined_fs:.4f}", logger=logger)
            print_log(f"  Improvement: CD-L1={baseline_cd - refined_cd:.2f}, F-Score={refined_fs - baseline_fs:.4f}",
                     logger=logger)

    if args.eval_unseen and all_results['unseen']['baseline']:
        print_log('\n21 Unseen Categories:', logger=logger)
        baseline_cd = all_results['unseen']['baseline']['overall'][0] * 1000
        baseline_fs = all_results['unseen']['baseline']['overall'][2]
        print_log(f"  Baseline:    CD-L1={baseline_cd:.2f}, F-Score={baseline_fs:.4f}", logger=logger)

        if all_results['unseen']['refined']:
            refined_cd = all_results['unseen']['refined']['overall'][0] * 1000
            refined_fs = all_results['unseen']['refined']['overall'][2]
            print_log(f"  Refined:     CD-L1={refined_cd:.2f}, F-Score={refined_fs:.4f}", logger=logger)
            print_log(f"  Improvement: CD-L1={baseline_cd - refined_cd:.2f}, F-Score={refined_fs - baseline_fs:.4f}",
                     logger=logger)

    print_log('='*80, logger=logger)

    # Save results
    save_results(
        args.output_dir,
        baseline_results_seen=all_results['seen']['baseline'],
        refined_results_seen=all_results['seen']['refined'],
        baseline_results_unseen=all_results['unseen']['baseline'],
        refined_results_unseen=all_results['unseen']['refined'],
        args=args
    )

    print_log(f'\nEvaluation complete. Results saved to {args.output_dir}', logger=logger)


if __name__ == '__main__':
    main()
