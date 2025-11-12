"""
Evaluation script for text-conditioned AdaPoinTr on seen and unseen categories.

This script evaluates a trained text-conditioned model on both:
1. Seen categories (Projected_ShapeNet-34)
2. Unseen categories (Projected_ShapeNet-Unseen21)

Usage:
    python evaluate_text_seen_unseen.py \\
        --config cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml \\
        --ckpts /path/to/checkpoint.pth \\
        --exp_name eval_seen_unseen

The script will:
- Load the specified checkpoint
- Evaluate on seen categories with text captions
- Evaluate on unseen categories with text captions
- Print per-category and overall metrics for both
- Save results to a JSON file
"""

import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from utils.config import *


def evaluate_split(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=None, split_name=''):
    """
    Evaluate model on a single data split (seen or unseen).

    Args:
        base_model: Model to evaluate
        test_dataloader: DataLoader for test data
        ChamferDisL1: Chamfer Distance L1 criterion
        ChamferDisL2: Chamfer Distance L2 criterion
        args: Arguments
        config: Configuration
        logger: Logger instance
        split_name: Name of the split ('Seen' or 'Unseen')

    Returns:
        dict: Dictionary containing overall metrics and per-category metrics
    """
    print_log(f'=' * 80, logger=logger)
    print_log(f'[{split_name}] Starting evaluation on {split_name} categories', logger=logger)
    print_log(f'=' * 80, logger=logger)

    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader)  # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME

            # Handle captions if present
            captions = None
            if dataset_name == 'PCN' or dataset_name == 'Projected_ShapeNet':
                if len(data) == 3:
                    partial = data[0].cuda()
                    gt = data[1].cuda()
                    captions = data[2]  # List of strings
                else:
                    partial = data[0].cuda()
                    gt = data[1].cuda()

                ret = base_model(partial, captions=captions)
                coarse_points = ret[0]
                dense_points = ret[-1]

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
                ret = base_model(partial)
                coarse_points = ret[0]
                dense_points = ret[-1]
            else:
                raise NotImplementedError(f'Test phase does not support {dataset_name}')

            sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
            sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
            dense_loss_l1 =  ChamferDisL1(dense_points, gt)
            dense_loss_l2 =  ChamferDisL2(dense_points, gt)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000,
                               dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

            _metrics = Metrics.get(dense_points, gt)
            if args.distributed:
                _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
            else:
                _metrics = [_metric.item() for _metric in _metrics]

            for _taxonomy_id in taxonomy_ids:
                if _taxonomy_id not in category_metrics:
                    category_metrics[_taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[_taxonomy_id].update(_metrics)

            test_metrics.update(_metrics)

            if (idx + 1) % 100 == 0:
                print_log(f'[{split_name}] Test [{idx + 1}/{n_samples}] Taxonomy = {taxonomy_id} Sample = {model_id}',
                         logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Print detailed results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log(f'=' * 80, logger=logger)
    print_log(f'[{split_name}] DETAILED RESULTS', logger=logger)
    print_log(f'=' * 80, logger=logger)

    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    # Per-category results
    results = {
        'split': split_name,
        'overall': {},
        'per_category': {}
    }

    for taxonomy_id in sorted(category_metrics.keys()):
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        category_values = {}
        for i, metric_name in enumerate(test_metrics.items):
            value = category_metrics[taxonomy_id].avg()[i]
            msg += '%.3f \t' % value
            category_values[metric_name] = float(value)
        msg += shapenet_dict.get(taxonomy_id, 'Unknown') + '\t'
        print_log(msg, logger=logger)

        results['per_category'][taxonomy_id] = {
            'name': shapenet_dict.get(taxonomy_id, 'Unknown'),
            'count': category_metrics[taxonomy_id].count(0),
            'metrics': category_values
        }

    # Overall results
    msg = ''
    msg += 'Overall\t\t'
    for i, metric_name in enumerate(test_metrics.items):
        value = test_metrics.avg()[i]
        msg += '%.3f \t' % value
        results['overall'][metric_name] = float(value)
    print_log(msg, logger=logger)

    print_log(f'=' * 80, logger=logger)
    print_log(f'[{split_name}] Summary:', logger=logger)
    for metric_name, value in results['overall'].items():
        print_log(f'[{split_name}]   {metric_name}: {value:.4f}', logger=logger)
    print_log(f'=' * 80, logger=logger)

    return results


def main():
    # Parse arguments
    parser = get_args_parser()
    parser.add_argument('--output_dir', type=str, default='eval_results',
                       help='Directory to save evaluation results')
    args = parser.parse_args()

    # Read config
    config = cfg_from_yaml_file(args.config)

    # Set distributed to False for evaluation
    args.distributed = False
    args.use_gpu = torch.cuda.is_available()

    # Setup logger
    logger = get_logger(args.log_name)
    print_log('=' * 80, logger=logger)
    print_log('Text-Conditioned AdaPoinTr Evaluation on Seen and Unseen Categories', logger=logger)
    print_log('=' * 80, logger=logger)
    print_log(f'Checkpoint: {args.ckpts}', logger=logger)
    print_log(f'Config: {args.config}', logger=logger)
    print_log('=' * 80, logger=logger)

    # Build datasets
    print_log('[DATASET] Building seen categories dataset...', logger=logger)
    seen_config = config.dataset.val  # Use val config for seen categories
    _, seen_dataloader = builder.dataset_builder(args, seen_config)
    print_log(f'[DATASET] Seen dataset size: {len(seen_dataloader)}', logger=logger)

    # Build unseen dataset
    print_log('[DATASET] Building unseen categories dataset...', logger=logger)
    # Create unseen dataset config
    unseen_config_dict = {
        '_base_': {
            'NAME': 'Projected_ShapeNet',
            'N_POINTS': 8192,
            'N_RENDERINGS': 16,
            'DATA_PATH': 'data/ShapeNet55-34/Projected_ShapeNet-Unseen21_noise',
            'PARTIAL_POINTS_PATH': '/home/soroushm/data/project_shapenet_pcd/%s/%s/models/%d.pcd',
            'COMPLETE_POINTS_ROOT': '/home/soroushm/data/ShapeNet55-34/ShapeNet55',
            'CARS': False,
            'USE_CAPTIONS': True,
            'CAPTION_FILE': '/home/soroushm/data/Cap3D_automated_ShapeNet.csv'
        },
        'others': {'subset': 'test', 'bs': 1}
    }
    from easydict import EasyDict
    unseen_config = EasyDict(unseen_config_dict)
    _, unseen_dataloader = builder.dataset_builder(args, unseen_config)
    print_log(f'[DATASET] Unseen dataset size: {len(unseen_dataloader)}', logger=logger)

    # Build model
    print_log('[MODEL] Building model...', logger=logger)
    base_model = builder.model_builder(config.model)

    # Load checkpoint
    print_log(f'[MODEL] Loading checkpoint from {args.ckpts}...', logger=logger)
    builder.load_model(base_model, args.ckpts, logger=logger)

    if args.use_gpu:
        base_model = base_model.cuda()

    # Wrap in DataParallel
    if torch.cuda.device_count() > 1:
        print_log(f'[MODEL] Using {torch.cuda.device_count()} GPUs', logger=logger)
        base_model = nn.DataParallel(base_model)

    print_log('[MODEL] Model loaded successfully', logger=logger)

    # Criteria
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    # Update config to point to test datasets for evaluation
    config.dataset.test = seen_config

    # Evaluate on seen categories
    seen_results = evaluate_split(base_model, seen_dataloader, ChamferDisL1, ChamferDisL2,
                                  args, config, logger=logger, split_name='Seen')

    # Evaluate on unseen categories
    config.dataset.test = unseen_config
    unseen_results = evaluate_split(base_model, unseen_dataloader, ChamferDisL1, ChamferDisL2,
                                    args, config, logger=logger, split_name='Unseen')

    # Combine results
    final_results = {
        'checkpoint': args.ckpts,
        'config': args.config,
        'seen': seen_results,
        'unseen': unseen_results,
        'comparison': {
            'CDL1_seen': seen_results['overall']['CDL1'],
            'CDL1_unseen': unseen_results['overall']['CDL1'],
            'CDL1_diff': unseen_results['overall']['CDL1'] - seen_results['overall']['CDL1'],
            'CDL2_seen': seen_results['overall']['CDL2'],
            'CDL2_unseen': unseen_results['overall']['CDL2'],
            'CDL2_diff': unseen_results['overall']['CDL2'] - seen_results['overall']['CDL2'],
        }
    }

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'{args.exp_name}_results.json')
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    print_log(f'Results saved to {output_file}', logger=logger)

    # Print comparison summary
    print_log('=' * 80, logger=logger)
    print_log('COMPARISON SUMMARY', logger=logger)
    print_log('=' * 80, logger=logger)
    print_log(f"Seen CDL1:   {final_results['comparison']['CDL1_seen']:.4f}", logger=logger)
    print_log(f"Unseen CDL1: {final_results['comparison']['CDL1_unseen']:.4f}", logger=logger)
    print_log(f"Difference:  {final_results['comparison']['CDL1_diff']:.4f}", logger=logger)
    print_log(f"", logger=logger)
    print_log(f"Seen CDL2:   {final_results['comparison']['CDL2_seen']:.4f}", logger=logger)
    print_log(f"Unseen CDL2: {final_results['comparison']['CDL2_unseen']:.4f}", logger=logger)
    print_log(f"Difference:  {final_results['comparison']['CDL2_diff']:.4f}", logger=logger)
    print_log('=' * 80, logger=logger)
    print_log('Evaluation complete!', logger=logger)


if __name__ == '__main__':
    main()
