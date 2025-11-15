"""
PCN Unseen Categories Dataset

This dataset evaluates point cloud completion on 8 ShapeNet categories
that are NOT in the original PCN benchmark:
- bus (02924116)
- bed (02818832)
- bookshelf (02871439)
- bench (02828884)
- guitar (03467517)
- motorbike (03790512)
- skateboard (04225987)
- pistol (03948459)

Uses the same preprocessing as PCN: 2048 input points, 8192 output points
"""

import torch.utils.data as data
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import json
from .build import DATASETS
from utils.logger import *


@DATASETS.register_module()
class PCNUnseen(data.Dataset):
    """
    Dataset for evaluating on 8 unseen ShapeNet categories not in PCN.

    Data format follows Projected_ShapeNet structure:
    - Partial point clouds from projected_partial_noise/
    - Complete point clouds from ShapeNet55-34/
    """

    # Unseen category taxonomy IDs (not in PCN's 8 categories)
    UNSEEN_CATEGORIES = {
        '02924116': 'bus',
        '02818832': 'bed',
        '02871439': 'bookshelf',
        '02828884': 'bench',
        '03467517': 'guitar',
        '03790512': 'motorbike',
        '04225987': 'skateboard',
        '03948459': 'pistol'
    }

    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_root = config.COMPLETE_POINTS_ROOT
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.n_renderings = config.N_RENDERINGS if self.subset == 'train' else 1
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')

        print_log(f'[PCNUnseen] Loading unseen categories from {self.data_list_file}',
                  logger='PCNUnseen')

        # Load and filter file list for unseen categories only
        self.file_list = []
        if os.path.exists(self.data_list_file):
            with open(self.data_list_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                taxonomy_id = line.split('-')[0].split('/')[-1]
                model_id = line.split('-')[1].split('.')[0]

                # Only include unseen categories
                if taxonomy_id in self.UNSEEN_CATEGORIES:
                    self.file_list.append({
                        'taxonomy_id': taxonomy_id,
                        'model_id': model_id,
                        'file_path': line
                    })

        print_log(f'[PCNUnseen] Loaded {len(self.file_list)} instances from {len(self.get_category_counts())} unseen categories',
                  logger='PCNUnseen')

        # Print per-category counts
        category_counts = self.get_category_counts()
        for tax_id, count in category_counts.items():
            category_name = self.UNSEEN_CATEGORIES.get(tax_id, tax_id)
            print_log(f'  - {category_name} ({tax_id}): {count} samples', logger='PCNUnseen')

        self.transforms = self._get_transforms(self.subset)

    def get_category_counts(self):
        """Return dictionary of category counts."""
        counts = {}
        for item in self.file_list:
            tax_id = item['taxonomy_id']
            counts[tax_id] = counts.get(tax_id, 0) + 1
        return counts

    def _get_transforms(self, subset):
        """Same transforms as PCN: 2048 input points."""
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial', 'gt']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset == 'train' else 0

        # Load complete point cloud (ground truth)
        gt_path = os.path.join(self.complete_points_root, sample['file_path'])
        data['gt'] = IO.get(gt_path).astype(np.float32)

        # Load partial point cloud
        partial_path = self.partial_points_path % (sample['taxonomy_id'], sample['model_id'], rand_idx)
        data['partial'] = IO.get(partial_path).astype(np.float32)

        # Verify point count
        assert data['gt'].shape[0] == self.npoints, \
            f"Expected {self.npoints} points, got {data['gt'].shape[0]}"

        # Apply transforms
        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

    def __len__(self):
        return len(self.file_list)
