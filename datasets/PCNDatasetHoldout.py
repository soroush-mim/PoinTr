"""
PCN Hold-out Class Dataset

This dataset supports training with one PCN class held out, and evaluating
on that held-out class as an unseen category.

PCN categories (8 total):
- airplane (02691156)
- cabinet (02933112)
- car (02958343)
- chair (03001627)
- lamp (03636649)
- sofa (04256520)
- table (04379243)
- watercraft (04530566)

Usage:
- For training: Set HOLDOUT_CLASS to the taxonomy_id to exclude
- For evaluation: Set HOLDOUT_ONLY to True to evaluate only on held-out class
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
class PCNHoldout(data.Dataset):
    """
    PCN dataset with support for holding out one class.

    Configuration:
    - HOLDOUT_CLASS: taxonomy_id of class to hold out (e.g., '02691156' for airplane)
    - HOLDOUT_ONLY: If True, only load the held-out class (for evaluation)
                    If False, load all classes EXCEPT the held-out class (for training)
    """

    # PCN category taxonomy IDs
    PCN_CATEGORIES = {
        '02691156': 'airplane',
        '02933112': 'cabinet',
        '02958343': 'car',
        '03001627': 'chair',
        '03636649': 'lamp',
        '04256520': 'sofa',
        '04379243': 'table',
        '04530566': 'watercraft'
    }

    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.cars = config.CARS

        # Hold-out configuration
        self.holdout_class = getattr(config, 'HOLDOUT_CLASS', None)
        self.holdout_only = getattr(config, 'HOLDOUT_ONLY', False)

        if self.holdout_class is None:
            raise ValueError("HOLDOUT_CLASS must be specified in config")

        if self.holdout_class not in self.PCN_CATEGORIES:
            raise ValueError(f"HOLDOUT_CLASS must be one of PCN categories: {list(self.PCN_CATEGORIES.keys())}")

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            if config.CARS:
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']

        # Create taxonomy name mapping
        self.taxonomy_names = {dc['taxonomy_id']: dc['taxonomy_name'] for dc in self.dataset_categories}

        self.n_renderings = 8 if self.subset == 'train' else 1

        # Filter based on hold-out configuration
        if self.holdout_only:
            # For evaluation: only load held-out class
            self.dataset_categories = [dc for dc in self.dataset_categories
                                       if dc['taxonomy_id'] == self.holdout_class]
            mode_str = f"HELD-OUT CLASS ONLY: {self.PCN_CATEGORIES[self.holdout_class]}"
        else:
            # For training: exclude held-out class
            self.dataset_categories = [dc for dc in self.dataset_categories
                                       if dc['taxonomy_id'] != self.holdout_class]
            mode_str = f"ALL CLASSES EXCEPT: {self.PCN_CATEGORIES[self.holdout_class]}"

        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)

        print_log(f'[PCNHoldout] {mode_str}', logger='PCNHoldout')
        print_log(f'[PCNHoldout] Loaded {len(self.file_list)} instances from {len(self.dataset_categories)} categories',
                  logger='PCNHoldout')

    def _get_transforms(self, subset):
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

    def _get_file_list(self, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']),
                      logger='PCNHoldout')
            samples = dc[subset]

            for s in samples:
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_path': [
                        self.partial_points_path % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gt_path':
                    self.complete_points_path % (subset, dc['taxonomy_id'], s),
                })

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list),
                  logger='PCNHoldout')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset == 'train' else 0

        for ri in ['partial', 'gt']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            data[ri] = IO.get(file_path).astype(np.float32)

        assert data['gt'].shape[0] == self.npoints

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

    def __len__(self):
        return len(self.file_list)
