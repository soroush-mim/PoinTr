import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
import csv
from .build import DATASETS
from utils.logger import *


@DATASETS.register_module()
class Projected_ShapeNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_root = config.COMPLETE_POINTS_ROOT
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.cars = config.CARS
        self.n_renderings = config.N_RENDERINGS if self.subset == 'train' else 1
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')

        # Text conditioning - load captions from CSV if available
        self.use_captions = getattr(config, 'USE_CAPTIONS', False)
        self.caption_file = getattr(config, 'CAPTION_FILE', None)
        self.captions_dict = {}

        if self.use_captions and self.caption_file is not None:
            print_log(f'[DATASET] Loading captions from {self.caption_file}', logger='Projected_ShapeNet')
            self._load_captions()
        else:
            print_log(f'[DATASET] No captions file specified, captions will be None', logger='Projected_ShapeNet')

        print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'Projected_ShapeNet')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0].split('/')[-1]
            model_id = line.split('-')[1].split('.')[0]
            if config.CARS:
                if taxonomy_id == '02958343':
                    self.file_list.append({
                        'taxonomy_id': taxonomy_id,
                        'model_id': model_id,
                        'file_path': line
                    })
                else:
                    pass
            else:
                self.file_list.append({
                    'taxonomy_id': taxonomy_id,
                    'model_id': model_id,
                    'file_path': line
                })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')

        self.transforms = self._get_transforms(self.subset)

    def _load_captions(self):
        """
        Load captions from CSV file.

        CSV format:
        - First column: {category_id}_{instance_id}
        - Second column: caption text

        Example:
            02958343_abc123,a red sports car with four wheels
            03001627_def456,a wooden chair with a tall back
        """
        if not os.path.exists(self.caption_file):
            print_log(f'[DATASET] Warning: Caption file {self.caption_file} not found!', logger='Projected_ShapeNet')
            return

        try:
            with open(self.caption_file, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    if len(row) >= 2:
                        # First column is {category_id}_{instance_id}
                        # Second column is caption
                        key = row[0].strip()
                        caption = row[1].strip()
                        self.captions_dict[key] = caption

            print_log(f'[DATASET] Loaded {len(self.captions_dict)} captions from CSV', logger='Projected_ShapeNet')

        except Exception as e:
            print_log(f'[DATASET] Error loading captions: {e}', logger='Projected_ShapeNet')
            self.captions_dict = {}

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
            },{
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
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

        gt_path = os.path.join(self.complete_points_root, sample['file_path'])
        data['gt'] = IO.get(gt_path).astype(np.float32)

        partial_path = self.partial_points_path % (sample['taxonomy_id'], sample['model_id'], rand_idx)
        data['partial'] = IO.get(partial_path).astype(np.float32)

        assert data['gt'].shape[0] == self.npoints

        if self.transforms is not None:
            data = self.transforms(data)

        # Get caption if available
        caption = None
        if self.use_captions:
            # Create key as {category_id}_{instance_id}
            caption_key = f"{sample['taxonomy_id']}_{sample['model_id']}"
            caption = self.captions_dict.get(caption_key, None)

            # If caption not found, use a default caption based on taxonomy_id
            if caption is None:
                caption = f"a 3D object from category {sample['taxonomy_id']}"

        # Return with caption as the last element
        if caption is not None:
            return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'], caption)
        else:
            return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

    def __len__(self):
        return len(self.file_list)