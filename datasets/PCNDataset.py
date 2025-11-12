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


# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py

@DATASETS.register_module()
class PCN(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.cars = config.CARS

        # Caption loading configuration
        self.return_captions = getattr(config, 'RETURN_CAPTIONS', False)
        self.caption_csv_path = getattr(config, 'CAPTION_CSV_PATH', None)
        self.captions = {}

        # Load captions from CSV if provided
        if self.return_captions and self.caption_csv_path and os.path.exists(self.caption_csv_path):
            self._load_captions()

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            if config.CARS:
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']

        # Create taxonomy name mapping
        self.taxonomy_names = {dc['taxonomy_id']: dc['taxonomy_name'] for dc in self.dataset_categories}

        self.n_renderings = 8 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)

    def _load_captions(self):
        """Load captions from CSV file."""
        print_log(f'Loading captions from {self.caption_csv_path}', logger='PCNDATASET')
        with open(self.caption_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    # First column: {category_id}_{instance_id}
                    # Second column: caption
                    instance_key = row[0].strip()
                    caption = row[1].strip()
                    self.captions[instance_key] = caption
        print_log(f'Loaded {len(self.captions)} captions', logger='PCNDATASET')

    def _get_caption(self, taxonomy_id, model_id):
        """Get caption for an instance, with fallback to class-based caption."""
        instance_key = f"{taxonomy_id}_{model_id}"

        # Try to get caption from loaded captions
        if instance_key in self.captions:
            return self.captions[instance_key]

        # Fallback: create caption based on taxonomy name
        taxonomy_name = self.taxonomy_names.get(taxonomy_id, 'object')
        return f"a 3d point cloud of a {taxonomy_name}"

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
                    'n_points': 128
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
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
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

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

        for ri in ['partial', 'gt']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            data[ri] = IO.get(file_path).astype(np.float32)
        
        assert data['gt'].shape[0] == self.npoints

        if self.transforms is not None:
            data = self.transforms(data)

        if self.return_captions:
            caption = self._get_caption(sample['taxonomy_id'], sample['model_id'])
            return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt']), caption
        else:
            return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

    def __len__(self):
        return len(self.file_list)

@DATASETS.register_module()
class PCNv2(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.cars = config.CARS

        # Caption loading configuration
        self.return_captions = getattr(config, 'RETURN_CAPTIONS', False)
        self.caption_csv_path = getattr(config, 'CAPTION_CSV_PATH', None)
        self.captions = {}

        # Load captions from CSV if provided
        if self.return_captions and self.caption_csv_path and os.path.exists(self.caption_csv_path):
            self._load_captions()

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            if config.CARS:
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']

        # Create taxonomy name mapping
        self.taxonomy_names = {dc['taxonomy_id']: dc['taxonomy_name'] for dc in self.dataset_categories}

        self.n_renderings = 8 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)

    def _load_captions(self):
        """Load captions from CSV file."""
        print_log(f'Loading captions from {self.caption_csv_path}', logger='PCNDATASET')
        with open(self.caption_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    # First column: {category_id}_{instance_id}
                    # Second column: caption
                    instance_key = row[0].strip()
                    caption = row[1].strip()
                    self.captions[instance_key] = caption
        print_log(f'Loaded {len(self.captions)} captions', logger='PCNDATASET')

    def _get_caption(self, taxonomy_id, model_id):
        """Get caption for an instance, with fallback to class-based caption."""
        instance_key = f"{taxonomy_id}_{model_id}"

        # Try to get caption from loaded captions
        if instance_key in self.captions:
            return self.captions[instance_key]

        # Fallback: create caption based on taxonomy name
        taxonomy_name = self.taxonomy_names.get(taxonomy_id, 'object')
        return f"a 3d point cloud of a {taxonomy_name}"

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'UpSamplePoints',
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
                'callback': 'UpSamplePoints',
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
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
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

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

        for ri in ['partial', 'gt']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            data[ri] = IO.get(file_path).astype(np.float32)

        assert data['gt'].shape[0] == self.npoints

        if self.transforms is not None:
            data = self.transforms(data)

        if self.return_captions:
            caption = self._get_caption(sample['taxonomy_id'], sample['model_id'])
            return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt']), caption
        else:
            return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

    def __len__(self):
        return len(self.file_list)