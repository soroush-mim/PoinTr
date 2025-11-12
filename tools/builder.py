import os, sys
# online package
import torch
# optimizer
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
# dataloader
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
# utils
from utils.logger import *
from utils.misc import *

def dataset_builder(args, config):
    dataset = build_dataset_from_cfg(config._base_, config.others)
    shuffle = config.others.subset == 'train'
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.others.bs if shuffle else 1,
                                            num_workers = int(args.num_workers),
                                            drop_last = config.others.subset == 'train',
                                            worker_init_fn = worker_init_fn,
                                            sampler = sampler)
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs if shuffle else 1,
                                                shuffle = shuffle, 
                                                drop_last = config.others.subset == 'train',
                                                num_workers = int(args.num_workers),
                                                worker_init_fn=worker_init_fn)
    return sampler, dataloader

def model_builder(config):
    model = build_model_from_cfg(config)
    return model

def build_optimizer(base_model, config):
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]
        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, base_model.parameters()), **opti_config.kwargs)
    else:
        raise NotImplementedError()

    return optimizer

def build_scheduler(base_model, optimizer, config, last_epoch=-1):
    sche_config = config.scheduler
    if sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs, last_epoch=last_epoch)  # misc.py
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, last_epoch=last_epoch, **sche_config.kwargs)
    elif sche_config.type == 'GradualWarmup':
        scheduler_steplr = torch.optim.lr_scheduler.StepLR(optimizer, last_epoch=last_epoch, **sche_config.kwargs_1)
        scheduler = GradualWarmupScheduler(optimizer, after_scheduler=scheduler_steplr, **sche_config.kwargs_2)
    elif sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                t_initial=sche_config.kwargs.t_max,
                lr_min=sche_config.kwargs.min_lr,
                warmup_t=sche_config.kwargs.initial_epochs,
                t_in_epochs=True)
    else:
        raise NotImplementedError()
    
    if config.get('bnmscheduler') is not None:
        bnsche_config = config.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
        scheduler = [scheduler, bnscheduler]
    
    return scheduler

def resume_model(base_model, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger = logger )

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    # parameter resume of base model
    # if args.local_rank == 0:
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)

    # parameter
    start_epoch = state_dict['epoch'] + 1
    best_metrics = state_dict['best_metrics']
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics.state_dict()
    # print(best_metrics)

    print_log(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})', logger = logger)
    return start_epoch, best_metrics

def resume_optimizer(optimizer, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0, 0
    print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path}...', logger = logger )
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

def save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, logger = None):
    if args.local_rank == 0:
        torch.save({
                    'base_model' : base_model.module.state_dict() if args.distributed else base_model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'epoch' : epoch,
                    'metrics' : metrics.state_dict() if metrics is not None else dict(),
                    'best_metrics' : best_metrics.state_dict() if best_metrics is not None else dict(),
                    }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger = logger)

def load_model(base_model, ckpt_path, logger = None):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger = logger )

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # parameter resume of base model
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')
    base_model.load_state_dict(base_ckpt)

    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    print_log(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})', logger = logger)
    return

def load_pretrained_adapointr_for_text(base_model, ckpt_path, logger=None):
    """
    Load pretrained non-text AdaPoinTr weights into a text-conditioned model.

    This function handles loading weights from a standard AdaPoinTr checkpoint
    into a text-conditioned AdaPoinTr model. It skips text-specific components
    (text encoder, text query MLP, ULIP loss module) and loads all shared
    geometric processing components.

    Args:
        base_model: Text-conditioned AdaPoinTr model instance
        ckpt_path: Path to pretrained non-text AdaPoinTr checkpoint
        logger: Logger instance for printing messages

    Returns:
        None
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'No checkpoint file at path {ckpt_path}')

    print_log(f'[PRETRAINED] Loading pretrained AdaPoinTr weights from {ckpt_path}...', logger=logger)

    # Load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    # Extract model weights
    if state_dict.get('model') is not None:
        pretrained_dict = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        pretrained_dict = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('Checkpoint does not contain model weights')

    # Get current model state dict
    model_dict = base_model.state_dict()

    # Filter out text-specific and mismatched parameters
    text_specific_keys = [
        'text_encoder',           # CLIP text encoder
        'mlp_query_text',         # Text-conditioned query generator
        'ulip_loss_module',       # ULIP alignment loss
        'ulip_encoder'            # ULIP PointBERT encoder
    ]

    # Count statistics
    loaded_keys = []
    skipped_keys = []
    missing_keys = []

    # Filter pretrained dict
    filtered_dict = {}
    for k, v in pretrained_dict.items():
        # Skip text-specific components
        if any(key in k for key in text_specific_keys):
            skipped_keys.append(k)
            continue

        # Skip mlp_query (will be replaced by mlp_query_text)
        if 'mlp_query' in k and 'mlp_query_text' not in k:
            skipped_keys.append(k)
            print_log(f'[PRETRAINED] Skipping {k} (replaced by mlp_query_text)', logger=logger)
            continue

        # Check if key exists in current model
        if k in model_dict:
            # Check shape match
            if v.shape == model_dict[k].shape:
                filtered_dict[k] = v
                loaded_keys.append(k)
            else:
                skipped_keys.append(k)
                print_log(f'[PRETRAINED] Shape mismatch for {k}: pretrained {v.shape} vs model {model_dict[k].shape}',
                         logger=logger)
        else:
            skipped_keys.append(k)

    # Find missing keys (keys in model but not in pretrained)
    for k in model_dict.keys():
        if k not in filtered_dict and not any(key in k for key in text_specific_keys):
            # Only report non-text-specific missing keys as concerning
            if 'mlp_query' not in k:  # mlp_query_text is expected to be new
                missing_keys.append(k)

    # Load filtered weights
    base_model.load_state_dict(filtered_dict, strict=False)

    # Print statistics
    print_log(f'[PRETRAINED] ========== Loading Summary ==========', logger=logger)
    print_log(f'[PRETRAINED] Total pretrained parameters: {len(pretrained_dict)}', logger=logger)
    print_log(f'[PRETRAINED] Successfully loaded: {len(loaded_keys)}', logger=logger)
    print_log(f'[PRETRAINED] Skipped (text-specific or mismatched): {len(skipped_keys)}', logger=logger)
    print_log(f'[PRETRAINED] Missing in pretrained (will be randomly initialized): {len(missing_keys)}', logger=logger)

    if missing_keys:
        print_log(f'[PRETRAINED] Missing keys (new text components - expected):', logger=logger)
        for k in missing_keys[:10]:  # Print first 10
            print_log(f'[PRETRAINED]   - {k}', logger=logger)
        if len(missing_keys) > 10:
            print_log(f'[PRETRAINED]   ... and {len(missing_keys) - 10} more', logger=logger)

    # Print checkpoint info
    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    print_log(f'[PRETRAINED] Checkpoint from epoch {epoch} (performance = {str(metrics)})', logger=logger)
    print_log(f'[PRETRAINED] =====================================', logger=logger) 