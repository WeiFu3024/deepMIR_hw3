import yaml
import torch

class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__


def convert_tensor_to_numpy(tensor, is_squeeze=True):
    if is_squeeze:
        tensor = tensor.squeeze()
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()


def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    config = DotDict(config)
    return config

def setup_lr_scheduler(optimizer, args, batches_per_epoch):
    if args.train.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=args.train.warmup_epochs * batches_per_epoch // args.train.gradient_accumulation_steps
        )
    if args.train.lr_scheduler == 'cosine':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(args.train.epochs - args.train.warmup_epochs) * batches_per_epoch // args.train.gradient_accumulation_steps, eta_min=args.train.min_lr
        )
    elif args.train.lr_scheduler == 'linear':
        main_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=args.train.min_lr / args.train.lr,
            total_iters=(args.train.epochs - args.train.warmup_epochs) * batches_per_epoch // args.train.gradient_accumulation_steps
        )
    elif args.train.lr_scheduler == 'step':
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30 * batches_per_epoch // args.train.gradient_accumulation_steps, gamma=0.1
        )
    elif args.train.lr_scheduler == 'none':
        main_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    else:
        raise ValueError(f'Unknown lr_scheduler: {args.train.lr_scheduler}')
    if args.train.warmup_epochs > 0:
        lr_scheduler = SequentialLR(optimizer, schedulers=[warmup, main_scheduler], milestones=[args.train.warmup_epochs * batches_per_epoch // args.train.gradient_accumulation_steps])
    else:
        lr_scheduler = main_scheduler
    return lr_scheduler

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)