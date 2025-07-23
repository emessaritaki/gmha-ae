import argparse
import datetime
import numpy as np
import time
import json
from pathlib import Path
import os

from sklearn.model_selection import KFold

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as udata

from timm.scheduler import create_scheduler
from timm.utils import *

from data.dataset import s2fDataset
from models.model import GMHAAE
from utils import utils
from engines2f import evaluate, train_one_epoch

cudnn.benchmark = True

def get_args_parser():
    parser = argparse.ArgumentParser('DynamicViT training and evaluation script', add_help=False)
    # Global & Device parameters
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='', type=str, help='model & log save path')
    parser.add_argument('--pin-mem', action='store_true', default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.set_defaults(pin_mem=True)

    # Model parameters
    parser.add_argument('--ele_dim', default=375, type=int, help='size of every patch in an input image')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', default=True, type=bool, help='exponential moving average of model weights')
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=True, help='')


    # Optimizer parameters
    parser.add_argument('--amp', action='store_true', default=False,
                        help='using automatic mixed precision, is loss is nan, please set False!')
    parser.add_argument('--sync-bn', action='store_true', default=True,
                        help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--dist-bn', type=str, default='reduce',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    parser.add_argument('--split-bn', action='store_true',
                        help='Enable separate BN layers per augmentation split.')

    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=5, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # some parameters for debug or other actions
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', default=True, help='Perform evaluation only')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distill', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

    return parser


def get_param_groups(model, weight_decay, args):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        else:
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
    return [
        {'params': no_decay, 'lr': args.lr, 'weight_decay': 0., 'name': 'base_no_decay'},
        {'params': decay, 'lr': args.lr, 'weight_decay': weight_decay, 'name': 'base_decay'},
    ]


def run(args, X_train, X_test, y_train, y_test, output_dir):
    if args.seed is not None:
        random_seed(args.seed, utils.get_rank())

    device = torch.device(args.device)

    model = GMHAAE(input_feat_dim=90, dropout=0.25)
    if args.eval:
        resume = output_dir + 'checkpoint_best.pth'
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model'])

    model.cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # optimizer settings
    opt_args = dict(lr=args.lr, weight_decay=args.weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    parameter_group = get_param_groups(model, args.weight_decay, args)
    optimizer = torch.optim.AdamW(parameter_group, **opt_args)

    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
    else:
        model_ema = None

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)

    # resume checkpoint
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        if not args.eval and 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                model_ema.module.load_state_dict(checkpoint['model_ema'], strict=True)

    # setup learning rate schedule and starting epoch
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch

    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    # Dataset
    dataset_train = s2fDataset(X_train, y_train)
    dataset_valid = s2fDataset(X_test, y_test)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_valid = torch.utils.data.SequentialSampler(dataset_valid)

    data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                    sampler=sampler_train,
                                                    batch_size=args.batch_size,
                                                    pin_memory=args.pin_mem,
                                                    drop_last=False)

    data_loader_val = torch.utils.data.DataLoader(dataset_valid,
                                                  sampler=sampler_valid,
                                                  batch_size=X_test.shape[0],
                                                  pin_memory=args.pin_mem,
                                                  drop_last=False)

    criterion = torch.nn.L1Loss()

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, output_dir, args)
        # print(f"Accuracy of the network on the {len(dataset_valid)} test samples: {test_stats['acc_rmse']:.1f}% ")
        return

    print(f"Start training for {num_epochs} epochs")
    start_time = time.time()
    if args.resume:
        max_accuracy = checkpoint['max_accuracy']
    else:
        max_accuracy = 100.0
    best_times = 0
    output_dir = Path(output_dir)
    for epoch in range(args.start_epoch, num_epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch,
            args.clip_grad, model_ema)

        test_stats = evaluate(data_loader_val, model, device, output_dir, args)

        print(f"Accuracy of the network on the {len(dataset_valid)} rmse: {test_stats['rmse']:.1f}% "
              f"mape: {test_stats['mape']:.1f}% ")
        is_best = test_stats["rmse"] < max_accuracy
        best_times += 1
        if is_best:
            best_times = 0
        max_accuracy = min(max_accuracy, test_stats["rmse"])
        print(f'Best accuracy: {max_accuracy:.6f}%')

        lr_scheduler.step(epoch + 1)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'args': args,
                    'max_accuracy': max_accuracy,
                }, checkpoint_path)

        if max_accuracy == test_stats["rmse"]:
            checkpoint_paths = [output_dir / 'checkpoint_best.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if best_times > 50:
            print('early_stop')
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Transformer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for m in range(4):
        for n in range(10):
            X = []
            for i in range(4):
                if i == 4 or i == 5:
                    continue
                else:
                    x = np.load(args.data_path + 's{}{}{}.npy'.format(n, m, i))
                    X.append(x)
            X = np.concatenate(X, axis=2).transpose(0, 2, 1)

            for j in range(4):
                Y = np.load(args.data_path + 'ff{}.npy'.format(j))
                fold = 0
                kfold = KFold(n_splits=5, shuffle=False)
                for train_ix, test_ix in kfold.split(X):

                    X_train, X_test = X[train_ix[:100]], X[train_ix[100:]]
                    y_train, y_test = Y[train_ix[:100]], Y[train_ix[100:]]

                    shuffle_idx = np.random.permutation(np.arange(len(X_train)))
                    X_train = X_train[shuffle_idx]
                    y_train = y_train[shuffle_idx]

                    output_dir = args.output_dir + 'ff/{}{}/{}/{}/'.format(n, m, j, fold)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    if args.eval:
                        X_test, y_test = X[test_ix], Y[test_ix]
                        import matplotlib.pyplot as plt
                        plt.close('all')
                    run(args, X_train, X_test, y_train, y_test, output_dir)
                    fold += 1
