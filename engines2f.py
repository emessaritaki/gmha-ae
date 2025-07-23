from typing import Iterable, Optional

import torch

from timm.utils import ModelEma
from utils import utils
import matplotlib.pylab as plt
import numpy as np
import os
import scipy.io as io

def compute_r(a, b):
    r = (((a - a.mean()) * (b - b.mean())).sum()) / (
                ((((a - a.mean()) ** 2).sum()) ** 0.5) * ((((b - b.mean()) ** 2).sum()) ** 0.5))
    return r

def compute_per(input, n):
    y = 0
    for i in range(n - 1):
        for j in range(i+1, n):
            y += compute_r(input[i], input[j])
    y = (2 * y) / (n * (n - 1))
    return y

def p_loss(output, target):
    y = compute_per(output, output.shape[0])
    t = compute_per(target, target.shape[0])
    l = torch.abs(t - y)
    return l

def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        loss = model(input_ids=samples, labels=targets).loss
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        loss_value = loss.item()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, output_dir, args):
    criterion = torch.nn.MSELoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    pred = []
    label = []
    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        # with amp_autocast():
        cls_pred = model(images)
        loss = criterion(cls_pred, target)

        pred.append(cls_pred.squeeze())
        label.append(target)
        metric_logger.update(loss=loss.item())

    pred = torch.cat(pred, dim=0)
    label = torch.cat(label, dim=0)

    if args.eval:
        plot_dir = args.output_dir + '/plots/' + output_dir.split('cnnff')[1]
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        s = np.zeros((label.shape[0], 90, 90)).astype(np.float32)
        for i in range(len(s)):
            v = 0
            for aa in range(90):
                for bb in range(90):
                    if bb > aa:
                        s[i][aa][bb] = pred[i][v]
                        s[i][bb][aa] = pred[i][v]
                        v += 1
            ax = plt.gca()
            ax.yaxis.set_ticks_position('left')
 
            plt.imshow(s[i])
            plt.colorbar()
            plt.savefig(plot_dir + '{}pred.png'.format(i), dpi=150, bbox_inches='tight')
            plt.close()
        plt.close('all')
        io.savemat(args.output_dir + '/plots/' + output_dir.split('cnnff')[1][:4] + plot_dir[-4] + 's' + plot_dir[-2] + '.mat', {'data': s})

        a = label.squeeze()
        ss = np.zeros((label.shape[0], 90, 90)).astype(np.float32)
        for i in range(len(ss)):
            v = 0
            for aa in range(90):
                for bb in range(90):
                    if bb > aa:
                        ss[i][aa][bb] = a[i][v]
                        ss[i][bb][aa] = a[i][v]
                        v += 1
            ax = plt.gca()
            ax.yaxis.set_ticks_position('left')
            # ax.invert_yaxis()
            plt.imshow(ss[i])
            plt.colorbar()
            plt.savefig(plot_dir + '{}label.png'.format(i), dpi=150, bbox_inches='tight')
            plt.close()
        plt.close('all')
    # pred = cls_pred.squeeze()
    # label = target.squeeze()
    #
    rmse = (torch.mean((pred - label) ** 2)) ** 0.5
    mape = torch.mean(torch.abs(label - pred) / label)

    metric_logger.meters['rmse'].update(rmse.item(), n=pred.shape[0])
    metric_logger.meters['mape'].update(mape.item(), n=pred.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@rmse {rmse.global_avg:.8f} Acc@mape {mape.global_avg:.8f}, loss {losses.global_avg:.8f}'
          .format(rmse=metric_logger.rmse, mape=metric_logger.mape, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
