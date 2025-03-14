import sys

sys.path.append('../..')
from config import config as cfg
import scipy.io as sio
import numpy as np
import torch
import os

from collections import OrderedDict


class MetricList(object):
    """Computes and stores the average and current value"""

    def __init__(self, loss_name):
        self.loss_names = loss_name
        self.reset()

    def reset(self):
        """Reset the loss value"""
        self.avg = OrderedDict()
        self.sum = OrderedDict()
        self.val = OrderedDict()
        self.count = 0

        for name in self.loss_names:
            self.avg[name] = 0.0
            self.sum[name] = 0.0
            self.val[name] = 0.0

    def update(self, losses, n):
        self.count += n
        for name, value in zip(self.loss_names, losses):
            self.val[name] = value
            self.sum[name] += value * n
            self.avg[name] = self.sum[name] / self.count


bound = cfg.bound
n_grid = cfg.n_grid


def cal_mse(pred_vield, gt_vield):
    pred_temp = pred_vield[:, bound:n_grid[0] - bound, bound: n_grid[1] - bound, :]
    gt_temp = gt_vield[:, bound:n_grid[0] - bound, bound: n_grid[1] - bound, :]
    pred_temp = torch.FloatTensor(pred_temp)
    gt_temp = torch.FloatTensor(gt_temp)
    mse_loss = torch.nn.functional.mse_loss(gt_temp, pred_temp)
    return mse_loss.numpy()


gt_field_root = f'../../../data/exp_data/test_420/'
subfolder_list = os.listdir(gt_field_root)

length_list = [60, 120, 180, 240, 300, 360, 420]

all_error = np.zeros((7, 10))
for n in range(10):
    error = MetricList(['60', '120', '180', '240', '300', '360', '420'])
    for subfolder in subfolder_list:
        pred_field_root = f'../resnet_sum_scale(1div50)_420/kld_weight0.1/2000/test/{subfolder}/{n}'
        pred_field_list = os.listdir(pred_field_root)
        pred_field_list.sort()
        for file_name in pred_field_list:
            pred_file_path = os.path.join(pred_field_root, file_name)
            pred_data = sio.loadmat(pred_file_path, squeeze_me=True, struct_as_record=False)['vel_field']
            gt_file_path = os.path.join(gt_field_root, subfolder, file_name)
            gt_data = sio.loadmat(gt_file_path, squeeze_me=True, struct_as_record=False)['vel_field_gt']

            error_list =[]
            for length in length_list:
                error_temp = cal_mse(pred_data[:length, :, :, :], gt_data[:length, :, :, :])
                error_list.append(error_temp)
            error.update(error_list, 1)

    for i in range(len(length_list)):
        all_error[i, n] = error.avg[str(length_list[i])]

for i in range(len(length_list)):
    print('{:.8f}'.format(np.mean(np.array(all_error[i, :]))))

print('####################################')

for i in range(len(length_list)):
    print('{:.8f}'.format(np.min(np.array(all_error[i, :]))))