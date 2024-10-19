import os
import numpy as np
import sys
import torch

sys.path.append('..')
import scipy.io as sio
from config import config as cfg

bound = cfg.bound
n_grid = cfg.n_grid

grid_type = np.zeros((n_grid[0], n_grid[1]))
grid_type[2:n_grid[0] - 2, 2:n_grid[1] - 2] = 1
dx = cfg.dx


def cal_mse(pred_vor, gt_vor):
    pred_temp = pred_vor[:, bound:n_grid[0] - bound, bound: n_grid[1] - bound]
    gt_temp = gt_vor[:, bound:n_grid[0] - bound, bound: n_grid[1] - bound]
    pred_temp = torch.FloatTensor(pred_temp)
    gt_temp = torch.FloatTensor(gt_temp)
    mse_loss = torch.nn.functional.mse_loss(gt_temp, pred_temp)
    return mse_loss.numpy()


def cal_vor(vel_field):
    temp = np.zeros((n_grid[0], n_grid[1]))
    for i in range(0, n_grid[0]):
        for j in range(0, n_grid[1]):
            if grid_type[i, j] == 1:
                if grid_type[i + 1, j] == 1:
                    temp[i, j] += (vel_field[i + 1, j][1] + vel_field[i, j][1]) / 2
                if grid_type[i - 1, j] == 1:
                    temp[i, j] -= (vel_field[i - 1, j][1] + vel_field[i, j][1]) / 2

                if grid_type[i, j + 1] == 1:
                    temp[i, j] -= (vel_field[i, j + 1][0] + vel_field[i, j][0]) / 2
                if grid_type[i, j - 1] == 1:
                    temp[i, j] += (vel_field[i, j - 1][0] + vel_field[i, j][0]) / 2

                temp[i, j] /= dx

    return temp


split = 'test'
gt_field_root = data_root = f'../../../data/exp_data/{split}_30/'
E = 200
pred_field_root = f'{split}_fixed_para_E{E}'
pred_file_list = os.listdir(pred_field_root)

pred_list = []
gt_list = []
for file_name in pred_file_list:
    pred_file_path = os.path.join(pred_field_root, file_name)
    pred_data = sio.loadmat(pred_file_path, squeeze_me=True, struct_as_record=False)['vel_field']
    gt_file_path = os.path.join(gt_field_root, file_name)
    gt_data = sio.loadmat(gt_file_path, squeeze_me=True, struct_as_record=False)['vel_field_gt']

    for i in range(30):
        pred_vor = cal_vor(pred_data[i, :, :, :])
        gt_vor = cal_vor(gt_data[i, :, :, :])

        pred_list.append(pred_vor)
        gt_list.append(gt_vor)

gt_list = np.stack(gt_list, axis=0)
pred_list = np.stack(pred_list, axis=0)
total_error = cal_mse(pred_list, gt_list)
