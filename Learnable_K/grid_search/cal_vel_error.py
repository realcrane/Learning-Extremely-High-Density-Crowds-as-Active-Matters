import os
import numpy as np
import sys
import torch

sys.path.append('..')
import scipy.io as sio
from config import config as cfg

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

bound = cfg.bound
n_grid = cfg.n_grid


def cal_mse(pred_vield, gt_vield):
    pred_temp = pred_vield[:, bound:n_grid[0] - bound, bound: n_grid[1] - bound, :]
    gt_temp = gt_vield[:, bound:n_grid[0] - bound, bound: n_grid[1] - bound, :]
    pred_temp = torch.FloatTensor(pred_temp)
    gt_temp = torch.FloatTensor(gt_temp)
    mse_loss = torch.nn.functional.mse_loss(gt_temp, pred_temp)
    return mse_loss.numpy()


split = 'val'
gt_field_root = data_root = f'../../../data/exp_data/{split}_60/'

if split == 'val':
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, ]

for K in K_list:
    cfg.K = K
    pred_field_root = f'{split}_fixed_para_K{cfg.K}'
    subfolder_list = os.listdir(pred_field_root)
    total_error_list = []
    for subfolder in subfolder_list:
        pred_subfolder_path = os.path.join(pred_field_root, subfolder)
        pred_field_list = os.listdir(pred_subfolder_path)
        pred_field_list.sort()

        gt_subfolder_path = os.path.join(gt_field_root, subfolder)

        error_list = []
        for file_name in pred_field_list:
            pred_file_path = os.path.join(pred_subfolder_path, file_name)
            pred_data = sio.loadmat(pred_file_path, squeeze_me=True, struct_as_record=False)['vel_field']
            gt_file_path = os.path.join(gt_subfolder_path, file_name)
            gt_data = sio.loadmat(gt_file_path, squeeze_me=True, struct_as_record=False)['vel_field_gt']
            error = cal_mse(pred_data, gt_data)
            error_list.append(error)

        avg_error = np.mean(np.array(error_list))
        total_error_list.append(avg_error)

    total_avg_error = np.mean(np.array(total_error_list))
    print(f'===> current K: {K},  total average error: {total_avg_error}')