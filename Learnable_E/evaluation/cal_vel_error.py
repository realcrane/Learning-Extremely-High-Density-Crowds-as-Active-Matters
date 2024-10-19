import os
import numpy as np
import sys
import torch

sys.path.append('..')
import scipy.io as sio
from config import config as cfg

bound = cfg.bound
n_grid = cfg.n_grid


def cal_mse(pred_vield, gt_vield):
    pred_temp = pred_vield[:, bound:n_grid[0] - bound, bound: n_grid[1] - bound, :]
    gt_temp = gt_vield[:, bound:n_grid[0] - bound, bound: n_grid[1] - bound, :]
    pred_temp = torch.FloatTensor(pred_temp)
    gt_temp = torch.FloatTensor(gt_temp)
    mse_loss = torch.nn.functional.mse_loss(gt_temp, pred_temp)
    return mse_loss.numpy()


total_error = 0
num = 0

split = 'test'
gt_field_root = data_root = f'../../../data/exp_data/{split}_30/'
E = 200

pred_field_root = f'{split}_fixed_para_E{E}'
pred_field_list = os.listdir(pred_field_root)

error_list = []
for file_name in pred_field_list:
    pred_file_path = os.path.join(pred_field_root, file_name)
    pred_data = sio.loadmat(pred_file_path, squeeze_me=True, struct_as_record=False)['vel_field']
    gt_file_path = os.path.join(gt_field_root, file_name)
    gt_data = sio.loadmat(gt_file_path, squeeze_me=True, struct_as_record=False)['vel_field_gt']
    error = cal_mse(pred_data, gt_data)
    error_list.append(error)

total_error = np.mean(np.array(error_list))
print(f'average error: ', +  total_error, len(error_list), E)
