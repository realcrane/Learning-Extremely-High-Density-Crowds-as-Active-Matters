import os
import scipy.io as sio
import numpy as np
import torch

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


def obtain_gt_flow(folder_name, data_name, length):
    gt_optical_flow = []
    for i in range(length):
        ind = int(data_name[10:-4]) + i + 1
        o_flow_x_path = '../../../../dataset/optical_flow_mat/{}/flow_x_{}.mat'.format(
            folder_name, str(ind).zfill(5))
        o_flow_x = sio.loadmat(o_flow_x_path, squeeze_me=True, struct_as_record=False)['data']
        o_flow_y_path = '../../../../dataset/optical_flow_mat/{}/flow_y_{}.mat'.format(
            folder_name, str(ind).zfill(5))
        o_flow_y = sio.loadmat(o_flow_y_path, squeeze_me=True, struct_as_record=False)['data']
        temp = np.zeros([480, 270, 2])
        temp[:, :, 0] = np.transpose(o_flow_x)
        temp[:, :, 1] = np.transpose(o_flow_y)
        temp = temp * 50 / 10
        gt_optical_flow.append(temp)
    return np.array(gt_optical_flow)


gt_field_root = f'../../../data/exp_data/test_420/'
subfolder_list = os.listdir(gt_field_root)

length_list = [60, 120, 180, 240, 300, 360, 420]
all_error = np.zeros((7, 10))
for n in range(10):
    print(n)
    error = MetricList(['60', '120', '180', '240', '300', '360', '420'])
    for subfolder in subfolder_list:
        pred_field_root = f'pred_420/{n}/{subfolder}'
        pred_field_list = os.listdir(pred_field_root)
        pred_field_list.sort()
        for file_name in pred_field_list:
            print(file_name)
            pred_data_path = os.path.join(pred_field_root, file_name)
            pred_data = sio.loadmat(pred_data_path, squeeze_me=True, struct_as_record=False)
            pred_optical_flow = pred_data['pred_optical_flow']

            gt_optical_flow = obtain_gt_flow(subfolder, file_name, 420)

            pred_temp = torch.FloatTensor(pred_optical_flow)
            gt_temp = torch.FloatTensor(gt_optical_flow)
            error_list = []
            for length in length_list:
                mse_loss = torch.nn.functional.mse_loss(pred_temp[:length, :, :, :], gt_temp[:length, :, :, :])
                error_list.append(mse_loss.numpy())

            error.update(error_list, 1)

    for i in range(len(length_list)):
        all_error[i, n] = error.avg[str(length_list[i])]

for i in range(len(length_list)):
    print('{:.8f}'.format(np.mean(np.array(all_error[i, :]))))

print('####################################')

for i in range(len(length_list)):
    print('{:.8f}'.format(np.min(np.array(all_error[i, :]))))
