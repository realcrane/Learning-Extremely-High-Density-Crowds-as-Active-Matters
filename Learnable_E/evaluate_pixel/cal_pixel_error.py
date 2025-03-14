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
        o_flow_x_path = '../../../dataset/optical_flow_mat/{}/flow_x_{}.mat'.format(
            folder_name, str(ind).zfill(5))
        o_flow_x = sio.loadmat(o_flow_x_path, squeeze_me=True, struct_as_record=False)['data']
        o_flow_y_path = '../../../dataset/optical_flow_mat/{}/flow_y_{}.mat'.format(
            folder_name, str(ind).zfill(5))
        o_flow_y = sio.loadmat(o_flow_y_path, squeeze_me=True, struct_as_record=False)['data']
        temp = np.zeros([480, 270, 2])
        temp[:, :, 0] = np.transpose(o_flow_x)
        temp[:, :, 1] = np.transpose(o_flow_y)

        temp = temp * 50 / 10
        gt_optical_flow.append(temp)

    return np.array(gt_optical_flow)


length_list = [60, 120, 180, 240, 300, 360, 420]
error = MetricList(['60', '120', '180', '240', '300', '360', '420'])

pred_root = 'pred_420/'
folder_list = os.listdir(pred_root)
for folder_name in folder_list:
    print('-----------> {}'.format(folder_name))
    folder_path = os.path.join(pred_root, folder_name)
    data_list = os.listdir(folder_path)
    data_list.sort()

    for data_name in data_list:
        print(data_name)
        pred_data_path = os.path.join(folder_path, data_name)
        pred_data = sio.loadmat(pred_data_path, squeeze_me=True, struct_as_record=False)
        pred_optical_flow = pred_data['pred_optical_flow']

        gt_optical_flow = obtain_gt_flow(folder_name, data_name, 420)
        error_list = []
        for length in length_list:
            pred_temp = torch.FloatTensor(pred_optical_flow[:length, :, :, :])
            gt_temp = torch.FloatTensor(gt_optical_flow[:length, :, :, :])
            mse_loss = torch.nn.functional.mse_loss(pred_temp, gt_temp)

            error_list.append(mse_loss.numpy())
        error.update(error_list, 1)

for n in range(len(length_list)):
    print('{:.8f}'.format(error.avg[str(length_list[n])]))
