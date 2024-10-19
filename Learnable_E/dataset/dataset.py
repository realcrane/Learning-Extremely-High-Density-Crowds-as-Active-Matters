import os
import scipy.io as sio

import torch

class HajjDataset(torch.utils.data.Dataset):
    def __init__(self, root, phase):
        data_folder = os.path.join(root, phase)
        all_data = os.listdir(data_folder)
        self.data_list =[]
        for temp in all_data:
            self.data_list.append(os.path.join(data_folder, temp))


    def __getitem__(self, index):
        data_path = self.data_list[index]
        # print(data_path)
        data = sio.loadmat(data_path, squeeze_me=True, struct_as_record=False)
        init_pos = torch.FloatTensor(data['init_pos'])
        init_vel = torch.FloatTensor(data['init_vel'])
        vel_field_gt = torch.FloatTensor(data['vel_field_gt'])
        return init_pos, init_vel, vel_field_gt

    def __len__(self):
        return len(self.data_list)
