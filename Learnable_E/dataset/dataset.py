import scipy.io as sio
import torch
import os

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, phase):
        data_folder = os.path.join(root, phase)
        subfolder_list = os.listdir(data_folder)
        self.data_list = []
        for subfolder in subfolder_list:
            all_data = os.listdir(os.path.join(data_folder, subfolder))
            for temp in all_data:
                self.data_list.append(os.path.join(data_folder, subfolder, temp))

    def __getitem__(self, index):
        data_path = self.data_list[index]
        data = sio.loadmat(data_path, squeeze_me=True, struct_as_record=False)
        init_pos = torch.FloatTensor(data['init_pos'])
        init_vel = torch.FloatTensor(data['init_vel'])
        init_ind = torch.FloatTensor(data['init_ind'])
        vel_field_gt = torch.FloatTensor(data['vel_field_gt'])
        return init_pos, init_vel, init_ind, vel_field_gt

    def __len__(self):
        return len(self.data_list)
