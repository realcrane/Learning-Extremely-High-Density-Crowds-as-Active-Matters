import os
import scipy.io as sio

import torch


class CVAEDataset(torch.utils.data.Dataset):
    def __init__(self, root, cfg, phase):
        self.cfg = cfg
        self.data_list = []
        data_folder = os.path.join(root, phase)
        subfolder_list = os.listdir(data_folder)
        for subfolder in subfolder_list:
            subfolder_path = os.path.join(data_folder, subfolder)
            all_data = os.listdir(subfolder_path)
            for temp in all_data:
                self.data_list.append(os.path.join(subfolder_path, temp))

    def __getitem__(self, index):
        data_path = self.data_list[index]

        data = sio.loadmat(data_path, squeeze_me=True, struct_as_record=False)
        alpha = torch.FloatTensor(data['alpha'])
        grid_v_in = torch.FloatTensor(data['grid_v_in'])
        grid_v_out = torch.FloatTensor(data['grid_v_out'])
        vel_field_gt = torch.FloatTensor(data['vel_field_gt'])

        mask = torch.zeros(30, 20).bool()
        mask[3:27, 3:17] = True
        grid_v_in[:, ~mask, :] = 0
        grid_v_out[:, ~mask, :] = 0
        vel_field_gt[:, ~mask, :] = 0
        alpha[:, ~mask] = 0

        diff = vel_field_gt - grid_v_out - alpha.unsqueeze(-1) * grid_v_in * self.cfg.dt
        diff = diff.permute(0, 3, 1, 2)

        cond_1 = torch.sum(grid_v_in ** 2, dim=-1, keepdim=True) * grid_v_in

        cond_2 = torch.zeros_like(grid_v_in).to(grid_v_in.device)
        temp = torch.zeros_like(alpha).unsqueeze(-1).to(grid_v_in.device)
        dvx_dx = (grid_v_in[:, 2:30, 1:19, 0] - grid_v_in[:, 0:28, 1:19, 0]) / (2 * self.cfg.dx)
        dvy_dy = (grid_v_in[:, 1:29, 2:20, 1] - grid_v_in[:, 1:29, 0:18, 1]) / (2 * self.cfg.dx)
        temp[:, 1:29, 1:19, 0] = dvx_dx + dvy_dy
        cond_2[:, 1:29, 1:19, 0] = (temp[:, 2:30, 1:19, 0] - temp[:, 0:28, 1:19, 0]) / (2 * self.cfg.dx)
        cond_2[:, 1:29, 1:19, 1] = (temp[:, 1:29, 2:20, 0] - temp[:, 1:29, 0:18, 0]) / (2 * self.cfg.dx)

        cond_3 = torch.zeros_like(grid_v_in).to(grid_v_in.device)
        cond_3[:, 1:29, 1:19, 0] = (grid_v_in[:, 2:30, 1:19, 0] + grid_v_in[:, 0:28, 1:19, 0] - grid_v_in[:, 1:29, 1:19,
                                                                                                0]) / self.cfg.dx ** 2
        cond_3[:, 1:29, 1:19, 1] = (grid_v_in[:, 1:29, 2:20, 1] + grid_v_in[:, 1:29, 0:18, 1] - grid_v_in[:, 1:29, 1:19,
                                                                                                1]) / self.cfg.dx ** 2

        cond_4 = torch.zeros_like(grid_v_in).to(grid_v_in.device)
        dvx_dx = (grid_v_in[:, 2:30, 1:19, 0] - grid_v_in[:, 0:28, 1:19, 0]) / (2 * self.cfg.dx)
        dvy_dy = (grid_v_in[:, 1:29, 2:20, 1] - grid_v_in[:, 1:29, 0:18, 1]) / (2 * self.cfg.dx)
        dvx_dy = (grid_v_in[:, 1:29, 2:20, 0] - grid_v_in[:, 1:29, 0:18, 0]) / (2 * self.cfg.dx)
        dvy_dx = (grid_v_in[:, 2:30, 1:19, 1] - grid_v_in[:, 0:28, 1:19, 1]) / (2 * self.cfg.dx)
        cond_4[:, 1:29, 1:19, 0] = grid_v_in[:, 1:29, 1:19, 0] * dvx_dx + grid_v_in[:, 1:29, 1:19, 1] * dvx_dy
        cond_4[:, 1:29, 1:19, 1] = grid_v_in[:, 1:29, 1:19, 0] * dvy_dx + grid_v_in[:, 1:29, 1:19, 1] * dvy_dy

        cond = torch.cat([cond_1, cond_2, cond_3, cond_4], dim=-1).permute(0, 3, 1, 2)

        mask = torch.zeros(30, 20).bool()
        mask[3:27, 3:17] = True
        diff[:, :, ~mask] = 0
        cond[:, :, ~mask] = 0
        return diff, cond

    def __len__(self):
        return len(self.data_list)
