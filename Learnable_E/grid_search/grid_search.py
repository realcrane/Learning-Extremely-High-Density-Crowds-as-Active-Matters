import sys

sys.path.append('..')
from config import config as cfg
import scipy.io as sio
import torch
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

split = 'val'

if split == 'val':
    w_ext_list = [0.01, 0.03, 0.05, 0.07, 0.09, 0.12]
    E_list = [500, 1000, 1500, 2000, 2500, 3000]

for w_ext in w_ext_list:
    cfg.w_ext = w_ext

    for E in E_list:
        cfg.E = E
        from model.mpm_fixedE import MPM

        model = MPM(cfg).cuda()
        model.eval()

        save_root = f'w_ext_{cfg.w_ext}/{split}_fixed_para_E{cfg.E}_60'
        #### load data
        data_root = f'../../data/exp_data/{split}_60/'
        subfolder_list = os.listdir(data_root)
        for subfolder in subfolder_list:
            print('start to process the folder {}, current E: {}, current weight: {}'.format(subfolder, E, w_ext))
            subfolder_path = os.path.join(data_root, subfolder)
            save_folder = os.path.join(save_root, subfolder)
            os.makedirs(save_folder, exist_ok=True)

            init_data_list = os.listdir(subfolder_path)
            init_data_list.sort()
            for i in range(len(init_data_list)):
                init_data_name = init_data_list[i]
                print(init_data_name)
                init_data_path = os.path.join(subfolder_path, init_data_name)

                init_data = sio.loadmat(init_data_path, squeeze_me=True, struct_as_record=False)
                init_pos = torch.FloatTensor(init_data['init_pos'])
                init_vel = torch.FloatTensor(init_data['init_vel'])
                init_ind = torch.FloatTensor(init_data['init_ind'])
                num = init_pos.size(0)
                C = torch.zeros((num, cfg.dim, cfg.dim))
                J = torch.ones((num))

                with torch.no_grad():
                    start_time = time.time()
                    model.set_input([init_pos, init_vel, init_ind, C, J])
                    model.n_substeps = 60 * cfg.steps
                    vel_field = model.forward().detach().cpu().numpy()
                    pos_seq = torch.stack(model.all_pos_seq).detach().cpu().numpy()
                    vel_seq = torch.stack(model.all_pos_seq).detach().cpu().numpy()
                    sio.savemat(os.path.join(save_folder, init_data_name),
                                {'pos': pos_seq, 'vel': vel_seq, 'vel_field': vel_field})

            print('-----> {} files has been processed'.format(len(init_data_list)))
