import sys
sys.path.append('..')
from config import config as cfg
import torch
import scipy.io as sio
import os
import time

mode = 'fixed'
split = 'val'

E_list =[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for E in E_list:
    if mode == 'fixed':
        cfg.E = E
        print(E)

        from model.mpm_fixed import MPM
        model = MPM(cfg).cuda()
        model.E.require_grad = False
        result_dir = f'{split}_fixed_para_E{cfg.E}'

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    #### load data
    data_root = f'../../../data/exp_data/{split}_30/'
    init_data_list = os.listdir(data_root)
    for i in range(len(init_data_list)):
        init_data_name = init_data_list[i]
        init_data_path = data_root + init_data_name

        init_data = sio.loadmat(init_data_path, squeeze_me=True, struct_as_record=False)
        init_pos = torch.FloatTensor(init_data['init_pos'])
        init_vel = torch.FloatTensor(init_data['init_vel'])
        num = init_pos.size(0)
        C = torch.zeros((num, cfg.dim, cfg.dim))
        J = torch.ones((num))

        with torch.no_grad():
            start_time = time.time()
            model.set_input([init_pos, init_vel, C, J])
            model.n_substeps = 30 * cfg.steps
            vel_field = model.forward().detach().cpu().numpy()
            pos_seq = torch.stack(model.pos_seq).detach().cpu().numpy()
            vel_seq = torch.stack(model.vel_seq).detach().cpu().numpy()
            sio.savemat(os.path.join(result_dir, init_data_name),
                        {'pos': pos_seq, 'vel': vel_seq, 'vel_field': vel_field})



