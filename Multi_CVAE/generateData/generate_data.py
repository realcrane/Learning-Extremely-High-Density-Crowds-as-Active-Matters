import sys
sys.path.append('..')
from utils.utils import seed_everything
from config import config as cfg
import scipy.io as sio
import numpy as np
import torch
import time
import os


def init_model(pretrain=None):
    from model.mpm_alpha import MPM
    model = MPM(cfg).cuda()
    if not pretrain == None and os.path.exists(pretrain):
        state_data = torch.load(pretrain)
        best_loss = state_data['best_loss']
        print('loading the model: {}'.format(pretrain))
        print('best_loss:{}'.format(best_loss))
        state_dict = state_data['state_dict']
        model.load_state_dict(state_dict)
    else:
        assert os.path.exists(pretrain), "no pretained model to load"
    model.eval()
    return model


def mkdir(foler):
    os.makedirs(foler, exist_ok=True)


seed_everything(cfg.seed)

# define some parameters
split_list =['train', 'val', 'test']
cfg.K = 1
cfg.E = 200

pretrain = '../pretrained/exp1_lr0.0001_lambda0.9_r3_E_K_Alpha/mpm_best.pth'
model = init_model(pretrain)

for split in split_list:
    save_folder = f'../data/{split}'
    mkdir(save_folder)
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
            grid_v_out = model.forward()
            grid_v_out = grid_v_out.detach().view(30, cfg.n_grid[0], cfg.n_grid[1], 2).cpu().numpy()
            grid_v_in = torch.stack(model.grid_v_in_seq).detach().view(30, cfg.n_grid[0], cfg.n_grid[1],
                                                                       2).cpu().numpy()
            grid_m = torch.stack(model.grid_m_seq).detach().view(30, cfg.n_grid[0], cfg.n_grid[1], 1).cpu().numpy()
            alpha = torch.stack(model.alpha_seq).detach().view(30, cfg.n_grid[0], cfg.n_grid[1], 1).cpu().numpy()
            print(np.sum(np.abs(grid_v_in)))
            sio.savemat(os.path.join(save_folder, init_data_name),
                        {'grid_v_in': grid_v_in, 'grid_v_out': grid_v_out, 'grid_m': grid_m,
                         'alpha': alpha, 'vel_field_gt': init_data['vel_field_gt']})
