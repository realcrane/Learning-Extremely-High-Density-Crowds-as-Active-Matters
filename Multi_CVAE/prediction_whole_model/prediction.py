import sys

sys.path.append('..')
from config import config as cfg
from utils.utils import seed_everything
import scipy.io as sio
import collections
import torch
import time
import os

# define the seed for all random operation
seed_everything(cfg.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

cfg.K = 1
cfg.E = 200
cfg.n_decoder = 4

from model.mpm import MPM

model = MPM(cfg).cuda()
pretrain = f'../pretrained/exp1_lr0.0001_lambda0.9_r3_E_K_Alpha/mpm_best.pth'
if os.path.exists(pretrain):
    print('log the pretrained model : {}'.format(pretrain))
    state_data = torch.load(pretrain)
    state_dict = state_data['state_dict']
    best_loss = state_data['best_loss']
    print(best_loss)
    state_dict_new = collections.OrderedDict()
    for ind, key in enumerate(state_dict):
        if 'get_super_paras_E' in key:
            key_new = key.replace('get_super_paras_E.', '')
            state_dict_new[key_new] = state_dict[key]
    model.get_super_paras_E.load_state_dict(state_dict_new)
    print('loaded the get_super_paras_E')

    state_dict_new = collections.OrderedDict()
    for ind, key in enumerate(state_dict):
        if 'get_super_paras_K' in key:
            key_new = key.replace('get_super_paras_K.', '')
            state_dict_new[key_new] = state_dict[key]
    model.get_super_paras_K.load_state_dict(state_dict_new)
    print('loaded the get_super_paras_K')

    state_dict_new = collections.OrderedDict()
    for ind, key in enumerate(state_dict):
        if 'get_super_paras_alpha' in key:
            key_new = key.replace('get_super_paras_alpha.', '')
            state_dict_new[key_new] = state_dict[key]
    model.get_super_paras_alpha.load_state_dict(state_dict_new)
    print('loaded the get_super_paras_alpha')

kld_weight = 0.1
split_list = ['val', 'test']
epoch_list = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]

for split in split_list:
    for epoch in epoch_list:
        pretrain = f'../checkpoints/resnet_sum/exp1_kld_weight{kld_weight}/mpm_{epoch}.pth'
        print('log the pretrained model : {}'.format(pretrain))
        state_data = torch.load(pretrain)
        state_dict = state_data['state_dict']
        model.get_active_force2.load_state_dict(state_dict)
        print('loaded the get_active_force2')
        model.eval()

        for n in range(10):
            save_folder = f'resnet_sum3/kld_weight{kld_weight}/{epoch}/{split}/{n}'
            os.makedirs(save_folder, exist_ok=True)
            print(save_folder)

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
                    sio.savemat(os.path.join(save_folder, init_data_name),
                                {'pos': pos_seq, 'vel': vel_seq, 'vel_field': vel_field})
