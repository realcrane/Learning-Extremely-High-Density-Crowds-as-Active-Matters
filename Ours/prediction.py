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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cfg.K = 3
cfg.E = 2000
cfg.w_ext = 0.03
cfg.n_decoder = 4

from model.mpm import MPM

model = MPM(cfg).cuda()
pretrain_path = f'../LearnableAlpha/checkpoints_E2000_w_ext0.03_K3_alpha/exp1_lr0.0001_lambda0.9_r5/mpm_best.pth'
if os.path.exists(pretrain_path):
    print('log the pretrained model : {}'.format(pretrain_path))
    state_data = torch.load(pretrain_path)
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
split_list = ['test']
epoch_list = [2000]

for split in split_list:
    for epoch in epoch_list:
        pretrain_path = f'../Multi_CVAE/checkpoints4/resnet_sum_bs4_latent64_small/exp1_kld_weight{kld_weight}/mpm_{epoch}.pth'
        print('log the pretrained model : {}'.format(pretrain_path))
        state_data = torch.load(pretrain_path)
        state_dict = state_data['state_dict']
        model.get_active_force2.load_state_dict(state_dict)
        print('loaded the get_active_force2')
        model.eval()

        root = f'../../data/exp_data/{split}_420/'
        subfolder_list = os.listdir(root)
        for subfolder in subfolder_list:
            print(subfolder)
            subfolder_path = os.path.join(root, subfolder)
            data_list = os.listdir(subfolder_path)
            data_list.sort()

            for n in range(10):
                print(n)
                save_folder = f'resnet_sum_scale(1div50)/kld_weight{kld_weight}/{epoch}/{split}/{subfolder}/{n}'
                os.makedirs(save_folder, exist_ok=True)
                print(save_folder)

                #### load data
                for i in range(len(data_list)):
                    data_name = data_list[i]
                    print(data_name)
                    data_path = os.path.join(subfolder_path, data_name)

                    init_data = sio.loadmat(data_path, squeeze_me=True, struct_as_record=False)
                    init_pos = torch.FloatTensor(init_data['init_pos'])
                    init_vel = torch.FloatTensor(init_data['init_vel'])
                    init_ind = torch.FloatTensor(init_data['init_ind'])
                    num = init_pos.size(0)
                    C = torch.zeros((num, cfg.dim, cfg.dim))
                    J = torch.ones((num))

                    with torch.no_grad():
                        start_time = time.time()
                        model.set_input([init_pos, init_vel, init_ind, C, J])
                        model.n_substeps = 420 * cfg.steps
                        vel_field = model.forward().detach().cpu().numpy()
                        pos_seq = torch.stack(model.all_pos_seq).detach().cpu().numpy()
                        vel_seq = torch.stack(model.all_vel_seq).detach().cpu().numpy()
                        sio.savemat(os.path.join(save_folder, data_name),
                                    {'pos': pos_seq, 'vel': vel_seq, 'vel_field': vel_field})

