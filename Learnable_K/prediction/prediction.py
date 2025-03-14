import random
import sys

sys.path.append('..')
from config import config as cfg
import scipy.io as sio
import numpy as np
import torch
import time
import os

def seed_everything(seed=92):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed_everything(cfg.seed)
split = 'test'
cfg.w_ext = 0.03
cfg.E = 2000
cfg.K = 3
from model.mpm_K import MPM

model = MPM(cfg).cuda()
model_path = '../checkpoints_E2000_w_ext0.03_K3/exp1_lr0.0001_lambda0.9_r5/mpm_best.pth'
state_data = torch.load(model_path)
best_loss = state_data['best_loss']
st_epoch = state_data['epoch']
model.load_state_dict(state_data['state_dict'])
print(f'epoch {st_epoch} : loss {best_loss}')
model.eval()

save_root = f'{split}_w_ext_{cfg.w_ext}_E{cfg.E}_K{cfg.K}_420'
#### load data
data_root = f'../../data/exp_data/{split}_420/'
subfolder_list = os.listdir(data_root)
for subfolder in subfolder_list:
    print('start to process the folder {}, current E: {}, current weight: {}'.format(subfolder, cfg.E, cfg.w_ext))
    subfolder_path = os.path.join(data_root, subfolder)
    save_folder = os.path.join(save_root, subfolder)
    os.makedirs(save_folder, exist_ok=True)

    init_data_list = os.listdir(subfolder_path)
    init_data_list.sort()
    for i in range(len(init_data_list)):
        init_data_name = init_data_list[i]
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
            model.n_substeps = 420 * cfg.steps
            vel_field = model.forward().detach().cpu().numpy()
            pos_seq = torch.stack(model.all_pos_seq).detach().cpu().numpy()
            vel_seq = torch.stack(model.all_vel_seq).detach().cpu().numpy()
            sio.savemat(os.path.join(save_folder, init_data_name),
                        {'pos': pos_seq, 'vel': vel_seq, 'vel_field': vel_field})

    print('-----> {} files has been processed'.format(len(init_data_list)))
