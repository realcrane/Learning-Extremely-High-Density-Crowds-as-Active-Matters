import os
import random

import numpy as np
import scipy.io as sio
from config.config import *


# sample velocity from the velocity field
def sample_velocity(pos, vel_field):
    vel = np.zeros(pos.shape).astype('float')
    for p in range(pos.shape[0]):
        Xp = pos[p, :] * inv_dx
        base = (Xp - 0.5).astype(int)
        fx = Xp - base

        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # Bspline
        new_v = np.zeros(2)
        for i, j in np.ndindex(3, 3):
            offset = np.array([i, j])
            weight = w[i][0] * w[j][1]
            g_v = vel_field[tuple(base + offset)]
            new_v += weight * g_v
        vel[p, :] = new_v
    return vel


# load trajectories
def load_trajectories(path):
    traj_list = os.listdir(path)
    file_num = len(traj_list)
    data = np.zeros((file_num, 5000, 5))
    data[:, :, 2] = -1
    for i in range(file_num):
        traj_name = traj_list[i]
        traj_path = os.path.join(path, traj_name)
        temp = sio.loadmat(traj_path, squeeze_me=True, struct_as_record=False)
        ind = temp['f_id'].astype('int')
        pos = temp['pos']
        data[i, ind, :2] = pos / 10
        data[i, ind, 2] = ind
    return data


MC_start_Ind = {'20131026_100932': 95,
                '20131026_101159': 80,
                '20131026_101431': 80,
                '20131026_101705': 100,
                '20131026_102039': 115,
                '20131026_102628': 105,
                '20131026_102859': 80,
                '20131026_103140': 130,
                '20131026_103439': 70,
                }

# define some paremeters
length = 60
step = 30
vel_field_root = '../exp_data/vel_field'
traj_root = '../../../dataset/projected_trajectories'
traj_folder_list = os.listdir(traj_root)

# load the split of train, val, test
MC_list = sio.loadmat('../process_code/random_index.mat', squeeze_me=True, struct_as_record=False)['random_index']

split_list = ['train']
for split in split_list:
    folder_list = []
    if split == 'train':
        folder_list.extend(MC_list[:4])
        folder_list.sort()
    elif split == 'val':
        folder_list.extend(MC_list[4:6])
    else:
        folder_list.extend(MC_list[6:8])

    print(f'-------> start to process {split} data')
    # start to process data
    for folder in folder_list:
        print('currrent folder is {}'.format(folder))
        field_data_list = os.listdir(os.path.join(vel_field_root, folder))
        field_data_list.sort()

        save_folder = f'../exp_data/{split}_{length}/{folder}'

        os.makedirs(save_folder, exist_ok=True)
        begin_frame = int(MC_start_Ind.get(folder))
        end_frame = begin_frame + 1000

        # load all trajectories
        for traj_folder in traj_folder_list:
            if folder in traj_folder:
                traj_folder_path = os.path.join(traj_root, traj_folder)
                trajectories = load_trajectories(traj_folder_path)
                break

        # sample velocity
        for ind in range(trajectories.shape[0]):  # for all trajectory
            trajectory = trajectories[ind, :, :]
            mask = trajectory[:, 2] >= 0
            f_index = trajectory[mask, 2].astype('int')
            for f_i in f_index:
                if f_i >= begin_frame and f_i < end_frame:
                    pos = trajectory[f_i:f_i + 1, :2]
                    vel_field_path = os.path.join(vel_field_root, folder, f'vel_field_{str(f_i).zfill(4)}.mat')
                    vel_field = sio.loadmat(vel_field_path, squeeze_me=True, struct_as_record=False)['vel_field']
                    sampled_vel = sample_velocity(pos, vel_field)
                    trajectories[ind, f_i, 3:] = sampled_vel

        for f_i in range(begin_frame, end_frame - length, step):
            init_pos = []
            init_vel = []
            init_ind = []

            traj_clips = trajectories[:, f_i: f_i + length, :]
            for ind in range(trajectories.shape[0]):
                traj_temp = traj_clips[ind, :, :]
                f_ind_list = np.where(traj_temp[:, 2] > 0)[0]
                if len(f_ind_list) > 0:
                    init_pos.append(traj_temp[f_ind_list[0], :2])
                    init_ind.append(f_ind_list[0])
                    init_vel.append(traj_temp[f_ind_list[0], 3:])

            init_pos = np.stack(init_pos)
            init_vel = np.stack(init_vel)
            init_ind = np.stack(init_ind)

            vel_field_gt = []
            for n in range(length):
                vel_field_path = os.path.join(vel_field_root, folder, f'vel_field_{str(f_i + n + 1).zfill(4)}.mat')
                vel_field = sio.loadmat(vel_field_path, squeeze_me=True, struct_as_record=False)['vel_field']
                vel_field_gt.append(vel_field)

            vel_field_gt = np.stack(vel_field_gt)
            save_path = os.path.join(save_folder, f'vel_field_{str(f_i).zfill(4)}.mat')
            sio.savemat(save_path,
                        {'init_pos': init_pos, 'init_vel': init_vel, 'init_ind': init_ind,
                         'vel_field_gt': vel_field_gt})
            print(f'save file: {save_path}')

        if f_i == (end_frame - length - 1):
            continue
        else:
            f_i = (end_frame - length - 1)
            init_pos = []
            init_vel = []
            init_ind = []

            traj_clips = trajectories[:, f_i: f_i + length, :]
            for ind in range(trajectories.shape[0]):
                traj_temp = traj_clips[ind, :, :]
                f_ind_list = np.where(traj_temp[:, 2] > 0)[0]
                if len(f_ind_list) > 0:
                    init_pos.append(traj_temp[f_ind_list[0], :2])
                    init_ind.append(f_ind_list[0])
                    init_vel.append(traj_temp[f_ind_list[0], 3:])

            init_pos = np.stack(init_pos)
            init_vel = np.stack(init_vel)
            init_ind = np.stack(init_ind)

            vel_field_gt = []
            for n in range(length):
                vel_field_path = os.path.join(vel_field_root, folder, f'vel_field_{str(f_i + n + 1).zfill(4)}.mat')
                vel_field = sio.loadmat(vel_field_path, squeeze_me=True, struct_as_record=False)['vel_field']
                vel_field_gt.append(vel_field)

            vel_field_gt = np.stack(vel_field_gt)
            save_path = os.path.join(save_folder, f'vel_field_{str(f_i).zfill(4)}.mat')
            sio.savemat(save_path,
                        {'init_pos': init_pos, 'init_vel': init_vel, 'init_ind': init_ind,
                         'vel_field_gt': vel_field_gt})
            print(f'save file: {save_path}')
