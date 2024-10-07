import os
import scipy.io as sio
from config.config import *

## configuration
begin_frame = 185
end_frame = 245
length = 30
step = 5

data_path = '../exp_data/hajj'
save_path = f'../exp_data/val_{length}'

if not os.path.exists(save_path):
    os.mkdir(save_path)

def is_valid(y, x):
    flag = False
    if (y - 240) ** 2 / 160 ** 2 + (x - 130) ** 2 / 90 ** 2 <= 1:
        if not (y > 220 and y < 260 and x > 80 and x < 120):
            flag = True
    return flag


def sample_position():
    points = []
    p_radius = 5
    for i in np.arange(0.501 * p_radius, 280, 1.502 * p_radius):
        for j in np.arange(0.501 * p_radius, 480, 1.502 * p_radius):
            x = i + (np.random.uniform(0, 1, 1) - 0.5) * 0.001 * p_radius
            y = j + (np.random.uniform(0, 1, 1) - 0.5) * 0.001 * p_radius
            if is_valid(y, x):
                points.append([y, x])

    sampled_pos = np.array(points).squeeze(2) + pixel_bound
    sampled_pos = sampled_pos
    return sampled_pos


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


for f_i in range(begin_frame, end_frame - length, step + 1):
    init_pos = sample_position()

    vel_field_path = f'{data_path}/vel_field_{f_i}.mat'
    vel_field = sio.loadmat(vel_field_path, squeeze_me=True, struct_as_record=False)['vel_field']
    init_vel = sample_velocity(init_pos, vel_field)

    vel_field_gt = []
    i = f_i
    while i < f_i + length:
        i = i + 1
        vel_field_path = f'{data_path}/vel_field_{i}.mat'
        vel_field = sio.loadmat(vel_field_path, squeeze_me=True, struct_as_record=False)['vel_field']
        vel_field_gt.append(vel_field)

    vel_field_gt = np.stack(vel_field_gt)

    sio.savemat(f'{save_path}/{f_i}.mat',
                {'init_pos': init_pos, 'init_vel': init_vel, 'vel_field_gt': vel_field_gt})


f_i = end_frame - length - 1
init_pos = sample_position()

vel_field_path = f'{data_path}/vel_field_{f_i}.mat'
vel_field = sio.loadmat(vel_field_path, squeeze_me=True, struct_as_record=False)['vel_field']
init_vel = sample_velocity(init_pos, vel_field)

vel_field_gt = []
i = f_i
while i < f_i + length:
    i = i + 1
    vel_field_path = f'{data_path}/vel_field_{i}.mat'
    vel_field = sio.loadmat(vel_field_path, squeeze_me=True, struct_as_record=False)['vel_field']
    vel_field_gt.append(vel_field)

vel_field_gt = np.stack(vel_field_gt)

sio.savemat(f'{save_path}/{f_i}.mat',
            {'init_pos': init_pos, 'init_vel': init_vel, 'vel_field_gt': vel_field_gt})