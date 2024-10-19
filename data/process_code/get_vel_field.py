from data_process import *
import os

ti.init(arch=ti.gpu, random_seed=int(1992), device_memory_GB=1.5)
begin_frame = 5
data_process = DataProcess()
flow_data_path = '../raw_data/optical_flow_mat/hajj_backup'
save_root = '../exp_data/hajj'  # grid 24 X 14
if not os.path.exists(save_root):
    os.mkdir(save_root)

for f_i in range(begin_frame, 310 + begin_frame):
    data_process.obtain_field(f_i, flow_data_path)
    vel_field = data_process.vel_field.to_numpy()
    print(np.sum(np.abs(vel_field)))
    # sio.savemat(save_root + f'/vel_field_{f_i}.mat', {'vel_field': vel_field, 'f_id': f_i})
