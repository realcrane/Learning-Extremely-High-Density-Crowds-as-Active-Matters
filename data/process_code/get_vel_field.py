import random

from data_process import *
import os

ti.init(arch=ti.gpu, random_seed=int(1992), device_memory_GB=1.5)

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

# grid 24 X 14
data_process = DataProcess()
save_root = '../exp_data/vel_field'
flow_root = '../../../dataset/optical_flow_mat'
folder_list = list(MC_start_Ind.keys())

if not os.path.exists('random_index.mat'):
    random.shuffle(folder_list)
    sio.savemat('random_index.mat', {'random_index': np.array(folder_list)})
else:
    random_index = sio.loadmat('random_index.mat', squeeze_me=True, struct_as_record=False)['random_index']

    for folder_name in random_index[:8]:
        print('start to process {}'.format(folder_name))
        folder_path = os.path.join(flow_root, folder_name)
        flow_data_list = os.listdir(folder_path)
        flow_data_list.sort()

        begin_frame = MC_start_Ind.get(folder_name)
        end_frame = begin_frame + 1000
        print('the begin frame is {}'.format(begin_frame))

        save_folder = os.path.join(save_root, folder_name)
        os.makedirs(save_folder, exist_ok=True)

        for f_i in range(begin_frame, end_frame):
            data_process.obtain_field(f_i, folder_path)
            vel_field = data_process.vel_field.to_numpy()
            sio.savemat(save_folder + f'/vel_field_{str(f_i).zfill(4)}.mat', {'vel_field': vel_field, 'f_id': f_i})
