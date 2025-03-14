import os
import scipy.io as sio
from data_process import *

ti.init(arch=ti.gpu, random_seed=int(1992), device_memory_GB=1.5)
data_process = DataProcess()

root = '../prediction/test_w_ext_0.03_E2000_K3_420/'
folder_list = os.listdir(root)

for folder_name in folder_list:
    folder_path = os.path.join(root, folder_name)
    data_list = os.listdir(folder_path)

    for data_name in data_list:
        data_path = os.path.join(root, folder_name, data_name)
        print(data_path)
        data = sio.loadmat(data_path, squeeze_me=True, struct_as_record=False)
        field_array = data['vel_field']

        pixel_image_list = data_process.obtain_pixel_image_list(field_array)

        save_folder = os.path.join('pred_420/', folder_name)
        os.makedirs(save_folder, exist_ok=True)

        sio.savemat(os.path.join(save_folder, data_name), {'pred_optical_flow': pixel_image_list})
