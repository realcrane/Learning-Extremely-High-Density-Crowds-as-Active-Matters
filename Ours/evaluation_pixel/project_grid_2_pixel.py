import os
import scipy.io as sio
from data_process import *

ti.init(arch=ti.gpu, random_seed=int(1992), device_memory_GB=1.5)
data_process = DataProcess()

root = '../resnet_sum_scale(1div50)_420/kld_weight0.1/2000/test'
folder_list = os.listdir(root)

for folder_name in folder_list:
    folder_path = os.path.join(root, folder_name)

    iters = os.listdir(folder_path)

    for iter in iters:
        iter_path = os.path.join(folder_path, iter)

        data_list = os.listdir(iter_path)
        data_list.sort()

        for data_name in data_list:
            data_path = os.path.join(iter_path, data_name)
            print(data_path)
            data = sio.loadmat(data_path, squeeze_me=True, struct_as_record=False)
            field_array = data['vel_field']

            pixel_image_list = data_process.obtain_pixel_image_list(field_array)

            save_folder = os.path.join('pred_420/', iter, folder_name)
            os.makedirs(save_folder, exist_ok=True)

            sio.savemat(os.path.join(save_folder, data_name), {'pred_optical_flow': pixel_image_list})


