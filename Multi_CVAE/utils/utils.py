from collections import OrderedDict
import numpy as np
import random
import torch
import time
import os


class Log(object):
    def __init__(self, save_folder):
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        name = time.strftime('train_log_%Y%m%d_%H%M%S')
        self.log_file = open(f"{save_folder}/{name}.txt", 'w')

    def log(self, s, flag=True, nl=True):
        self.log_file.write(s + '\n')
        self.log_file.flush()
        if flag:
            print(s, end='\n' if nl else '')

def mkdir(folder):
    os.makedirs(folder, exist_ok=True)

# Method to calculate the total number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def seed_everything(seed=92):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MetricList(object):
    """Computes and stores the average and current value"""

    def __init__(self, loss_name):
        self.loss_names = loss_name
        self.reset()

    def reset(self):
        """Reset the loss value"""
        self.avg = OrderedDict()
        self.sum = OrderedDict()
        self.val = OrderedDict()
        self.count = 0

        for name in self.loss_names:
            self.avg[name] = 0.0
            self.sum[name] = 0.0
            self.val[name] = 0.0

    def update(self, losses, n):
        self.count += n
        for name, value in zip(self.loss_names, losses):
            self.val[name] = value
            self.sum[name] += value * n
            self.avg[name] = self.sum[name] / self.count


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.avg = torch.tensor(0.)
        self.val = torch.tensor(0.)
        self.count = torch.tensor(0.)

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
