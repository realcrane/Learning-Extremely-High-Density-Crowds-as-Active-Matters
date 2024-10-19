import torch
import numpy
import time
import os

class Log(object):
    def __init__(self, args):
        if not os.path.exists(args.save_root):
            os.mkdir(args.save_root)
        name = time.strftime('train_log_%Y%m%d_%H%M%S')
        self.log_file = open(f"{args.save_root}/{name}.txt", 'w')

    def log(self, s, flag=True, nl=True):
        self.log_file.write(s + '\n')
        self.log_file.flush()
        if flag:
            print(s, end='\n' if nl else '')


def random_seed(seed):
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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