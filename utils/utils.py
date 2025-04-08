import os
import datetime
import random
import numpy as np

import torch
from torch.optim.lr_scheduler import _LRScheduler

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def create_file(path, filename, write_line=None, exist_create_flag=True):
    create_dir(path)
    filename = os.path.join(path, filename)

    if filename != None:
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if not os.path.exists(filename):
            with open(filename, "a") as myfile:
                print("create new file: %s" % filename)
        elif exist_create_flag:
            new_file_name = filename + ".bak-%s" % nowTime
            os.system('mv %s %s' % (filename, new_file_name))

        if write_line != None:
            with open(filename, "a") as myfile:
                myfile.write(write_line + '\n')

    return filename
