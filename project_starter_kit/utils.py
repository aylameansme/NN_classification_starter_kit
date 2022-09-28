# import
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
import numpy as np

#settings

def set_random(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = False  # If you want to set randomness, cudnn.benchmark = False
    cudnn.deterministic = True  # If you want to set randomness, cudnn.benchmark = True


    print(f"[✓] set random seed: {seed}\n[✓] randomness test : {torch.equal(torch.randint(1, 100, (3,)), torch.tensor([7, 96, 98]))}")

def device_check(CPU=None):
    if CPU is None and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    return device

#evaluation
def evaluation(mod, loader, critic):
    loss = 0
    pred = np.empty(shape=(0), dtype=np.float32)
    real = np.empty(shape=(0), dtype=np.float32)
    iter = 1

    for x_, y_ in loader :
        log = mod(x_.cuda())
        loss_ = critic(log, y_.cuda())
        _, pred_ = torch.max(F.softmax(log, -1).data, -1)

        loss = loss + loss_.cpu()
        pred = np.concatenate((pred, pred_.cpu()), axis=-1)
        real = np.concatenate((real, y_), axis=-1)

        iter = iter + 1

        del x_, y_, loss_, log, pred_

    acc = accuracy_score(real, pred)
    return acc, loss

#predicted label exporter




#