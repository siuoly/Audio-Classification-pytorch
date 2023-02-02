import torch, numpy as np
from torchaudio.transforms import TimeMasking, FrequencyMasking
import torch.nn as nn
from dataset import get_feature_shape
import torch.nn.functional as F

def get_transform():
    _, fdim, tdim = get_feature_shape()
    transform = nn.Sequential(
            TimeMasking( time_mask_param=int(tdim*0.3)),
            FrequencyMasking(freq_mask_param=int(fdim*0.3))
                              )
    return transform


# https://github.com/fschmid56/EfficientAT/blob/1ade11d4d29469602af6e66c4e70a581f1ed52d0/ex_esc50.py
class Mixup (object):
    def __init__(self, batch_size, alpha):
        self.random_indices = torch.randperm(batch_size)
        lamb = np.random.beta(alpha, alpha, size=batch_size).astype(np.float32)
        lamb = np.concatenate([lamb[:, None], 1 - lamb[:, None]], 1).max(1)
        self.lamb = torch.FloatTensor(lamb).cuda()

    def __call__(self, x):
        return self.get_mixup_x(x)

    def get_mixup_x(self,x):
        if x.ndim == 4:
            lamb  = self.lamb[...,None,None,None].to(x.device)
        elif x.ndim == 3:
            lamb = self.lamb[...,None,None].to(x.device)
        x = x * lamb + x[self.random_indices] * (1. - lamb)
        return x

    def get_loss(self, pred,y):
        samples_loss = (F.cross_entropy(pred, y, reduction="none") * self.lamb +
                        F.cross_entropy(pred, y[self.random_indices], reduction="none") *
                        (1. - self.lamb))
        return samples_loss.mean()
