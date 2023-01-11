import sys  # add new path to search top level module

from torchaudio.transforms import TimeMasking, FrequencyMasking
import torch.nn as nn
from dataset import get_feature_shape
fdim, tdim = get_feature_shape()


transform = nn.Sequential(
        TimeMasking( time_mask_param=int(tdim*0.3)),
        FrequencyMasking(freq_mask_param=int(fdim*0.3))
                          )
