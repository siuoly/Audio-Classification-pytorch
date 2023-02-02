"""
# TODO
1. 寫好model 確定可以跑
2. 建立超級何model，其會根據 config.yaml 選擇model。
3. 確定可以訓練。

......
查看 paper的寫法
"""

import sys  # add new path to search top level module
import torch
from dataset import (AudioDataset,
                     get_feature_shape,
                     get_a_dataset_sample,
                     get_a_batch_samples)
import torch.nn as nn
from torchinfo import summary
# from models.augment import transform
# from torchaudio.transforms import TimeMasking, FrequencyMasking
from config import config

class EmbeddingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_shape = get_feature_shape()  # input data shpae time, freq dimension
        self.conv = nn.Sequential()
        self.conv.append(
                nn.Sequential(nn.Conv2d(in_channels=self.x_shape[0],out_channels=32,
                                        kernel_size=(3,3)),
                              nn.ConstantPad2d(1,0),
                              nn.BatchNorm2d(32), nn.ELU(),
                              nn.Conv2d(in_channels=32,out_channels=64, kernel_size=(3,3)),
                              nn.ConstantPad2d(1,0),
                              nn.BatchNorm2d(64), nn.ELU(),
                              nn.MaxPool2d((8, 2)) )
                )
        self.conv.append(
                nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128, kernel_size=(3,3)),
                              nn.BatchNorm2d(128), nn.ELU(),
                              nn.ConstantPad2d(1,0),
                              nn.Conv2d(in_channels=128,out_channels=256, kernel_size=(3,3)),
                              nn.ConstantPad2d(1,0),
                              nn.BatchNorm2d(256), nn.ELU(),
                              nn.MaxPool2d((8, 4)))
                )
        self.conv.append(
                nn.Sequential(nn.Conv2d(in_channels=256,out_channels=512, kernel_size=(2,2)),
                              nn.BatchNorm2d(512), nn.ELU())
                )

    def forward(self, input_data) -> torch.FloatTensor:  
        # 比照paper: (3,128,431) --> (512,1,52)
        x = self.conv(input_data)
        return x

class AttentionUpPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Sequential(
                nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(1,1)),
                nn.Sigmoid(),
                )
    def forward(self,input_data) -> torch.FloatTensor:
        x = self.attention(input_data)
        attention = nn.functional.normalize(x,p=1.0,dim=3)  # (N,C,1,T) norm on T
        return attention


class MCTA(nn.Module):
    def __init__(self):
        super().__init__()
        model_arg = config["model"]["arg"]
        self.x_shape = get_feature_shape()  # input data shpae time, freq dimension
        assert self.x_shape[0] == 3, "input channel must be 3"
        self.embedding = EmbeddingBlock()
        self.attenUp = AttentionUpPath()
        self.attenDown = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(1,1))
        self.activation = nn.Sequential(nn.BatchNorm2d(num_features=512), nn.ReLU())
        self.fc = nn.Sequential(
                nn.Dropout(p=.3),
                nn.Linear(in_features=512,out_features=config["num_class"])
                )

    def forward(self, input_data:torch.FloatTensor ):  
        if len(input_data.shape) != 4:  # (N,f,t) -->(N,1,f,t)
            input_data = input_data.unsqueeze(1)
        x = self.embedding(input_data)  # (N,C,1,t)
        a = self.attenUp(x)  # (N,C,1,t)
        xl = self.attenDown(x)  # (N,C,1,t)
        xa = self.activation(xl * a)  #(N,C,1,t)
        h = xa.sum(dim=-1).squeeze(dim=-1)  # (N,C,1) --> (N,C)
        pred = self.fc(h)
        return pred

    def save(self):
        save_path = config["exp_folder"]+"model.pt"
        torch.save(self.state_dict(), save_path)

    def load(self):
        save_path = config["exp_folder"]+"model.pt"
        self.load_state_dict(torch.load(save_path))

    def summary(self):
        summary(self.cuda(),
                input_size=(config["batch_size"], *self.x_shape))

def test_embblock():
    x,y = get_a_batch_samples()
    m = EmbeddingBlock()
    print(m(x).shape)
def test_MCTA():
    x,y = get_a_batch_samples()
    m = MCTA()
    print(m(x).shape)

def test_summary():
    m = MCTA()
    m.summary()

if __name__ == "__main__":
    # test_embblock()
    # test_MCTA()
    test_summary()

    
