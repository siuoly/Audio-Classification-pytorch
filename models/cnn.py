import sys  # add new path to search top level module
import torch
from dataset import (AudioDataset,
                     get_feature_shape,
                     get_a_dataset_sample,
                     get_a_batch_samples)
import torch.nn as nn
from torchinfo import summary
# from models.augment import transform
from torchaudio.transforms import TimeMasking, FrequencyMasking
from config import config


def conv_block(inch,outch,ker,stride,padding,poolker):
    conv = nn.Sequential(
            nn.Conv2d(
                in_channels=inch,
                out_channels=outch,
                kernel_size=ker,
                stride=stride,
                padding=padding
                ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=poolker))
    return conv


def linear_block(in_layer, out_layer):
    return nn.Sequential( nn.Dropout(0.5), nn.Linear(in_layer, out_layer))



class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        model_arg = config["model"]["arg"]
        self.x_shape = get_feature_shape()  # input data shpae time, freq dimension
        self.conv = nn.Sequential()
        self.conv.append(conv_block(self.x_shape[0], 16, 3, 1, 2, 2))
        self.conv.append(conv_block(16, 32, 3, 1, 2, 2))
        self.conv.append(conv_block(32, 64, 3, 1, 2, 2))
        self.conv.append(conv_block(64, 128, 3, 1, 2, 2))
        self.conv.append(conv_block(128, 128, 3, 1, 2, 2))
        self.flatten = nn.Flatten()
        # linear_arg = model_arg[self.compute_linear_layer_dim(),
        #                     *model_arg["linear"], config["num_class"],]
        # self.linear = nn.Sequential()
        # for in_layer, out_layer in zip(linear_arg, linear_arg[1:]):
        #     self.linear.append(linear_block(in_layer, out_layer))
        self.linear = nn.Sequential(
                nn.Dropout(0.7),
                nn.Linear(self.compute_linear_layer_dim(), model_arg["linear"] ),
                nn.Dropout(0.7),
                nn.ReLU(),
                nn.Linear( model_arg["linear"],config["num_class"]),
                )
        self.softmax = nn.Softmax(dim=1)
        self.transform = nn.Sequential(
        FrequencyMasking(freq_mask_param=int(self.x_shape[1]*0.5)),
        TimeMasking( time_mask_param=int(self.x_shape[2]*0.5))
                          )


    def forward(self, input_data):
        if len(input_data.shape) != 4:  # (N,f,t) -->(N,1,f,t)
            input_data = input_data.unsqueeze(1)
        if self.training and config["transform"]:  # 手動關閉, 他不會自動檢查 model.eval()
            input_data = self.transform(input_data)
        x = self.conv(input_data)
        x = self.flatten(x)
        logit = self.linear(x)
        return logit

    def compute_linear_layer_dim(self):
        x = torch.ones((config["batch_size"], *self.x_shape))
        print(x.shape)
        x = self.conv(x)
        x = self.flatten(x)
        return x.shape[-1]

    def save(self):
        save_path = config["exp_folder"]+"model.pt"
        torch.save(self.state_dict(), save_path)

    def load(self):
        save_path = config["exp_folder"]+"model.pt"
        self.load_state_dict(torch.load(save_path))

    def summary(self):
        summary(self.cuda(),
                input_size=(config["batch_size"], *self.x_shape))

if __name__ == "__main__":
    cnn = CNNNetwork()
    x, y = get_a_batch_samples()
    cnn.summary()
    # summary(cnn.cuda(),input_size=(64,1,fdim,tdim))
    # cnn(x)

    # cnn.save()
    # cnn2 = CNNNetwork()
    # cnn2.load()
    # for (a,b),(c,d) in zip(cnn.state_dict().items(),cnn2.state_dict().items()):
    #     torch.testing.assert_close(b,d)

