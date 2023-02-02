import torch
from torch.utils.data import DataLoader, Dataset
from config import config
from tool.data_path import (dataset_arg,
                            get_feature_paths)
import pandas as pd
import numpy as np
process_arg = config["preprocessing"]


class BaseAudioDataset(Dataset):
    def __init__(self) :
        self.feature_folder = config["feature_folder"]
        self.meta_file = dataset_arg["meta_file"]
        self.meta = pd.read_csv(self.meta_file)
        self.files, self.targets = None, None
        self.init_files_and_targets()

    def __len__(self):
        return len(self.meta.index)

    def __getitem__(self, index):
        spectrogram = np.load(self.files[index])
        spectrogram = torch.from_numpy(spectrogram)
        return spectrogram, self.targets[index]

    def init_files_and_targets(self):
        # self.files = (self.feature_folder + self.meta["filename"]+".npy").values
        self.files = get_feature_paths(self.meta)
        if config["dataset"] == "ESC":
            self.targets = self.meta["target"].values
        elif config["dataset"] == "Urbsound8k":
            self.targets = self.meta["classID"].values


class AudioDataset(BaseAudioDataset):
    def __init__(self, train=True, test_fold_num=1):
        super().__init__()
        meta = self.meta
        if train is True:
            self.meta = meta[meta["fold"] != test_fold_num]
        else:
            self.meta = meta[meta["fold"] == test_fold_num]
        self.init_files_and_targets()


def get_feature_shape():
    filename = get_feature_paths(pd.read_csv(dataset_arg["meta_file"]))[0]
    return np.load(filename).shape  # 取最後兩維度 freq,time dim


def get_a_dataset_sample():
    dataset = AudioDataset()
    return dataset[0]


def get_a_batch_samples():
    dataloader = DataLoader(AudioDataset(),
                            batch_size=config['batch_size'])
    x,y = next(iter(dataloader))
    return x,y

def test():
    train= AudioDataset(train=True)
    test= AudioDataset(train=False)
    print(test.files[:10])
    print(test.meta[:10])
    print(test.targets[:10])
    print(test.files[:10])
    print(test.targets[:10])
def test_number():
    train= AudioDataset(train=True)
    test= AudioDataset(train=False)
    print(len(test.files))
    print(len(train.files))

    
if __name__ == "__main__":
    # test_number()
    test()
    exit()

    dataset = BaseAudioDataset()
    # print( dataset[0][0].shape, dataset[10][1])
    print((dataset[2]),dataset[1])
    # print( len(dataset))

    dataset = AudioDataset(train=True)
    print((dataset[2]),dataset[1])

    # print( get_a_dataset_sample() )
    # print( AudioDataset(train=True)[0])
    # print( AudioDataset(train=False)[0])
    #
    # print( "get a batch samples")
    # print( get_a_batch_samples() )
