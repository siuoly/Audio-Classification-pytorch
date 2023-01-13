import torch
from torch.utils.data import DataLoader, Dataset
from config import config
import pandas as pd
import numpy as np
dataset_arg = config["dataset"]
process_arg = config["preprocessing"]


class BaseAudioDataset(Dataset):
    def __init__(self) :
        self.feature_folder = dataset_arg["feature_folder"]
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
        self.files = (self.feature_folder + self.meta["filename"]+".npy").values
        self.targets = self.meta["target"].values


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
    import glob
    filename = glob.glob(dataset_arg["feature_folder"] + "*")[0]
    return np.load(filename).shape[-2:]  # 取最後兩維度 freq,time dim


def get_a_dataset_sample():
    dataset = AudioDataset()
    return dataset[0]


def get_a_batch_samples():
    dataloader = DataLoader(AudioDataset(),
                            batch_size=config['batch_size'])
    x,y = next(iter(dataloader))
    return x,y

if __name__ == "__main__":
    dataset = BaseAudioDataset()
    print( dataset[0][0].shape, dataset[0][1])
    print( len(dataset))

    # dataset = AudioDataset(train=True)
    # print( len(dataset))

    # print( get_a_dataset_sample() )
    # print( AudioDataset(train=True)[0])
    # print( AudioDataset(train=False)[0])
    #
    # print( "get a batch samples")
    # print( get_a_batch_samples() )
