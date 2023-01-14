from config import config
from pathlib import Path
import pandas as pd
from typing import List

process_arg = config['preprocessing']
if config['dataset'] == "ESC":
    dataset_arg = config['ESC_dataset']
elif config['dataset'] == "Urbsound8k":
    dataset_arg = config['Urbsound8k_dataset']


def get_audio_filenames() -> List[str]:  # 按照 metafile 檔案順序
    if config['dataset'] == "ESC":
        return pd.read_csv(dataset_arg['meta_file'])["filename"].values
    elif config["dataset"] == "Urbsound8k":
        meta = pd.read_csv(dataset_arg['meta_file'])
        return ("fold" + meta["fold"].astype(str) + "/" + meta["slice_file_name"]).values


def get_source_audio_paths() -> List[Path]:  # 按照metafile順序
    if config['dataset'] in ("ESC", "Urbsound8k"):
        return [Path(dataset_arg["audio_folder"]) / file
                for file in get_audio_filenames()]



def get_resample_audio_paths() -> List[Path]:
    resample_folder = Path(process_arg['resample_audio_folder'])
    resample_folder = resample_folder / f"{config['dataset']}_sr{process_arg['new_sr']}"
    resample_audio_paths = [resample_folder/path for path in get_audio_filenames()]
    return resample_audio_paths


def get_feature_paths(meta_df) -> List[Path]:
    if config['dataset'] == "ESC":  # 按照meta檔案裡的順序輸出。
        filenames = meta_df["filename"].values
    elif config['dataset'] == "Urbsound8k":
        filenames = ("fold" + meta_df["fold"].astype(str) +
                     "/" + meta_df["slice_file_name"]).values
    feature_paths = [Path(config["feature_folder"])/(file+'.npy')
                     for file in filenames]
    return feature_paths



def main():
    from pprint import pprint
    pprint(get_audio_filenames()[:5])
    pprint(get_source_audio_paths()[:5])
    pprint(get_resample_audio_paths()[:5])
    pprint(get_feature_paths(pd.read_csv(dataset_arg['meta_file']))[:5])


if __name__ == "__main__":
    main()
