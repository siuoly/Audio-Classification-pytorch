#!/bin/python

from config import config
from pprint import pprint
import glob
from pathlib import Path
import os
import soundfile as sf
import librosa as ra
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from tool.data_path import (dataset_arg,
                            get_audio_filenames,
                            get_source_audio_paths,
                            get_resample_audio_paths,
                            get_feature_paths)
import pandas as pd
process_arg = config['preprocessing']


audio_size = process_arg["new_sr"] * process_arg["times"]
def resample_fixlength_and_save(file, new_path):
    if file.name != new_path.name:
        raise RuntimeError("resmaple file", file.name, new_path.name)
    new_path.parent.mkdir(exist_ok=True)
    wav, _ = ra.load(file, sr=process_arg['new_sr'])
    if wav.shape[-1] != audio_size:
        wav = ra.util.fix_length(wav, size=audio_size)
    sf.write(new_path, wav, process_arg['new_sr'])

def create_resampled_folder(resample_audio_folder):
    print("create resample audio folder ", resample_audio_folder)
    resample_audio_folder.mkdir(parents=True)
    audio_paths = get_source_audio_paths()
    new_audio_paths = get_resample_audio_paths()

    arguments = zip(audio_paths, new_audio_paths)
    Parallel(n_jobs=12)(delayed(resample_fixlength_and_save)(file, new_path)
                        for file, new_path in tqdm(arguments, total=len(audio_paths)))


def get_an_filename():
    filenames = get_source_audio_paths()
    file = filenames[1]
    return file


# preprocessing : resample, melspectrogram
def show_preprocessing_message():
    file = get_an_filename()
    wav, sr = ra.load(file, sr=None)
    new_wav = ra.resample(wav, orig_sr=sr,
                          target_sr=process_arg['new_sr'])
    mel = ra.feature.melspectrogram(y=new_wav, sr=process_arg['new_sr'],
                                    **process_arg['mel_arg'])
    print("audio folder", dataset_arg["audio_folder"])
    print(f"wav shape :{wav.shape}, sr: {sr}")
    print(f"new wav shape:{new_wav.shape}, sr: {process_arg['new_sr']} ")
    print("mel argument:")
    pprint(process_arg['mel_arg'])
    print(f"spectrogram feature shape:{mel.shape}")


def get_mel_of_file(file):
    wav, sr = ra.load(file, sr=process_arg['new_sr'])
    mel = ra.feature.melspectrogram(y=wav, sr=sr, **process_arg['mel_arg'])
    if process_arg["dbscale"] is True:
        mel = ra.power_to_db(mel)
    return mel


def get_stft_and_phase_of_file(file):
    wav, sr = ra.load(file, sr=process_arg['new_sr'])
    stft = ra.stft(y=wav, **process_arg['stft_arg'])
    if config["in_channel"] == 2:
        phase = np.angle(stft)
        return np.stack([np.abs(stft)**2, phase])
    else:
        return np.abs(stft)**2

def processing_a_audio(file):
    if process_arg["feature"] == "mel":
        feature = get_mel_of_file(file)
    elif process_arg["feature"] == "stft":
        feature = get_stft_and_phase_of_file(file)
    else:
        raise RuntimeError(f"Unknown processing feature {process_arg['feature']}")
    if process_arg["normalize"]:
        feature = normalized_data(feature)
    return feature


def save_feature(save_path: Path, feature: np.ndarray):
    save_path.parent.mkdir(exist_ok=True)
    np.save(save_path, feature)
    return save_path


def processing_and_save_a_file(file: Path, save_path: Path):
    if file.stem != save_path.stem[:-4]:  # check xx.wav , xxx.wav.npy
        raise RuntimeError(f"filename is not save filename", file.stem, save_path.stem)
    feature = processing_a_audio(file)
    save_feature(save_path, feature)


def make_data_folder():
    data_folder = Path(config['feature_folder'])
    if data_folder.exists():
        print( f"feature folder {data_folder} exist." )
        return False
    else:
        data_folder.mkdir(parents=True, exist_ok=True)
        print( f"create data folder:{data_folder}" )
    return True


def normalized_data(x):
    axis = tuple(np.arange(x.ndim)[-2:])  # 取最後兩維度(0,1) or (0,1,2)
    return (x-x.mean(axis=axis, keepdims=True)) / \
        x.std(axis=axis, keepdims=True)

def main(show=True):
    meta = dataset_arg['meta_file']
    meta = pd.read_csv(meta)
    if show:
        show_preprocessing_message()

    resample_audio_folder = Path(process_arg['resample_audio_folder'])
    resample_audio_folder = resample_audio_folder/f"{config['dataset']}_sr{process_arg['new_sr']}"
    if not resample_audio_folder.exists():
        create_resampled_folder(resample_audio_folder)
    else:
        print("resample audio folder ", resample_audio_folder, "exists")

    dataset_arg['audio_folder'] = str(resample_audio_folder) +'/'

    if not make_data_folder():
        return  # 跳出 preprocessing
    audio_paths = get_source_audio_paths()
    arguments = zip(audio_paths, get_feature_paths(meta))

    Parallel(n_jobs=12)(
            delayed(processing_and_save_a_file)(file, new_path)
            for file, new_path in tqdm(arguments, total=len(audio_paths)))

if __name__ == "__main__":
    main(show=True)
