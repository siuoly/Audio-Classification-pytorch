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
import pandas as pd
process_arg = config['preprocessing']
new_sr = process_arg['new_sr']
dataset_arg = config['dataset']


def create_resampled_folder(resample_audio_folder):
    print("create resample audio folder ", resample_audio_folder)
    resample_audio_folder.mkdir(parents=True)
    audio_paths = list(Path(dataset_arg['audio_folder']).glob("*.wav"))
    new_audio_paths = [resample_audio_folder/path.name for path in audio_paths]

    def resample_and_save(file, new_path):
        wav, _ = ra.load(file, sr=process_arg['new_sr'])
        sf.write(new_path, wav, process_arg['new_sr'])
    arguments = zip(audio_paths, new_audio_paths)
    Parallel(n_jobs=12)(delayed(resample_and_save)(file, new_path)
                        for file, new_path in tqdm(arguments, total=len(audio_paths)))


def get_an_filename():
    filenames = glob.glob(dataset_arg['audio_folder'] + '/*.wav')
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


def save_feature(file, feature):
    save_path = Path( config['dataset']['feature_folder'] ) / (Path(file).name + ".npy")
    np.save(save_path, feature)
    return save_path


def processing_and_save_a_file(file):
    feature = processing_a_audio(file)
    save_feature( file , feature )


def make_data_folder():
    data_folder = Path( config['dataset']['feature_folder'] )
    if data_folder.exists():
        print( f"feature folder {data_folder} exist." )
        return False
    else:
        data_folder.mkdir( exist_ok=True )
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

    resample_audio_folder = Path(process_arg['resample_audio_folder'])/f"audio_sr{process_arg['new_sr']}"
    if not resample_audio_folder.exists():
        create_resampled_folder(resample_audio_folder)
    else:
        print("resample audio folder ", resample_audio_folder, "exists")

    dataset_arg['audio_folder'] = str(resample_audio_folder)
    if not make_data_folder():
        return  # 跳出 preprocessing
    audio_files = glob.glob( dataset_arg['audio_folder']+ '/*.wav' )

    features = Parallel(n_jobs=8)(delayed(processing_and_save_a_file)(file)
                                  for file in tqdm(audio_files))

if __name__ == "__main__":
    main(show=True)
