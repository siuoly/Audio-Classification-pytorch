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


def create_resampled_folder(resample_audio_folder):
    print("create resample audio folder ", resample_audio_folder)
    resample_audio_folder.mkdir(parents=True)
    audio_paths = list(Path(process_arg['audio_folder']).glob("*.wav"))
    new_audio_paths = [resample_audio_folder/path.name for path in audio_paths]

    def resample_and_save(file, new_path):
        wav, _ = ra.load(file, sr=new_sr)
        sf.write(new_path, wav, new_sr)
    arguments = zip(audio_paths, new_audio_paths)
    Parallel(n_jobs=12)(delayed(resample_and_save)(file, new_path)
                        for file, new_path in tqdm(arguments))


def get_an_filename():
    filenames = glob.glob(process_arg['audio_folder'] + '/*.wav')
    file = filenames[1]
    return file


# preprocessing : resample, melspectrogram
def show_preprocessing_message():
    file = get_an_filename()
    wav, sr = ra.load(file, sr=None)
    new_wav = ra.resample(wav, orig_sr=sr,
                          target_sr=new_sr)
    mel = ra.feature.melspectrogram(y=new_wav, sr=new_sr,
                                    **process_arg['mel_arg'])
    print("audio folder", process_arg["audio_folder"])
    print(f"wav shape :{wav.shape}, sr: {sr}")
    print(f"new wav shape:{new_wav.shape}, sr: {new_sr} ")
    print("mel argument:")
    pprint(process_arg['mel_arg'])
    print(f"spectrogram feature shape:{mel.shape}")


def get_mel_of_file(file):
    wav, sr = ra.load(file, sr=new_sr)
    mel = ra.feature.melspectrogram(y=wav, sr=sr, **process_arg['mel_arg'])
    if process_arg["dbscale"] is True:
        mel = ra.power_to_db(mel)
    return mel


def get_stft_and_phase_of_file(file):
    wav, sr = ra.load(file, sr=new_sr)
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
    save_path = data_folder / (os.path.basename(file) + ".npy")
    np.save(save_path, feature)
    return save_path


def processing_and_save_a_file(file):
    feature = processing_a_audio(file)
    save_feature( file , feature )


def make_data_folder():
    data_folder = Path( config['dataset']['train_folder'] )
    if not data_folder.exists():
        data_folder.mkdir( exist_ok=True )
        print( f"create data folder:{data_folder}" )
    else:
        print( f"feature folder {data_folder} exist." )
        exit(0)
    return data_folder


def normalized_data(x):
    axis = tuple(np.arange(x.ndim)[-2:])  # 取最後兩維度(0,1) or (1,2)
    return (x-x.mean(axis=axis, keepdims=True)) / \
        x.std(axis=axis, keepdims=True)

if __name__ == "__main__":
    meta = process_arg['meta_file']
    meta = pd.read_csv(meta)
    show_preprocessing_message()

    resample_audio_folder = Path(process_arg['resample_audio_folder'])/f"audio_sr{new_sr}"
    if not resample_audio_folder.exists():
        create_resampled_folder(resample_audio_folder)
    else:
        print("resample audio folder ", resample_audio_folder, "exists")

    process_arg['audio_folder'] = str(resample_audio_folder)
    # print(get_stft_and_phase_of_file(get_an_filename()).shape)
    # print(processing_a_audio(get_an_filename()))
    data_folder = make_data_folder()
    audio_files = glob.glob( process_arg['audio_folder']+ '/*.wav' )

    features = Parallel(n_jobs=8)(delayed(processing_and_save_a_file)(file)
                                  for file in tqdm(audio_files))

    # print(result)
