#!/bin/python
import sys
if '../' not in sys.path:
    sys.path.insert(0, '../')

import librosa as ra
from config import config
from preprocessing import get_an_filename
process_arg = config['preprocessing']
mel_arg = process_arg['mel_arg']

def compute_melspec_shape( samples, nfft,hops,mels_bin ):
    samples += nfft//2 * 2  # 左右 padding 半個 windows 長度, 使得陣列中初始點作為windows的中心
    time_length = int((samples-nfft)/hops) + 1
    return mels_bin, time_length


if __name__ == "__main__":
    

    file=get_an_filename()
    wav,origin_sr = ra.load(file)
    new_wav = ra.resample(y=wav, orig_sr=origin_sr, target_sr=process_arg['new_sr'] )
    mel = ra.feature.melspectrogram(y=new_wav, **mel_arg)



    print(
        f'''\
origin:
    time:10s sr:{origin_sr} length:{wav.shape}
new:
    time:10s sr:{process_arg["new_sr"]} length:{new_wav.shape}

melspectrogram:
      n_fft:{mel_arg["n_fft"]}  hop_length:{mel_arg["hop_length"]} n_mels:{mel_arg["n_mels"]}
      shape: {mel.shape}
''')
