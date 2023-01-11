
# Usage
1. Change the `audio_folder` item to yor audio dataset folder in config.yaml file.

2. At first time, preprocessing the audio data(wav file) to feature(melspectrogram, or stft) in npy format, save it in the `./data` folder.

```sh
python3 preprocessing.py
``` 

3. Then start to training.
```sh
python3 main.py
```
