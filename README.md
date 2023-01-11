
# Usage
1. Add repository folder to PYTHONPATH:  `export PYTHONPATH=/this/repository/path`.

2. Change the `audio_folder` item to yor audio dataset folder in config.yaml file.

3. At first time, preprocessing the audio data(wav file) to feature(melspectrogram, or stft) in npy format, save it in the `./data` folder.

```sh
python3 preprocessing.py
``` 

4. Then start to training.
```sh
python3 main.py
```
