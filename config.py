#!/bin/python
from yaml import safe_load
from pathlib import Path


def set_feature_folder():
    # data/sr16000_nfft1024_hpo512_n_mels128/ -->
    # data/ESCsr16000_nfft1024_hpo512_n_mels128
    folder = Path(config["feature_folder"])
    folder = folder.parent/(config["dataset"]+"_"+folder.name)
    config["feature_folder"] = str(folder)


with open('./config.yaml', 'r') as f:
    config = safe_load(f)
    # config["feature_folder"] = Path(config["feature_folder"]
    set_feature_folder()
