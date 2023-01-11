#!/bin/python
from yaml import safe_load
with open('./config.yaml', 'r') as f:
    config = safe_load(f)
