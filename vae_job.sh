#!/bin/bash
module add python/3.3.2
module add python/default
module add git/1.8.3.4
source ~/venv2/bin/activate
cd reimplementations
python3 vae.py
