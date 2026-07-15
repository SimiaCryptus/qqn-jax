#!/bin/bash


sudo apt install python3-full python3-venv
deactivate 2>/dev/null
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
which pip          # should show .../qqn-jax/.venv/bin/pip
python -m pip --version

pip install tensorflow torch torchvision
