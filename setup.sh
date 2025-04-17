#!/bin/bash

##
## Simple script to setup the local Python
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

pip install numpy torch torchvision torchaudio
pip install gymnasium "gymnasium[Box2D]"
pip install seaborn

echo ""
echo "Run this command to activate Python:"
echo ""
echo "source .venv/bin/activate"



