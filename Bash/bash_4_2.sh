#!/bin/bash
#
#
path_base=$(pwd)
#
#
find ./ -name __pycache__ -exec rm -rf {} \;
find ./ -name .idea -exec rm -rf {} \;
#
#
# Activate a python virtual environment.
source ${path_base}/pip3_virtulenv/RetinaFace/bin/activate
cd ${path_base}
#
#
# Test for Native PyTorch on the GPU
cd ${path_base}/Projects/Pytorch_Retinaface && python3 ./detect_optimized.py
cd ${path_base}
#
#
# Test for Native PyTorch on the CPU
cd ${path_base}/Projects/Pytorch_Retinaface && python3 ./detect_optimized.py --cpu
cd ${path_base}
#
#
# Test for AWS neuron on the AWS inf1 chip.
cd ${path_base}/Projects/Pytorch_Retinaface && python3 ./detect_optimized.py --cpu --aws_neuron
cd ${path_base}

