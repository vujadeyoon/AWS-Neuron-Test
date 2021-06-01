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
# Test for AWS-Neuron (100 iterations, input image shape: 768x432): This python3 script, detect_aws_neuron.py is based on the detect.py.
cd ${path_base}/Projects/Pytorch_Retinaface && python3 ./detect_aws_neuron.py --cpu
cd ${path_base}
#
#
# Test
# Compile PyTorch model to AWS-Neuron
cd ${path_base}/Projects/Pytorch_Retinaface && python3 ./detect_optimized.py --cpu
cd ${path_base}

