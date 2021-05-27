#!/bin/bash
#
#
path_root=$(pwd)
#
#
rm -rf ./Pytorch_Retinaface/
#
#
# gdown.pl for google drive
git clone https://github.com/circulosmeos/gdown.pl
#
#
# PyTorch RetinaFace 
git clone https://github.com/biubug6/Pytorch_Retinaface
mkdir -p ./Pytorch_Retinaface/weights/
#
#
# Download PyTorch pretrained model from the Google drive and copy the required python3 scripts to test the AWS-Neuron.
./gdown.pl/gdown.pl https://drive.google.com/file/d/14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW/view ././Pytorch_Retinaface/weights/Resnet50_Final.pth
cp ./compile_retinaface_resnet50.py ./Pytorch_Retinaface/
cp ./detect_aws_neuron.py ./Pytorch_Retinaface/
#
#
# Create pip env and install dependencies
cd Pytorch_Retinaface
sudo apt-get install python3-venv # install Python 3 virtualenv on Ubuntu
python3 -m venv retinaface_env
source retinaface_env/bin/activate
python3 -m pip install -U pip
pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com --upgrade 'torch-neuron' neuron-cc 'tensorflow==1.15.*' 'torchvision==0.8.2' opencv-python
cd ${path_root}
#
#
# Test for PyTorch (100 iterations)
cd Pytorch_Retinaface && python3 ./detect.py --cpu
cd ${path_root}
#
#
# Compile PyTorch model to AWS-Neuron
cd Pytorch_Retinaface && python3 ./compile_retinaface_resnet50.py
cd ${path_root}
#
#
# Test for AWS-Neuron (100 iterations): This python3 script, detect_aws_neuron.py is based on the detect.py.
cd Pytorch_Retinaface && python3 ./detect_aws_neuron.py --cpu
cd ${path_root}

