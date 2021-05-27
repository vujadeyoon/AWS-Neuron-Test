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
# Remove the existing directories.
rm -rf ${path_base}/pip3_virtulenv/
rm -rf ${path_base}/external_packages/
rm -rf ${path_base}/Projects/
#
#
# Make directories.
mkdir -p ${path_base}/pip3_virtulenv/
mkdir -p ${path_base}/external_packages/
mkdir -p ${path_base}/Projects/
#
#
# Create pip env and install dependencies
sudo apt-get install python3-venv # install Python 3 virtualenv on Ubuntu
python3 -m venv ${path_base}/pip3_virtulenv/RetinaFace
source ${path_base}/pip3_virtulenv/RetinaFace/bin/activate
python3 -m pip install -U pip
pip3 install --extra-index-url=https://pip.repos.neuron.amazonaws.com --upgrade 'torch-neuron==1.7.*' neuron-cc 'tensorflow==1.15.*' 'torchvision==0.8.2' opencv-python Cython
pip3 install psutil gpustat pytz
cd ${path_base}
#
#
# gdown.pl for google drive
git clone https://github.com/circulosmeos/gdown.pl ${path_base}/external_packages/gdown.pl
#
#
# PyTorch RetinaFace
git clone https://github.com/biubug6/Pytorch_Retinaface ${path_base}/Projects/Pytorch_Retinaface
mkdir -p ${path_base}/Projects/Pytorch_Retinaface/weights/
#
#
# vujade
git clone https://github.com/vujadeyoon/vujade ${path_base}/Projects/Pytorch_Retinaface/vujade
cd ${path_base}/Projects/Pytorch_Retinaface/vujade/ && bash ./bash_setup_vujade.sh && cd ${path_base}
#
#
# Download PyTorch pretrained model from the Google drive and copy the required python3 scripts to test the AWS-Neuron.
${path_base}/external_packages/gdown.pl/gdown.pl https://drive.google.com/file/d/14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW/view ./Projects/Pytorch_Retinaface/weights/Resnet50_Final.pth
cp ${path_base}/Codes/compile_retinaface_resnet50.py ${path_base}/Projects/Pytorch_Retinaface/
cp ${path_base}/Codes/detect_aws_neuron.py ${path_base}/Projects/Pytorch_Retinaface/
cp ${path_base}/Codes/detect_optimized.py ${path_base}/Projects/Pytorch_Retinaface/
#
#
# Test for PyTorch (100 iterations, input image shape: origin)
cd ${path_base}/Projects/Pytorch_Retinaface && python3 ./detect.py --cpu
cd ${path_base}
#
#
# Compile PyTorch model to AWS-Neuron
cd ${path_base}/Projects/Pytorch_Retinaface && python3 ./compile_retinaface_resnet50.py
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

