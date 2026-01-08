#!/bin/bash

# Download prerequisite packages
#apt update && apt install -y apt-transport-https ca-certificates gnupg curl cuda-toolkit tmux

## Download Kaggle Docker Image
# Download google-cloud-cli
#curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
#echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
#apt update && apt install -y google-cloud-cli

# Configure gcloud
#gcloud init
#gcloud auth configure-docker gcr.io

# Finally, pull the Docker image
#docker pull gcr.io/kaggle-gpu-images/python:v81

# Run jupyter by the Docker image
#echo "kjupyter () { docker run --gpus all --runtime nvidia -v $PWD:/tmp/working -w=/tmp/working -p 8888:8888 --rm -it gcr.io/kaggle-gpu-images/python:v82 jupyter notebook --no-browser --port 8888 --ip="*" --allow-root --notebook-dir=/tmp/working; }" >> ~/.bashrc

## Add the NVIDIA package repositories
#distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
#curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
#curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

# Install the toolkit
#apt update && apt install -y nvidia-docker2

#apt update && apt install -y zip python3.7 python3.7-dev python3.7-venv
#update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
#update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
#update-alternatives --config python3

#python3.7 -m pip install --upgrade pip
#python3.7 -m pip install jupyter_core
#python3.7 -m pip install notebook jupyterlab

# Clone TF3D
#git clone https://github.com/google-research/google-research.git

# Revert to the tested version
#cd google-research; git reset --hard 512a91d1c; cd -

# Move notebook source codes to home dir
mv ./CUCW-Dataset-Generation-Framework/3d-object-detection-and-synthetic* ~

# Download and unzip Waymo Night Dataset
#curl -L -o waymo-night-training-v2.zip https://www.kaggle.com/api/v1/datasets/download/mohammedosama/waymo-night-training-v2
#wget --continue https://www.kaggle.com/api/v1/datasets/download/mohammedosama/waymo-mv-night-training -O waymo-mv-night-training.zip
wget --continue https://www.kaggle.com/api/v1/datasets/download/mohammedosama/waymo-sf-night-training -O waymo-sf-night-training.zip
mkdir waymo-sf-night-training
cd waymo-sf-night-training; unzip ../waymo-sf-night-training.zip; cd ~

if [ "$1" = "cucw" ]; then
    wget --continue https://www.kaggle.com/api/v1/datasets/download/mohammedosama/cucw-v3 -O cucw.zip
    mkdir cucw
    cd cucw; unzip ../cucw.zip; cd ~
fi

# Download Waymo Night Validation Dataset
#curl -L -o waymo-night-validation-dataset.zip https://www.kaggle.com/api/v1/datasets/download/mohammedosama/waymo-night-validation-dataset
wget --continue https://www.kaggle.com/api/v1/datasets/download/mohammedosama/waymo-sf-night-validation -O waymo-sf-night-validation.zip
mkdir waymo-sf-night-validation
cd waymo-sf-night-validation; unzip ../waymo-sf-night-validation.zip; cd ~

