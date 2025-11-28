#!/bin/bash

# Download prerequisite packages
apt update && apt install -y apt-transport-https ca-certificates gnupg curl cuda-toolkit

# Generate SSH Keys
ssh-keygen -t ed25519
cat ~/.ssh/id_ed25519.pub
sleep 5

## Download Kaggle Docker Image
# Download google-cloud-cli
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
apt update && apt install google-cloud-cli

# Configure gcloud
gcloud init
gcloud auth configure-docker gcr.io

# Finally, pull the Docker image
docker pull gcr.io/kaggle-gpu-images/python:v82

# Run jupyter by the Docker image
echo "kjupyter () { docker run --gpus all --runtime nvidia -v $PWD:/tmp/working -w=/tmp/working -p 8888:8888 --rm -it gcr.io/kaggle-gpu-images/python:v82 jupyter notebook --no-browser --port 8888 --ip="*" --allow-root --notebook-dir=/tmp/working; }" >> ~/.bashrc

# Download Waymo Night Dataset
curl -L -o waymo-night-training-v2.zip https://www.kaggle.com/api/v1/datasets/download/mohammedosama/waymo-night-training-v2

# Download Waymo Night Validation Dataset
curl -L -o waymo-night-validation-dataset.zip https://www.kaggle.com/api/v1/datasets/download/mohammedosama/waymo-night-validation-dataset

# Clone TF3D
git clone https://github.com/google-research/google-research.git

# Revert to the tested version
cd google-research; git reset --hard 512a91d1c;

