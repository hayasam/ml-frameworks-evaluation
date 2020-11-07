#!/usr/bin/env bash

# Build the clients
## Version Python 3.7.9
sudo docker build -f Dockerfile.client --build-arg BASE_IMAGE_VERSION="10.1-cudnn7-runtime-ubuntu18.04" --build-arg CLIENT_PYTHON_VERSION="3.7.9" --tag emiliorivera/ml-frameworks-evaluation-client:py379-cu101 .
## Version Python 3.6.7
sudo docker build -f Dockerfile.client --build-arg BASE_IMAGE_VERSION="10.1-cudnn7-runtime-ubuntu18.04" --build-arg CLIENT_PYTHON_VERSION="3.6.7" --tag emiliorivera/ml-frameworks-evaluation-client:py367-cu101 .

# Build the server
sudo docker build -f Dockerfile.server --tag emiliorivera/ml-frameworks-evaluation-server .
