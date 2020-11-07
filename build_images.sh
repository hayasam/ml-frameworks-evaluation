#!/usr/bin/env bash

# Build the clients
## Version Python 3.7.9
sudo docker build -f Dockerfile.client --build-arg BASE_IMAGE_VERSION=py379-cu101 --tag emiliorivera/ml-frameworks-evaluation-client:py379-cu101 .
## Version Python 3.6.7
sudo docker build -f Dockerfile.client --build-arg BASE_IMAGE_VERSION=py367-cu101 --tag emiliorivera/ml-frameworks-evaluation-client:py367-cu101 .

# Build the server