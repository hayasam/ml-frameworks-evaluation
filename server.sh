#!/usr/bin/env bash

# TODO: Maybe check if dependencies are already satisfied
# Set PY_CACHE_DIR

SERVER_PY_VERSION=3.6.8
SERVER_VENV_NAME="server-venv"

if [ -z "${BUILD_DIR+x}" ]; then
  BUILD_DIR="/builds/$CUDA_VERSION"
  if [ -z "${CUDA_VERSION+x}" ]; then
    echo 'Need to set env variable CUDA_VERSION if BUILD_DIR is not set'
    echo 'exiting...'
    exit 1
  fi
  echo "BUILD_DIR not set, using value $BUILD_DIR"
fi

if [ -z "${SERVER_PY_VERSION+x}" ]; then
  echo 'Please set a value for $SERVER_PY_VERSION'
  exit 1
fi

# Setting PIP_FIND_LINKS will allow to check the local
# directory
PIP_FIND_LINKS="$PY_CACHE_DIR $PIP_FIND_LINKS"


#
# Server configuration
#
cd server
# mkdir --parents .logs
pyenv local $SERVER_PY_VERSION
python -m venv ".venvs/$SERVER_VENV_NAME"
source ".venvs/$SERVER_VENV_NAME/bin/activate"

if [[ ! $(python --version) = "Python $SERVER_PY_VERSION" ]]; then
  echo "$(python --version)"
  echo 'Python server version not correctly set.'
  echo 'Exiting...'
  exit 1
fi

pip install ../shared
pip install -r requirements.txt
echo "Starting server..."
python data_server.py &
