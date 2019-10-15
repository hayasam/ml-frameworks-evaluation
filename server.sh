#!/usr/bin/env bash

# TODO: Maybe check if dependencies are already satisfied
# Set PY_CACHE_DIR
PY_CACHE_DIR=${PY_CACHE_DIR:-/pip_cache}
SERVER_PY_VERSION="${SERVER_PY_VERSION:-3.6.8}"
SERVER_VENV_NAME="server-venv"

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
