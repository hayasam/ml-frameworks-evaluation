#!/usr/bin/env bash

# TODO: Maybe check if dependencies are already satisfied
# Set PY_CACHE_DIR
PY_CACHE_DIR=${PY_CACHE_DIR:-/pip_cache}
SERVER_PY_VERSION="${SERVER_PY_VERSION:-3.6.8}"
SERVER_VENV_NAME="server-venv"
SERVER_LOG_FILE="${SERVER_LOG_FILE:?}"
SEED_CONTROLLER_FILE="${SEED_CONTROLLER_FILE:?}"

if [ ! -d "${METRICS_LOG_DIR:?}" ]; then
  echo "Specified directory for logs does not exist!"
  exit 1
fi

if [ ! -d "$(dirname $SERVER_LOG_FILE)" ]; then
  echo "Specified directory for server log file does not exist!"
  exit 1
fi

if [ ! -d "$(dirname $SEED_CONTROLLER_FILE)" ]; then
  echo "Specified directory for seed controller file does not exist!"
  exit 1
fi

#
# Server configuration
#
cd server
# mkdir --parents .logs
pyenv local $SERVER_PY_VERSION
python -m venv ".venvs/$SERVER_VENV_NAME"
source ".venvs/$SERVER_VENV_NAME/bin/activate"

# Create temporary file that pip will read in order to allow caching
echo "[global]" >> .pip.config
echo "cache-dir = $PY_CACHE_DIR" >> .pip.config
export PIP_CONFIG_FILE=".pip.config"

if [[ ! $(python --version) = "Python $SERVER_PY_VERSION" ]]; then
  echo "$(python --version)"
  echo 'Python server version not correctly set.'
  echo 'Exiting...'
  exit 1
fi

pip install ../shared
pip install -r requirements.txt
echo "Starting server..."
python data_server.py --metrics-log-dir "$METRICS_LOG_DIR" --server-log-file "$SERVER_LOG_FILE" --seed-controller-file "$SEED_CONTROLLER_FILE"
