#!/usr/bin/env bash

# TODO: Maybe check if dependencies are already satisfied
# Set PY_CACHE_DIR
PY_CACHE_DIR=${PY_CACHE_DIR:-/pip-cache}
SERVER_PY_VERSION="${SERVER_PY_VERSION:-3.6.8}"
SERVER_VENV_NAME="server-venv"
DATA_SERVER_DATA_ROOT="${DATA_SERVER_DATA_ROOT:?}"
METRICS_LOG_DIR="${METRICS_LOG_DIR:-/results/server}"
SEED_CONTROLLER_FILE="${SEED_CONTROLLER_FILE:-/results/server/seed_control}"
SERVER_LOG_FILE="${METRICS_LOG_DIR}/${SERVER_LOG_FILE:-server.log}"

# Creating server directory
mkdir -p "$METRICS_LOG_DIR"

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
python data_server.py --metrics-log-dir "$METRICS_LOG_DIR" --server-log-file "$SERVER_LOG_FILE" --seed-controller-file "$SEED_CONTROLLER_FILE" --data-root "$DATA_SERVER_DATA_ROOT"
