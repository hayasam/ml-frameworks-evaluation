#!/usr/bin/env bash

#
# Server configuration
#
source /etc/profile.d/conda.sh
conda activate ${CONDA_SERVER_ENV:-server}

# TODO: Maybe check if dependencies are already satisfied
# Set PY_CACHE_DIR
PY_CACHE_DIR=${PY_CACHE_DIR:-/pip-cache}
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

cd server
echo "Starting server..."
python data_server.py --metrics-log-dir "$METRICS_LOG_DIR" --server-log-file "$SERVER_LOG_FILE" --seed-controller-file "$SEED_CONTROLLER_FILE" --data-root "$DATA_SERVER_DATA_ROOT"
