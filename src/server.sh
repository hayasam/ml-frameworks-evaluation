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
SEED_CONTROLLER_FILE="${SEED_CONTROLLER_FILE:?}"
SERVER_LOG_FILE="${SERVER_LOG_FILE:-${METRICS_LOG_DIR/server.log}}"

# Creating server directory
mkdir -p "$METRICS_LOG_DIR"

if [ ! -d "$(dirname $SEED_CONTROLLER_FILE)" ]; then
  echo "Specified directory for seed controller file does not exist!"
  exit 1
fi

cd server
echo "Starting server..."
python data_server.py --metrics-log-dir "$METRICS_LOG_DIR" --server-log-file "$SERVER_LOG_FILE" --seed-controller-file "$SEED_CONTROLLER_FILE"  ${DEFAULT_MINIMAL_SEED_LEN:+ --default-minimal-seed-len $DEFAULT_MINIMAL_SEED_LEN} ${DEFAULT_MIN_SEED_VALUE:+ --default-min-seed-value $DEFAULT_MIN_SEED_VALUE} ${DEFAULT_MAX_SEED_VALUE:+ --default-max-seed-value $DEFAULT_MAX_SEED_VALUE}
