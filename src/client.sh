#!/usr/bin/env bash

source /etc/profile.d/conda.sh

# TODO: Check how these variables should be set
# Information about the experiment
BUG_NAME="${BUG_NAME:?}"
EVALUATION_TYPE="${EVALUATION_TYPE:?}"
CLIENT_MANUAL_DEPENDENCY="${CLIENT_MANUAL_DEPENDENCY:?}"
CLIENT_PY_VERSION="${CLIENT_PY_VERSION:?}"
NUMBER_OF_RUNS="${NUMBER_OF_RUNS:?}"
# Challenge information
NUM_CLASSES="${NUM_CLASSES:?}"
CHALLENGE="${CHALLENGE:?}"
MODEL_LIBRARY="${MODEL_LIBRARY:?}"
MODEL_NAME="${MODEL_NAME:?}"
# Oher
USE_BUILD_MKL="${USE_BUILD_MKL:-0}"
DATA_SERVER_ENDPOINT="${DATA_SERVER_ENDPOINT:?}"
# Computed attribute
CLIENT_VENV_NAME="${BUG_NAME}_${EVALUATION_TYPE}"
# Mounts and directories
CLIENT_LOG_DIR="${CLIENT_LOG_DIR:?}"
PY_CACHE_DIR=${PY_CACHE_DIR:-/pip-cache}
BUILD_DIR="${BUILD_DIR:?}"
CONFIG_DIR="${CONFIG_DIR:?}"

# Setting PIP_FIND_LINKS will allow to check the local
# directory
PIP_FIND_LINKS="$PY_CACHE_DIR $PIP_FIND_LINKS"

if [ "$USE_BUILD_MKL" -eq 1 ]; then
  echo "Adding /builds/mkl to LD_LIBRARY_PATH"
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/builds/mkl"
fi

# Creating client directory
mkdir -p "$CLIENT_LOG_DIR"

#
# Client configuration
#
cd client
# Replicate the first train environment to branch off
conda create --name ${CLIENT_VENV_NAME} --clone ${CONDA_TRAIN_ENV:-train}
conda activate ${CLIENT_VENV_NAME}

# Create temporary file that pip will read in order to allow caching
echo "[global]" >> .pip.config
echo "cache-dir = $PY_CACHE_DIR" >> .pip.config
export PIP_CONFIG_FILE=".pip.config"

MANUAL_WHL=( $(find $BUILD_DIR -type f -name "*$CLIENT_MANUAL_DEPENDENCY*.whl") )
if [ "${#MANUAL_WHL[@]}" -ne 1 ]; then
  echo "MANUAL_WHL has more than one entry (#{aaa[@]})"
  exit 1
fi

CONFIG_BUG_DIR="${CONFIG_DIR}/${BUG_NAME}"

# Make sure bug specific directory exists
# It is used to share runtime dependencies
if [ ! -d "${CONFIG_BUG_DIR}" ]; then
  1>2& echo "Directory ${CONFIG_BUG_DIR} does not exist! Exiting..."
  exit 1
fi

if [ -e "${CONFIG_BUG_DIR}/requirements.txt" ]; then
  # Try installing it
  pip install -r "${CONFIG_BUG_DIR}/requirements.txt"
else
  # Create a bug specific requirements.txt
  # Bug specific requirements.in
  if [ -e "${CONFIG_BUG_DIR}/requirements.in" ]; then
    pip install -r "${CONFIG_BUG_DIR}/requirements.in"
    # Save output so next runner can take it
    pip freeze -l > "${CONFIG_BUG_DIR}/requirements.txt"
  elif [ -e ./client/requirements.in ]; then
    # fall back on client requirements.in
    # which does not specify versions
    pip install -r ./client/requirements.in
    # Save output so next runner can take it
    pip freeze -l > "${CONFIG_BUG_DIR}/requirements.txt"
  else
    1>2& echo "No suitable requirements file found was found! Exiting..."
    exit 1
  fi
  # Record what environment created the requirements.txt
  echo "${EVALUATION_TYPE}" > "${CONFIG_BUG_DIR}/setup_by.txt"
fi

# Add additional dependencies
pip install ./shared ${MANUAL_WHL[0]}

# Everything installed, export the environment information
conda list --export > "${CONFIG_BUG_DIR}/${EVALUATION_TYPE}.export.txt"
conda list --explicit > "${CONFIG_BUG_DIR}/${EVALUATION_TYPE}.explicit.txt"

echo "Starting client..."
# Note: runs should maybe be parameterized?
python trainer.py --evaluation-type "$EVALUATION_TYPE" --bug-name "$BUG_NAME" --challenge "$CHALLENGE" --model-library "$MODEL_LIBRARY" --model-name "$MODEL_NAME" --log-dir "$CLIENT_LOG_DIR" --runs "$NUMBER_OF_RUNS" --num-classes "$NUM_CLASSES"
