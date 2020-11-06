#!/usr/bin/env bash

source /etc/profile.d/conda.sh
conda activate ${CONDA_TRAIN_ENV:-train}

# TODO: Check how these variables should be set
BUG_NAME="${BUG_NAME:?}"
EVALUATION_TYPE="${EVALUATION_TYPE:?}"
CLIENT_VENV_NAME="${BUG_NAME}_${EVALUATION_TYPE}"
CLIENT_MANUAL_DEPENDENCY="${CLIENT_MANUAL_DEPENDENCY:?}"
CLIENT_PY_VERSION="${CLIENT_PY_VERSION:?}"
NUM_CLASSES="${NUM_CLASSES:?}"
CHALLENGE="${CHALLENGE:?}"
MODEL_LIBRARY="${MODEL_LIBRARY:?}"
MODEL_NAME="${MODEL_NAME:?}"
CLIENT_LOG_DIR="${CLIENT_LOG_DIR:?}"
NUMBER_OF_RUNS="${NUMBER_OF_RUNS:?}"
PY_CACHE_DIR=${PY_CACHE_DIR:-/pip_cache}
DATA_SERVER_ENDPOINT="${DATA_SERVER_ENDPOINT:?}"
USE_BUILD_MKL="${USE_BUILD_MKL:-0}"
BUILD_DIR="${BUILD_DIR:?}"

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
# mkdir --parents .logs
pyenv local "$CLIENT_PY_VERSION"
python -m venv ".venvs/$CLIENT_VENV_NAME"
source ".venvs/$CLIENT_VENV_NAME/bin/activate"

# Create temporary file that pip will read in order to allow caching
echo "[global]" >> .pip.config
echo "cache-dir = $PY_CACHE_DIR" >> .pip.config
export PIP_CONFIG_FILE=".pip.config"

MANUAL_WHL=( $(find $BUILD_DIR -type f -name "*$CLIENT_MANUAL_DEPENDENCY*.whl") )
if [ "${#MANUAL_WHL[@]}" -ne 1 ]; then
  echo "MANUAL_WHL has more than one entry (#{aaa[@]})"
  exit 1
fi

# Intall everything in one go
pip install ./shared ${MANUAL_WHL[0]} -r ./client/requirements.in

echo "Starting client..."
# Note: runs should maybe be parameterized?
python trainer.py --evaluation-type "$EVALUATION_TYPE" --bug-name "$BUG_NAME" --challenge "$CHALLENGE" --model-library "$MODEL_LIBRARY" --model-name "$MODEL_NAME" --log-dir "$CLIENT_LOG_DIR" --runs "$NUMBER_OF_RUNS" --num-classes "$NUM_CLASSES"
