#!/usr/bin/env bash

# Set CLIENT_PY_VERSION
# Set PY_CACHE_DIR
# Set CLIENT_MANUAL_DEPENDENCY
# Set EXPERIMENT_NAME
# Set EVALUATION_TYPE
# Set CHALLENGE MODEL_LIBRARY MODEL_NAME CLIENT_LOG_DIR --runs 1000
CLIENT_VENV_NAME="$EXPERIMENT_NAME"

if [ -z "${CLIENT_MANUAL_DEPENDENCY+x}" ]; then
  echo 'Please set a value for $CLIENT_MANUAL_DEPENDENCY'
  exit 1
fi

if [ -z "${EXPERIMENT_NAME+x}" ]; then
  echo 'Please set a value for $EXPERIMENT_NAME'
  exit 1
fi

if [ -z "${CLIENT_PY_VERSION+x}" ]; then
  echo 'Please set a value for $CLIENT_PY_VERSION'
  exit 1
fi

# Setting PIP_FIND_LINKS will allow to check the local
# directory
PIP_FIND_LINKS="$PY_CACHE_DIR $PIP_FIND_LINKS"

# TODO: Put this in its own function in other file
if [ -z "${BUILD_DIR+x}" ]; then
  BUILD_DIR="/builds/$CUDA_VERSION"
  if [ -z "${CUDA_VERSION+x}" ]; then
    echo 'Need to set env variable CUDA_VERSION if BUILD_DIR is not set'
    echo 'exiting...'
    exit 1
  fi
  echo "BUILD_DIR not set, using value $BUILD_DIR"
fi


#
# Client configuration
#
cd client
# mkdir --parents .logs
pyenv local "$CLIENT_PY_VERSION"
python -m venv ".venvs/$CLIENT_VENV_NAME"
source ".venvs/$CLIENT_VENV_NAME/bin/activate"
pip install ../shared
pip install -r partial_requirements.txt
# Need also to install the built library
MANUAL_WHL=( $(find $BUILD_DIR -type f -name "*$CLIENT_MANUAL_DEPENDENCY*.whl") )
if [ "${#MANUAL_WHL[@]}" -ne 1 ]; then
  echo "MANUAL_WHL has more than one entry (#{aaa[@]})"
  exit 1
fi

pip install "${MANUAL_WHL[0]}"
echo "Starting client..."
# Note: runs should maybe be parameterized?
python trainer.py --type "$EVALUATION_TYPE" --name "$EXPERIMENT_NAME" --challenge "$CHALLENGE" --model-library "$MODEL_LIBRARY" --model-name "$MODEL_NAME" --log-dir "$CLIENT_LOG_DIR" --runs 1000 &
