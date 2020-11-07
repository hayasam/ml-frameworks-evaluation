#!/usr/bin/env bash

X_DO_LAUNCH=${X_DO_LAUNCH:=0}

BUG_NAME=${1:?First argument is the bug name}
TYPE=${2:?Evaluation type is the second argument}
TYPE=${TYPE,,}
MODEL=${3:?Model is the third argument and is case sensitive}
CHALLENGE=${4:=CIFAR}

# TODO: Allow more
NUM_CLASSES=10

echo "Using bug name ${BUG_NAME}"
echo "Using evaluation type ${TYPE}"
echo "Using model ${MODEL}"
echo "Using challenge ${CHALLENGE:=cifar}"
echo "Using number of epochs ${NUMBER_OF_EPOCHS:=30}"
echo "Using number of runs ${NUMBER_OF_RUNS:=50}"
echo "Using a single seed value ${USE_FIRST_SEED:=1}"
echo "Using mount root ${MNT_ROOT:=/mnt/disks/experiments}"
echo '=== Runtime ==='
echo "Using server endpoint ${DATA_SERVER_ENDPOINT:=tcp://172.17.0.2:90002}"
# Docker relative
echo "Docker: Using client logging directory ${CLIENT_LOG_DIR:=/results/client}"
echo "Docker: Using config directory ${CONFIG_DIR:=/configs}"
echo "Docker: Using build dir ${BUILD_DIR:=/builds}"
echo "Docker: Installing extra conda MKL dependency? ${INSTALL_MKL:=1}"
X_CONTAINER_NAME="${BUG_NAME}_${TYPE}_${MODEL}"
echo "Docker: Using container name ${X_CONTAINER_NAME}"

BUG_BASE_ENV="${MNT_ROOT}/configs/${BUG_NAME}/${TYPE}.env"
BUG_IMG_TAG="$(head -n 1 ${MNT_ROOT}/configs/${BUG_NAME}/tag)"
BUG_BASE_IMAGE="emiliorivera/ml-frameworks-evaluation-client"

echo "Using docker environment file ${BUG_BASE_ENV}"

if [ -z "$BUG_IMG_TAG" ]; then
    echo "Please set a tag for the bug. Expected file at '${MNT_ROOT}/configs/${BUG_NAME}/tag'"
fi

X_TIME="$(date -u +"%FT%H%MZ")"
X_FILE_SAVE_PATH="${MNT_ROOT}/configs/run_${BUG_NAME}_${TYPE}_${X_TIME}"
echo "Will save to ${X_FILE_SAVE_PATH}"

_X_COMMAND="docker run --name "$X_CONTAINER_NAME" -e EVALUATION_TYPE="${TYPE}" --env-file "$BUG_BASE_ENV" -e USE_FIRST_SEED=${USE_FIRST_SEED} -e CHALLENGE=${CHALLENGE} -v "${MNT_ROOT}/configs:/configs" -v "${MNT_ROOT}/results:/results" -v "${MNT_ROOT}/builds:/builds:ro" --gpus all -e CLIENT_PY_VERSION=useless -e MODEL_NAME="$MODEL" -e NUMBER_OF_RUNS=${NUMBER_OF_RUNS} -e NUM_CLASSES=${NUM_CLASSES} -e DATA_SERVER_ENDPOINT=${DATA_SERVER_ENDPOINT} -e CLIENT_LOG_DIR=${CLIENT_LOG_DIR} -e BUILD_DIR=${BUILD_DIR} -e NUMBER_OF_EPOCHS=${NUMBER_OF_EPOCHS} -e CONFIG_DIR=${CONFIG_DIR} -e INSTALL_MKL=${INSTALL_MKL} "${BUG_BASE_IMAGE:?}:${BUG_IMG_TAG}""

if [ $X_DO_LAUNCH -eq 1 ]; then
    $_X_COMMAND
else
    echo "Set X_DO_LAUNCH to 1 to run the command."
    echo $_X_COMMAND
fi

# echo "BUG_NAME=$BUG_NAME" >> $X_FILE_SAVE_PATH
# echo "EVALUATION_TYPE=$TYPE" >> $X_FILE_SAVE_PATH
# echo "CLIENT_LOG_DIR=$CLIENT_LOG_DIR" >> $X_FILE_SAVE_PATH
# echo "MODEL_NAME=$MODEL" >> $X_FILE_SAVE_PATH
# echo ${BUG_NAME@A} >> $X_FILE_SAVE_PATH
# echo ${BUG_NAME@A} >> $X_FILE_SAVE_PATH
# echo ${BUG_NAME@A} >> $X_FILE_SAVE_PATH
# echo ${BUG_NAME@A} >> $X_FILE_SAVE_PATH
# echo ${BUG_NAME@A} >> $X_FILE_SAVE_PATH