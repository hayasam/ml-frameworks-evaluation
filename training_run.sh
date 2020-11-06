#!/usr/bin/env bash

MODELS=( "EvaluationVGG" )
BASE_DIR=$(readlink -f envs/pytorch/)
EXPERIMENTS_DIR=( $(ls "$BASE_DIR") )
CHALLENGE="cifar"
NUM_CLASSES=10

for n in "${EXPERIMENTS_DIR[@]}"; do
    exp_name="${n##*/}"
    
    echo "$exp_name"
    for model in "${MODELS[@]}"; do
        for evaluationtype in "buggy" "corrected"; do
            echo "RUNNING EXP ${exp_name} - ${model} for evaluation type $evaluationtype."
            envfile_path="${BASE_DIR}/${exp_name}/${evaluationtype}.env"
            echo "Content of envfile: "
            echo "------"
            cat "$envfile_path"
            echo "======"

            echo docker run --name "${exp_name}_${evaluationtype}_${model}" --mount source=pip-cache,target=/pip-cache --mount source=results,target=/results --mount source=build-vol,target=/builds --gpus all --env-file "$envfile_path" --env MODEL_NAME="$model" --env CHALLENGE="$CHALLENGE" --env NUM_CLASSES="$NUM_CLASSES" -it emiliorivera/ml-frameworks:eval100_client
        done
    done
done
