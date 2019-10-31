#!/usr/bin/env bash

MODELS=( "EvaluationVGG" )
EXPERIMENTS_ENV_FILES=( $(find envs/buggy_env -type f -name "*.env" -print) )

for n in "${EXPERIMENTS_ENV_FILES[@]}"; do
    exp_file="${n##*/}"
    exp_name="${exp_file%%.env}"
    
    echo "$exp_name $exp_file"
    for model in "${MODELS[@]}"; do
        for evaluationtype in "buggy" "corrected"; do
            echo "RUNNING EXP ${exp_name} ${model} for evaluation type $evaluationtype."
            envfile_path="/home/kacham/Documents/ml-frameworks-evaluation/envs/${evaluationtype}_env/${exp_file}"
            docker run --name "${exp_name}_${evaluationtype}_${model}" --mount source=pip-cache,target=/pip_cache --mount source=results,target=/results --mount source=build-vol,target=/builds --gpus all --env-file "$envfile_path" --env MODEL_NAME="$model" -it emiliorivera/ml-frameworks:eval100_client
        done
    done
done
