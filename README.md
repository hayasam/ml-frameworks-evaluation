# Building docker
The `Dockerfile` build context should be the `src` folder:

`sudo docker build --tag emiliorivera/ml-frameworks:eval100_client --file Dockerfile.client src`
or for the server
`sudo docker build --tag emiliorivera/ml-frameworks:eval100_server --file Dockerfile.server src`


## Running a client
Configuration takes place with environment variables. Use these parameters:
<pre>
EXPERIMENT_NAME=EXPERIMENT_NAME
CLIENT_VENV_NAME=EXPERIMENT_NAME
CLIENT_MANUAL_DEPENDENCY=COMMIT_SHA
CLIENT_PY_VERSION=3.5.6
EVALUATION_TYPE=corrected
CHALLENGE=mnist
MODEL_LIBRARY=pytorch
MODEL_NAME=Net
CLIENT_LOG_DIR=/results
NUMBER_OF_RUNS=10
PY_CACHE_DIR=/pip_cache
DATA_SERVER_ENDPOINT=tcp://IP_ADRESS_OF_SERVER:90002
USE_BUILD_MKL=1
USE_CUDA=True
</pre>

`sudo docker run --name <EXPERIMENT_NAME> --mount source=pip-cache,target=/pip_cache --mount source=results,target=/results --mount source=build-vol,target=/builds --gpus all --env-file <EXPERIMENT_NAME>.env -it emiliorivera/ml-frameworks:eval100_client`