# Building docker
The `Dockerfile` build context should be the `src` folder:

`sudo docker build --tag emiliorivera/ml-frameworks:eval100_client --file Dockerfile.client src`
or for the server
`sudo docker build --tag emiliorivera/ml-frameworks:eval100_server --file Dockerfile.server src`

_Update the `.dockerignore` file if need be, for example if the amount of files sent to the docker daemon during build is too large._

## Running a client

### Prerequisites
You should have the following docker volumes created:
1. `pip-cache`: Pip caching directory, to avoid downloading same dependencies when relaunching a server.
2. `results`: Contains the metrics recolted by the server (and the client, technically)
3. `build-vol`: Contains the builds of the libraries to use.

Configuration takes place with environment variables. Look at the template file `envs/template.env`, which contains these parameters:
<pre>
BUG_NAME=EXPERIMENT_NAME
CLIENT_MANUAL_DEPENDENCY=COMMIT_SHA
CLIENT_PY_VERSION=3.5.6
EVALUATION_TYPE=
MODEL_LIBRARY=pytorch
CLIENT_LOG_DIR=/results/client
NUMBER_OF_RUNS=2
PY_CACHE_DIR=/pip_cache
DATA_SERVER_ENDPOINT=tcp://IP_ADRESS_OF_SERVER:90002
USE_BUILD_MKL=1
USE_CUDA=True
</pre>
There are also the following environment variables that need to be added when launching an evaluation:
1. `CHALLENGE`, denotes the name of the challenge to use
2. `NUM_CLASSES` denotes the output size of the network to build, if the network supports it. Tied to `CHALLENGE` usually.
3. `MODEL_NAME` denotes the name of the model to use
`sudo docker run --rm --mount source=pip-cache,target=/pip_cache --mount source=results,target=/results --mount source=build-vol,target=/builds --gpus all --env-file /path/to/env/buggy.env --env CHALLENGE=NAME_OF_CHALLENGE --env NUM_CLASSES=NUMBER_OF_POSSIBLE_CLASSES --env MODEL_NAME=MODEL_TO_USE -it emiliorivera/ml-frameworks:eval100_client`

## Running a server

### Prerequisites
You should have the following docker volumes created:
1. `pip-cache`: Pip caching directory, to avoid downloading same dependencies when relaunching a server.
2. `results`: Contains the metrics recolted by the server (and the client, technically)
3. `data`: Contains the data for the challenges (datasets)

Here is the following preferred command in order to launch a server:
`sudo docker run --name eval100_server --mount source=pip-cache,target=/pip_cache --mount source=results,target=/results --mount source=data,target=/data --gpus all --env DATA_SERVER_DATA_ROOT=/data -it emiliorivera/ml-frameworks:eval100_server`

The default values for the following parameters are set:
1. `PY_CACHE_DIR` to `/pip_cache`
2. `METRICS_LOG_DIR` to `/results/server`
3. `SEED_CONTROLLER_FILE` to `/results/server/seed_control`
4. `SERVER_LOG_FILE` to `$METRICS_LOGDIR/server.log`}"
5. `SERVER_PY_VERSION` to `3.6.8`
