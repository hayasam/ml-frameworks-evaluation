import functools
import logging
import signal
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import zmq
from challenges import challenges as CHALLENGES
from metrics_logger_store import MetricsLoggerStore
from seed_controller import SeedController
from settings import (METRICS_LOG_BASE_PATH, SEED_CONTROLLER_FILE,
                      SERVER_ENDPOINT_CONNEXION, SERVER_LOG_FILE)
from stats import print_pair_metrics_from_files
from ml_evaluation_ipc_communication import EvaluationRunIdentifier

SEED_CONTROLLER = SeedController.from_saved_file(SEED_CONTROLLER_FILE)
LOGGER_STORE = MetricsLoggerStore(base_path=METRICS_LOG_BASE_PATH)
SERVER_LOGGER = logging.getLogger('server')
SERVER_LOGGER.addHandler(logging.FileHandler(SERVER_LOG_FILE))
SERVER_LOGGER.setLevel(logging.DEBUG)

def dataset_to_numpy(data_loader: torch.utils.data.DataLoader):
    data_x, data_y = [], []
    for data, target in data_loader:
        data_x.append(data.numpy())
        data_y.append(target.numpy())
    # print(len(data_x), len(data_y), data_x[0].shape, data_y[0].shape)
    ar_x = np.array(data_x)
    ar_y =  np.array(data_y)
    return ar_x, ar_y

def _dataset_size(train_set, test_set):
    return 'Train: {} - {} . Test {} - {}'.format(train_set[0].shape, train_set[1].shape,
    test_set[0].shape, test_set[1].shape)

def shuffle_dataset(x, y, seed):
    if len(x) != len(y):
        raise ValueError('x and y should have the same length!')
    # Create a pseudo rng generator for current seed
    rng = np.random.RandomState(seed)
    shuffled_ix = rng.permutation(len(x))
    SERVER_LOGGER.debug('Seed {} | First 10 shuffled indices: {}'.format(seed, shuffled_ix[:10]))
    return x[shuffled_ix], y[shuffled_ix]

def _dataset_hash(train_set, test_set):
    h_msg = 'Train x: {} - Train y: {} - Test x: {} - Test y: {}'.format(*[hash(bytes(x)) for x in [*train_set, *test_set]])
    return h_msg

# TODO: Challenges (or another class) should provide a method accepting a seed and deterministically return the train/test sets instead of defining it here
def prepare_data_for_run(run_identifier: EvaluationRunIdentifier, run: int, seed: int, train_batch_size: int, test_batch_size: int, **kwargs):
    # Verify seeds are coordinated
    assert seed == find_experiment_random_states(run_identifier)[run]
    shuffled_train, shuffled_test = get_data_for_challenge_seed(run_identifier.challenge, seed, run, train_batch_size, test_batch_size)
   
    print('Sending', _dataset_size(shuffled_train, shuffled_test))
    h_msg = _dataset_hash(shuffled_train, shuffled_test)
    SERVER_LOGGER.info('Challenge: {} - Run: {} - Seed: {} - Hashes: {}'.format(run_identifier.challenge, run, seed, h_msg))
    return shuffled_train, shuffled_test

@functools.lru_cache(maxsize=3)
def get_data_for_challenge_seed(challenge_name: str, seed: int, run: int, train_batch_size, test_batch_size):
    challenge = CHALLENGES[challenge_name]
    train_loader, test_loader = challenge.get_subset(run=run, seed=seed,
                                                     train_batch_size=train_batch_size,
                                                     test_batch_size=test_batch_size)
    np_train = dataset_to_numpy(train_loader)
    np_test = dataset_to_numpy(test_loader)
    shuffled_train = shuffle_dataset(np_train[0], np_train[1], seed)
    shuffled_test = shuffle_dataset(np_test[0], np_test[1], seed)
    return shuffled_train, shuffled_test

# From ZeroMQ's doc
def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)


def find_experiment_random_states(run_identifier, **kwargs):
    seed_identifier = EvaluationRunIdentifier.seed_identifier(run_identifier)
    random_state = SEED_CONTROLLER.get_random_states(seed_identifier)
    # print('Random states for {} are {}'.format(seed_identifier, random_state))
    return random_state


def receive_metrics(run_identifier: EvaluationRunIdentifier, **kwargs):
    # TODO: Do something with kwargs['run']?
    # identifier = '{}_{}'.format(kwargs['experiment_name'], kwargs['challenge'])
    identifier = EvaluationRunIdentifier.run_identifier(run_identifier)
    print('Received metrics from {}'.format(identifier))
    m = kwargs['value']
    metrics_msg = 'run: {} - accuracy: {} - precision: {} - recall: {} - f1: {}'.format(kwargs['run'], m['accuracy'], m['precision'], m['recall'], m['f1_score'])
    print(metrics_msg)
    specific_logger = LOGGER_STORE.get_logger(experiment_name=identifier)
    specific_logger.debug(metrics_msg)
    return True

def pair_stats_between_experiments(experiment_name_1: str, experiment_name_2: str, n_samples: int):
    def file_handle_for_experiment(experiment_name):
        specific_logger_experiment_1 = LOGGER_STORE.get_logger(experiment_name)
        return next(h for h in specific_logger_experiment_1.handlers if isinstance(h, logging.FileHandler)).baseFilename

    # TODO: Change for logger store to save a numpy array and just load that instead of reading a log file
    experiment_1_file = file_handle_for_experiment(experiment_name_1)
    experiment_2_file = file_handle_for_experiment(experiment_name_2)
    
    metrics = ['f1', 'accuracy', 'precision', 'recall']
    print_pair_metrics_from_files(experiment_1_file, experiment_2_file, metrics)

def save_current_info(signal, frame):
    print('Captured exit signal')
    try:
        if 'SEED_CONTROLLER' in globals():
            print('Saving SEED_CONTROLLER to {}'.format(SEED_CONTROLLER_FILE))
            SEED_CONTROLLER.dump(SEED_CONTROLLER_FILE, overwrite=True)
        SERVER_LOGGER.info('Server down and saved at {}'.format(str(datetime.now())))
    except Exception as e:
        print('ERROR in exit signal handler', e)
    finally:
        exit(0)


def setup_server_handlers(server):
    request_handlers = {}
    response_handlers = {}
    request_handlers['seed'] = find_experiment_random_states
    request_handlers['data'] = prepare_data_for_run
    request_handlers['metrics'] = receive_metrics
    response_handlers['seed'] = server.send_pyobj
    response_handlers['data'] = server.send_pyobj
    response_handlers['metrics'] = server.send_pyobj
    return request_handlers, response_handlers

def start_server():
    endpoint = SERVER_ENDPOINT_CONNEXION
    context = zmq.Context()
    server = context.socket(zmq.PAIR)
    server.bind(endpoint)
    
    handlers, resp_methods = setup_server_handlers(server)

    print("I: Service is ready at %s" % endpoint)
    # Setup interrupt handler
    signal.signal(signal.SIGINT, save_current_info)
    SERVER_LOGGER.info('Server up at {}'.format(str(datetime.now())))

    while True:
        request = server.recv_pyobj()
        if not request:
            break  # Interrupted
        print('Got', request)
        # Treat request
        try:
            resp = handlers[request['type']](**request)
        except Exception as e:
            traceback.print_exc()
            print('Error:', e)
            server.send_pyobj(e)
        else:
            resp_methods[request['type']](resp)

    server.setsockopt(zmq.LINGER, 0) # Terminate early

if __name__ == "__main__":
    start_server()
