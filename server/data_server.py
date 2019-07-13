import functools
import logging
import signal
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import zmq
from challenges import challenges as CHALLENGES
from metrics_logger_store import MetricsLoggerStore
from seed_controller import SeedController

DEFAULT_SEED_FILE = '.seed_control.pickle'

SEED_CONTROLLER = SeedController.from_saved_file(DEFAULT_SEED_FILE)
LOGGER_STORE = MetricsLoggerStore(base_path='.logs')

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

@functools.lru_cache()
def prepare_data_for_run(challenge: str, run: int, seed: int, train_batch_size: int, test_batch_size: int, **kwargs):
    challenge = CHALLENGES[challenge]
    train_loader, test_loader = challenge.get_subset(run=run, seed=seed,
                                                     train_batch_size=train_batch_size,
                                                     test_batch_size=test_batch_size)
    np_train = dataset_to_numpy(train_loader)
    np_test = dataset_to_numpy(test_loader)
    print('Sending', _dataset_size(np_train, np_test))
    return np_train, np_test

# From ZeroMQ's doc
def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)


def find_experiment_seed(**kwargs):
    experiment_name = kwargs['experiment_name']
    seed_info = SEED_CONTROLLER.get_seeds(experiment_name)
    print('Seed for {} has value {}'.format(experiment_name, seed_info))
    return seed_info


def receive_metrics(**kwargs):
    # TODO: Do something with kwargs['run']?
    identifier = '{}_{}'.format(kwargs['experiment_name'], kwargs['challenge'])
    print('Received metrics from {}'.format(identifier))
    m = kwargs['value']
    metrics_msg = 'run: {} - accuracy: {} - precision: {} - recall: {} - f1: {}'.format(kwargs['run'], m['accuracy'], m['precision'], m['recall'], m['f1_score'])
    print(metrics_msg)
    specific_logger = LOGGER_STORE.get_logger(experiment_name=identifier)
    specific_logger.debug(metrics_msg)
    return True

def pair_stats_between_experiments(experiment_name_1, experiment_name_2):
    # TODO: Use Wilcoxon / Mann Whitney
    import scipy.stats
    # metrics = {'accuracy', 'precision', 'recall', 'f1_score'}
    def aggregate_metric(experiment_name, metric):
        pass

    metrics = {'f1_score'}
    for metric in metrics:
        m_1 = aggregate_metric(experiment_name_1, metric)
        m_2 = aggregate_metric(experiment_name_2, metric)
        # Our metrics are continuous, so correction for continuity?
        w, p_w = scipy.stats.wilcoxon(m_1, m_2, correction=False)
        mn, p_mn = scipy.stats.wilcoxon(m_1, m_2, correction=False)
        print('Wilcoxon p-value of ', p_w)
        print('Mann-Whitney p-value of ', p_mn)

def save_current_info(signal, frame):
    print('Captured exit signal')
    if 'SEED_CONTROLLER' in globals():
        print('Saving SEED_CONTROLLER to {}'.format(DEFAULT_SEED_FILE))
        SEED_CONTROLLER.dump(DEFAULT_SEED_FILE, overwrite=True)
    exit(0)


def setup_server_handlers(server):
    request_handlers = {}
    response_handlers = {}
    request_handlers['seed'] = find_experiment_seed
    request_handlers['data'] = prepare_data_for_run
    request_handlers['metrics'] = receive_metrics
    response_handlers['seed'] = server.send_pyobj
    response_handlers['data'] = server.send_pyobj
    response_handlers['metrics'] = server.send_pyobj
    return request_handlers, response_handlers

def start_server():
    # TODO: Server arg params
    endpoint = "tcp://*:90002"
    context = zmq.Context()
    server = context.socket(zmq.PAIR)
    server.bind(endpoint)
    
    handlers, resp_methods = setup_server_handlers(server)

    print("I: Service is ready at %s" % endpoint)
    # Setup interrupt handler
    signal.signal(signal.SIGINT, save_current_info)

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
