import logging
import os
from collections import namedtuple
import torch

import numpy as np
import configargparse
import server_interactions
import zmq
from ml_evaluation_ipc_communication import EvaluationRunIdentifier
from experiment_logger import ExperimentLogger
from metrics_dto import create_metrics_dto, metrics_dto_str
from models.models_store import ModelStore
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


def log_params(model, logger):
    logger.parameters(model.get_params_str())

def set_local_seed(seed_info, **kwargs):
    try:
        import torch
        torch.manual_seed(seed_info)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    if 'numpy' in globals():
        np.random.seed(seed_info)
    if 'np' in globals():
        np.random.seed(seed_info)

def train(model, train_loader, epoch, logger, **kwargs):
    model.train_on_data(train_data=train_loader, current_epoch=epoch, logger=logger, **kwargs)

def test(model, test_loader, logger):
    np_pred, np_target = model.test_on_data(test_loader, logger)
    # print('Output shape', np_target.shape, 'Pred shape', np_pred.shape)
    acc, pr, rec, f1 = accuracy_score(y_true=np_target, y_pred=np_pred), precision_score(y_true=np_target, y_pred=np_pred, average='macro'), recall_score(y_true=np_target, y_pred=np_pred, average='macro'), f1_score(y_true=np_target, y_pred=np_pred, average='macro')
    metrics_msg = 'accuracy: {} - precision: {} - recall: {} - f1: {}'.format(acc, pr, rec, f1)
    logger.train(metrics_msg)
    return np_pred, np_target

def validate_args(args):
    from pathlib import Path
    if args.name == '':
        raise ValueError('--name', 'name must not be empty')
    if not Path(args.log_dir).exists():
        raise IOError('Log dir {} does no exist'.format(args.log_dir))


def parse_args():
    parser = configargparse.ArgParser(default_config_files=['.env.defaults'], description='Deep Learning evaluation framework')
    parser.add('-c', '--my-config', is_config_file=True, help='config file path')
    parser.add('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add('--use-cuda', action='store_true', default=False,
                        help='Forces CUDA training', env_var='USE_CUDA')
    parser.add('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add('--evaluation-type', type=str, choices=['buggy', 'corrected', 'automl'],
                        required=True)
    parser.add('--resume-run-at', type=int, help='what run to resume training from')
    parser.add('--name', required=True, type=str)
    parser.add('--challenge',  default="mnist", type=str, required=True)
    parser.add('--data-server-endpoint', required=True, type=str, env_var='DATA_SERVER_ENDPOINT')
    # TODO: Put this as a choice (maybe dynamic?)
    parser.add('--model-library',  default="pytorch", type=str, required=True)
    parser.add('--model-name',  default="Net", type=str, required=True)
    parser.add('--runs', required=True, type=int)
    # TODO: Change when supporting more classification problems
    parser.add('--num-classes', required=True, type=int)
    parser.add('--log-dir', required=True, type=str, env_var='CLIENT_LOG_DIR')
    # Optional
    parser.add('--save-model', action='store_true', default=False, help='For Saving the current Model')

    args = parser.parse_args()
    print(parser.print_values())

    validate_args(args)
    return vars(args)

def check_cuda_availability(model_library):
    if model_library == 'pytorch':
        import torch
        return torch.cuda.is_available()
    else:
        raise ValueError('Library {} is not yet supported'.format(model_library))

def run_experiment():
    args = parse_args()
    print(args)

    runtime_params_keys = ['epochs', 'log_interval', 'log_dir', 'save_model', 'runs']
    runtime_params = {k:v for k,v in args.items() if k in runtime_params_keys}
    print('Runtime params', runtime_params)

    if args['use_cuda']:
        if not check_cuda_availability(args['model_library']):
            raise ValueError("CUDA was requested but CUDA is not available")

    EXPERIMENT_NAME = '{}_{}'.format(args['name'], args['evaluation_type'])
    run_identifier = EvaluationRunIdentifier(name=args['name'], evaluation_type=args['evaluation_type'], challenge=args['challenge'],  lib_name=args['model_library'], model_name=args['model_name'])
    logger = ExperimentLogger(EXPERIMENT_NAME, **args)

    # Get server connection
    socket, context = connect_server(args['data_server_endpoint'])

    # Request server for seed
    seed = server_interactions.request_seed(socket, run_identifier)
    logger.status('Using seed value {}'.format(seed))

    for run in range(args['resume_run_at'] or 0, args['runs']):
        current_seed = seed[run]
        # Local seed is indexed at the run
        set_local_seed(current_seed)
        model_creation_args = {'use_gpu': args['use_cuda'], 'num_classes': args['num_classes']}
        # Recreate the net for each run with new initial weights
        model = ModelStore.get_model_for_name(library=args['model_library'], name=args['model_name'], **model_creation_args)
        model.initialize_weights(current_seed)

        logger.current_run = run
        data_params = model.get_data_params()
        logger.status('Requesting data from server')
        train_data, test_data = server_interactions.prepare_data_for_run(socket, run_identifier, run, current_seed, data_params)
        logger.status('Received data from server')

        # TODO: Turn back on if necessary
        log_params(model, logger)

        model.start_training()
        for epoch in range(1, args['epochs'] + 1):
            train(model, train_data, epoch, logger, **runtime_params)
            np_pred, np_target = test(model, test_data, logger)

        # TODO (opt): Put an option if we want per epoch or per run stats
        metrics = create_metrics_dto(predictions=np_pred, target=np_target)
        logger.metrics(metrics_dto_str(metrics))
        logger.status('Sending metrics to server')
        server_interactions.send_metrics_for_run(socket, run_identifier, seed, run, metrics)

        if (args['save_model']):
            model.save(evaluation_type=args['evaluation_type'], run=run)
        log_params(model, logger)


def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = numpy.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])


def connect_server(endpoint):
    # TODO: Connect args
    print('Connecting to {}'.format(endpoint))
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect(endpoint)
    return socket, context


if __name__ == "__main__":
    run_experiment()
