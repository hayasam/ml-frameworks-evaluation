import os
import argparse
import logging
from collections import namedtuple

import numpy as np
import server_interactions
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import zmq
from experiment_logger import ExperimentLogger
from metrics_dto import create_metrics_dto, metrics_dto_str
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# TODO: Create env file?
DEFAULT_ENDPOINT = 'tcp://localhost:9002'
DATA_SERVER_ENDPOINT = os.getenv('DATA_SERVER_ENDPOINT', DEFAULT_ENDPOINT)

def log_params(model, logger):
    # TODO
    logger.parameters(model.conv1.weight)
    logger.parameters(model.conv2.weight)
    logger.parameters(model.fc1.weight)
    logger.parameters(model.fc2.weight)

def set_local_seed(seed_info, **kwargs):
    torch.manual_seed(seed_info)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if 'numpy' in globals():
        np.random.seed(seed_info)
    if 'np' in globals():
        np.random.seed(seed_info)

class ChallengeRunIdentifier(dict):
    props = ['challenge', 'run', 'seed']
    @classmethod
    def from_values(cls, **kwargs):
        _underlying = {k:v for k,v in kwargs.items() if k in ChallengeRunIdentifier.props}
        return cls(_underlying)


def initialize_layer_weights(module):
    if hasattr(module, 'weight') and isinstance(module.weight, torch.nn.parameter.Parameter):
        torch.nn.init.xavier_uniform_(module.weight)

def train(args, model, device, train_loader, optimizer, epoch, logger):
    model.train()
    train_x, train_y = train_loader
    # print(train_x.shape, train_y.shape)
    for batch_idx, (np_data, np_target) in enumerate(zip(train_x, train_y)):
        data, target = torch.from_numpy(np_data), torch.from_numpy(np_target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            message = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * train_x.shape[1], train_y.size,
            100. * batch_idx / train_x.shape[0], loss.item())
            logger.train(message)

def test(args, model, device, test_loader, logger):
    model.eval()
    test_loss = 0
    correct = 0
    test_x, test_y = test_loader
    preds = []
    with torch.no_grad():
        for np_data, np_target in zip(test_x, test_y):
            data, target = torch.from_numpy(np_data), torch.from_numpy(np_target)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            preds.append(pred.cpu().numpy())
    total_n_examples = test_y.size  # test_x.shape[0] * test_x.shape[1]
    test_loss /= total_n_examples

    message = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total_n_examples,
        100. * correct / total_n_examples)
    logger.train(message)

    np_pred = np.array(preds).ravel()
    np_target = test_y.ravel()
    # print('Output shape', np_target.shape, 'Pred shape', np_pred.shape)
    acc, pr, rec, f1 = accuracy_score(y_true=np_target, y_pred=np_pred), precision_score(y_true=np_target, y_pred=np_pred, average='macro'), recall_score(y_true=np_target, y_pred=np_pred, average='macro'), f1_score(y_true=np_target, y_pred=np_pred, average='macro')
    metrics_msg = 'accuracy: {} - precision: {} - recall: {} - f1: {}'.format(acc, pr, rec, f1)
    logger.train(metrics_msg)
    return np_pred, np_target


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Forces CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--type', type=str, choices=['buggy', 'corrected', 'automl'],
                        required=True)
    parser.add_argument('--resume-run-at', type=int, help='what run to resume training from')
    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--runs', required=True, type=int)
    parser.add_argument('--log-dir', required=True, type=str)
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    return vars(args)


def run_experiment():
    args = parse_args()
    print(args)
    data_params_keys = ['batch_size', 'train_batch_size', 'test_batch_size', 'seed']
    model_params_keys = ['lr', 'momentum', 'use_cuda', 'type']
    runtime_params_keys = ['epochs', 'log_interval', 'log_dir', 'save_model', 'runs']
    
    data_params = {k:v for k,v in args.items() if k in data_params_keys}
    model_params = {k:v for k,v in args.items() if k in model_params_keys}
    runtime_params = {k:v for k,v in args.items() if k in runtime_params_keys}
    print('Data params', data_params)
    print('model params', model_params)
    print('Runtime params', runtime_params)
    if args['use_cuda'] and not torch.cuda.is_available():
        # TODO put logger
        raise ValueError("CUDA was requested but CUDA is not available")

    EXPERIMENT_NAME = '{}_{}'.format(args['name'], args['type'])
    logger = ExperimentLogger(EXPERIMENT_NAME, **args)

    # Get server connection
    socket, context = connect_server(DATA_SERVER_ENDPOINT)

    # Request server for seed
    seed = server_interactions.request_seed(socket, EXPERIMENT_NAME)
    logger.status('Using seed value {}'.format(seed))

    for run in range(args.get('resume_run_at', 0), args['runs']):
        # Local seed is indexed at the run
        set_local_seed(seed[run])
        # Recreate the net for each run with new initial weights
        x = Net()
        x.apply(initialize_layer_weights)
        device = torch.device("cuda" if args['use_cuda'] else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': True} if args['use_cuda'] else {}
        logger.current_run = run
        
        logger.status('Requesting data from server')
        train_data, test_data = server_interactions.prepare_data_for_run(socket, EXPERIMENT_NAME, run, seed[run], data_params)
        logger.status('Received data from server')

        model = x.to(device)
        # TODO: Turn back on if necessary
        # log_params(model, logger)
        optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
        for epoch in range(1, args['epochs'] + 1):
            train(args, model, device, train_data, optimizer, epoch, logger=logger)
            np_pred, np_target = test(args, model, device, test_data, logger=logger)

        # TODO (opt): Put an option if we want per epoch or per run stats
        metrics = create_metrics_dto(predictions=np_pred, target=np_target)
        logger.metrics(metrics_dto_str(metrics))
        logger.status('Sending metrics to server')
        server_interactions.send_metrics_for_run(socket, EXPERIMENT_NAME, seed, run, metrics)

        if (args['save_model']):
            torch.save(model.state_dict(), "mnist_cnn_{}.pt".format(args['type']))
        log_params(model, logger)


def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = numpy.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])


def connect_server(endpoint=DEFAULT_ENDPOINT):
    # TODO: Connect args
    print('Connecting to {}'.format(endpoint))
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect(endpoint)
    return socket, context


if __name__ == "__main__":
    run_experiment()
