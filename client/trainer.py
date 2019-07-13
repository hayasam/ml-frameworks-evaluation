import torch
import zmq
import argparse
import logging
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import namedtuple

# TODO: Put this in its own package
MetricsDTO = namedtuple('MetricsDTO', 'accuracy precision recall f1_score')
def metrics_dto_str(metrics_dto: MetricsDTO) -> str:
    s = 'accuracy: {} - precision: {} - recall: {} - f1: {}'.format(metrics_dto.accuracy, metrics_dto.precision, metrics_dto.recall, metrics_dto.f1_score)
    return s

DEFAULT_LOG_DIR = '.'


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


def log_params(model, logger):
    # TODO
    logger.parameters(model.conv1.weight)
    logger.parameters(model.conv2.weight)
    logger.parameters(model.fc1.weight)
    logger.parameters(model.fc2.weight)

def set_seed(seed_info, **kwargs):
    torch.manual_seed(seed_info)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if 'numpy' in globals():
        np.random.seed(seed_info)
    if 'np' in globals():
        np.random.seed(seed_info)

class ExperimentLogger(object):
    def __init__(self, experiment_name, **kwargs):
        self.name = experiment_name
        self.current_run = 0
        self.base_logger = logging.getLogger()
        self.bp = Path(kwargs.get('log_dir', DEFAULT_LOG_DIR))
        assert self.bp.exists()
        # Need to cast to str because of python 3.5
        self._training_log_handler = logging.FileHandler(str(self.bp / '{}.training.log'.format(self.name)))
        
        self.train_logger = self.base_logger.getChild('training')
        self.train_logger.addHandler(self._training_log_handler)
        self.train_logger.setLevel(logging.DEBUG)
        
        self.parameters_logger = self.base_logger.getChild('parameter')
        # Need to cast to str because of python 3.5
        self._parameter_log_handler = logging.FileHandler(str(self.bp / '{}.parameters.log'.format(self.name)))
        self.parameters_logger.addHandler(self._parameter_log_handler)
        self.parameters_logger.setLevel(logging.DEBUG)

        self.metrics_logger = self.base_logger.getChild('metrics')
        # Need to cast to str because of python 3.5
        self.metrics_log_handler = logging.FileHandler(str(self.bp / '{}.metrics.log'.format(self.name)))
        self.metrics_logger.addHandler(self.metrics_log_handler)
        self.metrics_logger.setLevel(logging.DEBUG)

        self.base_logger.setLevel(logging.DEBUG)
        self.base_log_handler = logging.StreamHandler()
        self.base_log_handler.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(message)s'))
        self.base_logger.addHandler(self.base_log_handler)


    def train(self, *args, **kwargs):
        self.train_logger.debug('run {}'.format(self.current_run))
        self.train_logger.debug(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        self.parameters_logger.debug('run {}'.format(self.current_run))
        self.parameters_logger.debug(*args, **kwargs)

    def metrics(self, *args, **kwargs):
        self.metrics_logger.debug('run {}'.format(self.current_run))
        self.metrics_logger.debug(*args, **kwargs)
    
    def status(self, *args, **kwargs):
        self.base_logger.debug(*args, **kwargs)


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
    print(train_x.shape, train_y.shape)
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
    print('Output shape', np_target.shape, 'Pred shape', np_pred.shape)
    acc, pr, rec, f1 = accuracy_score(y_true=np_target, y_pred=np_pred), precision_score(y_true=np_target, y_pred=np_pred, average='macro'), recall_score(y_true=np_target, y_pred=np_pred, average='macro'), f1_score(y_true=np_target, y_pred=np_pred, average='macro')
    metrics_msg = 'accuracy: {} - precision: {} - recall: {} - f1: {}'.format(acc, pr, rec, f1)
    logger.train(metrics_msg)
    return np_pred, np_target


def metrics_dto(predictions, target) -> MetricsDTO:
    np_pred = predictions
    if isinstance(np_pred, list):
        np_pred = np.array(np_pred)
    np_pred, np_target= np_pred.ravel(), target.ravel()
    acc, pr, rec, f1 = accuracy_score(y_true=np_target, y_pred=np_pred), precision_score(y_true=np_target, y_pred=np_pred, average='macro'), recall_score(y_true=np_target, y_pred=np_pred, average='macro'), f1_score(y_true=np_target, y_pred=np_pred, average='macro')
    return MetricsDTO(accuracy=acc, precision=pr, recall=rec, f1_score=f1)

def send_metrics_for_run(socket, experiment_name: str, seed: int, run: int, metrics_dto: MetricsDTO):
    metrics_obj = create_calculated_metrics_message(run, challenge='mnist', seed=seed, metrics=metrics_dto)
    socket.send_pyobj(metrics_obj)
    # Receive response
    obj = socket.recv_pyobj()
    if not obj:
        # TODO: Custom exception
        raise Exception('Metrics were not synced')


def create_calculated_metrics_message(run: int, challenge: str, seed: int, metrics: MetricsDTO):
    # TODO: Just send the DTO (implies making a middle package)
    obj = {'type': 'metrics', 'challenge': challenge, 'run': run, 'seed': seed, 'value': metrics._asdict()}
    return obj


# TODO: Create a clean interface object (ex: DTO)
def create_data_query(run: int, challenge: str, data_params: dict):
    obj = {'type': 'data', 'challenge': challenge, 'run': run, **data_params}
    return obj

def prepare_data_for_run(socket, run: int, data_params: dict):
    socket.send_pyobj(create_data_query(challenge='mnist', run=run, data_params=data_params))
    msg = socket.recv_pyobj()
    train_data, test_data = msg
    print(train_data[0].shape, train_data[1].shape)
    return train_data, test_data


def set_server_seed(socket, EXPERIMENT_NAME, seed):
    request = {'type': 'seed', 'seed': seed, 'experiment_name': EXPERIMENT_NAME}
    try:
        socket.send_pyobj(request)
        resp = socket.recv_pyobj()
    except Exception as e:
        print('Failed with', e)
        raise e
    else:
        print(resp)
        if not isinstance(resp, bool):
            raise ValueError("Failed to set seed on server:", resp)

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
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--type', type=str, choices=['buggy', 'corrected', 'automl'],
                        required=True)
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
    port = 90002
    endpoint = "tcp://localhost:{}".format(port)
    socket, context = connect_server('tcp://localhost:90002')

    # Advise server that this expirement will use this seed
    set_server_seed(socket, EXPERIMENT_NAME, args['seed'])

    set_seed(args['seed'])
    x = Net()
    x.apply(initialize_layer_weights)
    device = torch.device("cuda" if args['use_cuda'] else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args['use_cuda'] else {}

    for run in range(args['runs']):
        logger.current_run = run
        
        logger.status('Requesting data from server')
        train_data, test_data = prepare_data_for_run(socket, run, data_params)
        logger.status('Received data from server')

        model = x.to(device)
        # TODO: Turn back on if necessary
        # log_params(model, logger)
        optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
        for epoch in range(1, args['epochs'] + 1):
            train(args, model, device, train_data, optimizer, epoch, logger=logger)
            np_pred, np_target = test(args, model, device, test_data, logger=logger)

        # TODO (opt): Put an option if we want per epoch or per run stats
        metrics = metrics_dto(predictions=np_pred, target=np_target)
        logger.metrics(metrics_dto_str(metrics))
        logger.status('Sending metrics to server')
        send_metrics_for_run(socket, EXPERIMENT_NAME, args['seed'], run, metrics)

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


def connect_server(endpoint):
    # TODO: Connect args
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect(endpoint)
    return socket, context

if __name__ == "__main__":
    run_experiment()
