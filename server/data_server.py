import sys
import abc
import zmq
import numpy as np
import torch
import functools
from torchvision import datasets, transforms

class Challenge(abc.ABC):
    def __init__(self, **kwargs):
        pass
    
    @staticmethod
    def get_subset(run: int, seed: int, train_batch_size: int, test_batch_size: int):
        pass

class MNISTChallenge(Challenge):
    @staticmethod
    def get_subset(run: int, seed: int, train_batch_size: int, test_batch_size: int):
        # TODO: Something with params
        train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                            # transforms.Lambda(lambda v: v.numpy())
                        ])),
        batch_size=train_batch_size, shuffle=False, drop_last=True)    
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                            # transforms.Lambda(lambda v: v.numpy())
                        ])),
            batch_size=test_batch_size, shuffle=False, drop_last=True)
        return train_loader, test_loader


challenges = dict()
challenges['mnist'] = MNISTChallenge()

def dataset_to_numpy(data_loader: torch.utils.data.DataLoader):
    data_x, data_y = [], []
    for data, target in data_loader:
        data_x.append(data.numpy())
        data_y.append(target.numpy())
    # print(len(data_x), len(data_y), data_x[0].shape, data_y[0].shape)
    ar_x = np.array(data_x)
    ar_y =  np.array(data_y)
    return ar_x, ar_y

@functools.lru_cache()
def prepare_data_for_run(challenge: str, run: int, seed: int, train_batch_size: int, test_batch_size: int, **kwargs):
    challenge = challenges[challenge]
    train_loader, test_loader = challenge.get_subset(run=run, seed=seed,
                                                     train_batch_size=train_batch_size,
                                                     test_batch_size=test_batch_size)
    np_train = dataset_to_numpy(train_loader)
    np_test = dataset_to_numpy(test_loader)
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

def set_experiment_seed(experiment_name, seed, **kwargs):
    previous_seed = SEED_INFO.get(experiment_name)
    if previous_seed != seed:
        print('Setting a new seed (previously {}) now {}'.format(previous_seed, seed))
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if 'numpy' in globals():
        np.random.seed(seed)
    elif 'np' in globals():
        np.random.seed(seed)
    SEED_INFO[experiment_name] = seed
    return True

SEED_INFO = {}

def start_server():
    # TODO: Server arg params
    endpoint = "tcp://*:90002"
    context = zmq.Context()
    server = context.socket(zmq.PAIR)
    server.bind(endpoint)
    handlers = {}
    resp_methods = {}
    handlers['seed'] = set_experiment_seed
    handlers['data'] = prepare_data_for_run
    resp_methods['seed'] = server.send_pyobj
    resp_methods['data'] = server.send_pyobj
    print("I: Service is ready at %s" % endpoint)
    while True:
        request = server.recv_pyobj()
        if not request:
            break  # Interrupted
        print('Got', request)
        # Treat request
        try:
            resp = handlers[request['type']](**request)
        except Exception as e:
            print('Error:', e)
            server.send_pyobj(e)
        else:
            resp_methods[request['type']](resp)

    server.setsockopt(zmq.LINGER, 0) # Terminate early

if __name__ == "__main__":
    start_server()
