import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_model import EvaluationModel
from .weight_initializers import initialize_torch_weights_apply_fn


class Net(EvaluationModel, nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.use_cuda = kwargs['use_gpu']
        self.optimizer_params = kwargs.get('optimizer_params', {
            'lr': 0.01,
            'momentum': 0.5
        })
        self.loss_fn = F.nll_loss
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self._data_params = {'train_batch_size': 64, 'test_batch_size': 1000}

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def initialize_weights(self, random_state: int):
        # TODO: Use random state
        # torch.random.set_rng_state(random_state)  # This seems to take only byte tensors, check if we should just mod[ulo] it and cast to uint8 tensor
        self.apply(initialize_torch_weights_apply_fn)

    def save(self, evaluation_type: str, run: int):
        torch.save(self.state_dict(), "mnist_cnn_{}.pt".format(evaluation_type))

    def get_params_str(self):
        p = [
            str(self.conv1.weight),
            str(self.conv2.weight),
            str(self.fc1.weight),
            str(self.fc2.weight),
        ]
        return '\n'.join(p)

    def start_training(self):
        self.optimizer = optim.SGD(self.parameters(), **self.optimizer_params)
    
    def get_data_params(self):
        return self._data_params

    def train_on_data(self, train_data, current_epoch, logger, **kwargs):
        self.train()
        if self.use_cuda:
            self.cuda()
        train_x, train_y = train_data
        # print(train_x.shape, train_y.shape)
        for batch_idx, (np_data, np_target) in enumerate(zip(train_x, train_y)):
            data, target = torch.from_numpy(np_data), torch.from_numpy(np_target)
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % kwargs['log_interval'] == 0:
                message = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                current_epoch, batch_idx * train_x.shape[1], train_y.size,
                100. * batch_idx / train_x.shape[0], loss.item())
                logger.train(message)

    def test_on_data(self, test_data, logger, **kwargs):
        """Must return a tuple (np_predictions, np_target)"""
        self.eval()
        test_loss = 0
        correct = 0
        test_x, test_y = test_data
        preds = []
        with torch.no_grad():
            for np_data, np_target in zip(test_x, test_y):
                data, target = torch.from_numpy(np_data), torch.from_numpy(np_target)
                data, target = data.to(self.device), target.to(self.device)
                output = self(data)
                test_loss += self.loss_fn(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                preds.append(pred.cpu().numpy())
        total_n_examples = test_y.size
        test_loss /= total_n_examples

        message = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, total_n_examples,
            100. * correct / total_n_examples)
        logger.train(message)

        np_pred = np.array(preds).ravel()
        np_target = test_y.ravel()
        
        return np_pred, np_target
