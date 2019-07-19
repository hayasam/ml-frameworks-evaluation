import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import EvaluationModel
from .weight_initializers import initialize_torch_weights_apply_fn


class Net(EvaluationModel, nn.Module):
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
    
    def initialize_weights(self, random_state: int):
        # TODO: Use random state
        # torch.random.set_rng_state(random_state)  # This seems to take only byte tensors, check if we should just mod[ulo] it and cast to uint8 tensor
        self.apply(initialize_torch_weights_apply_fn)

    def use_device(self, device_type: str) -> EvaluationModel:
        """Makes the model be run on CPU or GPU"""
        return self.to('cuda' if device_type == 'cuda' else 'cpu')

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
