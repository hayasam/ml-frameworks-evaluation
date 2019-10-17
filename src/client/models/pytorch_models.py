import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.hub import load_state_dict_from_url

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

class EvaluationVGG(EvaluationModel):
    def __init__(self, cfg='A', **kwargs):
        super(EvaluationVGG, self).__init__()
        # TODO: Parametrize loss?
        self.loss_fn = F.nll_loss
        # TODO: Check why this was 512 before and now 1
        self.use_cuda = kwargs['use_gpu']
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.optimizer_params = kwargs.get('optimizer_params', {
            'lr': 0.01,
            'momentum': 0.5
        })
        self._data_params = {'train_batch_size': 64, 'test_batch_size': 1000}
        # TODO: Parametrize this
        vgg_params = { 'num_classes': 10 }
        # TODO: Parametrize this
        self.vgg_model = vgg11(False, False, **vgg_params)

    def get_params_str(self):
        # TODO: Get all layers
        p = []
        for m in self.vgg_model.modules():
            if hasattr(m, 'weight') and isinstance(m.weight, torch.nn.parameter.Parameter):
                p.append(str(m.weight))
        return '\n'.join(p)

    def start_training(self):
        self.optimizer = optim.SGD(self.vgg_model.parameters(), **self.optimizer_params)
    
    def get_data_params(self):
        return self._data_params

    def train_on_data(self, train_data, current_epoch, logger, **kwargs):
        # Say to pytorch we are in training mode
        self.vgg_model.train()

        if self.use_cuda:
            self.vgg_model.cuda()
        train_x, train_y = train_data
        # print(train_x.shape, train_y.shape)
        for batch_idx, (np_data, np_target) in enumerate(zip(train_x, train_y)):
            # TODO: Remove this hack
            _new_data = np.repeat(np_data, 3, 1)
            # print('Original shape:', np_data.shape, 'View Shape:', _new_data.shape)
            data, target = torch.from_numpy(_new_data), torch.from_numpy(np_target)

            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.vgg_model(data)
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
        self.vgg_model.eval()
        test_loss = 0
        correct = 0
        test_x, test_y = test_data
        preds = []
        with torch.no_grad():
            for np_data, np_target in zip(test_x, test_y):
                # TODO: Remove this hack
                _new_data = np.repeat(np_data, 3, 1)
                data, target = torch.from_numpy(_new_data), torch.from_numpy(np_target)

                
                data, target = data.to(self.device), target.to(self.device)
                output = self.vgg_model(data)
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

    def initialize_weights(self, random_state: int = 0): # removed the underscore in original name _initialize_weights
        # TODO check if we need to set the random_state
        for m in self.vgg_model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        # self.apply(initialize_torch_weights_apply_fn)


# __all__ = [
#     'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
#     'vgg19_bn', 'vgg19',
# ]
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


# OG Implementation of VGG from PyTorch from 
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html
class _VGGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(_VGGG, self).__init__()
        self.features = features
        pool_size = 5
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.classifier = nn.Sequential(
            nn.Linear(512 * pool_size * pool_size, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
            # Added layers:
            nn.LogSoftmax()
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# TODO: Parameterize config in EvaluationVGG
cfgs = {
    # 'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'A': [64, 256, 'M', 512 ],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = _VGGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)



def vgg11_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)



def vgg13(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)



def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)



def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)



def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)



def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)



def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)
