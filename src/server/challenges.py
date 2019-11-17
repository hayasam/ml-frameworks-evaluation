import abc
import torch
from torchvision import datasets, transforms

DEFAULT_DATA_ROOT = './data'
DEFAULT_DATASET_DOWNLOAD = True

class Challenge(abc.ABC):
    def __init__(self, **kwargs):
        self.data_root = kwargs.get('data_root', DEFAULT_DATA_ROOT)
        self.download = kwargs.get('download', DEFAULT_DATASET_DOWNLOAD)
    
    @abc.abstractmethod
    def get_subset(run: int, seed: int, train_batch_size: int, test_batch_size: int):
        pass

class MNISTChallenge(Challenge):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_subset(self, run: int, seed: int, train_batch_size: int, test_batch_size: int):
        # TODO: Something with params
        train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(self.data_root, train=True, download=self.download,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                            # transforms.Lambda(lambda v: v.numpy())
                        ])),
        batch_size=train_batch_size, shuffle=False, drop_last=True)    
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(self.data_root, train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                            # transforms.Lambda(lambda v: v.numpy())
                        ])),
            batch_size=test_batch_size, shuffle=False, drop_last=True)
        return train_loader, test_loader

class CIFARChallenge(Challenge):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_subset(self, run: int, seed: int, train_batch_size: int, test_batch_size: int):
        # TODO: Something with params
        train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(self.data_root, train=True, download=self.download,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                            # transforms.Lambda(lambda v: v.numpy())
                        ])),
        batch_size=train_batch_size, shuffle=False, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(self.data_root, train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                            # transforms.Lambda(lambda v: v.numpy())
                        ])),
            batch_size=test_batch_size, shuffle=False, drop_last=True)
        return train_loader, test_loader


def get_challenges(data_root: str, download=DEFAULT_DATASET_DOWNLOAD):
challenges = dict()
    challenges['mnist'] = MNISTChallenge(data_root=data_root, download=download)
    challenges['cifar'] = CIFARChallenge(data_root=data_root, download=download)
    return challenges
