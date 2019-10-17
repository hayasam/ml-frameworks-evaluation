import abc
import torch
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

class CIFARChallenge(Challenge):
    @staticmethod
    def get_subset(run: int, seed: int, train_batch_size: int, test_batch_size: int):
        # TODO: Something with params
        train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                            # transforms.Lambda(lambda v: v.numpy())
                        ])),
        batch_size=train_batch_size, shuffle=False, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                            # transforms.Lambda(lambda v: v.numpy())
                        ])),
            batch_size=test_batch_size, shuffle=False, drop_last=True)
        return train_loader, test_loader


challenges = dict()
challenges['mnist'] = MNISTChallenge()
challenges['cifar'] = CIFARChallenge()
