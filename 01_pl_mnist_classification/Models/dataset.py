#!/usr/bin/env python
# coding=utf-8

from torchvision.datasets import FashionMNIST as MNIST
import torchvision.transforms as transforms

def get_train_data(data_dir):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    return MNIST(data_dir, train=True, download=True, \
            transform=transform)

def get_test_data(data_dir):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    return MNIST(data_dir, train=False, download=True, \
            transform=transform)

