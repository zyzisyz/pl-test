"""
This file defines the core research contribution   
"""

import os
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST as MNIST
#from torchvision.datasets import MNIST

from collections import OrderedDict
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import _logger as log

class Model(pl.LightningModule):

    def __init__(self, hparams):
        super(Model, self).__init__()
        self.hparams = hparams

        # build nnet
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def loss(self, labels, logits):
        nll = F.nll_loss(logits, labels)
        return nll

    def training_step(self, batch, batch_idx):
        """
        # REQUIRED
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """

        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        # calculate loss
        loss_val = self.loss(y, y_hat)
        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        return output

    def validation_step(self, batch, batch_idx):
        """
        # OPTIONAL
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        x, y = batch
        y_hat = self.forward(x)

        loss_val = self.loss(y, y_hat)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': val_acc,
        })

        return output

    def validation_epoch_end(self, outputs):
        """
        # OPTIONAL
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.

        # if we want to return just the average in this case
        # return torch.stack(outputs).mean()
        """
        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result


    def test_step(self, batch, batch_idx):
        """
        # OPTIONAL
        Lightning calls this during testing, similar to `validation_step`,
        with the data from the test dataloader passed in as `batch`.
        """
        output = self.validation_step(batch, batch_idx)
        # Rename output keys
        output['test_loss'] = output.pop('val_loss')
        output['test_acc'] = output.pop('val_acc')

        return output

    def test_epoch_end(self, outputs):
        """
        # OPTIONAL
        Called at the end of test to aggregate outputs, similar to `validation_epoch_end`.
        :param outputs: list of individual outputs of each test step
        """
        results = self.validation_step_end(outputs)

        # rename some keys
        results['progress_bar'].update({
            'test_loss': results['progress_bar'].pop('val_loss'),
            'test_acc': results['progress_bar'].pop('val_acc'),
        })
        results['log'] = results['progress_bar']
        results['test_loss'] = results.pop('val_loss')

        return results

    def configure_optimizers(self):
        """
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        kwargs = {'num_workers': 7, 'pin_memory': True}
        log.info('Training data loader called.')
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        return DataLoader(MNIST(self.hparams.data_dir, train=True, download=True, \
                transform=transform), batch_size=self.hparams.batch_size, **kwargs)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        kwargs = {'num_workers': 7, 'pin_memory': True}
        log.info('Validation data loader called.')
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        return DataLoader(MNIST(self.hparams.data_dir, train=False, download=True, \
                transform=transform), batch_size=self.hparams.batch_size, **kwargs)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        log.info('Test data loader called.')
        kwargs = {'num_workers': 7, 'pin_memory': True}
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        return DataLoader(MNIST(self.hparams.data_dir, train=False, download=True, \
                transform=transform), batch_size=self.hparams.batch_size, **kwargs)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=0.01, type=float)
        parser.add_argument('--batch_size', default=400, type=int)
        parser.add_argument('--data_dir', default="./data", type=str)

        return parser

