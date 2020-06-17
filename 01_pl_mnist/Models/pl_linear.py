"""
This file defines the core research contribution   
"""

import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from argparse import ArgumentParser

import pytorch_lightning as pl

class Model(pl.LightningModule):

    def __init__(self, hparams):
        super(Model, self).__init__()
        # not the best model...
        self.hparams = hparams
        self.l1 = torch.nn.Linear(28 * 28, 256)
        self.l2 = torch.nn.Linear(256, 256)
        self.l3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.l1(x.view(x.size(0), -1)))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        return x

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {'avg_val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        tensorboard_logs = {'test_val_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(MNIST(self.hparams.data_dir, train=True, download=True, \
            transform=transforms.ToTensor()), batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(self.hparams.data_dir, train=True, download=True, \
                transform=transforms.ToTensor()), batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(self.hparams.data_dir, train=True, download=True, \
                transform=transforms.ToTensor()), batch_size=self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=0.01, type=float)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--data_dir', default="./data", type=str)

        # training specific (for this model)
        # parser.add_argument('--max_nb_epochs', default=2, type=int)

        return parser
