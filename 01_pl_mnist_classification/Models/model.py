import os
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from collections import OrderedDict
from argparse import ArgumentParser

from .nnet import CNN, Linear
from .dataset import get_train_data, get_test_data

import pytorch_lightning as pl
from pytorch_lightning import _logger as log

class Model(pl.LightningModule):

    def __init__(self, hparams):
        super(Model, self).__init__()
        self.hparams = hparams

        # build nnet
        self.nnet = CNN(self.hparams)

        # dataset
        self.train_data = get_train_data(self.hparams.data_dir)
        self.test_data = get_test_data(self.hparams.data_dir)

    def forward(self, x):
        return self.nnet(x)

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
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, **kwargs)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        kwargs = {'num_workers': 7, 'pin_memory': True}
        log.info('Validation data loader called.')
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size, **kwargs)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        log.info('Test data loader called.')
        kwargs = {'num_workers': 7, 'pin_memory': True}
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size, **kwargs)

