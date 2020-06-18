"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""

# from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import Trainer 
from argparse import ArgumentParser
from pytorch_lightning import LightningModule
import torch
from Models.pl_linear import Model
# from Models.pl_cnn import Model

# sets seeds for numpy, torch, etc...
# must do for DDP to work well
# seed_everything(123)

if __name__ == '__main__':
    # add args from trainer
    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    parser = Model.add_model_specific_args(parser) # StaticMethod

    # parse params
    args = parser.parse_args()
    model = Model(hparams=args)

    # most basic trainer, uses good defaults
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)

    PATH = "/work4/zhangyang/pl-test/01_pl_mnist/lightning_logs/version_0/checkpoints/epoch=1.ckpt"
    pretrained_model = model.load_from_checkpoint(PATH, hparams=args)
    a = torch.ones((2, 28, 28), dtype=torch.float32)
    a = pretrained_model(a)
    print(a.shape)
    print(a)
