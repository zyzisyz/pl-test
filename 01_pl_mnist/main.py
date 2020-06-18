"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""

# from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import Trainer 
from argparse import ArgumentParser
from Models.pl_cnn import Model
# from Models.pl_linear import Model

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
    input(args.learning_rate)
    model = Model(hparams=args)

    # most basic trainer, uses good defaults
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)
    trainer.test()

