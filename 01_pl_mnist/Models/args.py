#!/usr/bin/env python
# coding=utf-8

from argparse import ArgumentParser

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

