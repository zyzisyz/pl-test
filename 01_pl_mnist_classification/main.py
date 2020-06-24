"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""

# from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import Trainer 
from argparse import ArgumentParser
from pytorch_lightning import LightningModule
import torch
from Models import Model, add_model_specific_args
from torchvision.datasets import FashionMNIST as MNIST
import torchvision.transforms as transforms

# sets seeds for numpy, torch, etc...
# must do for DDP to work well
# seed_everything(123)

if __name__ == '__main__':
    # add args from trainer
    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    # give the module a chance to add own params
    parser = add_model_specific_args(parser) 

    # parse params
    args = parser.parse_args()
    model = Model(hparams=args)

    # most basic trainer, uses good defaults
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)

    # test
    PATH = "/work4/zhangyang/pl-test/01_pl_mnist/lightning_logs/version_0/checkpoints/epoch=51.ckpt"
    pretrained_model = model.load_from_checkpoint(PATH, hparams=args)
    pretrained_model.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    dataset = MNIST(args.data_dir, train=False, download=True, transform=transform)

    data, label = dataset[100]
    data = torch.unsqueeze(data, 0)
    input(data.shape)
    predict_label = torch.argmax(pretrained_model(data), dim=1)
    predict_label = predict_label.cpu().to_numpy()

    print("label: ", label)
    print("predict_label: ", predict_label)

