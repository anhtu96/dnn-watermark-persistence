import os
from argparse import ArgumentParser

import numpy as np
from train_utils import *
import utils
from easydict import EasyDict
import yaml

from models.vit_small import ViT
from torchvision.models import swin_t
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms


def main(cfg):
    # print('\033[1m', '*'*5, f'Restoring for {cfg.trigger_type.split('_')[-1]} triggers, {cfg.trig_lbl[:-3]} labels', '*'*5, '\033[0m')
    utils.set_seed(cfg.seed)
    save_path = cfg.save_name.rsplit('/', 1)[0]
    os.makedirs(save_path, exist_ok=True)

    train_loader, test_loader, wm_loader, train_wm_loader, ft_loader = utils.get_data_from_config(cfg, with_mark=False)
    if cfg.method == 'app':
        train_loader_app, test_loader_app, wm_loader_app, train_wm_loader_app, ft_loader_app = utils.get_data_from_config(cfg, with_mark=True)
    net = utils.get_model_from_config(cfg, len(train_loader.dataset.classes))

    # warm up with clean data
    if cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.wd))
    elif cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), float(cfg.lr), momentum=0.9, weight_decay=float(cfg.wd))
    
    if cfg.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step, gamma=cfg.scheduler_gamma)
    elif cfg.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs, eta_min=float(cfg.lr_min))
    else:
        scheduler = None
    criterion = nn.CrossEntropyLoss()

    if cfg.method == 'app':
        evaluator = Evaluator(net, criterion, mark=True)
        trainer = APPTrainer(net, criterion, optimizer, evaluator, train_wm_loader_app, test_loader_app, wm_loader_app, scheduler=scheduler, batchsize_p=cfg.batchsize_wm, batchsize_c=cfg.batchsize_c, app_eps=cfg.app_eps, app_alpha=cfg.app_alpha)
    elif cfg.method == 'adi':
        evaluator = Evaluator(net, criterion)
        trainer = Trainer(net, criterion, optimizer, evaluator, train_wm_loader, test_loader, wm_loader=wm_loader, scheduler=scheduler, use_trigger=True)
    elif cfg.method == 'certified':
        evaluator = Evaluator(net, criterion)
        trainer = CertifiedTrainer(net, criterion, optimizer, evaluator, train_wm_loader, test_loader, wm_loader=wm_loader, scheduler=scheduler, use_trigger=True)
    elif cfg.method == 'rowback':
        evaluator = Evaluator(net, criterion)
        trainer = ROWBACKTrainer(net, criterion, optimizer, evaluator, train_wm_loader, test_loader, wm_loader=wm_loader, scheduler=scheduler, use_trigger=True)
    trainer.train(None, cfg.save_name, cfg.epochs)
    if cfg.method == 'rowback':
        frozen_layers =[trainer.net.conv1, trainer.net.bn1, trainer.net.layer1, trainer.net.layer2, trainer.net.layer3, trainer.net.layer4]
        trainer.train_freeze(None, cfg.save_name, cfg.epochs, train_wm_loader, frozen_layers=frozen_layers, wandb_project=None, eval_robust=False, eval_pretrain=False, use_wandb=False, save_every=5)


    print(evaluator.eval(test_loader))
    print(evaluator.eval(wm_loader))
    # avg_wm_acc, med_wm_acc = evaluator.eval_robust(wmloader)
    # print(f"Avg WM acc {avg_wm_acc}, Med WM acc {med_wm_acc}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=str, help='path to config')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = EasyDict(yaml.safe_load(f))
    main(cfg)