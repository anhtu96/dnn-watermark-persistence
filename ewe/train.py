"""
This script trains null models given a configuration file (see configs)
"""

import argparse
import json
from datetime import datetime
from shutil import copyfile

import mlconfig
import torch
import os
from torchvision import datasets, transforms

from wrt.classifiers import PyTorchClassifier
from wrt.utils import reserve_gpu, get_max_index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/cifar10/train_configs/resnet.yaml',
                        help="Path to config file. Determines all training params.")
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument("--gpu", type=str, default=None, help="Which GPU to use. Defaults to GPU with least memory.")

    return parser.parse_args()


def __load_model(model, optimizer, image_size, num_classes):
    """ Loads a source model from a directory and wraps it into a pytorch classifier.
    """
    criterion = torch.nn.CrossEntropyLoss()

    model = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, image_size, image_size),
        nb_classes=num_classes
    )
    return model


def main():
    # Takes more time at startup, but optimizes runtime.
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    reserve_gpu(args.gpu)

    config = mlconfig.load(args.config)
    print(config)

    # Create output folder.
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    output_dir = os.path.join(config.output_dir,
                              f"{get_max_index(config.output_dir, suffix='null_model').zfill(5)}_null_model")
    os.makedirs(output_dir)

    # Save the cmd line arguments.
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    # Copy the config (.yaml) file.
    path, filename = os.path.split(args.config)
    copyfile(args.config, os.path.join(output_dir, filename))

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    model: torch.nn.Sequential = config.model()
    model = model.to(device)

    optimizer = config.optimizer(model.parameters())
    scheduler = config.scheduler(optimizer)

    model: PyTorchClassifier = __load_model(model,
                                            optimizer=optimizer,
                                            image_size=config.model.image_size,
                                            num_classes=config.model.num_classes)

    # train_loader = config.dataset(train=True)
    # valid_loader = config.dataset(train=False)
    data_path = '../data/CIFAR10'
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = datasets.ImageFolder(os.path.join(data_path, 'with_trigger/train'), transform=transform_train)
    testset = datasets.ImageFolder(os.path.join(data_path, 'test'), transform=transform_test)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=256, shuffle=True, num_workers=8, drop_last=True)

    valid_loader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=True, num_workers=8, drop_last=True)

    trainer = config.trainer(model=model, train_loader=train_loader, valid_loader=valid_loader,
                             scheduler=scheduler,  device=device, output_dir=output_dir)

    if args.resume is not None:
        trainer.resume(args.resume)

    train_metric = trainer.fit()

    test_acc = trainer.evaluate()[1].value
    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump({"test_acc": test_acc}, f)

    with open(os.path.join(output_dir, "history.json"), "w") as f:
        all_metrics = {
            **train_metric
        }
        json.dump(all_metrics, f)


if __name__ == "__main__":
    main()
