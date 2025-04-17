#!/bin/bash
python train_initial.py cfg/initial_train/vit_cifar10_textoverlay_many.yml
python train_finetune.py cfg/finetune/vit_cifar10_textoverlay_many.yml
python train_restore.py cfg/restore/vit_cifar10_textoverlay_many.yml

python train_initial.py cfg/initial_train/vit_cifar10_textoverlay_one.yml
python train_finetune.py cfg/finetune/vit_cifar10_textoverlay_one.yml
python train_restore.py cfg/restore/vit_cifar10_textoverlay_one.yml