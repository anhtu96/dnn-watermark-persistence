import os
import glob
import yaml
from easydict import EasyDict
import plot_trajectory, plot_surface, plot_2D


for d in glob.glob('cfg/*'):
    if 'finetune_medlr' in d and 'mix' not in d:
        files = glob.glob(d + '/*')
        for file in files:
            if 'multi' not in file:
                with open(file) as f:
                    cfg = EasyDict(yaml.safe_load(f))
                    args = plot_trajectory.plot(cfg)
                    args = plot_surface.plot(cfg, args)
                    plot_2D.plot(args)