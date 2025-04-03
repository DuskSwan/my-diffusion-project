# encoding: utf-8

import time
import os
import random

from loguru import logger
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initiate_cfg(cfg,merge_file = ''):
    '''
    Initiate the cfg object with the default config file and the extra config file. 
    The cfg will be frozen after initiation.
    '''
    if(merge_file): logger.info("Try to merge from {}.".format(merge_file))
    else: logger.info("No extra config file to merge.")

    extra_cfg = Path(merge_file)
    if extra_cfg.exists() and extra_cfg.suffix == '.yml':
        cfg.merge_from_file(extra_cfg)
        logger.info("Merge successfully.")
    else:
        logger.info("Wrong file path or file type of extra config file.")

    cfg.freeze()

    if(cfg.LOG.OUTPUT_TO_FILE): 
        logger.info("Output to file.")
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        logger.add(cfg.LOG.DIR + f'/{cfg.LOG.PREFIX}_{cur_time}.log', rotation='1 day', encoding='utf-8')
    else: logger.info("Output to console.")

def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


def make_grid(images, size=64):
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im