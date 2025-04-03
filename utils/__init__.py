# encoding: utf-8

import time
import os
import random

from loguru import logger
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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
    """
    Given a batch of gray images x, make a grid and convert to PIL
    x: Tensor of shape (B, C, H, W) in range [-1, 1] approximately
    """
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x) # Make a grid of images
    '''
    输入通常为形如 (B, C, H, W) 的 4D 张量，例如 (16, 3, 64, 64)。
    当调用 make_grid(x, nrow=8) 时，如果批量大小 B=16，那么输出会排列成两行，
    每行 8 张图像，结果是一个 3D 张量，形状为 (C, 2*64, 8*64)，也即(3, 128, 512)
    '''  
    
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8)) # Convert to PIL Image
    
    plt.imshow(grid_im)
    '''
    传入一个二维或三维数组：
    如果是二维数组 (H, W)，会被解释为灰度图像。
    如果是三维数组 (H, W, 3) 或 (H, W, 4)，则分别对应 RGB 或 RGBA 格式的彩色图像。
    '''
    plt.axis("off") 
    plt.show()


# def make_grid(images, size=64):
#     """Given a list of PIL images, stack them together into a line for easy viewing"""
#     output_im = Image.new("RGB", (size * len(images), size))
#     for i, im in enumerate(images):
#         output_im.paste(im.resize((size, size)), (i * size, 0))
#     return output_im