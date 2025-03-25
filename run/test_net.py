# encoding: utf-8

import argparse
import os
import sys
from os import mkdir

import torch

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.example_inference import inference
from modeling import build_model


def main():

    model = build_model(cfg)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT))
    val_loader = make_data_loader(cfg, is_train=False)

    inference(cfg, model, val_loader)


if __name__ == '__main__':
    main()
