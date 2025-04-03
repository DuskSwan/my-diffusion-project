import numpy as np
import matplotlib.pyplot as plt

import torch

from diffusers import DDPMScheduler
from diffusers import DDPMPipeline

from modeling import build_Unet
# from data.butterflies import butterflies_data_loader
from data import make_data_loader
from engine.train import do_train
from config import cfg
from utils import show_images, set_random_seed, initiate_cfg

def main(cfg):
    device = cfg.DEVICE
    train_dataloader = make_data_loader(dataset_name='butterflies', image_size=32, batch_size=32)
    model = build_Unet(image_size=32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    
    do_train(model, device, noise_scheduler, train_dataloader, optimizer, max_epoch=cfg.TRAIN.MAX_EPOCH)
    
    # image_pipe = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    # pipeline_output = image_pipe()
    # generated_image = pipeline_output.images[0]

    sample = torch.randn(8, 3, 32, 32).to(device)
    for i, t in enumerate(noise_scheduler.timesteps):
        with torch.no_grad():
            residual = model(sample, t).sample
        sample = noise_scheduler.step(residual, t, sample).prev_sample
    show_images(sample)
    

if __name__ == '__main__':
    set_random_seed(cfg.SEED)
    main(cfg)
