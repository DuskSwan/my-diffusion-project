# encoding: utf-8

import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

def do_train(model, device, noise_scheduler, train_dataloader, optimizer, max_epoch=50):
    losses = []
    for epoch in range(max_epoch):
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"].to(device)
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 5 == 0:
            loss_last_epoch = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
            print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")

    # fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    # axs[0].plot(losses)
    # axs[1].plot(np.log(losses))
    # plt.show()


