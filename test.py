import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from diffusers import DDPMScheduler
from diffusers import DDPMPipeline

from modeling import build_Unet
# from data.butterflies import butterflies_data_loader
from data import make_data_loader
from utils import show_images

def train(noise_scheduler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_Unet(image_size=32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
    train_dataloader = make_data_loader(dataset_name='butterflies', image_size=32, batch_size=32)

    losses = []

    for epoch in range(10):
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"].to(device)
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()
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

    return model

def gen_via_pipeline(model, noise_scheduler):
    image_pipe = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    pipeline_output = image_pipe()
    generated_image = pipeline_output.images[0]

    # Display the generated image using matplotlib
    plt.imshow(generated_image)
    plt.axis("off")  # Turn off axis
    plt.show()

def gen(model, noise_scheduler):
    sample = torch.randn(8, 3, 32, 32).to("cuda")
    for i, t in enumerate(noise_scheduler.timesteps):
        # Get model pred
        with torch.no_grad():
            residual = model(sample, t).sample
        # Update sample with step
        sample = noise_scheduler.step(residual, t, sample).prev_sample
    img = show_images(sample)
    plt.imshow(img)
    plt.axis("off")  # Turn off axis
    plt.show()


if __name__ == '__main__':
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    model = train(scheduler)
    gen(model, scheduler)
