# encoding: utf-8

import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

# import pytorch_lightning as pl
import lightning as pl

class DiffusionLightning(pl.LightningModule):
    def __init__(self, model, noise_scheduler, lr):
        super().__init__()
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.lr = lr

    def forward(self, noisy_images, timesteps):
        # forward 返回模型预测的噪声
        return self.model(noisy_images, timesteps, return_dict=False)[0]

    def training_step(self, batch, batch_idx):
        # 假设 batch 为一个 tensor，若你的 dataloader 返回字典，请自行修改这里的获取方式
        clean_images = batch.to(self.device)
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bs,),
            device=clean_images.device
        ).long()
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        noise_pred = self(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        
        # 记录损失
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer


def do_train(model, device, noise_scheduler, train_dataloader, max_epoch=50, lr=4e-4):
    """
    使用 PyTorch Lightning 训练 diffusion 模型。
    
    参数：
      model: 待训练的 diffusion 模型（例如 Unet）
      noise_scheduler: 噪声调度器，需提供属性 config.num_train_timesteps 和方法 add_noise
      train_dataloader: 训练数据加载器，返回的 batch 是一个 tensor（图像数据）
      max_epoch: 训练的总 epoch 数（默认 50）
      lr: 学习率（默认 4e-4）
    
    返回：
      训练好的模型（即传入的 model，其参数已被更新）
    """
    # 实例化 Lightning 模块
    lightning_model = DiffusionLightning(model, noise_scheduler, lr)
    
    # 根据当前环境选择是否使用 GPU
    trainer = pl.Trainer(max_epochs=max_epoch)
    trainer.fit(lightning_model, train_dataloader)
    
    return model

# def do_train(model, device, noise_scheduler, train_dataloader, optimizer, max_epoch=50):
#     losses = []
#     for epoch in range(max_epoch):
#         for step, batch in enumerate(train_dataloader):
#             clean_images = batch.to(device)
#             noise = torch.randn(clean_images.shape).to(clean_images.device)
#             bs = clean_images.shape[0]
#             timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()
#             noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
#             noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
#             loss = F.mse_loss(noise_pred, noise)
#             loss.backward()
#             losses.append(loss.item())
#             optimizer.step()
#             optimizer.zero_grad()

#         if (epoch + 1) % 1 == 0:
#             loss_last_epoch = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
#             print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")

#     # fig, axs = plt.subplots(1, 2, figsize=(12, 4))
#     # axs[0].plot(losses)
#     # axs[1].plot(np.log(losses))
#     # plt.show()


