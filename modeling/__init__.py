# encoding: utf-8
import torch
from diffusers import UNet2DModel
# from .example_model import ResNet18


def build_Unet(image_size, device):
    # Create a model
    model = UNet2DModel(
        sample_size=image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(64, 128, 128, 256),  # More channels -> more parameters
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",  # a regular ResNet upsampling block
        ),
    )
    model.to(device)
    return model

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    model = build_Unet(32, device)
    print(model)
    print("Model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))