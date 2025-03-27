# encoding: utf-8

import torch
import torchvision
from datasets import load_dataset
from torchvision import transforms
from PIL import Image

import sys
sys.path.append('.')
from utils import show_images


def make_data_loader():
    dataset = load_dataset("datasets/smithsonian_butterflies_subset", split="train")

    # We'll train on 32-pixel square images, but you can try larger sizes too
    image_size = 32
    # You can lower your batch size if you're running out of GPU memory
    batch_size = 64

    # Define data augmentations
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # Resize
            transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
            transforms.ToTensor(),  # Convert to tensor (0, 1)
            transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)

    # Create a dataloader from the dataset to serve up the transformed images in batches
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader

if __name__ == '__main__':
    train_dataloader = make_data_loader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xb = next(iter(train_dataloader))["images"].to(device)[:8]
    print("X shape:", xb.shape)
    show_images(xb).resize((8 * 64, 64), resample=Image.NEAREST)