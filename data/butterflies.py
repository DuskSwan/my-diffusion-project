# encoding: utf-8
from pathlib import Path
import requests

import torch
import torchvision
from datasets import load_dataset
from torchvision import transforms
from PIL import Image

from utils import show_images

'''
This dataset is a subset of the Smithsonian Butterflies dataset, which contains images of butterflies.
The original dataset is available at: https://huggingface.co/datasets/huggan/smithsonian_butterflies
'''

class CustomTransform:
    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, examples):
        images = [self.preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

def butterflies_data_loader(image_size, batch_size):

    dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

    # Define data augmentations
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # Resize
            transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
            transforms.ToTensor(),  # Convert to tensor (0, 1)
            transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
        ]
    )

    transform = CustomTransform(preprocess)
    dataset.set_transform(transform)

    # Create a dataloader from the dataset to serve up the transformed images in batches
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader

def download_butterflies_images():
    # 加载数据集
    dataset = load_dataset("huggan/smithsonian_butterflies_subset")

    # 创建文件夹保存图片
    output_dir = Path("datasets/butterflies")
    output_dir.mkdir(parents=True, exist_ok=True)  # 如果文件夹不存在，创建它

    # 下载图片
    for example in dataset['train']:
        img_url = example['image_url']
        img_name = example['name'] + ".jpg"
        img_path = output_dir / img_name  # 使用Path对象拼接路径
        
        # 下载并保存图片
        img_data = requests.get(img_url).content
        with open(img_path, 'wb') as handler:
            handler.write(img_data)

if __name__ == '__main__':
    train_dataloader = butterflies_data_loader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xb = next(iter(train_dataloader))["images"].to(device)[:8]
    print("X shape:", xb.shape)
    show_images(xb).resize((8 * 64, 64), resample=Image.NEAREST)