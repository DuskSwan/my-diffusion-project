# encoding: utf-8
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from torchvision import transforms

'''
Cartoon Set is a collection of random, 2D cartoon avatar images. The cartoons vary in 10 artwork categories, 4 color categories, and 4 proportion categories, with a total of ~1013 possible combinations.
The original dataset is available at: https://github.com/google/cartoonset
'''

class CartoonDataset(Dataset):
    def __init__(self, image_dir, image_size=32, transform=None, max_images=None):
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.glob("*.png"))
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]
        self.image_size = image_size
        self.transform = transform

        assert len(self.image_paths) > 0, f"No images found in {self.image_dir}. Please check the directory."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")  # Make sure the image is in RGB mode
        if self.transform:
            image = self.transform(image)
        return image

def cartoon_data_loader(image_size=64, batch_size=32):
    # Directory containing your images
    dir = Path("datasets/cartoonset10k")
    
    # Define transformations: resizing and normalization
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize images
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # turn from [0,255] to about [-1,1]
    ])
    
    # Create dataset
    dataset = CartoonDataset(image_dir=dir, image_size=image_size, transform=transform)
    # dataset = CartoonDataset(image_dir=dir, image_size=image_size, transform=transform, max_images=1000)
    
    # Create DataLoader
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return train_dataloader

if __name__ == '__main__':
    train_dataloader = cartoon_data_loader()
    xb = next(iter(train_dataloader))
    print("X shape:", xb.shape)
    import sys
    sys.path.append(".")
    from utils import show_images
    show_images(xb)