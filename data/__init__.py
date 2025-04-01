# encoding: utf-8

# from .build import make_data_loader

from datasets import load_dataset
import requests
from pathlib import Path

from .butterflies import butterflies_data_loader

def download_butterflies_images():
    # 加载数据集
    dataset = load_dataset("datasets/smithsonian_butterflies_subset")

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

def make_data_loader(dataset_name='', image_size=64, batch_size=32):
    if dataset_name == 'butterflies':
        return butterflies_data_loader(image_size, batch_size)

    

if __name__ == '__main__':
    download_butterflies_images()