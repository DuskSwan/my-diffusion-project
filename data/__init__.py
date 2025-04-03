# encoding: utf-8

# from .build import make_data_loader

from datasets import load_dataset
import requests


from .butterflies import butterflies_data_loader

def make_data_loader(dataset_name='', image_size=64, batch_size=32):
    if dataset_name == 'butterflies':
        return butterflies_data_loader(image_size, batch_size)

    

if __name__ == '__main__':
    # download_butterflies_images()
    pass