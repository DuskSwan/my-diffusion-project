# encoding: utf-8

# from .build import make_data_loader


from .butterflies import butterflies_data_loader
from .cartoons import cartoon_data_loader

def make_data_loader(dataset_name='', image_size=64, batch_size=32):
    if dataset_name == 'butterflies':
        return butterflies_data_loader(image_size, batch_size)
    if dataset_name == 'cartoon':
        return cartoon_data_loader(image_size, batch_size)
    raise ValueError(f"Unknown dataset name: {dataset_name}")
    

    

if __name__ == '__main__':
    # download_butterflies_images()
    pass