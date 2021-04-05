import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
import tifffile as tif

def tile_image(img_path,output_dir,tile_size=600):
    print(f'img_path:{img_path}')
    image = tif.imread(img_path)
    print(f'type(image):{type(image)} shape:{image.shape}')
    image_size = image.shape
    cols = image.shape[2]//tile_size
    rows = image.shape[1]//tile_size

    for col in range(cols):
        for row in range(rows):
            tile = image[:,col*tile_size:(col+1)*tile_size,row*tile_size:(row+1)*tile_size]
            tile_name = f'tile_{col}_{row}.tiff'
            tif.imsave(os.path.join(output_dir, tile_name),tile)


tile_image('/Users/tzvikif/Documents/Deep Learning/Projects/Bananas/Bananas.tiff',
'/Users/tzvikif/Documents/Deep Learning/Projects/Bananas/Images')
#tile_image('Bananas.tif','Images')


