import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import os
from collections import Counter
import tifffile as tif
import rasterio as rs
from rasterio.plot import show

PALM_TREE_PATH = '/Users/tzvikif/Documents/Deep Learning/Projects/Bananas/Palmtrees/Training'
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

def preprocessing_palm_trees(img_dir):
    with rs.open(os.path.join(img_dir, 'palm_tile2_0.img')) as image:
        print(f'image.count:{image.count}, image.height:{image.height}, image.width:{image.width}')
        array = image.read(1)
        print(f'{type(array)} shape:{array.shape}')
        show(array)
        pyplot.imshow(array, cmap='pink')
        pyplot.show()
    
#tile_image('/Users/tzvikif/Documents/Deep Learning/Projects/Bananas/Bananas.tiff',
#'/Users/tzvikif/Docuents/Deep Learning/Projects/Bananas/Images')

preprocessing_palm_trees(PALM_TREE_PATH)


