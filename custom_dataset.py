# Imports
import torch
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms # Transformations we can perform on our dataset
import torchvision
import os
import glob
import pandas as pd
from skimage import io
from torch.utils.data import Dataset, DataLoader # Gives easier dataset managment and creates mini batches
import xml.etree.ElementTree as et #for xml parsing
import rasterio as rs #reads geotiff files
import numpy as np

class BananaTreesDataset(Dataset):
    def __init__(self,root_dir,transforms=None):
        self.root_dir = root_dir
        self.annotations = self.get_annotations()
        self.transforms = transforms
        self.SCHEME = '{http://tempuri.org/XMLSchema.xsd}'

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self,index):
        annotated_path = os.path.join(self.root_dir,self.annotations[index])
        img_name,bboxes = self.get_annotatated_data(annotated_path)
        img_path = os.path.join(self.root_dir,img_name)
        with rs.open(img_path) as img:
            image = img.read()
            area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            area = torch.as_tensor(area, dtype=torch.float32)
            labels = torch.ones((bboxes.shape[0],), dtype=torch.int64)

            # suppose all instances are not crowd
            iscrowd = torch.zeros((bboxes.shape[0],), dtype=torch.int64)
            
            target = {}
            target['boxes'] = bboxes
            target['labels'] = labels
            # target['masks'] = None
            target['image_id'] = torch.tensor([index])
            target['area'] = area
            target['iscrowd'] = iscrowd

            if self.transforms:
                sample = {
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                }
                sample = self.transforms(**sample)
                image = sample['image']
                
                target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

            return image, target, img_name

            

    def get_annotations(self):
        names = glob.glob(f'{self.root_dir}/palm*.xml')
        return names
    def get_annotatated_data(self,annotated_path):
        xtree = et.parse(annotated_path)
        xroot = xtree.getroot()
        filename_element = xroot.find(self.SCHEME+'filename')
        img_name = filename_element.text
        for element in xroot.find('./'+self.SCHEME+'size'):
            if element.tag == self.SCHEME + 'width':
                width = element.text
            if element.tag == self.SCHEME + 'height':
                height = element.text
        rows = len(xroot.findall('.//'+self.SCHEME+'bndbox'))
        bboxes = np.zeros([rows,4])
        for i,bbox in enumerate(xroot.findall('.//'+self.SCHEME+'bndbox')):
            xmin = int(bbox.find(self.SCHEME+'xmin').text)
            ymin = int(bbox.find(self.SCHEME+'ymin').text)
            xmax = int(bbox.find(self.SCHEME+'xmax').text)
            ymax = int(bbox.find(self.SCHEME+'ymax').text)
            bboxes[i,:] = xmin,ymin,xmax-xmin,ymax-ymin
        return img_name,bboxes

bt_dataset = BananaTreesDataset('/Users/tzvikif/Documents/Deep Learning/Projects/Bananas/Palmtrees/Training')
sample = bt_dataset[0]

