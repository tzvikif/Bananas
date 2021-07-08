from typing import DefaultDict
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader # Gives easier dataset managment and creates mini batches
import os
import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from utils import split,cnvt_tile_predictions_to_image


class BananaTreesDataset(Dataset):
    def __init__(self,images_dir,annotations_path,transforms=None):
        self.root_dir = images_dir
        #self.annotations = self.get_annotations(annotations_path)
        self.annotations = self.get_tiles_annotations(annotations_path)
        self.transforms = transforms
    def __len__(self):
        #print(f'len:{len(self.annotations)}')
        return len(self.annotations)
    def __getitem__(self,index):
        current = self.annotations[index]
        img_name = current['filename']
        bboxes = current['bboxes']
        bboxes = torch.as_tensor(bboxes,dtype=torch.int16)
        img_path = os.path.join(self.root_dir,img_name)
        #print(f'image_path:{img_path}')
        img =Image.open(img_path)
        img = np.asfarray(img)
        image = torch.as_tensor(img,dtype=torch.uint8)
        shape = img.shape
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
            #print(f'image.shape:{image.shape:}, bboxes:{target["boxes"]}')
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            #print(f'** shape:{image.shape}')
            sample = self.transforms(**sample)
            #print(f'** after transform **:{sample["bboxes"]}')
            image = sample['image']
            
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, img_name
    def get_tiles_annotations(self,annotations_path):
        annotations = []
        with open(annotations_path) as json_file:
            data = json.load(json_file)
            for f in data:
                filename = f['filename']
                bboxes = f['bboxes']
                d = DefaultDict()
                d['filename'] = filename
                annotations.append({'filename':filename,'bboxes':bboxes})
        return annotations
    def get_annotations(self,annotations_path):
        annotations = []
        with open(annotations_path) as json_file:
            data = json.load(json_file)
            for key in data.keys():
                current_file = data[key]
                filename = current_file['filename']
                imsize = current_file['size']
                regions = current_file['regions']
                d = DefaultDict()
                d['filename'] = filename
                bboxes = np.zeros([len(regions),4])
                for i,attrib in enumerate(regions):
                    shape_attrib = attrib['shape_attributes']
                    xmin = shape_attrib['x']
                    ymin = shape_attrib['y']
                    xmax = xmin + shape_attrib['width']
                    ymax = ymin + shape_attrib['height']
                    bboxes[i,:] = xmin,ymin,xmax,ymax
                d['bboxes'] = bboxes
                annotations.append(d)
        return annotations
def tile_image(image_source_path,image_dest_path,annotation_source_path,annotation_dest_path,tile_size=(501,445)):
    b_data_set = BananaTreesDataset(image_source_path,annotation_source_path)
    annotations = []
    for image,target,image_name in b_data_set:
        rows = image.shape[0]//tile_size[0]
        cols = image.shape[1]//tile_size[1]
        for row in range(rows):
            for col in range(cols):
                filtered_bboxes = [bbx for bbx in target['boxes']
                if bbx[0] >= col*tile_size[1] and bbx[1] >= row*tile_size[0] and
                bbx[0] < (col+1)*tile_size[1] and bbx[1] < (row+1)*tile_size[0] and
                bbx[2] >= col*tile_size[1] and bbx[3] >= row*tile_size[0] and 
                bbx[2] < (col+1)*tile_size[1] and bbx[3] < (row+1)*tile_size[0]
                ] 
                filtered_bboxes = [
                    [int(bbx[0]%tile_size[1]),
                    int(bbx[1]%tile_size[0]),
                int(bbx[2]%tile_size[1]),
                int(bbx[3]%tile_size[0])] for bbx in filtered_bboxes ]
                #write tile image
                tile = image[row*tile_size[0]:(row+1)*tile_size[0],col*tile_size[1]:(col+1)*tile_size[1],:]
                new_tile = Image.fromarray(tile.numpy())
                tile_name = f'tile_{image_name[:-4]}_{row}_{col}.png'
                if len(filtered_bboxes) > 0:
                    new_tile.save(os.path.join(image_dest_path,tile_name))
                    annotations.append({'filename':tile_name,'bboxes':filtered_bboxes})
    with open(annotation_dest_path, "w") as write_file:
        json.dump(annotations, write_file)

def collate_fn(batch):
    return tuple(zip(*batch))      
def display_sample(data_loader):
    images, targets, image_ids = next(iter(data_loader))
    boxes = targets[1]['boxes'].cpu().numpy().astype(np.int32)
    sample = images[1]
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    print(f'filename:{image_ids[1]} bboxes:{boxes}')
    for box in boxes:
        #rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1], edgecolor='r', facecolor="none")
        rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1], edgecolor='r', facecolor="none")
        ax.add_patch(rect)    
    ax.set_axis_off() 
    ax.imshow(sample)
    plt.show()

images_dir = '/Users/tzvikif/Documents/Deep Learning/Projects/Bananas/batch_A_banana/images_bbx/'
annoations_path = '/Users/tzvikif/Documents/Deep Learning/Projects/Bananas/batch_A_banana/Batch_A_Banana_train.json'
#annoations_path = '/Users/tzvikif/Documents/Deep Learning/Projects/Bananas/batch_A_banana/annotations.json'
image_dest_train_path = '/Users/tzvikif/Documents/Deep Learning/Projects/Bananas/batch_A_banana/tiles/train'
image_dest_valid_path = '/Users/tzvikif/Documents/Deep Learning/Projects/Bananas/batch_A_banana/tiles/valid'
annotation_train_dest_path = '/Users/tzvikif/Documents/Deep Learning/Projects/Bananas/batch_A_banana/annotations_train.json'
annotation_valid_dest_path = '/Users/tzvikif/Documents/Deep Learning/Projects/Bananas/batch_A_banana/annotations_valid.json'
prediction_valid_annotation_path = '/Users/tzvikif/Documents/Deep Learning/Projects/Bananas/batch_A_banana/prediction_annotations.json'
#split(annoations_path,images_dir,image_dest_train_path,image_dest_valid_path,
#annotation_train_dest_path,annotation_valid_dest_path)
#tile_image(images_dir,image_dest_train_path,annoations_path,annotation_train_dest_path)
cnvt_tile_predictions_to_image(prediction_valid_annotation_path,images_dir)
'''
b_data_set = BananaTreesDataset(images_dir,annoations_path)
#image,target,img_name = b_data_set[0]
banana_data_loader = DataLoader(
      b_data_set,
      batch_size=4,
      shuffle=False,
      num_workers=1,
      collate_fn=collate_fn
)
display_sample(banana_data_loader)
'''
