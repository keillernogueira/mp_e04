import os
import numpy as np
import torch

from torch.utils import data

from skimage import io, transform
from skimage import img_as_float, img_as_float32
import cv2

'''
# This class ListDataset use the dataset name and task name to load the respectives images accordingly, modify this class to your use case
# For a simple loader, this version is implemented considering the following folder scheme:

/datasets_root_folder
|--- images
    |--- train
        |--- img_0.jpg
        ...
    |--- test
        |--- img_k.jpg
        ...

|--- labels
     |--- train
        |--- img_0.txt
        ...
    |--- test
        |--- img_k.txt
        ...

'''

# Class that reads a sequence of image paths from a text file and creates a data.Dataset with them.
class ListDataset(data.Dataset):
    
    def __init__(self, root, mode, img_size=480, class_names=[], num_classes=11, make=True, transform=None):
        
        self.root = root
        # Initializing variables.
        self.mode = mode
        self.imgs = None
        self.img_size = img_size
        self.num_classes = num_classes
        self.class_names = class_names
        self.transform = transform

        if make:
            # Creating list of paths.
            self.imgs = self.make_dataset()
            
            # Check for consistency in list.
            if len(self.imgs) == 0:
                
                raise (RuntimeError('Found 0 images, please check the data set'))
    
    # Function that create the list of img_path
    def make_dataset(self):

        # Making sure the mode is correct.
        assert self.mode in ['train', 'test']
        items = []
        
        # Setting string for the mode.
        mode_str = ''
        if 'train' in self.mode:
            mode_str = 'train'
        elif 'test' in self.mode:
            mode_str = 'test'

        # Joining input paths.
        img_path = os.path.join(self.root,'images',self.mode)

        # Reading paths from file.
        data_list = []
        data_list = os.listdir(img_path)
        
        # Creating list containing image and ground truth paths.
        for it in data_list:
            item = os.path.join(img_path, it)
            items.append(item)
        
        # Returning list.
        return items
    
        
    # Function to load images and annotations
    # Returns: img, bound_box, labels
    def get_data(self, index):
        img_path = self.imgs[index]
        im =  os.path.split(img_path)[1]
        ann_path = os.path.join(self.root,'labels', self.mode, os.path.splitext(im)[0]+".txt")
        labels = []
        bbx = []
        # Reading images.
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255

        with open(ann_path,"r") as f:
            while True:
                bb_list = f.readline().split()
                if not bb_list:
                    break
                labels.append(float(bb_list[0]))
                xmin = float(bb_list[1]) - float(bb_list[3])/2
                ymin = float(bb_list[2]) - float(bb_list[4])/2
                xmax = float(bb_list[1]) + float(bb_list[3])/2
                ymax = float(bb_list[2]) + float(bb_list[4])/2
                bbx.append([xmin,ymin,xmax,ymax])

        return img, bbx, labels

    def norm(self, img):
        if len(img.shape) == 2:
            img = (img - img.mean()) / img.std()
        else:
            for b in range(img.shape[2]):
                img[:,:,b] = (img[:,:,b] - img[:,:,b].mean()) / img[:,:,b].std()
        return img

    def torch_channels(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        else:
            img = np.moveaxis(img, -1, 0)
        return img

    def __getitem__(self, index):

        img, bbx, labels = self.get_data(index)
                
        # Normalization.
        #if norm:
        #    img = self.norm(img)

        #img = transform.resize(img, self.img_size, order=1, preserve_range=True)
        
        # Transform bb to the image size
        #h, w, _ = img.shape
        bbx = np.array(bbx)#*np.array([w,h,w,h])

        # Adding channel dimension.
        #img = self.torch_channels(img)
        
        # Turning to tensors.
        #img = torch.from_numpy(img)
        
        # Construction target dict (pytorch default)
        targets = {}
        #boxes = torch.as_tensor(bbx, dtype=torch.float32)
        #labels = torch.as_tensor(labels, dtype=torch.int64)

        #targets['boxes'] = boxes
        #targets["labels"] = labels

        # Returning to iterator.
        #return img, targets
        targets['img'] = img
        targets['annot'] = np.concatenate((bbx, np.array(labels).reshape((len(labels),1)) ), axis=1)
        if self.transform:
            targets = self.transform(targets)
        return targets
        
    def __len__(self):

        return len(self.imgs)
    

def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=480):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - image.mean()) / image.std()), 'annot': annots}