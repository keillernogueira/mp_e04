import os
import numpy as np
import torch

from torch.utils import data

from skimage import io, transform
from skimage import img_as_float, img_as_float32

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
    
    def __init__(self, root, mode, img_size=480, class_names=[], num_classes=11, make=True):
        
        self.root = root
        # Initializing variables.
        self.mode = mode
        self.imgs = None
        self.img_size = img_size
        self.num_classes = num_classes
        self.class_names = class_names

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
        img = io.imread(img_path)
        h, w, _ = img.shape

        with open(ann_path,"r") as f:
            while True:
                bb_list = f.readline().split()
                if not bb_list:
                    break
                labels.append(int(bb_list[0]))
                xmin = float(bb_list[1]) - float(bb_list[3])/2
                ymin = float(bb_list[2]) - float(bb_list[4])/2
                xmax = float(bb_list[1]) + float(bb_list[3])/2
                ymax = float(bb_list[2]) + float(bb_list[4])/2
                bbx.append([xmin,ymin,xmax,ymax])
        
        img = img_as_float32(img)
        img = img.astype(np.float32)

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

    def __getitem__(self, index, norm=False):

        img, bbx, labels = self.get_data(index)
                
        # Normalization.
        if norm:
            img = self.norm(img)

        img = transform.resize(img, (self.img_size, self.img_size) , order=1, preserve_range=True)
        
        # Transform bb to the image size
        h, w, _ = img.shape
        bbx = np.array(bbx)*np.array([w,h,w,h])

        # Adding channel dimension.
        img = self.torch_channels(img)
        
        # Turning to tensors.
        img = torch.from_numpy(img)
        
        # Construction target dict (pytorch default)
        targets = {}
        boxes = torch.as_tensor(bbx, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        targets['boxes'] = boxes
        targets["labels"] = labels

        # Returning to iterator.
        # print(index, targets)
        return img, targets

        
    def __len__(self):

        return len(self.imgs)
