import os
import numpy as np
import imageio
from sklearn import preprocessing

import torch

from preprocessing.preprocessing_general import PreProcess


class GenericDataLoader(object):
    def __init__(self, root, train=True, preprocessing_method=None, crop_size=(96, 112)):
        """
        Generic data loader.
        root: path to the dataset to be used.
        preprocessing_method: string with the name of the preprocessing method.
        crop_size: retrieval network specific crop size.
        """

        self.train = train
        self.preprocessing_method = preprocessing_method
        self.crop_size = crop_size
        self.img_list = []
        self.labels = []
        self.labels_string = []

        self.preprocess = PreProcess(self.preprocessing_method, crop_size=self.crop_size,
                                     is_processing_dataset=True, return_only_one_face=True, execute_default=True)

        self.img_list, self.labels, self.labels_string = self.read_directory(root)

        self.num_classes = len(np.unique(self.labels))
        print(self.num_classes)

    def read_directory(self, root):
        _files = []
        _labels_string = []

        subfolders = os.listdir(root)  # read subfolders
        for subf in subfolders:
            files = os.listdir(os.path.join(root, subf))  # read files of each subfolder
            for f in files:
                _files.append(os.path.join(root, subf, f))
                _labels_string.append(subf)

        le = preprocessing.LabelEncoder()
        _labels = le.fit_transform(_labels_string)
        print(len(_files), len(_labels), len(_labels_string))

        return _files, _labels, _labels_string

    def __getitem__(self, index):
        img = imageio.imread(self.img_list[index])
        cl = self.labels[index]

        # if image is grayscale, transform into rgb by repeating the image 3 times
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        
        #rgba to rgb
        if img.shape[2]==4: 
            img = img[:,:,:3]

        img, bb = self.preprocess.preprocess(img)
        img = img.squeeze()
        bb = bb.squeeze()

        if self.train is True:
            # basic data augmentation
            flip = np.random.choice(2) * 2 - 1
            img = img[:, ::flip, :]

            # normalization
            img = (img - 127.5) / 128.0
            if self.preprocessing_method != "mtcnn":
                img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()

            return img, cl
        else:
            # append image with its reverse
            imglist = [img, img[:, ::-1, :]]

            # normalization
            for i in range(len(imglist)):
                imglist[i] = (imglist[i] - 127.5) / 128.0
                if self.preprocessing_method != "mtcnn":
                    imglist[i] = imglist[i].transpose(2, 0, 1)
            imgs = [torch.from_numpy(i).float() for i in imglist]

            return imgs, cl, img, bb, self.img_list[index], self.labels_string[index]

    def __len__(self):
        return len(self.img_list)
