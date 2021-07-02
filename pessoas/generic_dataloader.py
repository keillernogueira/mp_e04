import os
import numpy as np
import imageio
from sklearn import preprocessing

import torch

from preprocessing.preprocessing_general import preprocess


class GenericDataLoader(object):
    def __init__(self, root, preprocessing_method=None, crop_size=(96, 112)):
        """
        Dataloader of the LFW dataset.
        root: path to the dataset to be used.
        preprocessing_method: string with the name of the preprocessing method.
        crop_size: retrieval network specific crop size.
        """

        self.preprocessing_method = preprocessing_method
        self.crop_size = crop_size
        self.img_list = []
        self.labels = []

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

        img, bb = preprocess(img, self.preprocessing_method, crop_size=self.crop_size,
                            is_processing_dataset=True, return_only_largest_bb=True, execute_default=True)

        # basic data augmentation
        flip = np.random.choice(2) * 2 - 1
        img = img[:, ::flip, :]

        # normalization
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        return img, cl, img, bb, self.img_list[index], self.labels_string[index]

    def __len__(self):
        return len(self.img_list)