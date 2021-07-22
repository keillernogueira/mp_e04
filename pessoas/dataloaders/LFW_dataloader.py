import os
import numpy as np
import imageio
from sklearn import preprocessing

import torch

from preprocessing.preprocessing_general import PreProcess


class LFW(object):
    def __init__(self, root, specific_folder, img_extension, preprocessing_method=None, crop_size=(96, 112)):
        """
        Dataloader of the LFW dataset.

        root: path to the dataset to be used.
        specific_folder: specific folder inside the same dataset.
        img_extension: extension of the dataset images.
        preprocessing_method: string with the name of the preprocessing method.
        crop_size: retrieval network specific crop size.
        """

        self.preprocessing_method = preprocessing_method
        self.crop_size = crop_size
        self.imgl_list = []
        self.classes = []
        self.people = []

        self.preprocess = PreProcess(self.preprocessing_method, crop_size=self.crop_size,
                                     is_processing_dataset=True, return_only_one_face=True, execute_default=True)

        # read the file with the names and the number of images of each people in the dataset
        with open(os.path.join(root, 'people.txt')) as f:
            people = f.read().splitlines()[1:]

        # get only the people that have more than 20 images
        for p in people:
            p = p.split('\t')
            if len(p) > 1:
                if int(p[1]) >= 20:
                    for num_img in range(1, int(p[1]) + 1):
                        self.imgl_list.append(os.path.join(root, specific_folder, p[0], p[0] + '_' +
                                                           '{:04}'.format(num_img) + '.' + img_extension))
                        self.classes.append(p[0])
                        self.people.append(p[0])

        le = preprocessing.LabelEncoder()
        self.classes = le.fit_transform(self.classes)

        print(len(self.imgl_list), len(self.classes), len(self.people))

    def __getitem__(self, index):
        imgl = imageio.imread(self.imgl_list[index])
        cl = self.classes[index]

        # if image is grayscale, transform into rgb by repeating the image 3 times
        if len(imgl.shape) == 2:
            imgl = np.stack([imgl] * 3, 2)

        imgl, bb = self.preprocess.preprocess(imgl)
        imgl = imgl.squeeze()
        bb = bb.squeeze()

        # append image with its reverse
        imglist = [imgl, imgl[:, ::-1, :]]

        # normalization
        for i in range(len(imglist)):
            imglist[i] = (imglist[i] - 127.5) / 128.0
            imglist[i] = imglist[i].transpose(2, 0, 1)
        imgs = [torch.from_numpy(i).float() for i in imglist]

        return imgs, cl, imgl, bb, self.imgl_list[index], self.people[index]

    def __len__(self):
        return len(self.imgl_list)
