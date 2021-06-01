import numpy as np

import torch

from preprocessing.preprocessing_general import preprocess
from dataloaders.conversor import read_image

from utils import plot_bbs


class ImageDataLoader(object):
    def __init__(self, image, preprocessing_method=None, crop_size=(96, 112), will_save_features=False):
        """
        Dataloader for specific images.

        :param preprocessing_method: string with the name of the preprocessing method.
        :param crop_size: retrieval network specific crop size.
        :param will_save_features: Loader will extract features to save.
        """

        self.image = image
        self.preprocessing_method = preprocessing_method
        self.crop_size = crop_size
        self.will_save_features = will_save_features

    def __getitem__(self, index):
        imgl = read_image(self.image)

        # if image is grayscale, transform into rgb by repeating the image 3 times
        if len(imgl.shape) == 2:
            imgl = np.stack([imgl] * 3, 2)

        try:
            imgl, bb = preprocess(imgl, self.preprocessing_method,
                                  crop_size=self.crop_size, return_only_largest_bb=self.will_save_features)
            assert imgl.size != 0 and bb.size != 0
        except AssertionError:
            # no face detected
            return [], [], []

        # plot_bbs(self.image, '/home/kno/recfaces/', bb)

        # append image with its reverse
        imglist = [imgl, imgl[:, :, ::-1, :]]

        # normalization
        for i in range(len(imglist)):
            imglist[i] = (imglist[i] - 127.5) / 128.0
            imglist[i] = imglist[i].transpose(0, 3, 1, 2)
        imgs = [torch.from_numpy(i).float() for i in imglist]

        return imgs, bb, self.image

    def __len__(self):
        return 1
