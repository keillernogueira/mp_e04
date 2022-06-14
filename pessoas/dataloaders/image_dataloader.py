import numpy as np

import torch

from ..preprocessing.preprocessing_general import PreProcess
from ..dataloaders.conversor import read_image

from ..utils import plot_bbs


class ImageDataLoader(object):
    def __init__(self, image, preprocessing_method=None, crop_size=(96, 112), return_only_one_face=False):
        """
        Dataloader for specific images.

        :param preprocessing_method: string with the name of the preprocessing method.
        :param crop_size: retrieval network specific crop size.
        :param will_save_features: Loader will extract features to save.
        """

        self.image = image
        self.preprocessing_method = preprocessing_method
        self.crop_size = crop_size
        self.return_only_one_face = return_only_one_face

        self.preprocess = PreProcess(self.preprocessing_method, crop_size=self.crop_size,
                                     return_only_one_face=self.return_only_one_face)

    def __getitem__(self, index):
        # print(self.image)
        imgl = read_image(self.image)

        # if image is grayscale, transform into rgb by repeating the image 3 times
        if len(imgl.shape) == 2:
            imgl = np.stack([imgl] * 3, 2)

        try:
            imgl, bb = self.preprocess.preprocess(imgl)
            assert imgl.size != 0 and bb.size != 0
        except AssertionError:
            # no face detected
            return [], [], [], []

        # plot_bbs(self.image, '/home/kno/recfaces/', bb)

        # append image with its reverse
        imglist = [imgl, imgl[:, :, ::-1, :]]

        # normalization
        for i in range(len(imglist)):
            imglist[i] = (imglist[i] - 127.5) / 128.0
            imglist[i] = imglist[i].transpose(0, 3, 1, 2)
        imgs = [torch.from_numpy(i).float() for i in imglist]

        return imgs, self.image, imgl, bb

    def __len__(self):
        return 1
