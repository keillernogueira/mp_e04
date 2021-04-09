import os
import numpy as np
import imageio
from sklearn import preprocessing

import torch

from preprocessing.preprocessing_general import preprocess


class YALE(object):
    def __init__(self, root, specific_folder, img_extension, preprocessing_method='sphereface', crop_size=(96, 112)):
        """
        Dataloader of the Yale dataset.

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
        self.model_align = None

        for r, d, f in os.walk(os.path.join(root, specific_folder)):
            for fi in f:
                if fi.endswith(img_extension) and 'Ambient' not in fi:
                    self.imgl_list.append(os.path.join(r, fi))
                    self.classes.append(os.path.basename(r))
                    self.people.append(os.path.basename(r))

        le = preprocessing.LabelEncoder()
        self.classes = le.fit_transform(self.classes)

        print(len(self.imgl_list), len(self.classes), len(self.people))

    def __getitem__(self, index):
        imgl = imageio.imread(self.imgl_list[index])
        cl = self.classes[index]

        # if image is grayscale, transform into rgb
        if len(imgl.shape) == 2:
            imgl = np.stack([imgl] * 3, 2)

        imgl, _ = preprocess(imgl, self.preprocessing_method, crop_size=self.crop_size,
                             is_processing_dataset=True, return_only_largest_bb=True, execute_default=True)

        # append image with its reverse
        imglist = [imgl, imgl[:, ::-1, :]]

        # normalization
        for i in range(len(imglist)):
            imglist[i] = (imglist[i] - 127.5) / 128.0
            imglist[i] = imglist[i].transpose(2, 0, 1)
        imgs = [torch.from_numpy(i).float() for i in imglist]

        # replicando cl para ocupar o espaço das bbs
        return imgs, cl, cl, cl, self.imgl_list[index], self.people[index]

    def __len__(self):
        return len(self.imgl_list)


if __name__ == '__main__':
    YALE('C:\\Users\\keill\\Desktop\\Datasets\\CroppedYale\\', 'CroppedB', img_extension='pgm')
