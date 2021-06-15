import os

import scipy.io

import torch
from imageio import imread, imwrite
import numpy as np

from dataloaders.image_dataloader import ImageDataLoader

from main import extract_features_from_image


def update_dataset(img_path, img_ID, feature_file = "features.mat"):
    assert img_ID is not None
    assert img_path is not None
    preprocessing_method = "sphereface"
    model_name = "mobilefacenet"
    gpu = True
    crop_size = (96, 112)
    operation = "extract_features"


    #img = Image.open(open(img_path, 'rb'))
    #img.show()
    dataset = ImageDataLoader(img_path, preprocessing_method,
                                crop_size, operation == 'extract_features')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

    features = None
    # load current features
    if feature_file is not None and os.path.isfile(feature_file):
        features = scipy.io.loadmat(feature_file)

    # extract the features
    feature = extract_features_from_image(model_name, dataloader, img_path, gpu)
    if feature is not None:
        if features is None:
            # if there is not features, use the recently extracted one as the current features
            features = feature
        else:
            # otherwise, concatenate the existing features with the recently extracted ones
            features['feature'] = np.concatenate((features['feature'], feature['feature']), 0)
            features['name'] = np.concatenate((features['name'], img_ID), 0)
            features['image'] = np.concatenate((features['image'], feature['image']), 0)
            features['bbs'] = np.concatenate((features['bbs'], feature['bbs']), 0)
            features['cropped_image'] = np.concatenate((features['cropped_image'], feature['cropped_image'][0]), 0)
            # print(features['feature'].shape, features['name'].shape,
            # features['image'].shape, features['bbs'].shape)
        # save the current version of the features
        scipy.io.savemat(feature_file, features)

if __name__ == '__main__':
    img_path = "https://upload.wikimedia.org/wikipedia/commons/3/37/Arnold_Schwarzenegger.jpg"

    update_dataset(img_path, img_ID="Arnold_Schwarzenegger", feature_file="features.mat")



