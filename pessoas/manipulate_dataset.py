import os
import argparse
import scipy.io
import numpy as np
import torch

from imageio import imread, imwrite
from dataloaders.image_dataloader import ImageDataLoader
#from dataloaders.generic_dataloader import GenericDataLoader
from dataloaders.LFW_dataloader import LFW
from main import extract_features_from_image
from argparse import ArgumentParser
from dataset_processor import extract_features

def manipulate_dataset(save_dir, feature_file, model_name="mobilefacenet", preprocessing_method="sphereface",
                    crop_size=(96,112), image_path=None, img_ID=None, dataset=None, dataset_path=None):
    """
    Creating a new features dataset or updating an existing one.

    :param image_path: Path to the image that will have its features extracted.
    :param img_ID: ID of the image.
    :param dataset: Name of the dataset that will have its features extracted.
    :param dataset_path: Path to the dataset.
    :param save_dir: Path to the dir used to save the feature file.
    :param feature_file: String with the name of the feature file that will be created.
    :param model_name: String with the name of the model used.
    :param preprocessing_method: String with the name of the preprocessing method used.
    :param crop_size: Size of the crop based on the model used.
    """
    feature_file = save_dir + feature_file
    image_dataloader = None 
    dataset_dataloader = None
    gpu = False
    operation = "extract_features"

    # loading dataset or image
    if dataset is not None:
        assert dataset_path is not None
        dataset = LFW(dataset, dataset_path, "jpg", preprocessing_method, crop_size)
        dataset_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False,
                                             num_workers=2, drop_last=False)
    else:
        assert image_path is not None
        dataset = ImageDataLoader(image_path, preprocessing_method, crop_size, operation == 'extract_features')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, 
                    num_workers=2, drop_last=False)
  
    # load current features
    features = None
    feature_file = save_dir + feature_file
    if feature_file is not None and os.path.isfile(feature_file):
        features = scipy.io.loadmat(feature_file)

    # extracting features
    if image_dataloader is not None:
        feature = extract_features_from_image(model_name, image_dataloader, image_path, gpu=False)
    else:
        feature = extract_features(model_name, dataset_dataloader, gpu=False)
    if feature is not None:
        if features is None:
            # if there is not features, use the recently extracted one as the current features
            features = feature
        else:
            # otherwise, concatenate the existing features with the recently extracted ones
            features['feature'] = np.concatenate((features['feature'], feature['feature']), 0)
            features['name'] = np.concatenate((features['name'], [img_ID]), 0)
            features['image'] = np.concatenate((features['image'], feature['image']), 0)
            features['bbs'] = np.concatenate((features['bbs'], feature['bbs']), 0)
        # save the current version of the features
        scipy.io.savemat(feature_file, features)

if __name__ == '__main__':
    manipulate_dataset("features/", "featurestestando123.mat", dataset="datasets/LFW", 
                    dataset_path="images")
    