import torch
import numpy as np
import argparse
import scipy.io

from dataloaders.image_dataloader import ImageDataLoader
from main import extract_features_from_image
from argparse import ArgumentParser

"""
CREATE_DATASET, que receberá, como parâmetros de entrada, o caminho para as imagens, extraíra features 
dessas imagens, e gerará o dataset de features
"""

def create_dataset(image_path, save_dir, feature_file, model_name="mobilefacenet", preprocessing_method="sphereface", 
                    crop_size=(96,112)):
    """
    Creating a new features dataset.

    :param image_path: Path to the image that will have its features extracted.
    :param save_dir: Path to the dir used to save the feature file.
    :param feature_file: String with the name of the feature file that will be created.
    :param model_name: String with the name of the model used.
    :param preprocessing_method: String with the name of the preprocessing method used.
    :param crop_size: Size of the crop based on the model used.
    """
    features = None
    feature_file = save_dir + feature_file

    dataset = ImageDataLoader(image_path, preprocessing_method, crop_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
    
    # extract the features
    feature = extract_features_from_image(model_name, dataloader, image_path, gpu=False)
    if feature is not None:
        if features is None:
            # if there is not features, use the recently extracted one as the current features
            features = feature
        else:
            # otherwise, concatenate the existing features with the recently extracted ones
            features['feature'] = np.concatenate((features['feature'], feature['feature']), 0)
            features['name'] = np.concatenate((features['name'], feature['name']), 0)
            features['image'] = np.concatenate((features['image'], feature['image']), 0)
            features['bbs'] = np.concatenate((features['bbs'], feature['bbs']), 0)
        # save the current version of the features
        scipy.io.savemat(feature_file, features)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--image_path', type=str, required=True, help='Image path or link')
    parser.add_argument('--feature_file', type=str, required=True, help='Feature file path')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to to save outcomes (such as trained models) of the algorithm')

    parser.add_argument('--model_name', type=str, required=False, default="mobilefacenet",
                        help='Name of the method.')
    parser.add_argument('--preprocessing_method', type=str, required=False, default="sphereface",
                        help='Pre-processing method')
    parser.add_argument('--crop_size', type=int, nargs="+", required=False, default=(96, 112),
                        help='Crop size')
    args = parser.parse_args()
    args.crop_size = tuple(args.crop_size)
    print(args)

    create_dataset(args.image_path, args.save_dir, args.feature_file,
                args.model_name, args.preprocessing_method, args.crop_size)
