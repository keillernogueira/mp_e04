import os
import argparse
import scipy.io
import numpy as np
import torch

from imageio import imread, imwrite
from dataloaders.image_dataloader import ImageDataLoader
from dataloaders.LFW_dataloader import LFW
from main import extract_features_from_image
from argparse import ArgumentParser
from dataset_processor import extract_features

def create_dataset(save_dir, feature_file, model_name="mobilefacenet", preprocessing_method="sphereface",
                    crop_size=(96,112), image_path=None, dataset=None):
    """
    Creating a new features dataset.

    :param image_path: Path to the image that will have its features extracted.
    :param dataset: Path to the dataset that will have its features extracted.
    :param save_dir: Path to the dir used to save the feature file.
    :param feature_file: String with the name of the feature file that will be created.
    :param model_name: String with the name of the model used.
    :param preprocessing_method: String with the name of the preprocessing method used.
    :param crop_size: Size of the crop based on the model used.
    """
    feature_file = save_dir + feature_file
    features = None
    image_dataloader = None 
    dataset_dataloader = None

    if dataset is not None:
        dataset = LFW(dataset, "images", "jpg", preprocessing_method, crop_size)
        dataset_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False,
                                             num_workers=2, drop_last=False)
    else:
        dataset = ImageDataLoader(image_path, preprocessing_method, crop_size)
        image_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, 
                        num_workers=2, drop_last=False)
    
    # extract the features
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
            features['name'] = np.concatenate((features['name'], feature['name']), 0)
            features['image'] = np.concatenate((features['image'], feature['image']), 0)
            features['bbs'] = np.concatenate((features['bbs'], feature['bbs']), 0)
        # save the current version of the features
        scipy.io.savemat(feature_file, features)
        
def update_dataset(image_path, save_dir, img_ID, feature_file, model_name="mobilefacenet", 
                    preprocessing_method="sphereface", crop_size=(96,112)):
    '''
    Extracting features from image and saving it.
    :param image_path: Path to the analysed image.
    :param img_ID: Name associated to the image.
    :param feature_file: Path to the file that contains extracted features from other images.
    :param model_name: String with the name of the model used.
    :param preprocessing_method: String with the name of the preprocessing method used.
    :param crop_size: Size of the crop based on the model used.
    '''
    assert img_ID is not None
    assert image_path is not None
    gpu = False
    operation = "extract_features"

    # img = Image.open(open(image_path, 'rb'))
    # img.show()
    dataset = ImageDataLoader(image_path, preprocessing_method, crop_size, operation == 'extract_features')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, 
                num_workers=2, drop_last=False)

    features = None
    feature_file = save_dir + feature_file
    # load current features
    if feature_file is not None and os.path.isfile(feature_file):
        features = scipy.io.loadmat(feature_file)

    # extract the features
    feature = extract_features_from_image(model_name, dataloader, image_path, gpu)
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
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--operation", type=str, required=True, 
                        help='Operation that will manipulate the dataset [create_dataset | update_dataset]')
    parser.add_argument('--feature_file', type=str, required=True, help='Feature file path')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to to save outcomes of the algorithm')
    
    #image processing options
    parser.add_argument('--image_path', type=str, default=None, help='Image path or link')
    parser.add_argument('--image_id', type=str, default=None, 
                        help='Image ID, **REQUIRED** if flag --operation is set as update_dataset.') 
    
    #dataset options
    parser.add_argument('--dataset_path', type=str, default=None, help='Path to the dataset')

    #processing options
    parser.add_argument('--model_name', type=str, required=False, default="mobilefacenet",
                        help='Name of the method.')
    parser.add_argument('--preprocessing_method', type=str, required=False, default="sphereface",
                        help='Pre-processing method')
    parser.add_argument('--crop_size', type=int, nargs="+", required=False, default=(96, 112),
                        help='Crop size')
    
    args = parser.parse_args()
    args.crop_size = tuple(args.crop_size)
    print(args)

    #processing whole dataset
    if args.dataset_path is not None:
        if args.operation == "create_dataset":
            create_dataset(args.save_dir, args.feature_file, args.model_name,
                            args.preprocessing_method, args.crop_size, dataset=args.dataset_path)
        elif args.operation == "update_dataset":
            raise NotImplementedError("Datasets features files cannot be updated passing datasets as argument")
        else:
            raise NotImplementedError("Operation " + args.operation + " not implemented")
    
    #processing single image
    if args.image_path is not None:
        if args.operation == "create_dataset":
            create_dataset(args.save_dir, args.feature_file, args.model_name,
                            args.preprocessing_method, args.crop_size, image_path=args.image_path, )
        elif args.operation == "update_dataset":
            update_dataset(args.image_path, args.save_dir, args.image_id, args.feature_file, 
                            args.model_name, args.preprocessing_method, args.crop_size)
        else:
            raise NotImplementedError("Operation " + args.operation + " not implemented")
    else:
        raise NotImplementedError("Dataset OR Image Path flags must be set. Both flags are None.")