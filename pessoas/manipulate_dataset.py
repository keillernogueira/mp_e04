import os
import argparse
import scipy.io
import numpy as np
import torch

from utils import str2bool
from dataloaders.generic_dataloader import GenericDataLoader
from processors.dataset_processor import extract_features
from networks.load_network import load_net


def manipulate_dataset(feature_file, dataset_path,
                       model_name="mobilefacenet", model_path=None, preprocessing_method="sphereface",
                       crop_size=(96, 112), gpu=True):
    """
    Extracting new features for a dataset or for a image.

    :param feature_file: String with the name of the feature file that will be created.
    :param dataset_path: Path to the dataset.
    :param model_name: String with the name of the model used.
    :param model_path: Path to a trained model
    :param preprocessing_method: String with the name of the preprocessing method used.
    :param crop_size: Size of the crop based on the model used.
    :param gpu: Use gpu?
    """
    # loading dataset or image
    dataset = GenericDataLoader(dataset_path, train=False,
                                preprocessing_method=preprocessing_method, crop_size=crop_size)
    dataset_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True,
                                                     num_workers=0, drop_last=False)

    # load current features
    features = None
    if feature_file is not None and os.path.isfile(feature_file):
        features = scipy.io.loadmat(feature_file)

    # extracting features
    feature = extract_features(dataset_dataloader, model=load_net(model_name, model_path, gpu))
    assert feature is not None, "Not capable of extracting features"

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
    parser = argparse.ArgumentParser(description='manipulate_dataset')
    parser.add_argument('--feature_file', type=str, required=True,
                        help='Feature file path. If exists, it will be updated. Otherwise, it will be created.')
    parser.add_argument('--dataset_path', type=str, required=False, default=None,
                        help='Path to the dataset. Each person must have a separate folder with his/her name. '
                             'This parameter and the parameters --image_path and --image_id are mutually exclusive.')

    parser.add_argument('--model_name', type=str, required=False, default="mobilefacenet", help='Name of the method.')
    parser.add_argument('--model_path', type=str, required=False, default=None,
                        help='Path to a trained model. If not set, the original trained model will be used.')
    parser.add_argument('--preprocessing_method', type=str, required=False, default="sphereface",
                        help='Pre-processing method')
    # parser.add_argument('--crop_size', type=int, nargs="+", required=False, default=(96, 112), help='Crop size')
    parser.add_argument('--gpu', type=str2bool, required=False, default=True, help='Use GPU?')
    args = parser.parse_args()
    # args.crop_size = tuple(args.crop_size)
    print(args)

    # selecting the size of the crop based on the network
    if args.model_name == 'mobilefacenet' or args.model_name == 'sphereface':
        crop_size = (96, 112)
    elif args.model_name == 'mobiface' or args.model_name == 'shufflefacenet':
        crop_size = (112, 112)
    elif args.model_name == 'openface':
        crop_size = (96, 96)
    elif args.model_name == 'facenet':
        crop_size = (160, 160)
    else:
        raise NotImplementedError("Model " + args.model_name + " not implemented")

    manipulate_dataset(args.feature_file, args.dataset_path,
                       args.model_name, args.model_path, args.preprocessing_method, crop_size, args.gpu)
