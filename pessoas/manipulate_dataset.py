import os
import argparse
import scipy.io
import numpy as np
import torch

from utils import str2bool
from dataloaders.image_dataloader import ImageDataLoader
from dataloaders.generic_dataloader import GenericDataLoader
from main import extract_features_from_image
from dataset_processor import extract_features


# TODO talvez o melhor seja ao inves de ter dois parametros para imagem e um pra dataset, ter somente um para dataset
def manipulate_dataset(feature_file=None, image_path=None, image_id=None, dataset_path=None,
                       model_name="mobilefacenet", preprocessing_method="sphereface",
                       crop_size=(96, 112), gpu=True):
    """
    Extracting new features for a dataset or for a image.

    :param feature_file: String with the name of the feature file that will be created.
    :param image_path: Path to the image that will have its features extracted.
    :param image_id: ID of the image.
    :param dataset_path: Path to the dataset.
    :param model_name: String with the name of the model used.
    :param preprocessing_method: String with the name of the preprocessing method used.
    :param crop_size: Size of the crop based on the model used.
    :param gpu: Use gpu?
    """
    image_dataloader = None
    dataset_dataloader = None

    # loading dataset or image
    if dataset_path is not None:
        dataset = GenericDataLoader(dataset_path, train=False,
                                    preprocessing_method=preprocessing_method, crop_size=crop_size)
        dataset_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True,
                                                         num_workers=8, drop_last=False)
    elif image_path is not None and image_id is not None:
        dataset = ImageDataLoader(image_path, preprocessing_method, crop_size, will_save_features=True)
        image_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                                       num_workers=2, drop_last=False)
    else:
        raise NotImplementedError("dataset OR image_path,image_id flags must be set. All those flags are None.")

    # load current features
    features = None
    if feature_file is not None and os.path.isfile(feature_file):
        features = scipy.io.loadmat(feature_file)

    # extracting features
    # TODO permitir que um modelo pre-treinado seja usado para extrair as features
    if image_dataloader is not None:
        feature = extract_features_from_image(model_name, image_dataloader, image_id, gpu=gpu)
        feature.pop('cropped_image', None)  # drop key 'cropped_image' since it is only used during the retrieval
    elif dataset_dataloader is not None:
        feature = extract_features(dataset_dataloader, model_name, gpu=gpu)
    else:
        raise NotImplementedError("Both dataloaders are None.")

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
    parser = argparse.ArgumentParser(description='manipulate_dataset')
    parser.add_argument('--feature_file', type=str, required=True,
                        help='Feature file path. If exists, it will be updated. Otherwise, it will be created.')
    parser.add_argument('--image_path', type=str, required=False, default=None,
                        help='Path to the image that will have features extracted and added to the pool of features.')
    parser.add_argument('--image_id', type=str, required=False, default=None,
                        help='Query label (or ID of the person) that will be added to the pool of features.')
    parser.add_argument('--dataset_path', type=str, required=False, default=None,
                        help='Path to the dataset. Each person must have a separate folder with his/her name. '
                             'This parameter and the parameters --image_path and --image_id are mutually exclusive.')

    parser.add_argument('--model_name', type=str, required=False, default="mobilefacenet", help='Name of the method.')
    parser.add_argument('--preprocessing_method', type=str, required=False, default="sphereface",
                        help='Pre-processing method')
    parser.add_argument('--crop_size', type=int, nargs="+", required=False, default=(96, 112), help='Crop size')
    parser.add_argument('--gpu', type=str2bool, required=False, default=True, help='Use GPU?')
    args = parser.parse_args()
    args.crop_size = tuple(args.crop_size)
    print(args)

    manipulate_dataset(args.feature_file, args.image_path, args.image_id, args.dataset_path,
                       args.model_name, args.preprocessing_method, args.crop_size, args.gpu)
