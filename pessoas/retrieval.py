import json
import os
import scipy.io
import torch
import numpy as np
import argparse
import urllib.request

from PIL import Image
from datetime import datetime
from argparse import ArgumentParser
from dataloaders.image_dataloader import ImageDataLoader
from main import extract_features_from_image, generate_ranking_for_image
from plots import plot_top15_person_retrieval


def retrieval(image_path, feature_file, save_dir, method="image", model_name="mobilefacenet",
              preprocessing_method="sphereface", crop_size=(96, 112)):
    """
    Retrieving results from an specific image.

    :param image_path: Path to the image analysed.
    :param feature_file: Path to the file that contains extracted features from dataset images.
    :param save_dir: Path to the dir used to save the results.
    :param method: Method to export the results, json or image.
    :param model_name: String with the name of the model used.
    :param preprocessing_method: String with the name of the preprocessing method used.
    :param crop_size: Size of the crop based on the model used.
    """
    assert image_path is not None
    image_name = None
    now = datetime.now()
    date = now.strftime("%d%m%Y-%H%M%S")

    # seting dataset and dataloader
    dataset = ImageDataLoader(image_name if image_name != None else image_path, preprocessing_method, crop_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
    
    features = None
    # load current features
    if feature_file is not None and os.path.isfile(feature_file):
        features = scipy.io.loadmat(feature_file)

    # extract features for the query image
    feature = extract_features_from_image(model_name, dataloader, None, gpu=False)
    if feature is not None:
        # generate ranking
        ranking = generate_ranking_for_image(features, feature)
    else:
        print("No face detected in this image.")
    
    # exporting results

    # if the method chosen was json
    if method.lower() == "json":
        data = {}
        data['Path'] = image_name if image_name != None else image_path
        data['Ranking'] = str(ranking[1])
        data['Bounding Boxes'] = ranking[0].tolist()
        with open(save_dir + 'faces-' + date + '.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    # if the method chosen was image
    elif method.lower() == "image":
        assert features is not None, 'To generate rank, use the flag --feature_file ' \
                                     'to load previously extracted faatures.'
        if feature is not None:
            # defining dataset variables
            query_features = feature['feature']
            database_features = features['feature']
            num_features_query = query_features.shape[0]

            features_stack = np.vstack((query_features, database_features))

            # classes = features['class']
            # if len(classes.shape) == 2:
            #     classes = classes[0]
            images = features['image']
            cropped_image = np.array(feature["cropped_image"])

            # normalizing features
            mu = np.mean(features_stack, 0)
            mu = np.expand_dims(mu, 0)
            features_stack = features_stack - mu
            features_stack = features_stack / np.expand_dims(np.sqrt(np.sum(np.power(features_stack, 2), 1)), 1)
            
            query_features = features_stack[0:num_features_query]
            database_features = features_stack[num_features_query:]
            
            # persons_scores = []
            for i, q in enumerate(query_features):
                scores_q = q @ np.transpose(database_features)
                # images is added twice
                scores_q = list(zip(scores_q, images, images, features['name']))
                scores_q = sorted(scores_q, key=lambda x: x[0], reverse=True)

                # persons_scores.append((feature["bbs"][i], generate_rank(scores_q)))
                person_name = scores_q[1][3].strip()
                plot_top15_person_retrieval(image_name if image_name is not None else image_path, person_name, scores_q,
                                            i+1, cropped_image=cropped_image[0][i], bb=feature["bbs"][i],
                                            save_dir=save_dir)
        else:
            print("No face detected in this image.")
            
    # in case the user didn't chose neither json nor image
    else:
        raise NotImplementedError("Method " + method + " not implemented")


if __name__ == '__main__':
    # image_path = "https://publisher-publish.s3.eu-central-1.amazonaws.com/pb-brasil247/swp/jtjeq9/media/20190830150812_6c55e3b5-c22c-4b41-a48e-3038c5088f31.jpeg"
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--image_path', type=str, required=True, help='Image path or link')
    parser.add_argument('--feature_file', type=str, required=True, help='Feature file path')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to to save outcomes (such as trained models) of the algorithm')

    parser.add_argument('--method', type=str, required=True, default="image",
                        help='Method to read the data.')
    parser.add_argument('--model_name', type=str, required=False, default="mobilefacenet",
                        help='Name of the method.')
    parser.add_argument('--preprocessing_method', type=str, required=False, default="sphereface",
                        help='Pre-processing method')
    parser.add_argument('--crop_size', type=int, nargs="+", required=False, default=(96, 112),
                        help='Crop size')
    args = parser.parse_args()
    args.crop_size = tuple(args.crop_size)
    print(args)

    retrieval(args.image_path, args.feature_file, args.save_dir,
              args.method, args.model_name, args.preprocessing_method, args.crop_size)