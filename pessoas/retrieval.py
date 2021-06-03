import json
import os
import scipy.io
import torch
import numpy as np

from datetime import datetime
from argparse import ArgumentParser
from dataloaders.image_dataloader import ImageDataLoader
from main import extract_features_from_image, generate_ranking_for_image
from plots import plot_top15_person_retrieval

def retrieval(image_path): 
    """
    RETRIEVAL, que terá como parâmetros a imagem de entrada (que pode vir como URL ou como arquivo local) 
    e gerará, como saída, um JSON (com o link/caminho para a imagem de entrada, os bounding boxes detectados,
    e os IDs do ranking) OU uma imagem de ranking (que seria composto da imagem de entrada com os bounding boxes
    e as imagens e IDs do top 10 do ranking)
    """
    # processing variables
    preprocessing_method = "sphereface"
    crop_size = (96, 112)  
    feature_file = "images/features.mat"
    model_name = "mobilefacenet"

    # seting dataset and dataloader
    dataset = ImageDataLoader(image_path, preprocessing_method, crop_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
    
    features = None
    # load current features
    if feature_file is not None and os.path.isfile(feature_file):
        features = scipy.io.loadmat(feature_file)

    # extract features for the query image
    query_features = extract_features_from_image(model_name, dataloader, None, gpu=False)
    if query_features is not None:
        # generate ranking
        ranking = generate_ranking_for_image(features, query_features)
    else:
        print("No face detected in this image.")
    
    # exporting results
    method = "imaGe"
    if(method.lower() == "json"):
        now = datetime.now()
        date = now.strftime("%d%m%Y-%H%M%S")

        data = {}
        data['Path'] = image_path
        data['Ranking'] = str(ranking[0])
        data['Bounding Boxes'] = ranking[1].tolist()
        with open('faces-' + date + '.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    elif(method.lower() == "image"):
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
                # print(features['feature'].shape, features['name'].shape,
                # features['image'].shape, features['bbs'].shape)
        plot_top15_person_retrieval(image_path, "Vladimir Putin", scores=features, bb=ranking[1], query_num=10)
    else:
        raise NotImplementedError("Method " + method + " not implemented")

if __name__ == '__main__':
    retrieval("./images/test7.jpg")