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
    model_name = "mobilefacenet"
    preprocessing_method = "sphereface"
    crop_size = (96, 112) 
    feature_file = "features/featuresMMPS.mat"
    save_dir = 'results/'
    method = "image"

    # seting dataset and dataloader
    dataset = ImageDataLoader(image_path, preprocessing_method, crop_size)
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
    if(method.lower() == "json"):
        now = datetime.now()
        date = now.strftime("%d%m%Y-%H%M%S")

        data = {}
        data['Path'] = image_path
        data['Ranking'] = str(ranking[1])
        data['Bounding Boxes'] = ranking[0].tolist()
        with open(save_dir + 'faces-' + date + '.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    # if the method chosen was image
    elif(method.lower() == "image"):
        assert features is not None, 'To generate rank, use the flag --feature_file' \
                                         ' to load previously extracted faatures.'
        if feature is not None:
            # defining dataset variables
            specific_features = features['feature']
            classes = features['class']
            if len(classes.shape) == 2:
                classes = classes[0]
            images = features['image']
            people = features['name']
            cropped_images = features['cropped_image']
            # normalizing features
            mu = np.mean(specific_features, 0)
            mu = np.expand_dims(mu, 0)
            specific_features = specific_features - mu
            specific_features = specific_features / np.expand_dims(np.sqrt(np.sum(np.power(specific_features, 2), 1)), 1)
            scores_all = specific_features@np.transpose(specific_features)

            #persons_scores = []
            for i, q in enumerate(feature["feature"]):
                scores_q = q @ np.transpose(features["feature"])
                scores_q = list(zip(scores_q, classes, images, features['name']))
                scores_q = sorted(scores_q, key=lambda x: x[0], reverse=True)

                #persons_scores.append((feature["bbs"][i], generate_rank(scores_q))) 
            #print(feature["bbs"][0])
            person_name = list(ranking[1][0])
            plot_top15_person_retrieval(image_path, person_name[1], scores_q, i + 1, bb = feature["bbs"][0], save_dir = save_dir)
            #plot_top15_person_retrieval(image_path, "Vladimir Putin", scores=features, bb=ranking[1], query_num=10)
        else:
            print("No face detected in this image.")
            
    # in case the user didn't chose neither json nor image
    else:
        raise NotImplementedError("Method " + method + " not implemented")

if __name__ == '__main__':
    image_path = "images/test3.jpg"
    retrieval(image_path)