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
    feature_file = "/content/drive/MyDrive/mp_e04-master/pessoas/features/features.mat"
    model_name = "mobilefacenet"

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
        assert features is not None, 'To generate rank, use the flag --feature_file' \
                                         ' to load previously extracted faatures.'
        if feature is not None:
            query_features = feature['feature']
            database_features = features['feature']
            num_features_query = query_features.shape[0]
            query_bbs = feature['bbs']

            features_stack = np.vstack((query_features, database_features))

            classes = features['class']
            if len(classes.shape) == 2:
                classes = classes[0]
            images = features['image']
            cropped_image = np.array(feature["cropped_image"])

            # normalizing features
            mu = np.mean(features_stack, 0)
            mu = np.expand_dims(mu, 0)
            features_stack = features_stack - mu
            features_stack = features_stack / np.expand_dims(np.sqrt(np.sum(np.power(features_stack, 2), 1)), 1)

            query_features = features_stack[0:num_features_query]
            database_features = features_stack[num_features_query:]

            for i, q in enumerate(query_features):
                scores_q = q @ np.transpose(database_features)

                scores_q = list(zip(scores_q, classes, images, features['name']))
                scores_q = sorted(scores_q, key=lambda x: x[0], reverse=True)

                plot_top15_person_retrieval(image_path, "Arnold Schwarzenegger", scores_q, i+1, cropped_image = cropped_image[0][i], bb = feature["bbs"][i], save_dir = "/content/drive/MyDrive/mp_e04-master/pessoas/test7")
        else:
            print("No face detected in this image.")
    else:
        raise NotImplementedError("Method " + method + " not implemented")

if __name__ == '__main__':
    retrieval("/content/sample_data/WhatsApp Image 2021-06-05 at 20.26.25.jpeg")