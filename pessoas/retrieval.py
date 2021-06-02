import json
import os
import scipy.io
import torch

from datetime import datetime
from argparse import ArgumentParser
from dataloaders.image_dataloader import ImageDataLoader
from main import extract_features_from_image, generate_ranking_for_image

def processing_image(image_path, features, dataloader, model_name):
    """
    Função que recebe como parâmetro a imagem a ser analisada, a lista de features, o dataloader
    e o modelo a serem utilizados para extrair suas características, retornando o top 10 do ranking de
    imagens mais parecidas, com o nome da pessoa, o valor de confiança e o caminho até a imagem
    """
    assert features is not None, 'To generate rank, use the flag --feature_file' \
                                ' to load previously extracted faatures.'
    # extract features for the query image
    query_features = extract_features_from_image(model_name, dataloader, None, gpu=False)
    if query_features is not None:
        # generate ranking
        ranking = generate_ranking_for_image(features, query_features)
    else:
        print("No face detected in this image.")
    return ranking

def retrieval(image_path, method, ranking): 
    """
    RETRIEVAL, que terá como parâmetros a imagem de entrada (que pode vir como URL ou como arquivo local) 
    e gerará, como saída, um JSON (com o link/caminho para a imagem de entrada, os bounding boxes detectados,
    e os IDs do ranking) OU uma imagem de ranking (que seria composto da imagem de entrada com os bounding boxes
    e as imagens e IDs do top 10 do ranking)
    """
    #print("\nthis is the current image path: ", image_path)
    #print("\nthis is the current ranking: ", ranking[0])
    #print("\nthese are the current bounding boxes: ", ranking[1])
    if(method == None):
        raise NotImplementedError("You need to specify the method")
    elif(method.lower() == "json"):
        now = datetime.now()
        date = now.strftime("%d%m%Y-%H%M%S")

        data = {}
        data['Path'] = image_path
        data['Ranking'] = str(ranking[0])
        data['Bounding Boxes'] = ranking[1].tolist()
        with open('faces-' + date + '.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    elif(method.lower() == "image"):
        raise NotImplementedError("Not yet")
    else:
        raise NotImplementedError("Method " + method + " not implemented")

if __name__ == '__main__':
    """
    parser = ArgumentParser(description='main')
    Argumentos do parser: operation, que determina a operação a ser feita; image_query, que representa a 
    imagem a ser analisada; feature_file, que são as features extraidas de um dataset para fazer o ranking
    com a imagem enviada; method, que é requerido quando a operação é setada como retrieval, que representa
    se o resultado do ranking será exportado em um .json ou em uma imagem.
    parser.add_argument('--operation', type=str, required=True,
                        help='Operation [Options: retrieval | update_dataset |'
                             'train (for dataset only)]')
    parser.add_argument('--image_query', type=str, required=True,
                        help='Path or string of query image. If set, only this image is processed.')
    parser.add_argument('--feature_file', type=str, required=True,
                        help='File path to save/load the extracted features (.mat file).')
    parser.add_argument('--method', type=str, default=None,
                        help='Method to export the results (image or json file)'
                            '**REQUIRED** if flag --operation is retrieval')
    """
    #args = parser.parse_args()
    # Definindo parâmetros estaticamente, por enquanto
    image_query = "https://static.dw.com/image/56863146_401.jpg"
    method = "JSON"
    preprocessing_method = "openface"
    crop_size = (96, 112)  
    operation = "retrieval"
    feature_file = "images/features_.mat"
    dataset = ImageDataLoader(image_query, preprocessing_method,
                                  crop_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
  
    if(operation == 'retrieval'):
        features = None
        # load current features
        if feature_file is not None and os.path.isfile(feature_file):
            features = scipy.io.loadmat(feature_file)
        ranking = processing_image(image_query, features, dataloader, model_name="mobilefacenet")
        retrieval(image_query, method, ranking)