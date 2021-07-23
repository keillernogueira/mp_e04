import os
import scipy.io
import torch
import numpy as np
import argparse
import json

from datetime import datetime
from dataloaders.image_dataloader import ImageDataLoader
from processors.image_processor import extract_features_from_image, generate_ranking_for_image
from plots import plot_top15_person_retrieval
from networks.load_network import load_net
from manipulate_json import save_retrieved_ranking


def retrieval(image_path, feature_file, save_dir, output_method="image", model_name="mobilefacenet", model_path=None,
              preprocessing_method="sphereface", crop_size=(96, 112), gpu=True):
    """
    Retrieving results from an specific image.

    :param image_path: Path to the image analysed.
    :param feature_file: Path to the file that contains extracted features from dataset images.
    :param save_dir: Path to the dir used to save the results.
    :param output_method: Method to export the results, json or image.
    :param model_name: String with the name of the model used.
    :param model_path: Path to a trained model
    :param preprocessing_method: String with the name of the preprocessing method used.
    :param crop_size: Size of the crop based on the model used.
    :param gpu: use GPU?
    """
    assert image_path is not None, "Must set parameter image_path"
    assert feature_file is not None, "Must set parameter feature_file"

    # seting dataset and dataloader
    dataset = ImageDataLoader(image_path, preprocessing_method, crop_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    
    features = None
    # load current features
    if feature_file is not None and os.path.isfile(feature_file):
        features = scipy.io.loadmat(feature_file)

    # extract features for the query image
    feature = extract_features_from_image(load_net(model_name, model_path, gpu), dataloader, None, gpu=gpu)
    assert feature is not None, "No face detected in this image."

    # generate ranking
    top_k_ranking, all_ranking = generate_ranking_for_image(features, feature)

    # exporting results

    # if the method chosen was json
    if output_method.lower() == "json":
        data=[]
        output_id=1
        for i in range(len([image_path])):
            output = []
            name = os.path.basename(image_path)
            output.append({'name':name})
            output[i]['path'] = image_path
            
            face_id = 1
            for rank in top_k_ranking:
                face_dict = {}
                face_dict['id'] = face_id
                face_dict['class'] = rank[1][0]['Name']
                face_dict['confidence'] = np.float64(rank[1][0]['Confidence'])
                face_dict['box'] = rank[0].tolist()
                output[0][f'face_{face_id}'] = face_dict
                print(os.path.join(save_dir, 'faces-'+datetime.now().strftime("%d%m%Y-%H%M%S%f") + '.json'))
                #save_retrieved_ranking(output, rank[1], rank[0],
                #                       os.path.join(save_dir, 'faces-'+datetime.now().strftime("%d%m%Y-%H%M%S%f") + '.json'))
                face_id += 1
            data={f'output{output_id}':output}
            output_id+=1 
        with open(os.path.join(save_dir, 'faces-'+datetime.now().strftime("%d%m%Y-%H%M%S%f") + '.json'), 'w', 
                      encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    # if the method chosen was image
    elif output_method.lower() == "image":
        for i in range(len(all_ranking)):
            plot_top15_person_retrieval(image_path, "Unknown", all_ranking[i], 1,
                                        image_name='faces-' + datetime.now().strftime("%d%m%Y-%H%M%S%f"),
                                        cropped_image=np.array(feature["cropped_image"])[0][i],
                                        bb=feature["bbs"][i], save_dir=save_dir)
    # in case the user didn't chose neither json nor image
    else:
        raise NotImplementedError("Output method " + output_method + " not implemented")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='retrieval')
    parser.add_argument('--image_query', type=str, required=True, help='Image path or link')
    parser.add_argument('--feature_file', type=str, required=True, help='Feature file path')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to to save outcomes (such as trained models) of the algorithm')

    parser.add_argument('--output_method', type=str, required=False, default="image",
                        help='Method to read the data.')

    parser.add_argument('--model_name', type=str, required=False, default="mobilefacenet",
                        help='Name of the method.')
    parser.add_argument('--model_path', type=str, required=False, default=None,
                        help='Path to a trained model. If not set, the original trained model will be used.')
    parser.add_argument('--preprocessing_method', type=str, required=False, default="sphereface",
                        help='Pre-processing method')
    parser.add_argument('--crop_size', type=int, nargs="+", required=False, default=(96, 112),
                        help='Crop size')
    args = parser.parse_args()
    args.crop_size = tuple(args.crop_size)
    print(args)

    retrieval(args.image_query, args.feature_file, args.save_dir,
              args.output_method, args.model_name, args.model_path, args.preprocessing_method, args.crop_size)
