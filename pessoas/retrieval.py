import os
import scipy.io
import torch
import numpy as np
import argparse
import json
from pathlib import Path

from datetime import datetime
from dataloaders.image_dataloader import ImageDataLoader
from dataloaders.video_dataloader import VideoDataLoader
from processors.image_processor import extract_features_from_image, generate_ranking_for_image
from processors.video_processor import extract_features_from_video
from plots import plot_top15_person_retrieval
from networks.load_network import load_net
from manipulate_json import save_retrieved_ranking, read_json

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']


def retrieval(data_to_load, feature_file, save_dir, input_data='image', output_method="image",
              model_name="mobilefacenet", model_path=None,
              preprocessing_method="sphereface", crop_size=(96, 112), gpu=True):
    """
    Retrieving results from an specific input data.

    :param data_to_load: Data to be analysed. Can be a link or path, image or video, or else a json file with multiple links/paths.
    :param feature_file: Path to the file that contains extracted features from dataset images.
    :param save_dir: Path to the dir used to save the results.
    :param input_data: Type of the input data: image or video
    :param output_method: Method to export the results: json or image.
    :param model_name: String with the name of the model used.
    :param model_path: Path to a trained model
    :param preprocessing_method: String with the name of the preprocessing method used.
    :param crop_size: Size of the crop based on the model used.
    :param gpu: use GPU?
    """
    assert data_to_load is not None, "Must set parameter data_to_load"
    assert feature_file is not None and os.path.isfile(feature_file), \
        "Must set parameter feature_file with existing file"

    if '.json' in data_to_load: 
        data_to_load = read_json(data_to_load)
        for path in data_to_load:
            if any(vid_format in path.lower() for vid_format in vid_formats) or 'youtube.com/' in path.lower() or 'youtu.be/' in path.lower():
                input_data = 'video'
            elif any(img_format in path.lower() for img_format in img_formats):
                input_data = 'image'
            individual_retrieval(path, feature_file, save_dir, input_data, output_method, model_name,
                                  model_path, preprocessing_method, crop_size, gpu)
    else:
        individual_retrieval(data_to_load, feature_file, save_dir, input_data, output_method, model_name,
                                  model_path, preprocessing_method, crop_size, gpu)


# TODO checar com MPMG import usando json com multiplas images
def individual_retrieval(data_to_load, feature_file, save_dir, input_data='image', output_method="image",
                        model_name="mobilefacenet", model_path=None,
                        preprocessing_method="sphereface", crop_size=(96, 112), gpu=True):
    """
    Retrieving results from an specific input data.

    :param data_to_load: Data to be analysed. Can be a link or path, image or video.
    :param feature_file: Path to the file that contains extracted features from dataset images.
    :param save_dir: Path to the dir used to save the results.
    :param input_data: Type of the input data: image or video
    :param output_method: Method to export the results: json or image.
    :param model_name: String with the name of the model used.
    :param model_path: Path to a trained model
    :param preprocessing_method: String with the name of the preprocessing method used.
    :param crop_size: Size of the crop based on the model used.
    :param gpu: use GPU?
    """
    assert data_to_load is not None, "Must set parameter data_to_load"

    # create save_dir if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # read feature file to perform the retrieval
    features = None
    # load current features
    if feature_file is not None and os.path.isfile(feature_file):
        features = scipy.io.loadmat(feature_file)

    feature = None
    if input_data == 'image':
        # setting dataset and dataloader
        dataset = ImageDataLoader(data_to_load, preprocessing_method, crop_size)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

        # extract features for the query image
        feature = extract_features_from_image(load_net(model_name, model_path, gpu), dataloader, None, gpu=gpu)
    elif input_data == 'video':
        detection_pipeline = VideoDataLoader(batch_size=60, resize=0.5, preprocessing_method=preprocessing_method,
                                             return_only_one_face=True, crop_size = crop_size)
        feature = extract_features_from_video(data_to_load, detection_pipeline,
                                              load_net(model_name, model_path, gpu))

    assert feature is not None, "No face detected in this file."

    # generate ranking
    top_k_ranking, all_ranking = generate_ranking_for_image(features, feature)

    # exporting results

    # if the method chosen was json
    if output_method.lower() == "json":
        data = []
        output_id = 1
        for i in range(len([data_to_load])):
            output = []
            name = os.path.basename(data_to_load)
            output.append({'name': name})
            output[i]['path'] = data_to_load
            
            face_id = 1
            for rank in top_k_ranking:
                face_dict = {'id': face_id, 'class': rank[1][0]['Name'],
                             'confidence': np.float64(rank[1][0]['Confidence']), 'box': rank[0].tolist()}
                output[0][f'face_{face_id}'] = face_dict
                print(os.path.join(save_dir, 'faces-'+datetime.now().strftime("%d%m%Y-%H%M%S%f") + '.json'))
                # save_retrieved_ranking(output, rank[1], rank[0],
                # os.path.join(save_dir, 'faces-'+datetime.now().strftime("%d%m%Y-%H%M%S%f") + '.json'))
                face_id += 1
            data = {f'output{output_id}': output}
            output_id += 1
        with open(os.path.join(save_dir, 'faces-'+datetime.now().strftime("%d%m%Y-%H%M%S%f") + '.json'), 'w', 
                      encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    # if the method chosen was image
    elif output_method.lower() == "image":
        for i in range(len(all_ranking)):
            plot_top15_person_retrieval(data_to_load if input_data == 'image' else feature['image'][i],
                                        "Unknown", all_ranking[i], 1,
                                        image_name='faces-' + datetime.now().strftime("%d%m%Y-%H%M%S%f"),
                                        cropped_image=feature["cropped_image"][i],
                                        bb=feature["bbs"][i], save_dir=save_dir)
    # in case the user didn't choose neither json nor image
    else:
        raise NotImplementedError("Output method " + output_method + " not implemented")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='retrieval')
    parser.add_argument('--data_to_process', type=str, required=True, help='Data path or link')
    parser.add_argument('--feature_file', type=str, required=True, help='Feature file path')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to to save outcomes (such as trained models) of the algorithm')

    parser.add_argument('--input_type', type=str, required=False, default="image",
                        help='Type of the input data: image or video.')
    parser.add_argument('--output_method', type=str, required=False, default="image",
                        help='Method to read the data.')

    parser.add_argument('--model_name', type=str, required=False, default="mobilefacenet",
                        help='Name of the method.')
    parser.add_argument('--model_path', type=str, required=False, default=None,
                        help='Path to a trained model. If not set, the original trained model will be used.')
    parser.add_argument('--preprocessing_method', type=str, required=False, default="sphereface",
                        help='Pre-processing method')
    # parser.add_argument('--crop_size', type=int, nargs="+", required=False, default=(96, 112),
    #                     help='Crop size')
    args = parser.parse_args()
    # args.crop_size = tuple(args.crop_size)
    print(args)

    # selecting the size of the crop based on the network
    if args.model_name == 'mobilefacenet' or args.model_name == 'sphereface':
        crop_size = (96, 112)
    elif args.model_name == 'mobiface' or args.model_name == 'shufflefacenet' or args.model_name == 'curricularface' or args.model_name == 'arcface' or args.model_name == 'cosface':
        crop_size = (112, 112)
    elif args.model_name == 'openface':
        crop_size = (96, 96)
    elif args.model_name == 'facenet':
        crop_size = (160, 160)
    else:
        raise NotImplementedError("Model " + args.model_name + " not implemented")

    retrieval(args.data_to_process, args.feature_file, args.save_dir, args.input_type,
              args.output_method, args.model_name, args.model_path, args.preprocessing_method, crop_size)
