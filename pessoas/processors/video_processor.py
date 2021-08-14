import os
import time
import numpy as np
import scipy.io
import json
from datetime import datetime

import cv2
from pathlib import Path
import PIL.ImageDraw as ImageDraw
from PIL import Image

import argparse

import torch

from processors.image_processor import generate_ranking_for_image
from dataloaders.video_dataloader import VideoDataLoader
from plots import plot_top15_person_retrieval
from networks.load_network import load_net
from utils import generate_video


def process_video(video_paths, model, pre_process, batch_size, feature_file, save_dir):
    # Define face detection pipeline
    detection_pipeline = VideoDataLoader(batch_size=batch_size, resize=0.5, preprocessing_method=pre_process,
                                         return_only_one_face=True)

    # create save_dir if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    features = None
    # load current features
    if feature_file is not None and os.path.isfile(feature_file):
        features = scipy.io.loadmat(feature_file)    

    # TODO check if this is really needed
    if type(video_paths) is str:
        video_paths = [video_paths]

    start = time.time()
    n_processed = 0
    data = []
    with torch.no_grad():
        output_id = 0
        output = []
        for i, filename in enumerate(video_paths):
            print(i, filename)
            # Load frames and find faces
            batches_frames, batches_imgs, batches_crops, batches_bbs = detection_pipeline(filename)

            # TODO linearize first dimension
            # generate_video(batches_frames, batches_bbs)
            
            # output.append({['name']: os.path.basename(filename), ['path']: filename})

            face_id = 1
            for j in range(len(batches_imgs)):  # batch loop
                frames, imgs, crops, bbs = batches_frames[j], batches_imgs[j], batches_crops[j], batches_bbs[j]
                print(frames.shape, imgs[0].shape, imgs[1].shape, crops.shape, bbs.shape)

                for i in range(len(imgs)):
                    imgs[i] = imgs[i].cuda()
                res = [model(d).data.cpu().numpy() for d in imgs]
                feature = np.concatenate((res[0], res[1]), 1)
                print(feature.shape)
                assert feature is not None, "No face detected in this image."
                
                # TODO gerar saida
                # Saída em JSON

                # for i in range(len(feature)):
                    # result = {'feature': np.reshape(feature[i], (-1, 1)).transpose(1,0),
                    #           'bbs': np.reshape(bbs[i], (-1, 1)).transpose(1,0)}

                # result = {'feature': feature, 'bbs': bbs, "cropped_image": crops}
                # # generate ranking
                # top_k_ranking, all_ranking = generate_ranking_for_image(features, result)
                # for r in range(len(all_ranking)):
                #     plot_top15_person_retrieval(frames[r], "Unknown", all_ranking[r], 1,
                #                                 image_name='faces-' + datetime.now().strftime("%d%m%Y-%H%M%S%f"),
                #                                 cropped_image=result["cropped_image"][r],
                #                                 bb=result["bbs"][r], save_dir=save_dir)
                
        #         output_method = "json"
        #         if output_method.lower() == "image":
        #             for i in range(len(all_ranking)):
        #                 pass
        #                 # Para rodar o plot do top 15, é necessário criar uma nova funçao do plot
        #                 # que receba o frame e não o link ou o diretorio.
        #                 # Eu mudei a original pra testar, mas não salvei pra evitar conflito com a
        #                 # parte do retrieval de imagem
        #
        #                 # plot_top15_person_retrieval(frames[i][0], "Unknown", all_ranking[i], 1,
        #                   #              image_name='faces-' + datetime.now().strftime("%d%m%Y-%H%M%S%f"),
        #                   #              cropped_image=np.array(result["cropped_image"])[i],
        #                   #              bb=result["bbs"][i], save_dir=save_dir)
        #         # Saída em JSON
        #
        #         # for i in range(len(feature)):
        #             # result = {'feature': np.reshape(feature[i], (-1, 1)).transpose(1,0),
        #             #           'bbs': np.reshape(bbs[i], (-1, 1)).transpose(1,0)}
        #
        #         #print(top_k_ranking)
        #         if output_method.lower() == "json":
        #             for rank in top_k_ranking:#for each face
        #                 #print(rank)
        #                 face_dict = {}
        #                 face_dict['id'] = face_id
        #                 face_dict['class'] = rank[1][0]['Name']
        #                 face_dict['confidence'] = np.float64(rank[1][0]['Confidence'])
        #                 face_dict['box'] = rank[0].tolist()
        #                 output[0][f'face_{face_id}'] = face_dict
        #                 #print(os.path.join(save_dir, 'faces-'+datetime.now().strftime("%d%m%Y-%H%M%S%f") + '.json'))
        #                 #save_retrieved_ranking(output, rank[1], rank[0],
        #                 #                       os.path.join(save_dir, 'faces-'+datetime.now().strftime("%d%m%Y-%H%M%S%f") + '.json'))
        #                 face_id += 1
        #
        #         n_processed += len(bbs)
        #     output_id += 1
        # data = {f'output{output_id}': output}
                
    # with open(os.path.join(save_dir, 'faces-'+datetime.now().strftime("%d%m%Y-%H%M%S%f") + '.json'), 'w', encoding='utf-8') as f:
    #                 json.dump(data, f, indent=4)
    print("Total time: " + str(time.time() - start))
    print("Frames per second: " + str(n_processed / (time.time() - start)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='video_processor')
    parser.add_argument('--video_query', type=str, required=True, help='Video path or link')
    parser.add_argument('--feature_file', type=str, required=True, help='Feature file path')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save the result of the algorithm')

    # model options
    parser.add_argument('--model_name', type=str, required=True, default='mobilefacenet',
                        help='Model to test [Options: mobilefacenet | mobiface | sphereface | '
                             'openface | facenet | shufflefacenet]')
    parser.add_argument('--model_path', type=str, required=False, default=None,
                        help='Path to a trained model. If not set, the original trained model will be used.')
    parser.add_argument('--preprocessing_method', type=str, default='sphereface',
                        help='Preprocessing method [Options: None | mtcnn | sphereface | openface]')
    parser.add_argument('--batch_size', type=int, default=60, help='Batch size to extract features')

    args = parser.parse_args()
    print(args)

    process_video(args.video_query, load_net(args.model_name, args.model_path, gpu=True),
                  args.preprocessing_method, args.batch_size, args.feature_file, args.save_dir)
