import os
import time
import numpy as np
import scipy.io
import json
from datetime import datetime

from pathlib import Path

import argparse

import torch

from processors.image_processor import generate_ranking_for_image
from dataloaders.video_dataloader import VideoDataLoader
from networks.load_network import load_net


def process_video(video_paths, model, feature_file, save_dir):
    # Define face detection pipeline
    detection_pipeline = VideoDataLoader(batch_size=60, resize=0.5, preprocessing_method='sphereface',
                                         return_only_one_face=True)

    #create save_dir if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    features = None
    # load current features
    if feature_file is not None and os.path.isfile(feature_file):
        features = scipy.io.loadmat(feature_file)    

    if(type(video_paths)==str):
        video_paths = [video_paths]

    start = time.time()
    n_processed = 0
    data=[]
    
    with torch.no_grad():
        output_id = 0
        print(video_paths)
        output = []
        for i, filename in enumerate(video_paths):
            print(i, filename)
            # Load frames and find faces
            batches_imgs, batches_bbs = detection_pipeline(filename)
            
            name = os.path.basename(filename)
            output.append(dict())
            output[i]['name'] = name
            output[i]['path'] = filename

            face_id = 1
            for j in range(len(batches_imgs)):  # batch loop
                imgs, bbs = batches_imgs[j], batches_bbs[j]
                imgs1, imgs2 = imgs

                for i in range(len(imgs)):
                    imgs[i] = imgs[i].cuda()
                res = [model(d).data.cpu().numpy() for d in imgs]
                feature = np.concatenate((res[0], res[1]), 1)
                print(feature.shape)
                assert feature is not None, "No face detected in this image."
                
                # TODO gerar saida
                # Saída em JSON

                #for i in range(len(feature)):
                    #result = {'feature': np.reshape(feature[i], (-1, 1)).transpose(1,0),
                    #           'bbs': np.reshape(bbs[i], (-1, 1)).transpose(1,0)}

                result = {'feature': feature, 'bbs': bbs}
                # generate ranking
                top_k_ranking, all_ranking = generate_ranking_for_image(features, result)
                
                #print(top_k_ranking)
                for rank in top_k_ranking:#for each face
                    #print(rank)
                    face_dict = {}
                    face_dict['id'] = face_id
                    face_dict['class'] = rank[1][0]['Name']
                    face_dict['confidence'] = np.float64(rank[1][0]['Confidence'])
                    face_dict['box'] = rank[0].tolist()
                    output[0][f'face_{face_id}'] = face_dict
                    #print(os.path.join(save_dir, 'faces-'+datetime.now().strftime("%d%m%Y-%H%M%S%f") + '.json'))
                    #save_retrieved_ranking(output, rank[1], rank[0],
                    #                       os.path.join(save_dir, 'faces-'+datetime.now().strftime("%d%m%Y-%H%M%S%f") + '.json'))      
                    face_id += 1

                n_processed += len(bbs)
            output_id+=1
        data={f'output{output_id}':output}
             
                
    with open(os.path.join(save_dir, 'faces-'+datetime.now().strftime("%d%m%Y-%H%M%S%f") + '.json'), 'w', 
                              encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
    print("Total time: " + str(time.time() - start))
    print("Frames per second: " + str(n_processed / (time.time() - start)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='video_processor')
    parser.add_argument('--video_query', type=str, required=True, help='Video path or link')
    parser.add_argument('--feature_file', type=str, required=True, help='Feature file path')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to save the result of the algorithm')

    args = parser.parse_args()
    print(args)

    process_video(args.video_query, load_net('mobilefacenet', gpu=True), args.feature_file, args.save_dir)
