import os
import argparse
import time

import numpy as np
import scipy.io
from pathlib import Path

import torch

import pickle
from dataloaders.video_dataloader import VideoDataLoader
from networks.load_network import load_net
from utils import generate_video


def extract_features_from_video(video_file, detection_pipeline, model):
    # if a single link is used as parameter
    # if type(video_file) is str:
    #     video_file = [video_file]
    
    model.eval()
    start = time.time()
    n_processed = 0
    first = True
    with torch.no_grad():
        # for i, filename in enumerate(video_file):  # loop over the videos
        #     print(i, filename)
        # Load frames and find faces
        batches_frames, batches_imgs, batches_crops, batches_bbs = detection_pipeline(video_file)

        # this method can be used to create a video with the detected bbs
        # generate_video(batches_frames, batches_bbs, 'video.avi')

        for j in range(len(batches_imgs)):  # batch loop
            for k in range(len(batches_imgs[j])):
                frames, imgs, crops, bbs = batches_frames[j][k], batches_imgs[j][k], batches_crops[j][k], batches_bbs[j][k]
                # print(frames.shape, imgs[0].shape, imgs[1].shape, crops.shape, bbs.shape, len(frames))
                n_processed += 1
                for i in range(len(imgs)):
                    imgs[i] = torch.stack([imgs[i]]) 
                    imgs[i] = imgs[i].cuda()
                    
                res = [model(d.view(-1, d.shape[2], d.shape[3], d.shape[4])).data.cpu().numpy() for d in imgs]
                feature = np.concatenate((res[0], res[1]), 1)
    
                if first:
                    all_features = feature
                    all_frames = frames
                    all_crops = crops
                    all_bbs = bbs
                    first = False
                else:
                    all_features = np.concatenate((all_features, feature))
                    all_frames = np.concatenate((all_frames, frames))
                    all_crops = np.concatenate((all_crops, crops))
                    all_bbs = np.concatenate((all_bbs, bbs))

    # print(all_features.shape, all_frames.shape, all_crops.shape, all_bbs.shape)
    print("Total time: " + str(time.time() - start))
    print("Frames per second: " + str(n_processed / (time.time() - start)))
    return {'feature': all_features, 'name': [None]*len(all_features), 'image': all_frames,
            'bbs': all_bbs, 'cropped_image': all_crops}

    #     # for i in range(len(feature)):
    #         # result = {'feature': np.reshape(feature[i], (-1, 1)).transpose(1,0),
    #         #           'bbs': np.reshape(bbs[i], (-1, 1)).transpose(1,0)}
    #
    #     #print(top_k_ranking)
    #     if output_method.lower() == "json":
    #         for rank in top_k_ranking:#for each face
    #             #print(rank)
    #             face_dict = {}
    #             face_dict['id'] = face_id
    #             face_dict['class'] = rank[1][0]['Name']
    #             face_dict['confidence'] = np.float64(rank[1][0]['Confidence'])
    #             face_dict['box'] = rank[0].tolist()
    #             output[0][f'face_{face_id}'] = face_dict
    #             #print(os.path.join(save_dir, 'faces-'+datetime.now().strftime("%d%m%Y-%H%M%S%f") + '.json'))
    #             #save_retrieved_ranking(output, rank[1], rank[0],
    #             #                       os.path.join(save_dir, 'faces-'+datetime.now().strftime("%d%m%Y-%H%M%S%f") + '.json'))
    #             face_id += 1
    #
    #     n_processed += len(bbs)
    # output_id += 1
    # data = {f'output{output_id}': output}
    # with open(os.path.join(save_dir, 'faces-'+datetime.now().strftime("%d%m%Y-%H%M%S%f") + '.json'), 'w', encoding='utf-8') as f:
    #                 json.dump(data, f, indent=4)


def process_video(video_query, feature_file, save_dir, model, pre_process="sphereface", batch_size=60):
    # create save_dir if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # read feature file to perform the retrieval
    features = None
    # load current features
    if feature_file is not None and os.path.isfile(feature_file):
        with open(feature_file, 'rb') as handle:
            features = pickle.load(handle)

    # Define face detection pipeline
    detection_pipeline = VideoDataLoader(batch_size=batch_size, resize=0.5, preprocessing_method=pre_process,
                                         return_only_one_face=True)

    feature = extract_features_from_video(video_query, detection_pipeline, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='video_processor')
    parser.add_argument('--video_query', type=str, required=True, help='Video path or link')
    parser.add_argument('--feature_file', type=str, required=True, help='Feature file path')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to to save outcomes (such as trained models) of the algorithm')

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

    process_video(args.video_query, args.feature_file, args.save_dir,
                  load_net(args.model_name, args.model_path, gpu=True),
                  args.preprocessing_method, args.batch_size)
