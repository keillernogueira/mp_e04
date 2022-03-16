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
from processors.FaceQNet import get_images_scores


def extract_features_from_video(video_file, detection_pipeline, model, n_best_frames=None):
    # if a single link is used as parameter
    # if type(video_file) is str:
    #     video_file = [video_file]
    
    model.eval()
    start = time.time()
    first = True
    all_bbs = np.array([])
    all_features = np.array([])
    all_frames = np.array([])
    all_crops = np.array([])
    with torch.no_grad():
        # for i, filename in enumerate(video_file):  # loop over the videos
        #     print(i, filename)
        # Load frames and find faces
        batches_frames, batches_imgs, batches_crops, batches_bbs, v_len = detection_pipeline(video_file)

        # this method can be used to create a video with the detected bbs
        # generate_video(batches_frames, batches_bbs, 'video.avi')
        
        print("Loading time:", time.time() - start)
        if n_best_frames is not None:
            st = time.time()
            assert n_best_frames > 0, f"n_best_frames({n_best_frames}) must be a positive integer"
            assert n_best_frames <= len(batches_frames[0]), \
                f"n_best_frames({n_best_frames}) must be smaller than batch size({len(batches_frames[0])})."
            idxs = []
            s = len(batches_frames)
            for j in range(s):
                frames, imgs, crops, bbs = batches_frames[j], batches_imgs[j], batches_crops[j], batches_bbs[j]

                scores = get_images_scores(imgs)

                # Selects n_best_frames for full batches, and the same proportion for incomplete ones.
                n = -n_best_frames if j < s-1 else int(-np.ceil(len(scores)/(len(batches_imgs[0][0])/n_best_frames)))
                idx = np.argpartition(scores, n)[n:]
                idxs.append(idx)
                
            print("Score Calculation:", time.time() - st)
        st = time.time()
        for j in range(len(batches_imgs)):  # batch loop
            if n_best_frames is None:
                sel_frames = batches_frames[j]
                imgs_standard = batches_imgs[j][0]          
                imgs_inverted = batches_imgs[j][1]
                sel_crops = batches_crops[j]
                sel_bbs = batches_bbs[j]
            else:
                sel_frames = batches_frames[j][idxs[j]]
                imgs_standard = batches_imgs[j][0][idxs[j]]
                imgs_inverted = batches_imgs[j][1][idxs[j]]
                sel_crops = batches_crops[j][idxs[j]]
                sel_bbs = batches_bbs[j][idxs[j]]
                
            imgs = [imgs_standard, imgs_inverted]
            
            frames, crops, bbs = sel_frames, sel_crops, sel_bbs
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
                all_features = np.concatenate((all_features,feature),0)
                all_frames = np.concatenate((all_frames,frames),0)
                all_crops = np.concatenate((all_crops,crops),0)
                all_bbs = np.concatenate((all_bbs,bbs),0)

    print("Feature Extraction done in:", time.time() - st)
    # print(all_features.shape, all_frames.shape, all_crops.shape, all_bbs.shape)
    print("Total time: " + str(time.time() - start))
    print("Selected", all_frames.shape[0], "Frames of the", v_len, "on the video")
    print("Frames per second: " + str(v_len / (time.time() - start)))
    return {'feature': all_features, 'name': [None]*len(all_features), 'image': all_frames,
            'bbs': all_bbs, 'cropped_image': all_crops}


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
    parser.add_argument('--model_name', type=str, required=True, default='curricularface',
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
