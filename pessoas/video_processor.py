import os
import time
import json
import scipy.io
import cv2
import torch
from datetime import datetime

from main import extract_features_from_image, extract_features_from_video, generate_ranking_for_image
from dataloaders.image_dataloader import ImageDataLoader
from dataloaders.video_dataloader import VideoDataLoader


def video_processor(video_path):

    model_name = "mobilefacenet"
    preprocessing_method = "sphereface"
    crop_size = (96, 112) 
    feature_file = "/content/drive/MyDrive/pessoas/features/features.mat"
    output_folder = '/content/drive/MyDrive/pessoas/results/'
    method = "json"

    features = None
    # load current features
    if feature_file is not None and os.path.isfile(feature_file):
        features = scipy.io.loadmat(feature_file)

    now = datetime.now()
    date = now.strftime("%d%m%Y-%H%M%S")

    #vid_capture = cv2.VideoCapture(video_path)

    #if (vid_capture.isOpened() == False):
    #    print("Error opening the video file")
    # Read fps and frame count
    #else:
        # Get frame rate information
        # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
    #    fps = vid_capture.get(5)
    #    print('Frames per second : ', fps,'FPS')
 
        # Get frame count
        # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
    #    frame_count = vid_capture.get(7)
    #    print('Frame count : ', frame_count)

    data = {}
    frame_number=1
    
    inicio_video_data_loader = time.time()
    dataset = VideoDataLoader(video_path, preprocessing_method=preprocessing_method, crop_size=crop_size, output_folder = "/content/drive/MyDrive/pessoas/videos")
    fim_video_data_loader = time.time()
    print("dataloader: ",fim_video_data_loader - inicio_video_data_loader)
    
    inicio_extract_features = time.time()
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
    feature = extract_features_from_video(model_name, dataset, None, gpu=False)
    fim_extract_features = time.time()
    print("extract features: ",fim_extract_features - inicio_extract_features)
    
    for i in feature:
        bbs_, ranking = generate_ranking_for_image(features, i)
        if(method.lower() == "json"):
            data[f'Frame{frame_number}'] = {}
            data[f'Frame{frame_number}']['Path'] = video_path
            data[f'Frame{frame_number}']['Ranking'] = str(ranking[0])
            data[f'Frame{frame_number}']['Bounding Boxes'] = bbs_[0][0].tolist()
        frame_number+=1

    if(method.lower() == "json"):
        with open(output_folder + 'faces-' + date + '.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    '''if feature is not None:
        # generate ranking
        ranking = generate_ranking_for_image(features, feature)
    else:
        print("No face detected in this image.")

    if(method.lower() == "json"):
        data[f'Frame{frame_number}'] = {}
        data[f'Frame{frame_number}']['Path'] = video_path
        data[f'Frame{frame_number}']['Ranking'] = str(ranking[1])
        data[f'Frame{frame_number}']['Bounding Boxes'] = ranking[0].tolist()
        with open(save_dir + 'faces-' + date + '.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    '''
    '''while(vid_capture.isOpened()):
        # vid_capture.read() methods returns a tuple, first element is a bool that indicates if any frame was returned
        # and the second is frame
        frame_number+=1
        ret, frame = vid_capture.read()
        #print(type(frame))

        if ret == True:
            #cv2_imshow(frame)
            # 20 is in milliseconds, try to increase the value, say 50 and observe
            #key = cv2.waitKey(20)
            #dataset = VideoDataLoader(frame, video_path, preprocessing_method, crop_size)
            #dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

            #feature = extract_features_from_image(model_name, dataloader, None, gpu=False)

            if feature is not None:
                # generate ranking
                ranking = generate_ranking_for_image(features, feature)
            else:
                print("No face detected in this image.")

            if(method.lower() == "json"):
                data[f'Frame{frame_number}'] = {}
                data[f'Frame{frame_number}']['Path'] = video_path
                data[f'Frame{frame_number}']['Ranking'] = str(ranking[1])
                data[f'Frame{frame_number}']['Bounding Boxes'] = ranking[0].tolist()
                with open(save_dir + 'faces-' + date + '.json', 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
            
            #if key == ord('q'):
            #    break
        else:
            break
    
    if(method.lower() == "json"):
        with open(save_dir + 'faces-' + date + '.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    vid_capture.release()
    cv2.destroyAllWindows()
'''       

if __name__ == '__main__':
    video_path = "https://www.youtube.com/watch?v=-TXBxxPAtb0"
    video_processor(video_path)