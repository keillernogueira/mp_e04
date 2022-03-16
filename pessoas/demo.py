import os
import io
import base64
import numpy as np
import argparse
import time
import pickle
import torch.utils.data
import torchvision
from PIL import Image
from dataloaders.LFW_loader_retrieval import LFW
from dataloaders.yale_loader_retrieval import YALE
from dataloaders.preprocessing import preprocess
from config import *
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import imageio
import scipy.io

import sys
# sys.path.append("..")

# MobileFaceNet
from MobileFaceNet_Pytorch.core import model
# SphereFace
from sphereface_pytorch.net_sphere import sphere20a
# MobiFace
from MobiFace_Pytorch.core import model as mobiface_model
# OpenFace
from OpenFacePytorch_v2 import loadOpenFace
# FaceNet
from facenet_pytorch import InceptionResnetV1
# ShuffleFaceNet
from ShuffleFaceNet_Pytorch.core import model as shuffleface_model

def str2bool(v):
    """
    Function to transform strings into booleans.

    v: string variable
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_features(model_name, dataloader_loader, gpu=True, gather_cropped_images=False,
                 gather_bbs=False, feature_save_file=None):

    """
    Function to extract features of images.

    model_name: string with the name of the model used.
    dataloader_loader: dataloader used to load the images.
    gpu: boolean to allow use gpu.
    gather_cropped_images: boolean to return the cropped images.
    gather_bbs: boolean to return the bouding boxes of images.
    feature_save_file: directory to save the features file.    
    """

    # initialize the network
    if model_name == 'mobilefacenet':
        net = model.MobileFacenet()
        ckpt = torch.load(MOBILEFACENET_MODEL_PATH)
        net.load_state_dict(ckpt['net_state_dict'])
    elif model_name == 'sphereface':
        net = sphere20a(feature=True)
        net.load_state_dict(torch.load(SPHEREFACE_MODEL_PATH))
    elif model_name == 'mobiface':
        net = mobiface_model.MobiFace(final_linear=True)
        ckpt = torch.load(MOBIFACE_MODEL_PATH)
        net.load_state_dict(ckpt['net_state_dict'])
    elif model_name == 'openface':
        net = loadOpenFace.OpenFaceModel(use_cuda=gpu)
        net.load_state_dict(torch.load(OPENFACE_MODEL_PATH))
    elif model_name == 'facenet':
        net = InceptionResnetV1(pretrained='casia-webface')
        net.load_state_dict(torch.load(FACENET_MODEL_PATH))
    elif model_name == 'shufflefacenet':
        net = shuffleface_model.ShuffleFaceNet()
        ckpt = torch.load(SHUFFLEFACENET_MODEL_PATH)
        net.load_state_dict(ckpt['net_state_dict'])
    else:
        raise NotImplementedError("Model " + model_name + " not implemented")

    if gpu:
        net = net.cuda()

    net.eval()

    features = None
    classes = None
    images = None
    names = None
    cropped_images = None
    bbs = None
    count = 0

    # forward
    for imgs, cls, crop_img, bb, imgl_list, people in dataloader_loader:
        if gpu:
            for i in range(len(imgs)):
                imgs[i] = imgs[i].cuda()
        count += imgs[0].size(0)
        if count % 1000 == 0:
            print('extracing deep features from the face {}...'.format(count))

        res = [net(d).data.cpu().numpy() for d in imgs]
        feature = np.concatenate((res[0], res[1]), 1)

        if features is None:
            features = feature
            classes = cls
            images = imgl_list
            names = people
            if gather_cropped_images is True:
                cropped_images = crop_img
            if gather_bbs is True:
                bbs = bb
        else:
            names = np.concatenate((names, people), 0)
            features = np.concatenate((features, feature), 0)
            classes = np.concatenate((classes, cls), 0)
            images = np.concatenate((images, imgl_list), 0)
            if gather_cropped_images is True:
                cropped_images = np.concatenate((cropped_images, crop_img), 0)
            if gather_bbs is True:
                bbs = np.concatenate((bbs, bb), 0)

    print(features.shape, classes.shape, names.shape, images.shape)
    if gather_cropped_images is True:
        print(cropped_images.shape)
    if gather_bbs is True:
        print(bbs.shape)

    result = {'features': features, 'classes': classes, 'images': images, 'people': names}
    
    if feature_save_file is not None:
        with open(feature_save_file, 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return result, cropped_images, bbs


def plot_top15_face_retrieval(query_image, scores, cropped_image=None, bb=None, save_dir="result/"):
    """
    Function to plot the top 15 visual results of the queries.
    Face retrieval considers the most similar images, including images of the same person.

    scores: ranked list containing the information of the images.
    query_num: int representing the queru number when evaluating the whole dataset.
    metrics: vector with the calculated metrics.
    cropped_image: query image cropped by preprocessing.
    bb: bouding boxes of query image by preprocessing.
    save_dir: directory where are saved the image results.
    """
    fig, axes = plt.subplots(4, 5, figsize=(15, 15), sharex=True, sharey=True)
    ax = axes.ravel()

    # decode base64 file   
    if query_image.endswith("txt"):
        f = open(query_image, "r")
        base64_img = f.read()
        if base64_img[0] == "b":
            img = imageio.imread(io.BytesIO(base64.b64decode(base64_img[1:])))
        else:
            img = imageio.imread(io.BytesIO(base64.b64decode(base64_img)))    
    else:
        img = imageio.imread(query_image)

    img = Image.fromarray(img)
    img = img.resize((250, 250))
    img = np.array(img)

    ax[0].set_title('| Query image |')
    ax[0].imshow(img)

    # ax[1].imshow(img)
    # if not np.array_equal(bb, [0., 0., 250., 250., 0.]):
    #     ax[1].set_title('| Bounding Box |')
    #     rect = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1],
    #                              linewidth=1, edgecolor='r', facecolor='none')
    #     ax[1].add_patch(rect)
    # else:
    #     ax[1].set_title('| NO Bounding Box |')

    ax[1].set_title('| Cropped Face |')
    shift = 75  # this shift is only used to center the cropped image into de subplot
    ax[1].imshow(cropped_image, extent=(shift, shift + cropped_image.shape[1], shift + cropped_image.shape[0], shift))

    for i in range(15):
        if scores[i][2].endswith("txt"):
            with open(scores[i][2].strip(), "r") as f:
                base64_img = f.read()
                if base64_img[0] == "b":
                    img = imageio.imread(io.BytesIO(base64.b64decode(base64_img[1:])))
                else:
                    img = imageio.imread(io.BytesIO(base64.b64decode(base64_img)))
        else:
            img = imageio.imread(scores[i][2].strip())
        ax[i+5].set_title('| %i |\n%s\n%f' % (i+1, scores[i][3].strip(), scores[i][0]))
        ax[i+5].imshow(img)

    for a in ax:
        a.set_axis_off()

    fig.savefig(os.path.join(save_dir, os.path.basename(os.path.splitext(query_image)[0]) + "_face_retrieval_result.jpg"))
    
    plt.close(fig)


def plot_top15_person_retrieval(query_image, scores, cropped_image=None, bb=None, save_dir="result/"):
    """
    Function to plot the top 15 visual results of the queries.
    Person retrieval considers the most similar persons, not repeating the images of the same person.

    scores: ranked list containing the information of the images.
    query_num: int representing the queru number when evaluating the whole dataset.
    metrics: vector with the calculated metrics.
    cropped_image: query image cropped by preprocessing.
    bb: bouding boxes of query image by preprocessing.
    save_dir: directory where are saved the image results.
    """
    
    fig, axes = plt.subplots(4, 5, figsize=(15, 15), sharex=True, sharey=True)
    ax = axes.ravel()

    # decode base64 file   
    if query_image.endswith("txt"):
        f = open(query_image, "r")
        base64_img = f.read()
        if base64_img[0] == "b":
            img = imageio.imread(io.BytesIO(base64.b64decode(base64_img[1:])))
        else:
            img = imageio.imread(io.BytesIO(base64.b64decode(base64_img)))    
    else:
        img = imageio.imread(query_image)

    img = Image.fromarray(img)
    img = img.resize((250, 250))
    img = np.array(img)
        
    ax[0].set_title('| Query image |')
    ax[0].imshow(img)

    # ax[1].imshow(img)
    # if not np.array_equal(bb, [0., 0., 255., 255., 0.]):
    #     ax[1].set_title('| Bounding Box |')
    #     rect = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1],
    #                              linewidth=1, edgecolor='r', facecolor='none')
    #     ax[1].add_patch(rect)
    # else:
    #     ax[1].set_title('| NO Bounding Box |')

    ax[1].set_title('| Cropped Face |')
    shift = 75  # this shift is only used to center the cropped image into de subplot
    ax[1].imshow(cropped_image, extent=(shift, shift + cropped_image.shape[1], shift + cropped_image.shape[0], shift))

    unique_persons = []
    i = j = 0
    while i < 15:
        if unique_persons:
            if scores[j][3] not in unique_persons:
                if scores[j][2].strip().endswith("txt"):
                    with open(scores[j][2], "r") as f:
                        base64_img = f.read()
                        if base64_img[0] == "b":
                            img = imageio.imread(io.BytesIO(base64.b64decode(base64_img[1:])))
                        else:
                            img = imageio.imread(io.BytesIO(base64.b64decode(base64_img)))
                else:
                    img = imageio.imread(scores[j][2].strip())
                ax[i + 5].set_title('| %i |\n%s\n%f' % (i + 1, scores[j][3].strip(), scores[j][0]))
                ax[i + 5].imshow(img)
                unique_persons.append(scores[j][3].strip())
                i += 1
        else:
            if scores[j][2].strip().endswith("txt"):
                with open(scores[i][2], "r") as f:
                    base64_img = f.read()
                    if base64_img[0] == "b":
                        img = imageio.imread(io.BytesIO(base64.b64decode(base64_img[1:])))
                    else:
                        img = imageio.imread(io.BytesIO(base64.b64decode(base64_img)))
            else:
                img = imageio.imread(scores[j][2].strip())
            ax[i + 5].set_title('| %i |\n%s\n%f' % (i + 1, scores[j][3].strip(), scores[j][0]))
            ax[i + 5].imshow(img)
            unique_persons.append(scores[j][3])
            i += 1
        j += 1

    for a in ax:
        a.set_axis_off()

    fig.savefig(os.path.join(save_dir, os.path.basename(os.path.splitext(query_image)[0]) + "_person_retrieval_result.jpg"))
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    # model options
    parser.add_argument('--path_image_query', type=str, required=True, help='The path to query image')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name [Options: LFW | YALEB]')
    parser.add_argument('--dataset_feature_load_file', type=str, default=None,
                        help='The dataset features')
    parser.add_argument('--model_name', type=str, default='mobilefacenet',
                        help='Model to test [Options: mobilefacenet | mobiface | sphereface | '
                             'openface | facenet | shufflefacenet]')
    parser.add_argument('--preprocessing_method', type=str, default='sphereface',
                        help='Preprocessing method [Options: None | mtcnn | sphereface | openface]')
    parser.add_argument('--visual_results_path', type=str, default='./',
                        help='Path where the results are saved')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size to extract features')
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='Use or not GPU')
    parser.add_argument('--feature_save_file', type=str, default=None,
                        help='The file path to save the extracted features, if wanted. '
                             'Suggestion = ./result/best_result.mat')
    args = parser.parse_args()
    # print(args)


    if args.model_name == 'mobilefacenet' or args.model_name == 'sphereface':
        crop_size = (96, 112)
    elif args.model_name == 'mobiface' or args.model_name == 'shufflefacenet':
        crop_size = (112, 112)
    elif args.model_name == 'openface':
        crop_size = (96, 96)
    elif args.model_name == 'facenet':
        crop_size = (160, 160)
    else:
        raise NotImplementedError("Model " + args.model_name + " not implemented")

    if args.dataset_feature_load_file is None:
        # YALEB dataset only runs with batch 8 because of GPU memory space
        if args.dataset == 'YALEB' and args.batch_size == 32:
            args.batch_size = 8

        if args.dataset.upper() == 'LFW':
            dataset = LFW(LFW_GENERAL_DATA_DIR, 'lfw', 'jpg',
                          preprocessing_method=args.preprocessing_method, crop_size=crop_size)
        elif args.dataset.upper() == 'YALEB':
            dataset = YALE(YALEB_GENERAL_DATA_DIR, 'ExtendedYaleB', 'pgm',
                           preprocessing_method=args.preprocessing_method, crop_size=crop_size)
        else:
            raise NotImplementedError("Dataset " + args.dataset + " dataloader not implemented")

        dataloader_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                                        num_workers=2, drop_last=False)

        # extract the features of the dataset
        features_database, cropped_images, bbs = get_features(args.model_name, dataloader_loader,
                                                     gather_cropped_images=True,
                                                     gather_bbs=True,
                                                     feature_save_file=args.feature_save_file)
        classes_database = features_database['classes']

    else:
        cropped_images = None
        bbs = None

        with open(args.dataset_feature_load_file, 'rb') as handle:
            features_database = pickle.load(handle)
        classes_database = features_database['classes'][0]

    images_database = features_database['images']
    people_database = features_database['people']
    features_database = features_database['features']

    # normalization
    mu = np.mean(features_database, 0)
    mu = np.expand_dims(mu, 0)
    std = np.expand_dims(np.sqrt(np.sum(np.power(features_database, 2), 1)), 1)

    features_database = (features_database - mu) / std

    if args.preprocessing_method == 'openface':
        model_align = AlignDlib('../dlib/shape_predictor_68_face_landmarks.dat')
    else:
        model_align=None


    img_query = imageio.imread(args.path_image_query)
    imgl_cropped, bb = preprocess(img_query, args.preprocessing_method, crop_size=crop_size, model=model_align)
    imglist = [imgl_cropped, imgl_cropped[:, ::-1, :]]

    # normalization of image 
    for i in range(len(imglist)):
        imglist[i] = (imglist[i] - 127.5) / 128.0
        imglist[i] = imglist[i].transpose(2, 0, 1)
    imgs = [torch.from_numpy(i).float() for i in imglist]

    # initialize the network
    if args.model_name == 'mobilefacenet':
        net = model.MobileFacenet()
        ckpt = torch.load(MOBILEFACENET_MODEL_PATH)
        net.load_state_dict(ckpt['net_state_dict'])
    elif args.model_name == 'sphereface':
        net = sphere20a(feature=True)
        net.load_state_dict(torch.load(SPHEREFACE_MODEL_PATH))
    elif args.model_name == 'mobiface':
        net = mobiface_model.MobiFace(final_linear=True)
        ckpt = torch.load(MOBIFACE_MODEL_PATH)
        net.load_state_dict(ckpt['net_state_dict'])
    elif args.model_name == 'openface':
        net = loadOpenFace.OpenFaceModel(use_cuda=gpu)
        net.load_state_dict(torch.load(OPENFACE_MODEL_PATH))
    elif args.model_name == 'facenet':
        net = InceptionResnetV1(pretrained='casia-webface')
        net.load_state_dict(torch.load(FACENET_MODEL_PATH))
    elif args.model_name == 'shufflefacenet':
        net = shuffleface_model.ShuffleFaceNet()
        ckpt = torch.load(SHUFFLEFACENET_MODEL_PATH)
        net.load_state_dict(ckpt['net_state_dict'])
    else:
        raise NotImplementedError("Model " + args.model_name + " not implemented")


    if args.use_gpu:
        net=net.cuda()

    net.eval()

    for i in range(len(imgs)):
        imgs[i] = imgs[i][None,:,:,:]
        imgs[i] = imgs[i].cuda()

    res = [net(d).data.cpu().numpy() for d in imgs]
    query_features = np.concatenate((res[0], res[1]), 1)

    query_features = query_features - mu
    query_features = query_features / std

    scores = query_features@np.transpose(features_database)

    scores = np.squeeze(scores)

    score = list(zip(scores[0], classes_database, images_database, people_database))

    score = sorted(score, key=lambda x: x[0], reverse=True)

    plot_top15_face_retrieval(args.path_image_query, score, imgl_cropped, bb, save_dir=args.visual_results_path)
    plot_top15_person_retrieval(args.path_image_query, score, imgl_cropped, bb, save_dir=args.visual_results_path)