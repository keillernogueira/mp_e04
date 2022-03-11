import py7zr

import gzip

import torch

from config import *

# MobileFaceNet
from networks.mobilefacenet import MobileFacenet
# SphereFace
from networks.sphereface import sphere20a
# MobiFace
from networks.mobiface import MobiFace
# OpenFace
from networks.openface import OpenFaceModel
# FaceNet
from networks.inception_resnet_facenet import InceptionResnetV1
# ShuffleFaceNet
from networks.shufflefacenet import ShuffleFaceNet
# CurricularFace
from networks.curricularface import IR_101, IR_50
# ArcFace
from networks.arcface import get_model


def load_net(model_name, model_path=None, gpu=True):
    # initialize the network
    if model_name == 'mobilefacenet':
        net = MobileFacenet()
        if gpu:
            ckpt = torch.load(MOBILEFACENET_MODEL_PATH if model_path is None else model_path)
        else:
            ckpt = torch.load(MOBILEFACENET_MODEL_PATH if model_path is None else model_path, map_location='cpu')
        try:
            net.load_state_dict(ckpt['net_state_dict'])
        except:
            raise RuntimeError("Unable to load " + model_name + ". Check if the especified model_path and model_name are of the same network")
        
    elif model_name == 'sphereface':
        net = sphere20a(feature=True)
        if model_path is None and not os.path.exists(SPHEREFACE_MODEL_PATH):
            # unzip default model
            archive = py7zr.SevenZipFile(os.path.join(MODEL_DIR, 'sphereface.7z'))
            archive.extractall(path=MODEL_DIR)
        if gpu:
            ckpt = torch.load(SPHEREFACE_MODEL_PATH if model_path is None else model_path)
        else:
            ckpt = torch.load(SPHEREFACE_MODEL_PATH if model_path is None else model_path, map_location='cpu')
        try:
            net.load_state_dict(ckpt)
        except:
            try:
                net.load_state_dict(ckpt['net_state_dict'])
            except:
                raise RuntimeError("Unable to load " + model_name + ". Check if the especified model_path and model_name are of the same network")
    elif model_name == 'mobiface':
        net = MobiFace(final_linear=True)
        if gpu:
            ckpt = torch.load(MOBIFACE_MODEL_PATH if model_path is None else model_path)
        else:
            ckpt = torch.load(MOBIFACE_MODEL_PATH if model_path is None else model_path, map_location='cpu')
        try:
            net.load_state_dict(ckpt['net_state_dict'])
        except:
            raise RuntimeError("Unable to load " + model_name + ". Check if the especified model_path and model_name are of the same network")
    elif model_name == 'openface':
        net = OpenFaceModel()
        if gpu:
            ckpt = torch.load(OPENFACE_MODEL_PATH if model_path is None else model_path)
        else:
            ckpt = torch.load(OPENFACE_MODEL_PATH if model_path is None else model_path, map_location='cpu')
        try:
            net.load_state_dict(ckpt)
        except:
            try:
                net.load_state_dict(ckpt['net_state_dict'])
            except:
                raise RuntimeError("Unable to load " + model_name + ". Check if the especified model_path and model_name are of the same network")
    elif model_name == 'facenet':
        net = InceptionResnetV1(pretrained='casia-webface')
        if not os.path.exists(FACENET_MODEL_PATH):
            # unzip default model
            extract_gz()
        if gpu:
            ckpt = torch.load(FACENET_MODEL_PATH if model_path is None else model_path)
        else:
            ckpt = torch.load(FACENET_MODEL_PATH if model_path is None else model_path, map_location='cpu')
        try:
            net.load_state_dict(ckpt)
        except:
            try:
                net.load_state_dict(ckpt['net_state_dict'])
            except:
                raise RuntimeError("Unable to load " + model_name + ". Check if the especified model_path and model_name are of the same network")
    elif model_name == 'shufflefacenet':
        net = ShuffleFaceNet()
        if gpu:
            ckpt = torch.load(SHUFFLEFACENET_MODEL_PATH if model_path is None else model_path)
        else:
            ckpt = torch.load(SHUFFLEFACENET_MODEL_PATH if model_path is None else model_path, map_location='cpu')
        try:
            net.load_state_dict(ckpt['net_state_dict'])
        except:
            raise RuntimeError("Unable to load " + model_name + ". Check if the especified model_path and model_name are of the same network")
    elif model_name == 'curricularface':
        net = IR_101([112, 112])
        split_files = [os.path.join(MODEL_DIR, 'a_CurricularFace_Backbone.pth'),
                       os.path.join(MODEL_DIR, 'b_CurricularFace_Backbone.pth'),
                       os.path.join(MODEL_DIR, 'c_CurricularFace_Backbone.pth')]
        join_files(CURRICULARFACE_MODEL_PATH, split_files)

        if gpu:
            ckpt = torch.load(CURRICULARFACE_MODEL_PATH if model_path is None else model_path)
        else:
            ckpt = torch.load(CURRICULARFACE_MODEL_PATH if model_path is None else model_path, map_location='cpu')
        try:
            net.load_state_dict(ckpt)
        except:
            try:
                net.load_state_dict(ckpt['net_state_dict'])
            except:
                raise RuntimeError("Unable to load " + model_name + ". Check if the especified model_path and model_name are of the same network")
    elif model_name == 'arcface':
        net = get_model('r100', dropout=0.0, fp16=True, num_features=512)
        split_files = [os.path.join(MODEL_DIR, 'a_arcface_backbone.pth'),
                       os.path.join(MODEL_DIR, 'b_arcface_backbone.pth'),
                       os.path.join(MODEL_DIR, 'c_arcface_backbone.pth')]
        join_files(ARCFACE_MODEL_PATH, split_files)
        
        if gpu:
            ckpt = torch.load(ARCFACE_MODEL_PATH if model_path is None else model_path)
        else:
            ckpt = torch.load(ARCFACE_MODEL_PATH if model_path is None else model_path, map_location='cpu')
        try:
            net.load_state_dict(ckpt)
        except:
            try:
                net.load_state_dict(ckpt['net_state_dict'])
            except:
                raise RuntimeError("Unable to load " + model_name + ". Check if the especified model_path and model_name are of the same network")
    elif model_name == 'cosface':
        net = get_model('r100', dropout=0.0, fp16=True, num_features=512)
        split_files = [os.path.join(MODEL_DIR, 'a_cosface_backbone.pth'),
                       os.path.join(MODEL_DIR, 'b_cosface_backbone.pth'),
                       os.path.join(MODEL_DIR, 'c_cosface_backbone.pth')]
        join_files(COSFACE_MODEL_PATH, split_files)
        
        if gpu:
            ckpt = torch.load(COSFACE_MODEL_PATH if model_path is None else model_path)
        else:
            ckpt = torch.load(COSFACE_MODEL_PATH if model_path is None else model_path, map_location='cpu')
        try:
            net.load_state_dict(ckpt)
        except:
            try:
                net.load_state_dict(ckpt['net_state_dict'])
            except:
                raise RuntimeError("Unable to load " + model_name + ". Check if the especified model_path and model_name are of the same network")

    else:
        raise NotImplementedError("Model " + model_name + " not implemented")

    if gpu:
        net = net.cuda()
    return net


def extract_gz():
    if os.path.isfile(os.path.join(MODEL_DIR, 'facenet.gz')):
        os.remove(os.path.join(MODEL_DIR, 'facenet.gz'))
    files = [os.path.join(MODEL_DIR, 'facenet.pt.gz.part-aa'), os.path.join(MODEL_DIR, 'facenet.pt.gz.part-ab')]
    with open(os.path.join(MODEL_DIR, 'facenet.gz'), 'ab') as result:  # append in binary mode
        for f in files:
            with open(f, 'rb') as tmpf:        # open in binary mode also
                result.write(tmpf.read())

    input = gzip.GzipFile(os.path.join(MODEL_DIR, 'facenet.gz'), 'rb')
    s = input.read()
    input.close()

    output = open(os.path.join(MODEL_DIR, './facenet.pt'), 'wb')
    output.write(s)
    output.close()


def join_files(new_file_name, file_names):
    with open(new_file_name, 'wb') as result:  # append in binary mode
        for f in file_names:
            with open(f, 'rb') as tmpf:        # open in binary mode also
                result.write(tmpf.read())

    result.close()
