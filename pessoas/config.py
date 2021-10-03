import os

# general configuration
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(ROOT_DIR, 'models')
DATASET_DIR = os.path.join(ROOT_DIR, 'datasets')

# model dirs
MOBILEFACENET_MODEL_PATH = os.path.join(MODEL_DIR, 'mobilefacenet.ckpt')
MOBIFACE_MODEL_PATH = os.path.join(MODEL_DIR, 'mobiface.ckpt')

SPHEREFACE_MODEL_PATH = os.path.join(MODEL_DIR, 'sphere20a_20171020.pth')
OPENFACE_MODEL_PATH = os.path.join(MODEL_DIR, 'openface.pth')
FACENET_MODEL_PATH = os.path.join(MODEL_DIR, 'facenet.pt')
SHUFFLEFACENET_MODEL_PATH = os.path.join(MODEL_DIR, 'shufflefacenet.ckpt')
CURRICULARFACE_MODEL_PATH = os.path.join(MODEL_DIR, 'CurricularFace_Backbone.pth')
ARCFACE_MODEL_PATH = os.path.join(MODEL_DIR, 'arcface_backbone.pth')
COSFACE_MODEL_PATH = os.path.join(MODEL_DIR, 'cosface_backbone.pth')

# data dirs
LFW_GENERAL_DATA_DIR = os.path.join(DATASET_DIR, 'LFW')
LFW_UPDATE_GENERAL_DATA_DIR = os.path.join(DATASET_DIR, 'LFW')
YALEB_GENERAL_DATA_DIR = os.path.join(DATASET_DIR, 'YaleFaceCroppedB')
