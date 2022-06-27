import requests # to get image from the web
import shutil # to save it locally
import tempfile
import json

import glob
import hashlib
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset

from .general import check_requirements
from .datasets import letterbox

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes


def generate_sha256(file_path):
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def read_json(input_file):
    files = []
    with open(input_file) as json_file:
        data = json.load(json_file)
        for p in data['input']:
            if 'src' in p.keys():
                if p['src'].split('.')[-1] in img_formats + vid_formats:
                    files.append(p['src'])

    return files

def download_img(url, output_folder='tmp'):
    ## Importing Necessary Modules
    supported_files = img_formats + vid_formats

    ## Set up the image URL and filename
    filename = url.split("/")[-1]
    assert filename.split(".")[-1] in supported_files, f"Image/Video file {filename.split('.')[-1]} not supported. The file extension should be one of the following: {supported_files}."
    
    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(url, stream = True)

    # Check if the image was retrieved successfully
    if r.status_code == 200:        
        # Create temporary folder if necessary
        if output_folder == 'tmp':
            output_folder = tempfile.TemporaryDirectory().name
        
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True) # Create output folder if necessary

        # Open a local file with wb ( write binary ) permission.
        with open(os.path.join(output_folder, filename),'wb') as fd:    
            for chunk in r.iter_content(chunk_size=1024):
                fd.write(chunk)

            
        print('Image sucessfully Downloaded: ',filename)
        return os.path.join(output_folder, filename)
    else:
        print('Image Couldn\'t be retreived')
        return None

def download_youtube(url, output_folder='tmp'):
    check_requirements(('pafy', 'youtube_dl'))
    import pafy

    v = pafy.new(url)
    filename = f"{v.videoid}[{v.title.replace(' ', '_')}].mp4"
    if output_folder == 'tmp':
        output_folder = tempfile.TemporaryDirectory().name
    
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True) # Create output folder if necessary

    video = v.getbest(preftype="mp4").download(filepath=os.path.join(output_folder, filename), quiet=True)

    return os.path.join(output_folder, filename)

def check_url(url):
    if url.lower().startswith(('http://', 'https://')) or 'youtube.com/' in url.lower() or 'youtu.be/' in url.lower():
        return True
    return False

class DetectLoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32, output_folder='tmp'):
        if isinstance(path, str):
            p = str(Path(path).absolute())  # os-agnostic absolute path
            if '*' in p:
                files = sorted(glob.glob(p, recursive=True))  # glob
            elif os.path.isdir(p):
                files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
            elif os.path.isfile(p):
                files = [p]  # files
            elif check_url(path):
                files = [path]
            else:
                raise Exception(f'ERROR: {p} does not exist')
        elif isinstance(path, list):
            p = f"<List Object {path}>"
            files = [x for x in path if isinstance(x, str)]

        # Download Images if necessary
        notDl = []
        haveDl = False
        filtered_files = []
        for x in files:
            if 'youtube.com/' in x.lower() or 'youtu.be/' in x.lower():  # if is YouTube video
                haveDl = True
                try:
                    x = download_youtube(x, output_folder)
                    filtered_files.append(x)
                except:
                    notDl.append(x)
            elif x.lower().startswith(('http://', 'https://')):
                haveDl = True
                try:
                    x = download_img(x, output_folder)
                    filtered_files.append(x)
                except:
                    notDl.append(x)
            else:
                filtered_files.append(x)
                
        if haveDl and len(notDl) > 0:
            print(f'The following file(s) could not be downloaded : {notDl}.\nProceeding with the availabel files: {filtered_files}')

        images = [x for x in filtered_files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in filtered_files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # hash
        hash_data = generate_sha256(path)

        return path, img, img0, self.cap, hash_data

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files
