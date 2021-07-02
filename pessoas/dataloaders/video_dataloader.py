import numpy as np
import time
import cv2

import torch

import os
from pathlib import Path
import validators

from preprocessing.preprocessing_general import preprocess
from dataloaders.conversor import read_image

from utils import plot_bbs

vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def download_youtube(url, output_folder='tmp'):
    #check_requirements(('pafy', 'youtube_dl'))
    import pafy

    v = pafy.new(url)
    filename = f"{v.videoid}[{v.title.replace(' ', '_')}].mp4"
    if output_folder == 'tmp':
        output_folder = tempfile.TemporaryDirectory().name
    
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True) # Create output folder if necessary

    video = v.getbest(preftype="mp4").download(filepath=os.path.join(output_folder, filename), quiet=True)

    return os.path.join(output_folder, filename)

class VideoDataLoader(object):
    def __init__(self, video_path, img_size=640, stride=32, output_folder='tmp', preprocessing_method=None, crop_size=(96, 112), will_save_features=False):
        """
        Dataloader for specific images.

        :param preprocessing_method: string with the name of the preprocessing method.
        :param crop_size: retrieval network specific crop size.
        :param will_save_features: Loader will extract features to save.
        """
        if isinstance(video_path, str):
            p = str(Path(video_path).absolute())  # os-agnostic absolute path
            if '*' in p:
                files = sorted(glob.glob(p, recursive=True))  # glob
            elif os.path.isdir(p):
                files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
            elif os.path.isfile(p):
                files = [p]  # files
            elif validators.url(video_path):
                files = [video_path]
            else:
                raise Exception(f'ERROR: {p} does not exist')
        elif isinstance(video_path, list):
            p = f"<List Object {path}>"
            files = [x for x in video_path if isinstance(x, str)]

        self.img_size = img_size
        self.stride = stride
        self.video_path = video_path
        self.preprocessing_method = preprocessing_method
        self.crop_size = crop_size
        self.will_save_features = will_save_features

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
            else:
                filtered_files.append(x)
        
        if haveDl and len(notDl) > 0:
                print(f'The following file(s) could not be downloaded : {notDl}.\nProceeding with the availabel files: {filtered_files}')
        print(filtered_files)

        videos = [x for x in filtered_files if x.split('.')[-1].lower() in vid_formats]
        nv = len(videos)
        self.files = videos

        self.nf = nv

        self.mode = 'video'
        
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None

        self.count = 0

        assert self.nf > 0, f'No videos found in {p}. ' \
                            f'Supported formats are:\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        inicio_pegar_imagem_dataloader = time.time()
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        #print("antes")
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
        #print("depois")
        self.frame += 1
        print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        self.imgl = img

        # if image is grayscale, transform into rgb by repeating the image 3 times
        if len(self.imgl.shape) == 2:
            self.imgl = np.stack([self.imgl] * 3, 2)
        print(type(self.imgl))
        try:
            inicio_preprocess = time.time()
            self.imgl, bb = preprocess(self.imgl, self.preprocessing_method,
                                  crop_size=self.crop_size, return_only_largest_bb=self.will_save_features)
            assert self.imgl.size != 0 and bb.size != 0
            fim_preprocess = time.time()
            print("mandar para preprocessamento: ",fim_preprocess - inicio_preprocess)
        except AssertionError:
            # no face detected
            return [], [], [], []

        # append image with its reverse
        imglist = [self.imgl, self.imgl[:, :, ::-1, :]]

        # normalization
        for i in range(len(imglist)):
            imglist[i] = (imglist[i] - 127.5) / 128.0
            imglist[i] = imglist[i].transpose(0, 3, 1, 2)
        imgs = [torch.from_numpy(i).float() for i in imglist]

        # Convert
        #img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        #img = np.ascontiguousarray(img)
        fim_pegar_imagem_dataloader = time.time()
        print("funcao __next__ dataloader: ", fim_pegar_imagem_dataloader-inicio_pegar_imagem_dataloader)
        return imgs, self.video_path, self.imgl, bb

#    def __getitem__(self, index):
#        if self.count == self.nf:
#            raise StopIteration
#        path = self.files[self.count]

#        # Read video
#        self.mode = 'video'
#        ret_val, img0 = self.cap.read()
#        if not ret_val:
#            self.count += 1
#            self.cap.release()
#            if self.count == self.nf:  # last video
#                raise StopIteration
#            else:
#                path = self.files[self.count]
#                self.new_video(path)
#                ret_val, img0 = self.cap.read()

#        self.frame += 1
#        #print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: ', end='')

#        # Padded resize
#        img = letterbox(img0, self.img_size, stride=self.stride)[0]
#        self.imgl = img0

#        # if image is grayscale, transform into rgb by repeating the image 3 times
#        if len(self.imgl.shape) == 2:
#            self.imgl = np.stack([self.imgl] * 3, 2)

#        try:
#            self.imgl, bb = preprocess(self.imgl, preprocessing_method=None,
#                                  crop_size=self.crop_size, return_only_largest_bb=self.will_save_features)
#            assert self.imgl.size != 0 and bb.size != 0
#        except AssertionError:
#            # no face detected
#            return [], [], [], []

#        # plot_bbs(self.image, '/home/kno/recfaces/', bb)

#        # append image with its reverse
#        imglist = [self.imgl, self.imgl[:, :, ::-1, :]]

#        # normalization
#        for i in range(len(imglist)):
#            imglist[i] = (imglist[i] - 127.5) / 128.0
#            imglist[i] = imglist[i].transpose(0, 3, 1, 2)
#        imgs = [torch.from_numpy(i).float() for i in imglist]

#        return imgs, self.video_path, self.imgl, bb
#
    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(self.frames)

    def __len__(self):
        return self.nf