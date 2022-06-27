import os
import tempfile
import numpy as np
import time

import cv2
import pafy

import torch

from pathlib import Path
from PIL import Image

from ..preprocessing.preprocessing_general import PreProcess
from .conversor import generate_sha256


vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']


def download_youtube(url, output_folder='tmp'):
    v = pafy.new(url)
    filename = f"{v.videoid}[{v.title.replace(' ', '_')}].mp4"
    if output_folder == 'tmp':
        output_folder = tempfile.TemporaryDirectory().name
    
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)  # Create output folder if necessary

    v.getbest(preftype="mp4").download(filepath=os.path.join(output_folder, filename), quiet=True)

    return os.path.join(output_folder, filename)


class VideoDataLoader:
    def __init__(self, n_frames=None, batch_size=60, resize=None, output_folder='videos',
                 preprocessing_method=None, crop_size=(96, 112), return_only_one_face=False):
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.resize = resize
        self.output_folder = output_folder

        self.preprocessing_method = preprocessing_method
        self.crop_size = crop_size
        self.return_only_one_face = return_only_one_face
        self.detector = PreProcess(self.preprocessing_method, crop_size=self.crop_size,
                                   return_only_one_face=self.return_only_one_face)

    def __call__(self, filename):
        if 'youtube.com/' in filename.lower() or 'youtu.be/' in filename.lower():  # if is YouTube video
            try:
                st = time.time()
                filename = download_youtube(filename, self.output_folder)
                print("Video Download Concluded in", time.time() - st)
            except:
                raise AssertionError("Could not download youtube video " + filename)

        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_hash = generate_sha256(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert v_len > 0, "Could not identify or load video"
        print("Video length:", v_len, "frames")
        # print(v_len)

        # Pick 'n_frames' evenly spaced frames to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            if self.n_frames > v_len:
                print("########## Number of skipped frames greater than video length ##########")
                sample = np.arange(0, v_len)
            else:
                self.n_frames = int(np.ceil(v_len/self.n_frames))
                sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        # Loop through frames
        first = True
        frames = []
        faces = []
        bbs = []
        crops = []
        frames_batches = []
        img_batches = []
        crop_batches = []
        bb_batches = []
        for j in range(0, v_len):
            success = v_cap.grab()
            if j in sample:
                # Load frame
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)

                # Resize frame to desired size
                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])
                # frames.append(frame)
                
                imgl, bb = self.detector.preprocess(np.array(frame))
                if imgl.size == 0 or bb.size == 0:
                    continue
                imglist = [imgl, imgl[:, :, ::-1, :]]
                
                # normalization
                for i in range(len(imglist)):
                    imglist[i] = (imglist[i] - 127.5) / 128.0
                    imglist[i] = imglist[i].transpose(0, 3, 1, 2)
                imgs = [torch.from_numpy(i).float() for i in imglist]
                
                if first:
                    # repeat because of multiple faces detected in one frame
                    frames = np.repeat(np.expand_dims(np.array(frame), axis=0), imgl.shape[0], axis=0)
                    faces = imgs
                    crops = imgl
                    bbs = bb
                    first = False
                else:
                    frame = np.repeat(np.expand_dims(np.array(frame), axis=0), imgl.shape[0], axis=0)
                    frames = np.concatenate((frames, frame))
                    faces[0] = torch.cat((faces[0], imgs[0]), 0)
                    faces[1] = torch.cat((faces[1], imgs[1]), 0)
                    crops = np.concatenate((crops, imgl))
                    bbs = np.concatenate((bbs, bb))
                
                # When batch is full, reset list
                if len(faces) % self.batch_size == 0 or j == sample[-1]:
                    frames_batches.append(frames)
                    frames = []
                    img_batches.append(faces)
                    faces = []
                    crop_batches.append(crops)
                    crops = []
                    bb_batches.append(bbs)
                    bbs = []
                    first = True

        v_cap.release()

        # print(np.asarray(frames_batches).shape, np.asarray(img_batches).shape, img_batches[0][0].shape,
        #       img_batches[0][1].shape,  np.asarray(crop_batches).shape, np.asarray(bb_batches).shape)
        if len(faces) != 0:
            frames_batches.append(frames)
            frames = []
            img_batches.append(faces)
            faces = []
            crop_batches.append(crops)
            crops = []
            bb_batches.append(bbs)
            bbs = []

        return frames_batches, img_batches, crop_batches, bb_batches, v_len, v_hash, sample
