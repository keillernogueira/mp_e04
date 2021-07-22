import os
import tempfile
import numpy as np
import cv2
import pafy

import torch

from pathlib import Path
from PIL import Image

from preprocessing.preprocessing_general import PreProcess


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
                filename = download_youtube(filename, self.output_folder)
            except:
                raise AssertionError("Could not download youtube video " + filename)

        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(v_len)

        # Pick 'n_frames' evenly spaced frames to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        # Loop through frames
        first = True
        faces = []
        bbs = []
        img_batches = []
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
                    faces = imgs
                    bbs = bb
                    first = False
                else:
                    faces[0] = torch.cat((faces[0], imgs[0]))
                    faces[1] = torch.cat((faces[1], imgs[1]))
                    bbs = np.concatenate((bbs, bb))
                    # print('4', np.asarray(faces).shape, np.asarray(faces[0]).shape, np.asarray(faces[1]).shape,
                    #       type(faces[0]), type(faces[1]), bbs.shape)

                # When batch is full, reset list
                if len(faces[0]) % self.batch_size == 0 or j == sample[-1]:
                    img_batches.append(faces)
                    bb_batches.append(bbs)
                    first = True

        v_cap.release()

        return img_batches, bb_batches
