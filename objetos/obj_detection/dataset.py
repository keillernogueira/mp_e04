import os
from unicodedata import normalize
import numpy as np
import torch
import random
import math
import matplotlib.pyplot as plt

from torch.utils import data
from torchvision import transforms as T

from skimage import io, transform
from skimage import img_as_float, img_as_float32, img_as_ubyte
from skimage.color import gray2rgb

import cv2

'''
# This class ListDataset use the dataset name and task name to load the respectives images accordingly, modify this class to your use case
# For a simple loader, this version is implemented considering the following folder scheme:

/datasets_root_folder
|--- images
    |--- train
        |--- img_0.jpg
        ...
    |--- test
        |--- img_k.jpg
        ...

|--- labels
     |--- train
        |--- img_0.txt
        ...
    |--- test
        |--- img_k.txt
        ...

'''
#hyperparameters for data augmenting
hyp = {
    'degrees': 0.0,  # image rotation (+/- deg)
    'translate': 0.1,  # image translation (+/- fraction)
    'scale': 0.5,  # image scale (+/- gain)
    'shear': 0.0,  # image shear (+/- deg)
    'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
}

# hsv augmentation flag
HSV_AUG = True

# normalization means and stds for different datasets
norms = {'imagenet': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

# Class that reads a sequence of image paths from a text file and creates a data.Dataset with them.
class ListDataset(data.Dataset):
    def __init__(self, root, mode, img_size=480, class_names=[], num_classes=11,
                 make=True, normvalues=None, quad=False, mosaic=1.0, mixup=0.0):

        self.root = root
        # Initializing variables.
        self.mode = mode
        self.imgs = None
        self.img_size = img_size
        self.num_classes = num_classes
        self.class_names = class_names
        self.normalize = None
        self.quad = quad
        self.mosaic = mosaic
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.mixup = mixup
        self.hyp = hyp

        if isinstance(normvalues, dict):
            self.normalize = T.Normalize(mean=normvalues['mean'],
                                         std=normvalues['std'])
        elif isinstance(normvalues, str):
            self.normalize = T.Normalize(mean=norms[normvalues]['mean'],
                                         std=norms[normvalues]['std'])
                    
        if make:
            # Creating list of paths.
            self.imgs = self.make_dataset()
            self.indices = range(len(self.imgs))

            # Check for consistency in list.
            if len(self.imgs) == 0:
                raise (RuntimeError('Found 0 images, please check the data set'))

    # Function that create the list of img_path
    def make_dataset(self):

        # Making sure the mode is correct.
        assert self.mode in ['train', 'test']
        items = []

        # Setting string for the mode.
        mode_str = ''
        if 'train' in self.mode:
            mode_str = 'train'
        elif 'test' in self.mode:
            mode_str = 'test'

        # Joining input paths.
        self.imgs_path = os.path.join(self.root, 'images', self.mode)
        self.annots_path = os.path.join(self.root, 'labels', self.mode)

        # Reading paths from file.
        data_list = []
        data_list = os.listdir(self.imgs_path)

        # Creating list containing image and ground truth paths.
        for it in data_list:
            item = os.path.join(self.imgs_path, it)
            items.append(item)

        # Returning list.
        return items

    # Function to load images and annotations
    # Returns: img, bound_box, labels
    def get_data(self, index, only_bb=False):
        img_path = self.imgs[index]
        im = os.path.split(img_path)[1]
        ann_path = os.path.join(self.annots_path, os.path.splitext(im)[0] + ".txt")
        labels = []
        bbx = []
        # Reading images.
        img = io.imread(img_path)  # cv2.imread(img_path).astype(np.uint8) #io.imread(img_path)
        # check gray
        if len(img.shape) == 2:
            img = gray2rgb(img)

        with open(ann_path, "r") as f:
            while True:
                bb_list = f.readline().split()
                if not bb_list:
                    break
                labels.append(int(bb_list[0]) + 1)
                xmin = float(bb_list[1]) - (float(bb_list[3]) / 2)
                ymin = float(bb_list[2]) - (float(bb_list[4]) / 2)
                xmax = float(bb_list[1]) + (float(bb_list[3]) / 2)
                ymax = float(bb_list[2]) + (float(bb_list[4]) / 2)
                if xmin == xmax:xmin -= 0.1
                if ymin == ymax:ymin -= 0.1
                bbx.append([xmin, ymin, xmax, ymax])

        if only_bb:
            return bbx, labels

        # img = img_as_float32(img)
        img = img_as_ubyte(img)
        # img = img.astype(np.float32)
        img = img.astype(np.uint8)
        if img.shape[-1] > 3:
            img = img[:, :, 0:3]

        return img, bbx, labels

    def normalize(self, img):
        if len(img.shape) == 2:
            img = (img - img.mean()) / img.std()
        else:
            for b in range(img.shape[2]):
                img[:, :, b] = (img[:, :, b] - img[:, :, b].mean()) / img[:, :, b].std()
        return img

    def norm01(self, img):
        mn = img.min()
        mx = img.max()

        img = (img + mn)/(mx + mn)

        return img

    def torch_channels(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        else:
            img = np.moveaxis(img, -1, 0)
        return img

    def __getitem__(self, index, norm=True):
        mosaic = self.quad and random.random() < self.mosaic
        if mosaic:
            # Load mosaic
            img, bbx, labels = load_mosaic(self, index)  # BGR Image
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < self.mixup:
                img2, bbx2, labels2 = load_mosaic(self, random.randint(0, self.n - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)
                bbx = np.concatenate((bbx, bbx2), 0)

            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=img)
        else:
            img, bbx, labels = self.get_data(index)
            idtype = img.dtype
            img = transform.resize(img, (self.img_size, self.img_size), order=1, preserve_range=True).astype(idtype)

            # Transform bb to the image size
            h, w, _ = img.shape
            bbx = np.array(bbx) * np.array([w, h, w, h])

        if HSV_AUG:
            augment_hsv(img)
            img = img_as_float32(img)
            img = img.astype(np.float32)

        # Normalization.
        if norm:
            # img = self.norm(img)
            img = self.norm01(img)

        # Adding channel dimension.
        img = self.torch_channels(img)

        # Turning to tensors.
        img = torch.from_numpy(img).float()

        if self.normalize is not None and not HSV_AUG:
            img = self.normalize(img)

        # Construction target dict (pytorch default)
        targets = {}
        boxes = torch.as_tensor(bbx, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        targets['boxes'] = boxes
        targets["labels"] = labels

        # Returning to iterator.
        # print(index, targets)
        return img, targets

    def __len__(self):
        return len(self.imgs)

    def visualize(self, index, save=False):
        img, target = self[index]
        bbxs = target['boxes']
        # print(img)
        img = np.moveaxis(img.numpy(), 0, -1)
        h, w, _ = img.shape
        for bbx in bbxs:
            img[max(int(bbx[1]), 0), max(int(bbx[0]), 0):int(bbx[2]), :]=1
            img[max(int(bbx[3]), 0), max(int(bbx[0]), 0):int(bbx[2]), :]=1
            img[max(int(bbx[1]), 0):int(bbx[3]), min(max(int(bbx[0]), 0), w-1), :]=1
            img[max(int(bbx[1]), 0):int(bbx[3]), min(max(int(bbx[2]), 0), w-1), :]=1

        if save:
            io.imsave('vis.png', img)
        return img


class ValidationListDataset(ListDataset):
    def __init__(self, root, mode, img_size=480, class_names=[], num_classes=11, train_size=0.8,
                 normvalues=None, quad=True, mosaic=1.0, mixup=0.0):

        self.root = root
        # Initializing variables.
        self.mode = mode
        self.imgs = None
        self.img_size = img_size
        self.num_classes = num_classes
        self.class_names = class_names
        self.train_size = train_size
        self.quad = quad
        self.mosaic = mosaic
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.mixup = mixup
        self.hyp = hyp

        if isinstance(normvalues, dict):
            self.normalize = T.Normalize(mean=normvalues['mean'], std=normvalues['std'])
        elif isinstance(normvalues, str):
            self.normalize = T.Normalize(mean=norms[normvalues]['mean'], std=norms[normvalues]['std'])

        # Creating list of paths.
        self.imgs = self.make_dataset()
        self.indices = range(len(self.imgs))

        # Check for consistency in list.
        if len(self.imgs) == 0:
            raise (RuntimeError('Found 0 images, please check the data set'))

    # Function that create the list of img_path
    def make_dataset(self):
        items = []

        # Joining input paths.
        self.imgs_path = os.path.join(self.root, 'images', 'validation')
        self.annots_path = os.path.join(self.root, 'labels', 'validation')

        # Reading paths from file.
        data_list = []
        data_list = os.listdir(self.imgs_path)

        # Creating list containing image and ground truth paths.
        for it in data_list:
            item = os.path.join(self.imgs_path, it)
            items.append(item)

        random.shuffle(items)
        if self.mode == 'train':
            items = items[:int(np.ceil(len(items) * self.train_size))]
        else:
            items = items[int(np.ceil(len(items) * self.train_size)):]

        # Returning list.
        return items


def augment_hsv(im, hgain=0.015, sgain=0.7, vgain=0.4):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_RGB2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=im)  # no return needed


def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    path = self.imgs[index]
    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    r = self.img_size / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not HSV_AUG else cv2.INTER_LINEAR)
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized


def load_mosaic(self, index):
    # loads images in a 4-mosaic

    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        bbx, clabels = self.get_data(index, only_bb=True)
        bbx = (np.array(bbx) * np.array([w, h, w, h])) + np.array([padw, padh, padw, padh])
        labels = np.empty((len(clabels), 5))
        labels[:, 0] = clabels
        labels[:, 1:] = bbx
        labels4.append(labels)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in labels4[:, 1:]:
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    img4, labels4 = random_perspective(img4, labels4,
                                    degrees=self.hyp['degrees'],
                                    translate=self.hyp['translate'],
                                    scale=self.hyp['scale'],
                                    shear=self.hyp['shear'],
                                    perspective=self.hyp['perspective'],
                                    border=self.mosaic_border)  # border to remove

    return img4, labels4[:, 1:], labels4[:, 0]  # mosaic, boundbox, labels


def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        new = np.zeros((n, 4))
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates