import logging

import imageio
from PIL import Image

from .mtcnn import mtcnn_crop_image
from .sphereface import alignment, load_landmarks
from .align_dlib import *

from .mtcnn_network.detector import detect_faces
from .mtcnn_network_v2.mtcnn import MTCNN

import torch


class PreProcess(object):
    def __init__(self, preprocessing_method, crop_size=(96, 112), has_load_landmarks=False, img_name=None,
                 is_processing_dataset=False, return_only_one_face=False, execute_default=False, gpu=True):
        """
        Constructor

        :param preprocessing_method: string with the name of the preprocessing method.
        :param crop_size: retrieval network specific crop size.
        :param has_load_landmarks: boolean if exists the file with the landmarks of lfw.
        :param img_name: name of person whose the landmarks are to be loaded.
        :param is_processing_dataset: bool to indicate if it is processing a dataset.
        :param return_only_one_face: bool to indicate if it is returning ONLY the largest BB.
                            If true, return only the face of the largest bounding box.
        :param execute_default: bool to indicate if the algorithm should execute the default pre-processing.
                                If False, if the pre-processing produces an error, no default will be executed.
        :param gpu: bool to indicate if it is going to use GPU
        """
        self.preprocessing_method = preprocessing_method
        self.crop_size = crop_size
        self.has_load_landmarks = has_load_landmarks
        self.img_name = img_name
        self.is_processing_dataset = is_processing_dataset
        self.return_only_one_face = return_only_one_face
        self.execute_default = execute_default

        self.mtcnn = MTCNN(keep_all=True, selection_method="largest",
                           device=torch.device('cuda') if gpu is True else torch.device('cpu'))

    def preprocess(self, img):
        """
        Function to do the preprocessing of the images. Detect the faces and
        the points of interest, rotate and crop the image.

        :param img: an imageio image to be preprocessed.
        """
        # this will be returned
        img_res = []
        bounding_boxes = []

        exception_in_pre_processing = False
        if self.preprocessing_method is not None:
            try:
                if self.preprocessing_method == 'openface':
                    # model to detect faces used in openface
                    model = AlignDlib('landmarks/shape_predictor_68_face_landmarks.dat')
                    img, __ = model.align(112, img)
                    bounding_boxes = np.array([[0., 0., 255., 255., 0.]]).astype(np.float64)

                    # resize to the crop size
                    img = Image.fromarray(img)
                    img = img.resize(self.crop_size)
                    img_res = np.expand_dims(np.array(img), axis=0)

                elif self.preprocessing_method == 'mtcnn' or self.preprocessing_method == 'sphereface':
                    # model to detect faces used in mtcnn
                    # bounding_boxes, landmarks = detect_faces(Image.fromarray(img))  # v1
                    # print('1', bounding_boxes.shape, landmarks.shape)  # 1 (2, 5) (2, 10)
                    bounding_boxes, probs, landmarks = self.mtcnn.detect(img, landmarks=True)
                    if self.return_only_one_face is True:
                        bounding_boxes, probs, landmarks = self.mtcnn.select_boxes(bounding_boxes, probs, landmarks, img,
                                                                                   method=self.mtcnn.selection_method)

                    landmarks = np.concatenate((landmarks[:, :, 0], landmarks[:, :, 1]), axis=1)
                    assert bounding_boxes.size != 0 and landmarks.size != 0, "No face detected in this image"
                    # print('2', bounding_boxes.shape, landmarks.shape)

                    if self.preprocessing_method == 'mtcnn':
                        img_res = mtcnn_crop_image(img, bounding_boxes, detect_multiple_faces=True,
                                                   margin=24, image_size=self.crop_size)
                        print('1', img_res.shape)
                        img_res = self.mtcnn.extract(img, bounding_boxes, "/tmp")
                        print('2', img_res.shape)
                    elif self.preprocessing_method == 'sphereface':
                        if self.has_load_landmarks is True:
                            if self.img_name is None:
                                raise AssertionError("Image name is required to load landmark")
                            # load the landmarks if they exist
                            landmarks = [load_landmarks('landmarks/lfw_landmark.txt', self.img_name)]

                        # model to align the image used in sphereface
                        img_res = alignment(img, landmarks, crop_size=self.crop_size)
                        # print('3', img_res.shape)
                else:
                    raise NotImplementedError("Preprocessing method " + self.preprocessing_method + " not implemented")
            except Exception as e:
                # if there is any error in the processing of the selected pre-process methods
                # active the default pre-processing
                logging.error('Error in ' + self.preprocessing_method + ': ' + str(e))
                exception_in_pre_processing = True
                img_res = []
                bounding_boxes = []

        if self.preprocessing_method is None or (self.execute_default is True and exception_in_pre_processing is True):
            logging.info('Calling pre-processing default.')

            bounding_boxes = np.array([[0., 0., 255., 255., 0.]]).astype(np.float64)
            # this happens when a bounding box of the face is not recognized
            cropx, cropy = self.crop_size[0], self.crop_size[1]

            img = Image.fromarray(img)
            img = img.resize((200, 200))
            img = np.array(img)

            y = img.shape[0]
            x = img.shape[1]
            startx = x // 2 - (cropx // 2)
            starty = y // 2 - (cropy // 2)

            img_res = np.asarray([img[starty:starty + cropy, startx:startx + cropx]])

        if len(bounding_boxes) > 1 and self.return_only_one_face is True:
            # select the largest BB if it is processing dataset
            # this is because dataset processing expects to receive only ONE image and bb
            largest_bb_area = -999999999.9
            index = -1
            for i, bb in enumerate(bounding_boxes):
                area = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)
                if area > largest_bb_area:
                    index = i
                    largest_bb_area = area
            img_res, bounding_boxes = img_res[index], bounding_boxes[index]

            if self.is_processing_dataset is False:
                img_res = np.expand_dims(np.array(img_res), axis=0)
                bounding_boxes = np.expand_dims(np.array(bounding_boxes), axis=0)

        return np.asarray(img_res), np.asarray(bounding_boxes)
