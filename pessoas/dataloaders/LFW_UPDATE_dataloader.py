import argparse
import imageio
import torch
import base64
import os
import io
from sklearn import preprocessing

from ..preprocessing.preprocessing_general import preprocess
from ..preprocessing.align_dlib import *


class LFW_UPDATE(object):
    def __init__(self, root, specific_folder, img_extension, query_path=None, query_label=None,
                 preprocessing_method='sphereface', crop_size=(96, 112)):
        """
        Dataloader of the LFW dataset updating with the new image query.

        root: path to the dataset to be used.
        specific_folder: specific folder inside the same dataset.
        img_extension: extension of the dataset images.
        preprocessing_method: string with the name of the preprocessing method.
        crop_size: retrieval network specific crop size.
        """
        self.preprocessing_method = preprocessing_method
        self.img_extension = img_extension
        self.crop_size = crop_size
        self.imgl_list = []
        self.classes = []
        self.people = []
        self.model_align = None

        # read the file with the names and the number of images of each people in the dataset
        with open(os.path.join(root, 'people_update.txt')) as f:
            people = f.read().splitlines()[1:]

        if query_label is not None and query_path is not None:
            self.imgl_list.insert(0, query_path)
            self.classes.insert(0, '_'+query_label)
            self.people.insert(0, '_'+query_label)

            # get only the images of the folders that have more than 20 images or 
            # that have been created (starts with underline)
            if os.path.exists(os.path.join(root, specific_folder, "_"+query_label)):
                with open(os.path.join(root, 'people_update.txt'), 'r') as f:
                    lines = f.read().splitlines()
                with open(os.path.join(root, 'people_update.txt'), 'w') as f:
                    for line in lines:
                        query = line.split('\t')
                        if query[0] == '_'+query_label:
                            f.write('_'+query_label+'\t'+str(int(query[1])+1)+'\n')
                            if self.img_extension == "txt":
                                f_ = open(query_path, "r")
                                new_f = open(os.path.join(root, specific_folder, "_"+query_label, query_label+'_{:04}'.format(int(query[1])+1)+'.'+img_extension), "w+")
                                new_f.write(f_.read())
                                f_.close()
                                new_f.close()
                            else:
                                imgl = imageio.imread(query_path)
                                imageio.imwrite(os.path.join(root, specific_folder, "_"+query_label, query_label+'_{:04}'.format(int(query[1])+1)+'.'+img_extension), imgl)
                        else:
                            f.write(line+'\n')
            else:
                os.mkdir(os.path.join(root, specific_folder, "_"+query_label))
                if self.img_extension == "txt":
                    f_ = open(query_path, "r")
                    new_f = open(os.path.join(root, specific_folder, "_"+query_label, query_label+'_{:04}'.format(1)+'.'+img_extension), "w+")
                    new_f.write(f_.read())
                    f_.close()
                    new_f.close()
                else:
                    imgl = imageio.imread(query_path)   
                    imageio.imwrite(os.path.join(root, specific_folder, "_"+query_label, query_label+'_{:04}'.format(1)+'.'+img_extension), imgl)


                with open(os.path.join(root, 'people_update.txt'), 'a') as f:
                    f.write('_'+query_label+'\t'+str(1)+'\n')

        # append in the classes list if the folder have more than 20 images or starts with underline
        for p in people:
            p = p.split('\t')
            if len(p) > 1:

                if int(p[1]) >= 20 or p[0][0] == '_' :
                    for num_img in range(1, int(p[1]) + 1):
                        # remove the underline before append
                        if p[0][0] == '_' :
                            self.imgl_list.append(os.path.join(root, specific_folder, p[0], p[0][1:] + '_' +
                                                               '{:04}'.format(num_img) + '.' + img_extension))
                        else:
                            self.imgl_list.append(os.path.join(root, specific_folder, p[0], p[0] + '_' +
                                                               '{:04}'.format(num_img) + '.' + img_extension))

                        self.classes.append(p[0])
                        self.people.append(p[0])


        le = preprocessing.LabelEncoder()
        self.classes = le.fit_transform(self.classes)

        # load the dlib model if the preprocess is openface
        if self.preprocessing_method == 'openface':
            self.model_align = AlignDlib('../dlib/shape_predictor_68_face_landmarks.dat')

        print(len(self.imgl_list), len(self.classes), len(self.people))

    def __getitem__(self, index):
        if self.img_extension == "txt":
            with open(self.imgl_list[index], "r") as f:
                base64_img = f.read()

                if base64_img[0] == "b":
                    imgl = imageio.imread(io.BytesIO(base64.b64decode(base64_img[1:])))
                else:
                    imgl = imageio.imread(io.BytesIO(base64.b64decode(base64_img)))
        else:
            imgl = imageio.imread(self.imgl_list[index])
        cl = self.classes[index]

        # if image is grayscale, transform into rgb by repeating the image 3 times
        if len(imgl.shape) == 2:
            imgl = np.stack([imgl] * 3, 2)

        imgl_cropped_0, bb_0 = preprocess(imgl, self.preprocessing_method, crop_size=self.crop_size, model=self.model_align)

        num_rotate += 1

        # if none bb are found, rotate 90, 180, 270 degree and use the bb with the biggest score
        if bb_0[4] == 0:
            imgl_ = np.rot90(imgl)
            imgl_cropped_90, bb_90 = preprocess(imgl_, self.preprocessing_method, crop_size=self.crop_size, model=self.model_align)

            imgl_ = np.rot90(imgl_)
            imgl_cropped_180, bb_180 = preprocess(imgl_, self.preprocessing_method, crop_size=self.crop_size, model=self.model_align)

            imgl_ = np.rot90(imgl_)
            imgl_cropped_270, bb_270 = preprocess(imgl_, self.preprocessing_method, crop_size=self.crop_size, model=self.model_align)

            list_acc = [bb_90[4], bb_180[4], bb_270[4]]

            bigger_acc = max(list_acc)

            index_bigger = list_acc.index(bigger_acc)

            if index_bigger == 0:
                imgl_cropped = imgl_cropped_90
                bb = bb_90
            if index_bigger == 1:
                imgl_cropped = imgl_cropped_180
                bb = bb_180
            if index_bigger == 2:
                imgl_cropped = imgl_cropped_270
                bb = bb_270
        else:
            imgl_cropped = imgl_cropped_0
            bb = bb_0

        # append image with its reverse
        imglist = [imgl_cropped, imgl_cropped[:, ::-1, :]]

        # normalization
        for i in range(len(imglist)):
            imglist[i] = (imglist[i] - 127.5) / 128.0
            imglist[i] = imglist[i].transpose(2, 0, 1)
        imgs = [torch.from_numpy(i).float() for i in imglist]

        return imgs, cl, imgl_cropped, bb, self.imgl_list[index], self.people[index]

    def __len__(self):
        return len(self.imgl_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LFW_UPDATE')
    # model options
    parser.add_argument('--path_query', type=str, required=True, help='query path')
    parser.add_argument('--query_label', type=str, required=True, help='query label')

    args = parser.parse_args()
    print(args)

    root = '/home/users/matheusb/recfaces/datasets/LFW/'
    specific_folder = 'lfw'
    img_extension = 'jpg'
    query_path = args.path_query
    query_label = args.query_label

    LFW_UPDATE(root, specific_folder, img_extension, query_path, query_label)

