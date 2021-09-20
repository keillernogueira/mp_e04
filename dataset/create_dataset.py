import os
import shutil
import argparse

import numpy as np
from sklearn.model_selection import train_test_split


def create_dataset(origin_folder, destination_folder):
    class_counter = {}
    subfolders = os.listdir(origin_folder)
    valid_subfolders = []
    for fd in subfolders:
        if 'annotations' not in fd and fd != 'knife' and fd != 'pistol' and fd != 'google' and fd != 'bing':
            valid_subfolders.append(fd)

    # print(valid_subfolders)
    # process original images - naver
    for vfd in valid_subfolders:
        counter = 0
        files = os.listdir(os.path.join(origin_folder, vfd + '_annotations'))
        if vfd == 'stiletto_knife' or vfd == 'pocket_knife':
            cl = 'pocket_knife'
        else:
            cl = vfd
        for f in files:
            # print(os.path.join(origin_folder, vfd, os.path.splitext(f)[0] + '.jpg'),
            #       os.path.join(destination_folder, 'images', vfd + '_' + str(counter) + '.jpg'))
            # print(os.path.join(origin_folder, vfd + '_annotations', f),
            #       os.path.join(destination_folder, 'annotations', vfd + '_' + str(counter) + '.txt'))
            try:
                shutil.copy(os.path.join(origin_folder, vfd, os.path.splitext(f)[0] + '.jpg'),
                            os.path.join(destination_folder, 'images', cl + '_' + str(counter) + '.jpg'))
            except FileNotFoundError:
                shutil.copy(os.path.join(origin_folder, vfd, os.path.splitext(f)[0] + '.jfif'),
                            os.path.join(destination_folder, 'images', cl + '_' + str(counter) + '.jpg'))
            shutil.copy(os.path.join(origin_folder, vfd + '_annotations', f),
                        os.path.join(destination_folder, 'annotations', cl + '_' + str(counter) + '.txt'))
            counter += 1
        if cl in class_counter.keys():
            class_counter[cl] += counter
        else:
            class_counter[cl] = counter

    print(class_counter)
    # process google and bing
    for vfd in ['google', 'bing']:
        files = os.listdir(os.path.join(origin_folder, vfd + '_annotations'))
        for f in files:
            cl = os.path.splitext(f)[0].split('_')[0]
            if cl == 'fireaxe':
                cl = 'axe'
            if cl == 'submachine':
                cl = 'submachine_gun'
            if cl in class_counter.keys():
                counter = class_counter[cl]
            else:
                counter = 0
            shutil.copy(os.path.join(origin_folder, vfd, os.path.splitext(f)[0] + '.jpg'),
                        os.path.join(destination_folder, 'images', cl + '_' + str(counter) + '.jpg'))
            shutil.copy(os.path.join(origin_folder, vfd + '_annotations', f),
                        os.path.join(destination_folder, 'annotations', cl + '_' + str(counter) + '.txt'))
            counter += 1
            class_counter[cl] = counter
    print(class_counter)


def split_dataset(path):
    class_relation = []
    file_relation = []
    for f in os.listdir(os.path.join(path, 'annotations')):
        bbs = np.genfromtxt(os.path.join(path, 'annotations', f))
        # print(file_num, bbs, len(bbs))
        if bbs.ndim == 1:
            class_relation.append(int(bbs[0]))
            file_relation.append(f)
        else:
            bincount = np.bincount(bbs[:, 0].astype(int))
            # bin_sort = np.sort(bincount)
            bin_argsort = bincount.argsort()
            if np.bincount(bincount)[-1] > 1:  # this means that multiple classes have the same number bbs in this image
                filtered_classes = bin_argsort[-np.bincount(bincount)[-1]:]
                largest_bb = -1
                largest_bb_class = -1
                for bb in bbs:
                    if bb[0] in filtered_classes and bb[3] * bb[4] > largest_bb:
                        largest_bb = bb[3] * bb[4]
                        largest_bb_class = bb[0]
                class_relation.append(int(largest_bb_class))
                file_relation.append(f)
            else:
                class_relation.append(int(bin_argsort[-1]))
                file_relation.append(f)
    # print(class_file_relation)
    print(np.asarray(file_relation).shape, np.asarray(class_relation).shape, np.bincount(np.asarray(class_relation)))
    x_train, x_test, y_train, y_test = train_test_split(file_relation, class_relation, stratify=class_relation)
    print(np.asarray(x_train).shape, np.asarray(y_train).shape, np.bincount(np.asarray(y_train)))
    print(np.asarray(x_test).shape, np.asarray(y_test).shape, np.bincount(np.asarray(y_test)))
    for x in x_train:
        try:
            shutil.copy(os.path.join(path, 'images', os.path.splitext(x)[0] + '.jpg'),
                        os.path.join(path, 'split', 'images', 'train', os.path.splitext(x)[0] + '.jpg'))
        except FileNotFoundError:
            try:
                shutil.copy(os.path.join(path, 'images', os.path.splitext(x)[0] + '.jfif'),
                            os.path.join(path, 'split', 'images', 'train', os.path.splitext(x)[0] + '.jpg'))
            except FileNotFoundError:
                shutil.copy(os.path.join(path, 'images', os.path.splitext(x)[0] + '.JPG'),
                            os.path.join(path, 'split', 'images', 'train', os.path.splitext(x)[0] + '.jpg'))
        shutil.copy(os.path.join(path, 'annotations', x),
                    os.path.join(path, 'split', 'annotations', 'train', x))
    for x in x_test:
        try:
            shutil.copy(os.path.join(path, 'images', os.path.splitext(x)[0] + '.jpg'),
                        os.path.join(path, 'split', 'images', 'test', os.path.splitext(x)[0] + '.jpg'))
        except FileNotFoundError:
            try:
                shutil.copy(os.path.join(path, 'images', os.path.splitext(x)[0] + '.jfif'),
                            os.path.join(path, 'split', 'images', 'test', os.path.splitext(x)[0] + '.jpg'))
            except FileNotFoundError:
                shutil.copy(os.path.join(path, 'images', os.path.splitext(x)[0] + '.JPG'),
                            os.path.join(path, 'split', 'images', 'test', os.path.splitext(x)[0] + '.jpg'))
        shutil.copy(os.path.join(path, 'annotations', x),
                    os.path.join(path, 'split', 'annotations', 'test', x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create_dataset')

    # general options
    parser.add_argument('--origin_folder', type=str, required=True, help='Folder where the images are')
    parser.add_argument('--destination_folder', type=str, required=True,
                        help='Folder where the images will be copied to')
    parser.add_argument('--operation', type=str, required=True,
                        help='Operation to be performed. Options: create_dataset | split_dataset')
    args = parser.parse_args()
    print(args)

    if args.operation == 'create_dataset':
        create_dataset(args.origin_folder, args.destination_folder)
    elif args.operation == 'split_dataset':
        split_dataset(args.destination_folder)
    else:
        raise NotImplementedError
