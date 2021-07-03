import os
import numpy as np
import argparse

import PIL.ImageDraw as ImageDraw
from PIL import Image
import imageio


def str2bool(v):
    """
    Function to transform strings into booleans.

    v: string variable
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def compute_cmc(ranked_list, query_label):
    """
    Calculate cumulative match curve.

    ranked_list: list ordered by similarity
    query_label: query image label
    """
    cmc = np.zeros(len(ranked_list))

    for i in range(len(ranked_list)):
        if ranked_list[i][3] == query_label:   # if not empty
            cmc[i:] = 1
            break
    return cmc


def compute_map(scores, query_label):
    """
    Calculate mean average precision.

    scores: list ordered by similarity
    query_label: label of query image
    """
    tp = np.zeros(len(scores))
    num_match = 0

    for j in range(len(scores)):
        if scores[j][3] == query_label:
            num_match += 1
            tp[j] = num_match/(j+1)

    return tp, num_match


def plot_bbs(input_image, output_path, bbs):
    im = Image.open(input_image)  # '/home/kno/recfaces/datasets/LFW/lfw/Alejandro_Toledo/Alejandro_Toledo_0028.jpg')

    draw = ImageDraw.Draw(im)

    for i, bb in enumerate(bbs):
        print(bb)
        draw.rectangle([(bb[0], bb[1]), (bb[2], bb[3])], fill=None, outline='red')
        draw.text((bb[0], bb[1]), str(i) + '-' + str("{0:.3f}".format(bb[4])), fill='red')

    imageio.imwrite(os.path.join(output_path, 'bbs.jpg'), im)
