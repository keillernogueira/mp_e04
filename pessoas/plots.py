import io
import base64
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.patches as patches
import imageio
import cv2
import validators
from datetime import datetime

from config import *
from dataloaders.conversor import read_image


def plot_top15_face_retrieval(query_image, query_person, scores, query_num,
                              metrics=None, cropped_image=None, bb=None, save_dir="outputs/"):
    """
    Function to plot the top 15 visual results of the queries.
    Face retrieval considers the most similar images, including images of the same person.
    query_image: adress to the query image.
    query_person: name of the person in query image.
    scores: ranked list containing the information of the images.
    query_num: int representing the queru number when evaluating the whole dataset.
    metrics: vector with the calculated metrics.
    cropped_image: query image cropped by preprocessing.
    bb: bouding boxes of query image by preprocessing.
    save_dir: directory where are saved the image results.
    """
    fig, axes = plt.subplots(4, 5, figsize=(15, 15), sharex=True, sharey=True)
    ax = axes.ravel()

    # decode base64 file   
    if query_image.endswith("txt"):
        f = open(query_image, "r")
        base64_img = f.read()
        if base64_img[0] == "b":
            img = imageio.imread(io.BytesIO(base64.b64decode(base64_img[1:])))
        else:
            img = imageio.imread(io.BytesIO(base64.b64decode(base64_img)))    
    else:
        img = imageio.imread(query_image.strip())

    ax[0].set_title('| Query image |\nPerson: %s\nImage: %s' %
                    (query_person, os.path.basename(os.path.splitext(query_image)[0])))
    ax[0].imshow(img)

    ax[1].imshow(img)
    if not np.array_equal(bb, [0, 0, 0, 0]):
        ax[1].set_title('| Bounding Box |')
        rect = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1],
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax[1].add_patch(rect)
    else:
        ax[1].set_title('| NO Bounding Box |')

    if cropped_image is not None:
        ax[2].set_title('| Cropped Face |')
        shift = 75  # this shift is only used to center the cropped image into de subplot
        ax[2].imshow(cropped_image, extent=(shift, shift + cropped_image.shape[1],
                                            shift + cropped_image.shape[0], shift))
    else:
        ax[2].set_title('| NO Cropped Face |')

    if metrics is not None:
        ax[4].text(50, 200, 'Query %i\n\nmAP: %.2f\n\ntop1: %.2f\ntop5: %.2f\ntop10: '
                            '%.2f\ntop20: %.2f\ntop50: %.2f\ntop100: %.2f' % (query_num, metrics[0]*100, metrics[1]*100,
                                                                              metrics[2]*100, metrics[3]*100,
                                                                              metrics[4]*100, metrics[5]*100,
                                                                              metrics[6]*100),
                   style='italic', fontsize=11, bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 10})

    for i in range(15):
        img = read_image(scores[i][2].strip())
        ax[i+5].set_title('| %i |\n%s\n%f' % (i+1, scores[i][1], scores[i][0]))
        ax[i+5].imshow(img)

    for a in ax:
        a.set_axis_off()

    # fig.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig.savefig(os.path.join(save_dir, os.path.basename(os.path.splitext(query_image)[0]) +
                             "_" + str(query_num) + ".jpg"))
    
    plt.close(fig)


def plot_top15_person_retrieval(query_image, query_person, scores, query_num, image_name,
                                cropped_image=None, bb=None, save_dir="outputs/"):
    """
    Function to plot the top 10 visual results of the queries.
    Person retrieval considers the most similar persons, not repeating the images of the same person.
    query_image: address to the query image.
    query_person: name of the person in query image.
    scores: ranked list containing the information of the images.
    query_num: int representing the queru number when evaluating the whole dataset.
    metrics: vector with the calculated metrics.
    cropped_image: query image cropped by preprocessing.
    bb: bouding boxes of query image by preprocessing.
    save_dir: directory where are saved the image results.
    """
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 15), sharex=True, sharey=True)
    ax = axes.ravel()

    # decode base64 file
    img = read_image(query_image)
    
    basewidth = 250
    wpercent = (basewidth / float(img.shape[1]))
    hsize = int((float(img.shape[0]) * float(wpercent)))
    hpercent = (hsize / float(img.shape[0]))
    img = cv2.resize(img, (basewidth, hsize))

    if os.path.isfile(query_image):
        ax[0].set_title('| Query image |\nPerson: %s\nImage: %s' %
                        (query_person, os.path.basename(os.path.splitext(query_image)[0])))
    else:
        ax[0].set_title('| Query image |\nPerson: %s' % query_person)
    ax[0].imshow(img)

    ax[1].imshow(img)

    bb[0] = bb[0]*wpercent
    bb[2] = bb[2]*wpercent
    bb[1] = bb[1]*hpercent
    bb[3] = bb[3]*hpercent

    if bb is not None:
        ax[1].set_title('| Bounding Box |')
        rect = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1],
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax[1].add_patch(rect)
    else:
        ax[1].set_title('| NO Bounding Box |')

    if cropped_image is not None:
        ax[2].set_title('| Cropped Face |')
        shift = 75  # this shift is only used to center the cropped image into de subplot
        ax[2].imshow(cropped_image.astype('uint8'), extent=(shift, shift + cropped_image.shape[1], shift + cropped_image.shape[0], shift))
    else:
        ax[2].set_title('| NO Cropped Face |')

    unique_persons = []
    i = j = 0
    while i < 10:
        if unique_persons:
            if scores[j][1] not in unique_persons:
                img = read_image(scores[j][2].strip())
                ax[i + 5].set_title('| %i |\n%s\n%f' % (i + 1, scores[j][1], scores[j][0]))
                ax[i + 5].imshow(img)
                unique_persons.append(scores[j][1])
                i += 1
        else:
            img = read_image(scores[j][2].strip())
            ax[i + 5].set_title('| %i |\n%s\n%f' % (i + 1, scores[j][1], scores[j][0]))
            ax[i + 5].imshow(img)
            unique_persons.append(scores[j][1])
            i += 1
        j += 1

    for a in ax:
        a.set_axis_off()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig.savefig(os.path.join(save_dir, image_name +
                             "_" + str(query_num) + "_person_retrieval.jpg"))
    plt.close(fig)