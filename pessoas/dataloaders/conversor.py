import os
import io
import base64
from imageio import imread, imwrite
import validators
import numpy as np


def process_base64(base64_img):
    img = None
    if base64_img[0] == "b":
        img = imread(io.BytesIO(base64.b64decode(base64_img[1:])))
    elif base64_img[:4] == "data":
        img = imread(io.BytesIO(base64.b64decode(base64_img[base64_img.find(","):])))
    else:
        img = imread(io.BytesIO(base64.b64decode(base64_img)))
    return img


def read_image(image):
    try:
        img = None
        # if type is array, just return
        if type(image) is np.ndarray:
            return image
        # otherwise, check if it is file or link
        else:
            image = image.strip()
            if os.path.isfile(image):
                if image.endswith("txt"):
                    f = open(image, "r")
                    base64_img = f.read()
                    img = process_base64(base64_img)
                    f.close()
                elif image.endswith("json"):
                    f = open(image, "r")
                    base64_img = f.read()
                    img = process_base64(base64_img)
                    f.close()
                else:
                    img = imread(image)
            elif validators.url(image):
                img = imread(image)
            
        return img
    except:
        raise NotImplementedError("Could not identify and read image")


if __name__ == '__main__':
    path_image = "image.json"

    img = read_image(path_image)

    imwrite('image.png', img)
