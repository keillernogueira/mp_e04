import numpy as np
import sys
import cv2

from .matlab_cp2tform import get_similarity_transform_for_cv2


def alignment(src_img, src_all_pts, crop_size):
    ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]]

    all_faces = np.empty([src_all_pts.shape[0], crop_size[1], crop_size[0], 3])
    
    for i, src_pts in enumerate(src_all_pts):
        # src_pts = np.array(src_pts).reshape(5, 2)
        src_pts = [[src_pts[0], src_pts[5]], [src_pts[1], src_pts[6]], [src_pts[2], src_pts[7]],
                   [src_pts[3], src_pts[8]], [src_pts[4], src_pts[9]]]
        s = np.array(src_pts).astype(np.float32)
        r = np.array(ref_pts).astype(np.float32)

        tfm = get_similarity_transform_for_cv2(s, r)
        face_img = cv2.warpAffine(src_img, tfm, crop_size)
        all_faces[i] = face_img
    return all_faces


def load_landmarks(path, image_name):
    landmark = {}
    with open(path) as f:
        landmark_lines = f.readlines()

    for line in landmark_lines:
        l = line.replace('\n', '').split('\t')
        landmark[l[0]] = [int(k) for k in l[1:]]

    return landmark[image_name]
