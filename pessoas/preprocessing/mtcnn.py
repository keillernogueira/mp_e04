import numpy as np
from PIL import Image


def mtcnn_crop_image(img, bounding_boxes, detect_multiple_faces=False, margin=44, image_size=112):
    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        if nrof_faces > 1:
            if detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                img_center = img_size // 2
                offsets = np.vstack(
                    [(det[:, 0] + det[:, 2]) // 2 - img_center[1], (det[:, 1] + det[:, 3]) // 2 - img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                det_arr.append(det[index, :])
        else:
            det_arr.append(np.squeeze(det))

        if isinstance(image_size, int):
            all_faces = np.empty([nrof_faces, image_size, image_size, 3])
        else:
            all_faces = np.empty([nrof_faces, image_size[1], image_size[0], 3])

        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = cropped

            out_img = Image.fromarray(scaled)
            if isinstance(image_size, int):
                all_faces[i] = out_img.resize((image_size, image_size))
            else:
                all_faces[i] = out_img.resize(image_size)

        return all_faces
