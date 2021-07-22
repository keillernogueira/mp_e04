import time
import numpy as np

import torch

from dataloaders.video_dataloader import VideoDataLoader
from networks.load_network import load_net


def process_video(video_paths, model):
    # Define face detection pipeline
    detection_pipeline = VideoDataLoader(batch_size=60, resize=0.5, preprocessing_method='sphereface',
                                         return_only_one_face=True)

    start = time.time()
    n_processed = 0
    with torch.no_grad():
        for i, filename in enumerate(video_paths):
            # Load frames and find faces
            batches_imgs, batches_bbs = detection_pipeline(filename)

            for j in range(len(batches_imgs)):  # batch loop
                imgs, bbs = batches_imgs[j], batches_bbs[j]

                for i in range(len(imgs)):
                    imgs[i] = imgs[i].cuda()
                res = [model(d).data.cpu().numpy() for d in imgs]
                feature = np.concatenate((res[0], res[1]), 1)

                # TODO gerar saida

                n_processed += len(bbs)

    print("Total time: " + str(time.time() - start))
    print("Frames per second: " + str(n_processed / (time.time() - start)))


if __name__ == '__main__':
    process_video(["https://www.youtube.com/watch?v=-TXBxxPAtb0"], load_net('mobilefacenet', gpu=True))
