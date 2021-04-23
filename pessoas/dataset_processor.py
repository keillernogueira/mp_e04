import time
import scipy.io

import torch.utils.data

from config import *
from utils import *
from networks.load_network import load_net
from plots import plot_top15_face_retrieval, plot_top15_person_retrieval

from dataloaders.LFW_dataloader import LFW
# from dataloaders.LFW_UPDATE_dataloader import LFW_UPDATE
from dataloaders.yale_dataloader import YALE


def extract_features(model_name, dataloader, gpu=True, save_img_results=False):

    """
    Function to extract features of images for a WHOLE dataset.

    :param model_name: string with the name of the model used.
    :param dataloader: dataloader used to load the images.
    :param gpu: boolean to allow use gpu.
    :param save_img_results: boolean that indicates that we want to save image samples of the produced results.
    :return a dict composed of the extracted features, names, classes, images, cropped images, and bounding boxes.
    """

    # initialize the network
    net = load_net(model_name, gpu)
    if gpu:
        net = net.cuda()

    net.eval()

    features = None
    classes = None
    images = None
    names = None
    cropped_images = []
    bbs = []
    count = 0

    # forward
    for imgs, cls, crop_img, bb, imgl_list, people in dataloader:
        if gpu:
            for i in range(len(imgs)):
                imgs[i] = imgs[i].cuda()
        count += imgs[0].size(0)
        if count % (dataloader.batch_size*10) == 0:
            print('extracing deep features from the face {}...'.format(count))

        res = [net(d).data.cpu().numpy() for d in imgs]
        feature = np.concatenate((res[0], res[1]), 1)

        if features is None:
            features = feature
            classes = cls
            images = imgl_list
            names = people
            if save_img_results is True:
                cropped_images = crop_img
                bbs = bb
        else:
            names = np.concatenate((names, people), 0)
            features = np.concatenate((features, feature), 0)
            classes = np.concatenate((classes, cls), 0)
            images = np.concatenate((images, imgl_list), 0)
            if save_img_results is True:
                cropped_images = np.concatenate((cropped_images, crop_img), 0)
                bbs = np.concatenate((bbs, bb), 0)

    print(features.shape, classes.shape, names.shape, images.shape)
    if save_img_results is True:
        print(cropped_images.shape)
        print(bbs.shape)

    result = {'name': names, 'feature': features, 'class': classes, 'image': images,
              'cropped_image': cropped_images, 'bb': bbs}

    return result


def evaluate_dataset(result, metric='map', bib="numpy", gpu=False, save_dir=None):
    """
    Make a query for each image of the dataset and calculate mean average precision.

    :param result: ouput of the features extraction network, contains the features, classes, images,
            people, cropped images and bounding boxes (bbs).
    :param metric: the used metric.
    :param bib: library used to calculate the cosine distance.
    :param gpu: boolean to allow use gpu.
    :param save_dir: saving directory of the visual results.
    :return Void, only prints the final results and save the sample images.
    """
    features = result['feature']
    classes = result['class']
    if len(classes.shape) == 2:
        classes = classes[0]
    images = result['image']
    people = result['name']
    cropped_images = result['cropped_image']
    bbs = result['bb']

    all_cmc = None
    aps = np.zeros((len(features), len(features)-1))
    corrects = np.zeros(len(features))
    # normalizing features
    mu = np.mean(features, 0)
    mu = np.expand_dims(mu, 0)
    features = features - mu
    features = features / np.expand_dims(np.sqrt(np.sum(np.power(features, 2), 1)), 1)

    start = time.time()

    # calculate cosine distance
    if bib == "pytorch":
        features = torch.from_numpy(features)
        if gpu:
            features = features.cuda()
        scores_all = features @ features.t()
    else:
        scores_all = features@np.transpose(features)

    for i in range(len(features)):
        scores = scores_all[i]
        scores = np.delete(scores, i, 0)

        scores = list(zip(scores, classes, images, people))

        scores = sorted(scores, key=lambda x: x[0], reverse=True)

        # calculate metrics, cmc or map
        if metric == 'cmc':
            cmc = compute_cmc(scores, classes[i])
            if all_cmc is None:
                all_cmc = np.reshape(cmc, (1, len(cmc)))
            else:
                all_cmc = np.concatenate((all_cmc, np.reshape(cmc, (1, len(cmc)))), 0)
        else:
            aps[i, :], corrects[i] = compute_map(scores, classes[i])

        if save_dir is not None:
            mean_ap = np.sum(aps[i])/corrects[i]
            top1 = aps[i, 0]
            top5 = np.sum(aps[i, 0:5]) / np.minimum(corrects[i], 5)
            top10 = np.sum(aps[i, 0:10]) / np.minimum(corrects[i], 10)
            top20 = np.sum(aps[i, 0:20]) / np.minimum(corrects[i], 20)
            top50 = np.sum(aps[i, 0:50]) / np.minimum(corrects[i], 50)
            top100 = np.sum(aps[i, 0:100]) / np.minimum(corrects[i], 100)
            metrics = [mean_ap, top1, top5, top10, top20, top50, top100]

            # generate top15 rank images
            plot_top15_face_retrieval(images[i], people[i], scores, i + 1, metrics, cropped_images[i], bbs[i], save_dir)
            plot_top15_person_retrieval(images[i], people[i], scores, i + 1, cropped_images[i], bbs[i], save_dir)

    end = time.time()

    # calculate metrics, cmc or map
    if metric == 'cmc':
        mean_cmc = np.mean(all_cmc, axis=0)
        print('top1: %f top5: %f top10: %f' % (mean_cmc[0], mean_cmc[4], mean_cmc[9]))
    else:
        mean_ap = np.mean(np.sum(aps, axis=1)/corrects)
        top1 = np.mean(aps[:, 0])
        top5 = np.mean(np.sum(aps[:, 0:5], axis=1) / np.minimum(corrects, 5))
        top10 = np.mean(np.sum(aps[:, 0:10], axis=1) / np.minimum(corrects, 10))
        top20 = np.mean(np.sum(aps[:, 0:20], axis=1) / np.minimum(corrects, 20))
        top50 = np.mean(np.sum(aps[:, 0:50], axis=1) / np.minimum(corrects, 50))
        top100 = np.mean(np.sum(aps[:, 0:100], axis=1) / np.minimum(corrects, 100))
        print('mAP: %f top1: %f top5: %f top10: %f top20: %f top50: %f top100: %f' %
              (mean_ap, top1, top5, top10, top20, top50, top100))
    print("Total execution time: %f seconds. Execution time per query: %f seconds." %
          (end - start, (end - start)/len(features)))


def process_dataset(operation, model_name, batch_size,
                    dataset, dataset_folder, img_extension, preprocessing_method, crop_size,
                    result_sample_path, feature_file, gpu):

    if dataset.upper() == 'LFW':
        dataset = LFW(LFW_GENERAL_DATA_DIR, dataset_folder, img_extension, preprocessing_method, crop_size)
    # elif dataset.upper() == 'LFW_UPDATE':
    #     dataset = LFW_UPDATE(LFW_UPDATE_GENERAL_DATA_DIR, dataset_folder, img_extension,
    #                          path_image_query, query_label, preprocessing_method, crop_size)
    elif dataset.upper() == 'YALEB':
        dataset = YALE(YALEB_GENERAL_DATA_DIR, dataset_folder, img_extension, preprocessing_method, crop_size)
    else:
        raise NotImplementedError("Dataset " + dataset + " dataloader not implemented")

    # YALEB dataset only runs with batch 8 because of GPU memory space
    if dataset == 'YALEB' and batch_size == 32:
        batch_size = 8
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=2, drop_last=False)

    if operation == 'extract_features':
        # extract the features for a WHOLE DATASET
        features = extract_features(model_name, dataloader, gpu,
                                    save_img_results=(False if result_sample_path is None else True))
        assert feature_file is not None
        scipy.io.savemat(feature_file, features)
    elif operation == 'generate_rank':
        assert feature_file is not None
        features = scipy.io.loadmat(feature_file)
        evaluate_dataset(features, save_dir=result_sample_path)
    elif operation == 'extract_generate_rank':
        if feature_file is None:
            # extract the features for a WHOLE DATASET...
            features = extract_features(model_name, dataloader, gpu,
                                        save_img_results=(False if result_sample_path is None else True))
        else:
            # ...OR load the previous saved features, if possible
            features = scipy.io.loadmat(feature_file)
        evaluate_dataset(features, save_dir=result_sample_path)
    else:
        raise NotImplementedError("Operation " + operation + " not implemented")
