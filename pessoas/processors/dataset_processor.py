import time
import scipy.io
import numpy as np

import torch.utils.data

from config import *
from utils import *
from networks.load_network import load_net
from plots import plot_top15_face_retrieval, plot_top15_person_retrieval
from sklearn.preprocessing import normalize
from PyRetri import index as pyretri

import pickle
from processors.image_processor import generate_ranking_for_image

from dataloaders.LFW_dataloader import LFW
# from dataloaders.LFW_UPDATE_dataloader import LFW_UPDATE
from dataloaders.yale_dataloader import YALE


def extract_features(dataloader, model, save_img_results=False, gpu=True):
    """
    Function to extract features of images for a WHOLE dataset.

    :param dataloader: dataloader used to load the images.
    :param model: the model
    :param save_img_results: boolean that indicates that we want to save image samples of the produced results.
    :param gpu: boolean to allow use gpu.
    :return a dict composed of the extracted features, names, classes, images, cropped images, and bounding boxes.
    """

    model.eval()

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

        res = [model(d).data.cpu().numpy() for d in imgs]
        # print(res[0].shape, res[1].shape)
        feature = np.concatenate((res[0], res[1]), 1)

        if features is None:
            features = feature
            images = imgl_list
            names = people
            bbs = bb
            if save_img_results is True:
                classes = cls
                cropped_images = crop_img
        else:
            names = np.concatenate((names, people), 0)
            features = np.concatenate((features, feature), 0)
            images = np.concatenate((images, imgl_list), 0)
            bbs = np.concatenate((bbs, bb), 0)
            if save_img_results is True:
                classes = np.concatenate((classes, cls), 0)
                cropped_images = np.concatenate((cropped_images, crop_img), 0)

    print(np.asarray(features).shape, np.asarray(bbs).shape, np.asarray(names).shape, np.asarray(images).shape)
    if save_img_results is True:
        print(cropped_images.shape)
        print(classes.shape)

    if save_img_results is True:
        result = {'name': names, 'feature': features, 'class': classes, 'image': images,
                  'cropped_image': cropped_images, 'bbs': bbs}
    else:
        result = {'name': names, 'feature': features, 'image': images, 'bbs': bbs}

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
    database_data = result
    query_data = result

    if len(database_data['feature']) > 10000:  # assume normalizing with query_data is not going to change too much
            database_features = database_data['normalized_feature']
            query_features = query_data['feature']
            num_features_query = query_features.shape[0]
            print("n:", num_features_query)
            query_bbs = query_data['bbs']
            # normalize query data using dataset data mean
            print(query_features.shape)
            query_features = query_features - (database_data['feature_mean'] - 1e-18)
            print(query_features.shape)
            query_features = normalize(query_features, norm='l2', axis=1)
            print(query_features.shape)
    else:
        database_features = database_data['feature']
        query_features = query_data['feature']
        num_features_query = query_features.shape[0]
        query_bbs = query_data['bbs']
        features = np.vstack((query_features, database_features))
        # calculate the mean
        mu = np.mean(features, 0)
        mu = np.expand_dims(mu, 0)
        # extract mean from features and add a bias
        features = features - (mu - 1e-18)
        # divide by the standard deviation
        print(features.shape)
        features = normalize(features, norm='l2', axis=1)
        print(np.linalg.norm(features[0]))
        query_features = features[0:num_features_query]
        database_features = features[num_features_query:]

    top_k = pyretri.main(query_features, database_features, "PyRetri/configs/oxford.yaml", len(database_features))

    people_2_class = dict()
    class_value = 1
    for name in database_data['name']:
        if name not in people_2_class.keys():
            people_2_class[name] = class_value
            class_value += 1

    start = time.time()
    all_scores_q = [] 
    query_label = [] 
    query_images = []  
    for i, q in enumerate(query_features):
        # database_features = search_features
        query_images.append(query_data['image'][i])
        query_label.append(people_2_class[query_data['name'][i]])
        sf = database_features[top_k[i]]
        if bib == "pytorch":
            q = torch.from_numpy(q)
            sf = torch.from_numpy(sf)
            if gpu:
                q = q.cuda()
                sf = sf.cuda()
            scores_q = q @ sf.t()
        else:
            scores_q = q @ np.transpose(sf)

        # associate confidence score with the label of the dataset and sort based on the confidence
        scores_q = list(zip(scores_q, database_data['name'][top_k[i]],
                            database_data['image'][top_k[i]], database_data['cropped_image'][top_k[i]],
                            database_data['bbs'][top_k[i]]))
        # scores_q = list(zip(scores_q, included_names, included_images))
        scores_q = sorted(scores_q, key=lambda x: x[0], reverse=True)
        all_scores_q.append(scores_q)

    aps = np.zeros((len(database_data['feature']), len(database_data['feature'])-1))
    corrects = np.zeros(len(database_data['feature']))
    
    for i in range(len(database_features)):
        scores = all_scores_q[i].copy()
        scores = np.delete(scores, 0, 0)
        
        classes = [people_2_class[i[1]] for i in scores]
        classes = np.array(classes).astype(np.float)
        bbs = [i[4] for i in scores]
        cropped_image = [i[3] for i in scores]
        images = [i[2] for i in scores]
        people = [i[1] for i in scores]
        scores = [i[0] for i in scores]
        scores = np.array(scores).astype(np.float)
                
        scores = list(zip(scores, people, images, classes))

        # scores = sorted(scores, key=lambda x: x[0], reverse=True)
        # calculate metrics, cmc or map
        if metric == 'cmc':
            cmc = compute_cmc(scores, classes[i])
            if all_cmc is None:
                all_cmc = np.reshape(cmc, (1, len(cmc)))
            else:
                all_cmc = np.concatenate((all_cmc, np.reshape(cmc, (1, len(cmc)))), 0)
        else:
            aps[i, :], corrects[i] = compute_map(scores, query_label[i])

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
            plot_top15_face_retrieval(query_data['image'][i], query_data['name'][i], scores, i + 1, metrics,
                                      query_data['cropped_image'][i], query_data['bbs'][i], save_dir)
            plot_top15_person_retrieval(query_data['image'][i], query_data['name'][i], scores, i + 1,
                                        os.path.basename(os.path.splitext(query_data['image'][i])[0]),
                                        query_data['cropped_image'][i], query_data['bbs'][i], save_dir)

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
                                             num_workers=0, drop_last=False)

    if operation == 'extract_features':
        # extract the features for a WHOLE DATASET
        features = extract_features(dataloader, model=load_net(model_name, gpu=gpu),
                                    save_img_results=(False if result_sample_path is None else True), gpu=gpu)
        assert feature_file is not None
        with open(feature_file, 'wb') as handle:
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif operation == 'generate_rank':
        assert feature_file is not None
        with open(feature_file, 'rb') as handle:
            features = pickle.load(handle)
        evaluate_dataset(features, save_dir=result_sample_path)
    elif operation == 'extract_generate_rank':
        if feature_file is None:
            # extract the features for a ENTIRE DATASET...
            features = extract_features(dataloader, model=load_net(model_name, gpu=gpu),
                                        save_img_results=(False if result_sample_path is None else True), gpu=gpu)
        else:
            # ...OR load the previous saved features, if possible
            with open(feature_file, 'rb') as handle:
                features = pickle.load(handle)
        evaluate_dataset(features, save_dir=result_sample_path)
    else:
        raise NotImplementedError("Operation " + operation + " not implemented")
