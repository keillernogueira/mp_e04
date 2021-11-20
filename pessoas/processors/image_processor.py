import os
import numpy as np
import scipy.io

import torch

from dataloaders.image_dataloader import ImageDataLoader
from networks.load_network import load_net

import time


def create_feature_segments(features, n_sub_codebooks):
    segments = np.reshape(features, (features.shape[0], n_sub_codebooks, int(features.shape[1]/n_sub_codebooks)))
    return np.swapaxes(segments, 0, 1)


def extract_features_from_image(model, dataloader, query_label, gpu):
    """
    Function to extract features for ONE specific image.
    :param model: the model
    :param dataloader: dataloader used to load the images.
    :param query_label: the image query label, i.e., the ID.
    :param gpu: boolean to allow use gpu.
    :return: a dict with the features, id, and cropped image of the current query.
    """

    # set network to evaluation, and not training
    model.eval()

    feature = None
    img_name = None
    cropped = None
    bbs = None

    # forward
    for imgs, img_nm, imgl, bb in dataloader:
        if not imgs:
            # no face detected
            return None
        if gpu:
            for i in range(len(imgs)):
                imgs[i] = imgs[i].cuda()

        res = [model(d.view(-1, d.shape[2], d.shape[3], d.shape[4])).data.cpu().numpy() for d in imgs]
        feature = np.concatenate((res[0], res[1]), 1)
        img_name = img_nm
        cropped = imgl
        bbs = bb

    # plot_bbs(bbs[0].cpu().numpy())

    result = {'feature': feature, 'name': [query_label], 'image': img_name,
              'bbs': bbs[0].cpu().numpy(), 'cropped_image': cropped[0].cpu().numpy()}

    return result


def generate_rank(scores, k_rank):
    i = j = 0
    persons_scores = []
    unique_persons = []

    # return the top K most similar persons to the query image.
    # if there are less than K persons in the dataset return rank
    # with the maximum possible number of persons.

    while i < k_rank and j < len(scores):
        # if unique_persons is not empty
        if unique_persons:
            # if the i person in the scores list is note in unique_persons
            if scores[j][1] not in unique_persons:
                unique_persons.append(scores[j][1])
                # append tuple with person id, score and image path
                persons_scores.append({"Name": scores[j][1].strip(), "Confidence": scores[j][0],
                                       "Image": scores[j][2].strip()})
                i += 1
        else:
            unique_persons.append(scores[j][1])
            persons_scores.append({"Name": scores[j][1].strip(), "Confidence":
                                   scores[j][0], "Image": scores[j][2].strip()})
            i += 1
        j += 1
    return persons_scores


def generate_ranking_for_image(database_data, query_data, features_meta, k_rank=10, bib="numpy", gpu=False):
    """
    Make a specific query and calculate the average precision.
    :param database_data: features of the entire dataset.
    :param query_data: features of the image query.
    :param k_rank: number of elements in the ranking
    :param bib: library used to calculate the cosine distance.
    :param gpu: boolean to allow use gpu.
    :return: top k list (with ID and confidence) of most similar images.
    """
    # TODO check copy and pointer
    st = time.time()
    database_features = database_data['feature']

    query_features = query_data['feature']
    query_bbs = query_data['bbs']
    num_features_query = query_features.shape[0] # this is just to store the number of faces detected in the query image

    # normalization
    # concatenate the features
    features = np.vstack((query_features, database_features))
    # calculate the mean
    mu = np.mean(features, 0)
    mu = np.expand_dims(mu, 0)
    # extract mean from features and add a bias
    features = features - mu + 1e-18
    # divide by the standard deviation
    features = features / np.expand_dims(np.sqrt(np.sum(np.power(features, 2), 1)), 1)
    query_features = features[0:num_features_query]
    database_features = features[num_features_query:]

    persons_scores = []
    all_scores = []

    vocabulary = list(features_meta)
    M = len(vocabulary[0])
    for img, q in enumerate(query_features):
        sub_codebooks = create_feature_segments(q, M)
        print(sub_codebooks.shape)

        dists = []
        for word in vocabulary:
            dist = 0
            for i in range(len(sub_codebooks)):
                sub_codebook = sub_codebooks[i][0]
                center = word[i]

                for d in range(len(sub_codebook)):
                    dist += (sub_codebook[d] - center[d]) ** 2
                dist = np.sqrt(dist)
            dists.append(dist)
        
        n_assignments = 10
        
        idx = np.argpartition(dists, n_assignments)
        print(np.array(dists)[idx])
        # print(idx)
        # indexes = vocabulary[idx[:n_assignments]]
        indexes = idx[:n_assignments]
        
        # print(indexes)
        # print(indexes[0])
        
        inverted_Table = dict()
        search_features = list()
        included_names = list()
        included_images = list()
        for idx in indexes:
            key = vocabulary[idx]
            list_features = features_meta[key]
            if len(list_features) > 0:
                list_features = list_features
            for j in list_features:
                search_features.append(j)
               
        print(time.time() - st)
        sf = database_features[search_features]

        # calculate cosine distance
        if bib == "pytorch":
            q = torch.from_numpy(q)
            sf = torch.from_numpy(sf)
            if gpu:
                q = query_features.cuda()
                sf = sf.cuda()
            scores_q = q @ sf.t()
        else:
            scores_q = q @ np.transpose(sf)
            
        print(np.argmax(scores_q))
        r = np.argmax(scores_q)
        # print(included_names[r], scores_q[r], included_images[r])

        # associate confidence score with the label of the dataset and sort based on the confidence
        scores_q = list(zip(scores_q, database_data['name'][search_features], database_data['image'][search_features]))
        # scores_q = list(zip(scores_q, included_names, included_images))
        scores_q = sorted(scores_q, key=lambda x: x[0], reverse=True)

        persons_scores.append((query_bbs[img], generate_rank(scores_q, k_rank)))
        all_scores.append(scores_q)

    return persons_scores, all_scores


def process_image(operation, model_name, model_path, image_query, query_label,
                  preprocessing_method, crop_size, feature_file, gpu):
    # process unique image
    dataset = ImageDataLoader(image_query, preprocessing_method,
                              crop_size, operation == 'extract_features')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    features = None
    # load current features
    if feature_file is not None and os.path.isfile(feature_file):
        features = scipy.io.loadmat(feature_file)

    # process specific image
    if operation == 'extract_features':
        assert query_label is not None, 'To process and add a new image to ' \
                                             'the database, the flag --query_label is required.'
        # extract the features
        feature = extract_features_from_image(load_net(model_name, model_path, gpu),
                                              dataloader, query_label, gpu=gpu)
        if feature is not None:
            if features is None:
                # if there is not features, use the recently extracted one as the current features
                features = feature
            else:
                # otherwise, concatenate the existing features with the recently extracted ones
                features['feature'] = np.concatenate((features['feature'], feature['feature']), 0)
                features['name'] = np.concatenate((features['name'], feature['name']), 0)
                features['image'] = np.concatenate((features['image'], feature['image']), 0)
                features['bbs'] = np.concatenate((features['bbs'], feature['bbs']), 0)
                # print(features['feature'].shape, features['name'].shape,
                # features['image'].shape, features['bbs'].shape)
            # save the current version of the features
            scipy.io.savemat(feature_file, features)
    elif operation == 'generate_rank':
        assert features is not None, 'To generate rank, use the flag --feature_file' \
                                     ' to load previously extracted features.'
        # extract features for the query image
        query_features = extract_features_from_image(load_net(model_name, model_path, gpu),
                                                     dataloader, None, gpu=gpu)

        if query_features is not None:
            # generate ranking
            ranking, _ = generate_ranking_for_image(features, query_features)
            print(ranking)
        else:
            print("No face detected in this image.")
    else:
        raise NotImplementedError("Operation " + operation + " not implemented")
