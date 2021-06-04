import os
import scipy.io

import torch

from utils import *

from dataloaders.image_dataloader import ImageDataLoader

from networks.load_network import load_net
from dataset_processor import process_dataset


def extract_features_from_image(model_name, dataloader, query_label, gpu):
    """
    Function to extract features for ONE specific image.

    :param model_name: string with the name of the model used.
    :param dataloader: dataloader used to load the images.
    :param query_label: the image query label, i.e., the ID.
    :param gpu: boolean to allow use gpu.
    :return: a dict with the features, id, and cropped image of the current query.
    """

    net = load_net(model_name, gpu)
    if gpu:
        # net to GPU
        net = net.cuda()
    # set network to evaluation, and not training
    net.eval()

    feature = None
    img = None
    bbs = None

    # forward
    for imgs, cls, imgl, bb in dataloader:
        if not imgs:
            # no face detected
            return None
        if gpu:
            for i in range(len(imgs)):
                imgs[i] = imgs[i].cuda()

        res = [net(d.view(-1, d.shape[2], d.shape[3], d.shape[4])).data.cpu().numpy() for d in imgs]
        feature = np.concatenate((res[0], res[1]), 1)
        img = imgs
        bbs = bb

    # plot_bbs(bbs[0].cpu().numpy())

    result = {'feature': feature, 'name': [query_label], 'image': img[0][0].cpu().numpy(), 'bbs': bbs[0].cpu().numpy(), 'cropped_image': imgl}

    return result


def generate_rank(scores):
    i = j = 0
    persons_scores = []
    unique_persons = []

    # return the top 10 most similar persons to the query image.
    # if there are less than 10 persons in the dataset return rank
    # with the maximum possible number of persons.

    while i < 10 and j < len(scores):
        # if unique_persons is not empty
        if unique_persons:
            # if the i person in the scores list is note in unique_persons
            if scores[j][1] not in unique_persons:
                unique_persons.append(scores[j][1])
                # append tuple with person id, score and image path
                persons_scores.append(("Name: ", scores[j][1].strip(), "Confidence: ", scores[j][0], "Image: ", scores[j][2].strip()))

                i += 1
        else:
            unique_persons.append(scores[j][1])
            persons_scores.append(("Name: ", scores[j][1].strip(), "Confidence: ", scores[j][0], "Image: ", scores[j][2].strip()))
            i += 1
        j += 1
    return persons_scores


def generate_ranking_for_image(database_data, query_data, bib="numpy", gpu=False):
    """
    Make a specific query and calculate the average precision.

    :param database_data: features of the whole dataset.
    :param query_data: features of the image query.
    :param bib: library used to calculate the cosine distance.
    :param gpu: boolean to allow use gpu.
    :return: top 10 list (with ID and confidence) of most similar images.
    """

    # normalize features
    database_features = database_data['feature']
    query_features = query_data['feature']
    query_bbs = query_data['bbs']

    # this is just to store the number of faces detected in the query image
    num_features_query = query_features.shape[0]

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
    for i, q in enumerate(query_features):
        # calculate cosine distance
        if bib == "pytorch":
            q = torch.from_numpy(q)
            database_features = torch.from_numpy(database_features)
            if gpu:
                q = query_features.cuda()
                database_features = database_features.cuda()
            scores_q = q @ database_features.t()
        else:
            scores_q = q @ np.transpose(database_features)

        # associate confidence score with the label of the dataset and sort based on the confidence
        scores_q = list(zip(scores_q, database_data['name'], database_data['image']))
        scores_q = sorted(scores_q, key=lambda x: x[0], reverse=True)

        persons_scores.append((query_bbs[i]))
        persons_scores.append((generate_rank(scores_q)))

    return persons_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')

    # general options
    parser.add_argument('--operation', type=str, required=True,
                        help='Operation [Options: extract_features | generate_rank |'
                             ' extract_generate_rank (for dataset only)]')
    parser.add_argument('--feature_file', type=str, default=None,
                        help='File path to save/load the extracted features (.mat file).')
    parser.add_argument('--result_sample_path', type=str, default=None,
                        help='Path to save image samples of the results.')
    parser.add_argument('--gpu', type=str2bool, default=True,
                        help='Boolean to indicate the use of GPU.')

    # image processing options
    parser.add_argument('--image_query', type=str, default=None,
                        help='Path or string of query image. If set, only this image is processed.')
    parser.add_argument('--query_label', type=str, default=None,
                        help='Query label (or ID of the person). '
                             '**REQUIRED** if flag --operation is extract_features and --image_query is used.')

    # dataset options
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name. If set, the whole dataset is processed [Options: LFW | LFW_UPDATE | YALEB]')
    parser.add_argument('--specific_dataset_folder_name', type=str, default=None,
                        help='Specific dataset folder. **REQUIRED** if flag --dataset is used.')
    parser.add_argument('--img_extension', type=str, default='jpg',
                        help='Extension of the images [Options: jpg | pgm | txt (in base64 format)]')
    parser.add_argument('--preprocessing_method', type=str, default=None,
                        help='Preprocessing method [Options: None | mtcnn | sphereface | openface]')

    # model options
    parser.add_argument('--model_name', type=str, required=True, default='mobilefacenet',
                        help='Model to test [Options: mobilefacenet | mobiface | sphereface | '
                             'openface | facenet | shufflefacenet]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size to extract features')

    args = parser.parse_args()
    print(args)

    # selecting the size of the crop based on the network
    if args.model_name == 'mobilefacenet' or args.model_name == 'sphereface':
        crop_size = (96, 112)
    elif args.model_name == 'mobiface' or args.model_name == 'shufflefacenet':
        crop_size = (112, 112)
    elif args.model_name == 'openface':
        crop_size = (96, 96)
    elif args.model_name == 'facenet':
        crop_size = (160, 160)
    else:
        raise NotImplementedError("Model " + args.model_name + " not implemented")

    if args.dataset is not None:
        # process whole dataset
        assert args.specific_dataset_folder_name is not None, 'To process a dataset, ' \
                                                              'the flag --specific_dataset_folder_name is required.'
        process_dataset(args.operation, args.model_name, args.batch_size,
                        args.dataset, args.specific_dataset_folder_name,
                        args.img_extension, args.preprocessing_method, crop_size,
                        args.result_sample_path, args.feature_file)
    elif args.image_query is not None:
        # process unique image
        dataset = ImageDataLoader(args.image_query, args.preprocessing_method,
                                  crop_size, args.operation == 'extract_features')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

        features = None
        # load current features
        if args.feature_file is not None and os.path.isfile(args.feature_file):
            features = scipy.io.loadmat(args.feature_file)

        # process specific image
        if args.operation == 'extract_features':
            assert args.query_label is not None, 'To process and add a new image to ' \
                                                 'the database, the flag --query_label is required.'
            # extract the features
            feature = extract_features_from_image(args.model_name, dataloader, args.query_label, gpu=args.gpu)
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
                scipy.io.savemat(args.feature_file, features)
        elif args.operation == 'generate_rank':
            assert features is not None, 'To generate rank, use the flag --feature_file' \
                                         ' to load previously extracted faatures.'
            # extract features for the query image
            query_features = extract_features_from_image(args.model_name, dataloader, None, gpu=args.gpu)
            if query_features is not None:
                # generate ranking
                ranking = generate_ranking_for_image(features, query_features)
                print(ranking)
            else:
                print("No face detected in this image.")
        else:
            raise NotImplementedError("Operation " + args.operation + " not implemented")
    else:
        raise NotImplementedError("Dataset OR Image Path flags must be set. Both flags are None.")
