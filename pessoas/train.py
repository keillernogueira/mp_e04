import os
import numpy as np
import logging
import time
import urllib.request
import tarfile
import argparse

import torch
from torch import nn
import torch.optim as optim

from dataloaders.LFW_dataloader import LFW
from dataloaders.generic_dataloader import GenericDataLoader
from networks.mobilefacenet import MobileFacenet, ArcMarginProduct
from dataset_processor import extract_features, evaluate_dataset


def train(dataset_path, save_dir, resume_path=None, num_epoch=71):
    """
    Train a model.

    :param dataset_path: Path to the dataset used to train.
    :param save_dir: Path to the dir used to save the trained model.
    :param resume_path: Path to a previously trained model.
    :param num_epoch: number of epochs to train
    """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    # create dataset
    train_dataset = GenericDataLoader(dataset_path, preprocessing_method='sphereface', crop_size=(96, 112))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8,
                                                   shuffle=True, num_workers=4, drop_last=False)

    # validation dataset
    lfw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'LFW')
    if not os.path.exists(lfw_path):  # if folder does not exist
        os.mkdir(lfw_path)  # create folder
        # download data
        urllib.request.urlretrieve('http://vis-www.cs.umass.edu/lfw/lfw.tgz', os.path.join(lfw_path, 'lfw.tgz'))
        urllib.request.urlretrieve('http://vis-www.cs.umass.edu/lfw/people.txt', os.path.join(lfw_path, 'people.txt'))
        # unzip
        tar = tarfile.open(os.path.join(lfw_path, 'lfw.tgz'))
        tar.extractall(lfw_path)
        tar.close()

    validate_dataset = LFW(lfw_path, specific_folder='lfw', img_extension='jpg',
                           preprocessing_method='sphereface', crop_size=(96, 112))
    validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=8, shuffle=False,
                                                      num_workers=2, drop_last=False)

    # TODO carregar modelo usando load_net(model_name, gpu)
    net = MobileFacenet()
    arc_margin = ArcMarginProduct(128, train_dataset.num_classes)
    net = net.cuda()
    arc_margin = arc_margin.cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # define optimizers
    ignored_params = list(map(id, net.linear1.parameters()))
    ignored_params += list(map(id, arc_margin.weight))
    prelu_params = []
    for m in net.modules():
        if isinstance(m, nn.PReLU):
            ignored_params += list(map(id, m.parameters()))
            prelu_params += m.parameters()
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer_ft = optim.SGD([
        {'params': base_params, 'weight_decay': 4e-5},
        {'params': net.linear1.parameters(), 'weight_decay': 4e-4},
        {'params': arc_margin.weight, 'weight_decay': 4e-4},
        {'params': prelu_params, 'weight_decay': 0.0}
    ], lr=0.1, momentum=0.9, nesterov=True)

    exp_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_ft, milestones=[36, 52, 58], gamma=0.1)

    SAVE_FREQ = 10
    TEST_FREQ = 3
    start_epoch = 1

    if resume_path:
        ckpt = torch.load(resume_path)
        net.load_state_dict(ckpt['net_state_dict'])
        start_epoch = ckpt['epoch'] + 1

    for epoch in range(start_epoch, num_epoch):
        # train model
        logging.info('Train Epoch: {}/{} ...'.format(epoch, num_epoch))
        net.train()

        train_total_loss = 0.0
        total = 0
        since = time.time()
        for data in train_dataloader:
            img, label = data[0].cuda(), data[1].cuda()
            batch_size = img.size(0)
            optimizer_ft.zero_grad()

            raw_logits = net(img)
            output = arc_margin(raw_logits, label)
            total_loss = criterion(output, label)

            total_loss.backward()
            optimizer_ft.step()

            train_total_loss += total_loss.item() * batch_size
            total += batch_size

        train_total_loss = train_total_loss / total
        exp_lr_scheduler.step()

        time_elapsed = time.time() - since
        loss_msg = 'Loss: {:.4f} time: {:.0f}m {:.0f}s'.format(train_total_loss, time_elapsed // 60, time_elapsed % 60)
        logging.info(loss_msg)

        # test model on lfw
        if epoch % TEST_FREQ == 0:
            features = extract_features(validate_dataloader, model=net, gpu=True, save_img_results=True)
            evaluate_dataset(features)

        # save model
        if epoch % SAVE_FREQ == 0:
            logging.info('Saving checkpoint: {}'.format(epoch))
            net_state_dict = net.state_dict()
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save({'epoch': epoch, 'net_state_dict': net_state_dict}, os.path.join(save_dir, '%03d.ckpt' % epoch))
    logging.info('finishing training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to to save outcomes (such as trained models) of the algorithm')

    parser.add_argument('--resume_path', type=str, required=False, default=None,
                        help='Path to to save outcomes (such as trained models) of the algorithm')
    parser.add_argument('--num_epoch', type=int, required=False, default=71,
                        help='Path to to save outcomes (such as trained models) of the algorithm')
    args = parser.parse_args()
    print(args)

    train(args.dataset_path, args.save_dir, args.resume_path, args.num_epoch)

    # train('/home/kno/mp_e04/pessoas/datasets/CASIA-WebFace/', '/home/kno/mp_e04/pessoas/output/')
