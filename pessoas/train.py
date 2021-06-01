import os
import numpy as np
import logging
import time

import torch
from torch import nn
import torch.optim as optim

from dataloaders.generic_dataloader import GenericDataLoader
from networks.mobilefacenet import MobileFacenet, ArcMarginProduct


def train(dataset_path, save_dir):
    """
    Train a new model.

    :param dataset_path: Path to the dataset used to train.
    :param save_dir: Path to the dir used to save the trained model.
    """

    # create dataset
    train_dataset = GenericDataLoader(dataset_path, preprocessing_method='sphereface', crop_size=(96, 112))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8,
                                                   shuffle=True, num_workers=4, drop_last=False)

    net = MobileFacenet()
    arc_margin = ArcMarginProduct(128, train_dataset.num_classes)
    net = net.cuda()
    arc_margin = arc_margin.cuda()
    criterion = torch.nn.CrossEntropyLoss()

    # if RESUME:
    #     ckpt = torch.load(RESUME)
    #     net.load_state_dict(ckpt['net_state_dict'])
    #     start_epoch = ckpt['epoch'] + 1

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

    best_acc = 0.0
    best_epoch = 0
    TOTAL_EPOCH = 71
    SAVE_FREQ = 10
    for epoch in range(1, TOTAL_EPOCH):
        # train model
        logging.info('Train Epoch: {}/{} ...'.format(epoch, TOTAL_EPOCH))
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
        # if epoch % TEST_FREQ == 0:
        #     net.eval()
        #     featureLs = None
        #     featureRs = None
        #     logging.info('Test Epoch: {} ...'.format(epoch))
        #     for data in testloader:
        #         for i in range(len(data)):
        #             data[i] = data[i].cuda()
        #         res = [net(d).data.cpu().numpy() for d in data]
        #         featureL = np.concatenate((res[0], res[1]), 1)
        #         featureR = np.concatenate((res[2], res[3]), 1)
        #         if featureLs is None:
        #             featureLs = featureL
        #         else:
        #             featureLs = np.concatenate((featureLs, featureL), 0)
        #         if featureRs is None:
        #             featureRs = featureR
        #         else:
        #             featureRs = np.concatenate((featureRs, featureR), 0)
        #
        #     result = {'fl': featureLs, 'fr': featureRs, 'fold': folds, 'flag': flags}
        #     # save tmp_result
        #     scipy.io.savemat('./result/tmp_result.mat', result)
        #     accs = evaluation_10_fold('./result/tmp_result.mat')
        #     logging.info('    ave: {:.4f}'.format(np.mean(accs) * 100))

        # save model
        if epoch % SAVE_FREQ == 0:
            logging.info('Saving checkpoint: {}'.format(epoch))
            net_state_dict = net.state_dict()
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save({'epoch': epoch, 'net_state_dict': net_state_dict}, os.path.join(save_dir, '%03d.ckpt' % epoch))
    logging.info('finishing training')


if __name__ == '__main__':
    """
    Testing purposes
    """
    train('/home/kno/mp_e04/pessoas/datasets/CASIA-WebFace/', '/home/kno/mp_e04/pessoas/')
