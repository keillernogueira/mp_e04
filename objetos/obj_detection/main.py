import argparse
import os, random
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import dataloader
import torch.nn as nn
import torch
import yaml
import numpy as np
from network_factory_v2 import model_factory, train, final_eval


# Fixed image size


def main():
    parser = argparse.ArgumentParser(description='Object Detection Training/Test')
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset.yaml file path')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to folder where models and stats will be saved')
    parser.add_argument('--size', type=int, default=480,
                        help='Image Size')
    parser.add_argument('--batch', type=int, required=True,
                        help='Batch Size')
    parser.add_argument('--epochs', type=int, required=True,
                        help='Number of epochs')
    parser.add_argument('--early_stop', type=int, required=True,
                        help='Number of epochs to activate early stop.')

    parser.add_argument('--model', type=str, required=False, default='vgg',
                        help='Choose network model. [faster|faster-mobile|retina|ssd]')

    parser.add_argument('--optim', type=str, required=False, default='adam',
                        help='Optimizer used [adam|sgd].')
    parser.add_argument('--lr', type=float, required=False, default=0.001,
                        help='Learning Rate.')
    parser.add_argument('--momentum', type=float, required=False, default=0.9,
                        help='Learning Rate.')
    parser.add_argument('--wd', type=float, required=False, default=5e-5,
                        help='Weight Decay.')

    parser.add_argument('--inference', action='store_true',
                        help='Only test the model on the images in the test folder of the dataset.')

    parser.add_argument('--is_val', action='store_true',
                        help='Only for hyperparameter validation. Uses a smaller subset of the train set to train/test the model.')

    parser.add_argument('--plot', action='store_true', help='Plot metric graphics.')

    parser.add_argument('--save_best', action='store_true', help='Save the best model in relation do mAP.')

    parser.add_argument('--weights', type=str, default='default', help='Initial weights path')

    parser.add_argument('--normalization', type=str, default='imagenet', help='Normalization values')

    parser.add_argument('--quad', action='store_true', help='Uses a four image mosaic for training.')

    args = parser.parse_args()

    dataset_dict = yaml.safe_load(open(args.dataset, 'r'))
    total_classes = dataset_dict['nc']

    img_size = args.size
    out_dir = args.output_path
    batch_size = args.batch
    epochs = args.epochs
    early_stop = args.early_stop
    total_classes = dataset_dict['nc'] + 1 # +1 to include the __background__ class used in the pytorch models 
    infer = args.inference
    is_val = args.is_val
    weights = args.weights

    infer = args.inference
    
    optim_type = args.optim
    lr = args.lr
    momentum = args.momentum
    wd = args.wd

    plot = args.plot
    save_best = args.save_best
    norm = args.normalization
    quad = args.quad

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    save_dir = os.path.join(out_dir, model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(os.path.join(save_dir, 'weights')):
        os.makedirs(os.path.join(save_dir, 'weights'))

    print('.......Creating model.......')
    print('total classes: ', total_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model_factory(model, total_classes, img_size=(img_size, img_size))
    if weights != 'default' and os.path.exists(weights):
        net.load_state_dict(torch.load(weights))

    print(net)
    net = net.to(device)
    print('......Model created.......')

    print('......Creating dataloader......')
    dataloaders_dict = dataloader.create_dataloader(dataset_dict['root'], img_size, batch_size,
                                                    num_classes=total_classes, is_val=is_val,
                                                    normalization=norm, quad=quad)
    print('......Dataloader created......')

    params_to_update = net.parameters()
    # print("Params to learn:")
    # print (params_to_update)

    if optim_type == 'sgd':
        optimizer = optim.SGD(params_to_update, lr=lr, momentum=0.9, weight_decay=wd)
    elif optim_type == 'adam':
        optimizer = optim.Adam(params_to_update, lr=lr, betas=(momentum, 0.99), weight_decay=wd)
    else:
        print("Optimizer Not Implemented.")
        exit()

    tensor_board = SummaryWriter(log_dir=save_dir)
    if not infer:
        final_model, map_history = train(net, dataloaders_dict, optimizer, epochs, early_stop, tensor_board, save_dir,
                                         plot=plot, save_best=save_best, quad=quad)
    else:
        final_model = net

    final_stats_file = os.path.join(save_dir, 'finalstats.txt')

    final_eval(final_model, dataloaders_dict, final_stats_file, save_dir, plot=plot)


if __name__ == '__main__':
    main()
