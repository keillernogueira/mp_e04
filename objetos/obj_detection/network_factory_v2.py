import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models
from torchvision.models.detection import _utils as det_utils
import statistics
import time
import copy
from tqdm import tqdm
from collections import OrderedDict
from pathlib import Path

from utils import box_iou_v2, ap_per_class


def model_factory(model_name, num_classes, feature_extract=False, use_pretrained=True, img_size=(480, 480)):
    model_ft = None
    if model_name == "faster":
        """ Faster R-CNN ResNet-50 FPN
        """
        model_ft = models.detection.fasterrcnn_resnet50_fpn(pretrained=use_pretrained)
        # get number of input features for the classifier
        in_features = model_ft.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model_ft.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    elif model_name == "faster-mobile":
        """ Faster R-CNN MobileNet-v3 FPN
        """
        model_ft = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=use_pretrained)
        # get number of input features for the classifier
        in_features = model_ft.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model_ft.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    elif model_name == "retina":
        """ RetinaNet ResNet-50 FPN
        """
        model_ft = models.detection.retinanet_resnet50_fpn(pretrained=use_pretrained)
        # get number of input features for the classifier
        in_channels = model_ft.head.classification_head.cls_logits.in_channels
        num_achors = model_ft.head.classification_head.num_anchors

        # replace the pre-trained head with a new one
        model_ft.head.classification_head.cls_logits = nn.Conv2d(in_channels, num_achors * num_classes, kernel_size=3,
                                                                 stride=1, padding=1)
        model_ft.head.classification_head.num_classes = num_classes

    elif model_name == "ssd":
        """ SSD300 VGG16
        """
        model_ft = models.detection.ssd300_vgg16(pretrained=use_pretrained)
        # get number of input features for the classifier
        if hasattr(model_ft.backbone, 'out_channels'):
            out_channels = model_ft.backbone.out_channels
        else:
            out_channels = det_utils.retrieve_out_channels(model_ft.backbone, img_size)

        num_anchors = model_ft.anchor_generator.num_anchors_per_location()
        # replace the pre-trained head with a new one

        model_ft.head.classification_head = models.detection.ssd.SSDClassificationHead(out_channels, num_anchors,
                                                                                       num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    model_ft.num_classes = num_classes
    set_parameter_requires_grad(model_ft, feature_extract)

    return model_ft


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def train(model, dataloaders, optimizer, num_epochs, epochs_early_stop, tensor_board, save_dir, plot=False,
          save_best=True):
    # Directories
    save_dir = Path(save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    counter_early_stop_epochs = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    since = time.time()
    # counter = 0
    val_map_history = []
    total_time = 0

    num_classes = dataloaders['train'].dataset.num_classes
    class_names = dataloaders['train'].dataset.class_names
    class_names = class_names if len(class_names) == num_classes else [str(i) for i in range(num_classes)]

    iou_values = np.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    n_ious = iou_values.size

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = 9999999.99
    best_map = -1.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        predictions = []
        labels_list = []

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
                stats, ap, ap_class = [], [], []
                p, r, f1, mp, mr, mAP50, mAP = 0., 0., 0., 0., 0., 0., 0.  # Precision, Recall, F1, Mean P, Mean R, mAP@0.5, mAP@[0.5:0.95]

            running_loss = 0.0

            # Iterate over data.
            for inputs, targets in tqdm(dataloaders[phase]):
                # inputs = inputs.to(device)
                inputs = list(inp.to(device) for inp in inputs)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                if phase == 'train':
                    time1 = time.time()

                    loss_dict = model(inputs, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                    total_time += (time.time() - time1)

                if phase == 'test':
                    outputs = model(inputs)

                    outputs = [{k: v.detach().cpu().numpy() for k, v in t.items()} for t in outputs]
                    targets = [{k: v.detach().cpu().numpy() for k, v in t.items()} for t in targets]

                    # Assuming all imgs have at least one detection target
                    # For each img, there is an out dict with the predictions
                    for out, tgt in zip(outputs, targets):
                        # Zero detections for the img
                        if len(out['scores']) == 0:
                            stats.append((np.zeros((0, n_ious), dtype=bool), np.zeros(0), np.zeros(0),
                                          tgt['labels'].tolist()))
                            continue

                        # Correct detected targets, initially assume all targets are missed for all iou threshold values
                        correct = np.zeros((len(out['scores']), n_ious), dtype=bool)
                        detected = []  # Detected Target Indices

                        # Per target class
                        for cls in np.unique(tgt['labels']):
                            target_ids = np.flatnonzero(cls == tgt['labels'])
                            pred_ids = np.flatnonzero(cls == out['labels'])

                            # Search for detections
                            if pred_ids.shape[0]:
                                # Prediction to target ious
                                inter = box_iou_v2(out['boxes'][pred_ids], tgt['boxes'][target_ids])
                                ious = np.amax(inter, axis=1)
                                max_iou_ids = np.argmax(inter, axis=1)
                                # ious, max_iou_ids = box_iou_v2(out['boxes'][pred_ids], tgt['boxes'][target_ids]).max(1)  # best ious, indices

                                # Append detections
                                detected_set = set()
                                for idx in np.flatnonzero(ious > iou_values[0]):
                                    detected_tgt_id = target_ids[max_iou_ids[idx]]  # detected target
                                    if detected_tgt_id.item() not in detected_set:
                                        detected_set.add(detected_tgt_id.item())
                                        detected.append(detected_tgt_id)

                                        correct[pred_ids[idx]] = ious[idx] > iou_values  # iou_thres is 1xn

                                        if len(detected) == len(tgt['labels']):  # all targets already located in image
                                            break

                        # Append statistics (correct, conf, pred_cls, gt_cls)
                        stats.append((correct, out['scores'], out['labels'], tgt['labels']))

                    del outputs

                running_loss += losses.item() * len(inputs)

                del inputs
                del targets

            # Compute statistics
            if phase == 'test':
                stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
                if len(stats) and stats[0].any():
                    p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plot, save_dir=save_dir, names=class_names)
                    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
                    mp, mr, mAP50, mAP = p.mean(), r.mean(), ap50.mean(), ap.mean()

                pf = '%20s' + '%12.3g' * 4  # print format
                print(pf % ('all', mp, mr, mAP50, mAP))

                # Write metrics to file
                with open(results_file, 'a') as f:
                    f.write('%g/%g' % (epoch, num_epochs - 1) + '%10.4g' * 5 % (mp, mr, mAP50, mAP, running_loss / len(
                        dataloaders['test'].dataset)) + '\n')  # append metrics, val_loss

            # Epoch loss/ metric registry in tensorboard
            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if phase == 'train':
                tensor_board.add_scalar('Loss/train', epoch_loss, epoch)
            else:
                tensor_board.add_scalar('mAP/val', mAP, epoch)

            # Early stopping and validation loss history
            save_model = False
            if phase == 'test':
                counter_early_stop_epochs += 1
                val_map_history.append(mAP)
            if phase == 'test' and mAP > best_map:
                counter_early_stop_epochs = 0
                best_map = mAP
                save_model = True

            # Saving models
            if phase == 'train':
                torch.save(copy.deepcopy(model.state_dict()), last)
            if phase == 'test':
                if save_model and save_best:
                    torch.save(copy.deepcopy(model.state_dict()), best)

        print('Epoch ' + str(epoch) + ' - Time Spent ' + str(total_time))
        if counter_early_stop_epochs >= epochs_early_stop:
            print('Stopping training because mAP score did not improve in ' +
                  str(epochs_early_stop) + ' consecutive epochs.')
            break

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best mAP Loss: {:4f}'.format(best_map))

    # load best model weights
    if save_best and os.path.exists(best):
        model.load_state_dict(torch.load(best))
    return model, val_map_history


def final_eval(model, dataloaders, stats_file, save_dir, plot=False):
    print("Begining final eval.")
    save_dir = Path(save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    num_classes = dataloaders['test'].dataset.num_classes
    class_names = dataloaders['test'].dataset.class_names
    class_names = class_names if len(class_names) == num_classes else [str(i) for i in range(num_classes)]

    iou_values = np.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    n_ious = iou_values.size

    model.eval()
    stats = []
    for inputs, targets in tqdm(dataloaders['test']):
        # inputs = inputs.to(device)
        inputs = list(inp.to(device) for inp in inputs)

        outputs = model(inputs)

        outputs = [{k: v.detach().cpu().numpy() for k, v in t.items()} for t in outputs]
        targets = [{k: v.detach().cpu().numpy() for k, v in t.items()} for t in targets]

        # Assuming all imgs have at least one detection target
        # For each img, there is an out dict with the predictions
        for out, tgt in zip(outputs, targets):
            # Zero detections for the img
            if len(out['scores']) == 0:
                stats.append((np.zeros((0, n_ious), dtype=bool), np.zeros(0), np.zeros(0),
                                tgt['labels'].tolist()))
                continue

            # Correct detected targets, initially assume all targets are missed for all iou threshold values
            correct = np.zeros((len(out['scores']), n_ious), dtype=bool)
            detected = []  # Detected Target Indices

            # Per target class
            for cls in np.unique(tgt['labels']):
                target_ids = np.flatnonzero(cls == tgt['labels'])
                pred_ids = np.flatnonzero(cls == out['labels'])

                # Search for detections
                if pred_ids.shape[0]:
                    # Prediction to target ious
                    inter = box_iou_v2(out['boxes'][pred_ids], tgt['boxes'][target_ids])
                    ious = np.amax(inter, axis=1)
                    max_iou_ids = np.argmax(inter, axis=1)
                    # ious, max_iou_ids = box_iou_v2(out['boxes'][pred_ids], tgt['boxes'][target_ids]).max(1)  # best ious, indices

                    # Append detections
                    detected_set = set()
                    for idx in np.flatnonzero(ious > iou_values[0]):
                        detected_tgt_id = target_ids[max_iou_ids[idx]]  # detected target
                        if detected_tgt_id.item() not in detected_set:
                            detected_set.add(detected_tgt_id.item())
                            detected.append(detected_tgt_id)

                            correct[pred_ids[idx]] = ious[idx] > iou_values  # iou_thres is 1xn

                            if len(detected) == len(tgt['labels']):  # all targets already located in image
                                break

            # Append statistics (correct, conf, pred_cls, gt_cls)
            stats.append((correct, out['scores'], out['labels'], tgt['labels']))

    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plot, save_dir=save_dir, names=class_names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, mAP50, mAP = p.mean(), r.mean(), ap50.mean(), ap.mean()
    print('%12.3g' * 4 % (mp, mr, mAP50, mAP))

    # Write metrics to file
    with open(stats_file, 'a') as f:
        f.write('Precision, Recall, mAP@50, mAP@[50, 95]\n')
        f.write('%10.4g' * 5 % (mp, mr, mAP50, mAP))  # append metrics
