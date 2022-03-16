# original author: signatrix
# adapted from https://github.com/signatrix/efficientdet/blob/master/train.py
# modified by Zylo117

import argparse
import datetime
import os
import traceback

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm

from backbone import EfficientDetBackbone
from efficientdet.dataset_ import ListDataset, collater
from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string, box_iou, ap_per_class, postprocess
from efficientdet.utils import BBoxTransform, ClipBoxes


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('--hyp_path', type=str, default='dataset.yaml', help='path to file that contains parameters')
    parser.add_argument('--data_name', type=str, default='coco', help='dataset name for saving folder')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=1, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=boolean_string, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--schlr', type=str, default="plateau")
    parser.add_argument('--gama', type=int, default=0.1)
    parser.add_argument('--step_size', type=int, default=30)
    parser.add_argument('--milestones', type=list, default=[30,80])
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='logs/')
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='whether visualize the predicted boxes of training, '
                             'the output images will be in test/')
    parser.add_argument('--plot', action='store_true', help='Plot metric graphics.')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_imgs', type=int, default=20720)
    parser.add_argument('--anchors_ratios', type=str, default='[(0.7, 1.4), (1.0, 1.0), (1.5, 0.7)]')
    parser.add_argument('--anchors_scales', type=str, default='[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]')
    parser.add_argument('--normalization', type=str, default='imagenet', help='Normalization values')

    parser.add_argument('--quad', action='store_true', help='Uses a four image mosaic for training.')

    args = parser.parse_args()
    return args


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None, test = False):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        if test:
            return _, regression, classification, anchors
        return cls_loss, reg_loss


def train(opt):
    params = Params(opt.hyp_path)

    if opt.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)
    
    opt.saved_path = opt.saved_path + f'/{opt.data_name}/'
    opt.log_path = opt.log_path + f'/{opt.data_name}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)
    
    norm = opt.normalization
    quad = opt.quad
    
    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': False,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': True,
                  'drop_last': False,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    training_set = ListDataset(root=opt.data_path, img_size=input_sizes[opt.compound_coef],  mode = 'train', num_classes = params.nc ,class_names = params.names,normvalues=norm, quad=quad)
    _ = '''training_set = ListDataset(root=opt.data_path, mode = 'train', num_classes = params.nc ,class_names = params.names,
                               transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                             Augmenter(),
                                                             Resizer(input_sizes[opt.compound_coef])]))'''
    training_generator = DataLoader(training_set, **training_params)

    val_set = ListDataset(root=opt.data_path, mode="test", num_classes = params.nc ,class_names = params.names,
                         img_size=input_sizes[opt.compound_coef], normvalues=norm, quad=quad)
    _ = '''val_set = ListDataset(root=opt.data_path, mode="test", num_classes = params.nc ,class_names = params.names,
                         img_size=input_sizes[opt.compound_coef],
                         transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                       #Augmenter(),
                                                       Resizer(input_sizes[opt.compound_coef])]))'''
    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_classes=len(params.names), compound_coef=opt.compound_coef,load_weights=True,
                                 ratios=eval(opt.anchors_ratios), scales=eval(opt.anchors_scales))

    # load last weights
    if opt.load_weights is not None:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if opt.num_gpus > 1 and opt.batch_size // opt.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)

    if opt.num_gpus > 0:
        model = model.cuda()
        if opt.num_gpus > 1:
            model = CustomDataParallel(model, opt.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True, weight_decay = 4e-5)
    
    if opt.schlr == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    elif opt.schlr == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gama)
    elif opt.schlr == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gama)
    elif opt.schlr == 'lambda-cosine':
        calc_lr = lambda epoch: epoch/(opt.num_imgs//opt.num_epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = calc_lr, last_epoch=-1)        


    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iou_values = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    n_ious = iou_values.numel()
    cpu_device = torch.device("cpu")
    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']
                    annot = data['annot']

                    if opt.num_gpus == 1:
                        # if only one gpu, just send it to cuda
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    cls_loss, reg_loss = model(imgs, annot, obj_list=params.names)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1
                    if epoch == 0 and opt.schlr == 'lambda-cosine':
                        scheduler.step()
                        
                    if step % opt.save_interval == 0 and step > 0:
                        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            if epoch == 0:
                if opt.schlr == 'lambda-cosine':
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_epochs-1, eta_min=0)
                elif opt.schlr == 'plateau':
                    scheduler.step(np.mean(epoch_loss))
                else:
                    scheduler.step()
            else:
                if opt.schlr == 'plateau':
                    scheduler.step(np.mean(epoch_loss))
                else:
                    scheduler.step()

            with open(os.path.join(opt.saved_path, "lr.txt"), 'a+') as f:
                f.write('Epoch {}, lr {}\n'.format(epoch, scheduler.get_last_lr()))

            if epoch % opt.val_interval == 0:
                model.requires_grad_(False)
                model.eval()
                stats, ap, ap_class = [], [], []
                p, r, f1, mp, mr, mAP50, mAP = 0., 0., 0., 0., 0., 0., 0. # Precision, Recall, F1, Mean P, Mean R, mAP@0.5, mAP@[0.5:0.95]

                loss_regression_ls = []
                loss_classification_ls = []
                p_bar = tqdm(val_generator, unit="batch")

                for iter, targets in enumerate(p_bar):
                    inputs = targets["img"]
                    inputs = inputs.cuda()
                    annots = targets['annot']
                    
                    targets = [{'boxes': (annot[:,:4]).to(device), 'labels':(annot[:,4]).to(device)} for annot in annots]
                    
                    regressBoxes = BBoxTransform()
                    clipBoxes = ClipBoxes()
                    x = inputs
                    x = x.to(device)
                    x = x.float()

                    features, regression, classification, anchors = model(x,annots, test=True)

                    outputs = postprocess(x,
                                        anchors, regression, classification,
                                        regressBoxes, clipBoxes,
                                        0.05, 0.5)


                    outputs = [{k: torch.from_numpy(v).to(device) for k, v in out.items()} for out in outputs]
                    for out,target in zip(outputs,targets):
                        target['labels'] = target['labels'][target['labels'] != -1]
                        target['boxes'] = target['boxes'][:len(target['labels']),:]
                                                
                        # Zero detections for the img
                        if len(out['scores']) == 0:
                            stats.append((torch.zeros(0, n_ious, dtype=torch.bool), torch.Tensor(), torch.Tensor(), list(target['labels'].cpu().numpy())))
                            continue

                        # Correct detected targets, initially assume all targets are missed for all iou threshold values
                        correct = torch.zeros(len(out['scores']), n_ious, dtype=torch.bool, device=device)
                        detected = []  # Detected Target Indices

                        # Per target class
                        for cls in torch.unique(target['labels']):
                            target_ids = (cls == target['labels']).nonzero(as_tuple=False).view(-1)
                            pred_ids = (cls == out['class_ids']).nonzero(as_tuple=False).view(-1) 

                            # Search for detections
                            if pred_ids.shape[0]:
                                # Prediction to target ious
                                ious, max_iou_ids = box_iou(out['rois'][pred_ids], target['boxes'][target_ids]).max(1)  # best ious, indices

                                # Append detections
                                detected_set = set()
                                for idx in (ious > iou_values[0]).nonzero(as_tuple=False):
                                    detected_tgt_id = target_ids[max_iou_ids[idx]]  # detected target
                                    if detected_tgt_id.item() not in detected_set:
                                        detected_set.add(detected_tgt_id.item())
                                        detected.append(detected_tgt_id)

                                        correct[pred_ids[idx]] = ious[idx] > iou_values  # iou_thres is 1xn

                                        if len(detected) == len(target['labels']):  # all targets already located in image
                                            break

                        # Append statistics (correct, conf, pred_cls, gt_cls)
                        stats.append((correct.cpu(), out['scores'].cpu(), out['class_ids'].cpu(), target['labels'].cpu()))

                for iter, data in enumerate(val_generator):
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']

                        if opt.num_gpus == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = model(imgs, annot, obj_list=params.names)
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss

                print(
                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                        epoch, opt.num_epochs, cls_loss, reg_loss, loss))
                writer.add_scalars('Loss', {'val': loss}, step)
                writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)

                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')

                stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
                if len(stats) and stats[0].any():
                    p, r, ap, f1, ap_class = ap_per_class(*stats, plot=opt.plot, save_dir=os.path.join(opt.saved_path, "results"), names=params.names)
                    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
                    mp, mr, mAP50, mAP = p.mean(), r.mean(), ap50.mean(), ap.mean()

                pf = '%20s'  + '%12.3g' * 4  # print format  
                print(pf % ('all', mp, mr, mAP50, mAP))

                # Write metrics to file
                with open(os.path.join(opt.saved_path, "results_novo.txt"), 'a+') as f:
                    if epoch==0:
                        f.write("\n %s %s %f \n"% (opt.optim, opt.schlr, opt.lr))
                    f.write('%g/%g' % (epoch, opt.num_epochs - 1) + '%10.4g' * 5 % (mp, mr, mAP50, mAP, loss) + '\n')  # append metrics, val_loss
                
                model.requires_grad_(True)
                model.train()

                # Early stopping
                if epoch - best_epoch > opt.es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                    break
    except KeyboardInterrupt:
        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
        writer.close()
    writer.close()

    return model


def save_checkpoint(model, name):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(opt.saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(opt.saved_path, name))


if __name__ == '__main__':
    opt = get_args()
    model = train(opt)
