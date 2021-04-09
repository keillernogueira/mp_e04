import os
import time
import numpy
import pathlib
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from torch.legacy.nn.Module import Module
# from torch.legacy.nn.utils import clear

from torch.nn.modules import CrossMapLRN2d

SpatialCrossMapLRN_temp = CrossMapLRN2d
containing_dir = str(pathlib.Path(__file__).resolve().parent)


# class SpatialCrossMapLRN_temp(Module):
#
#     def __init__(self, size, alpha=1e-4, beta=0.75, k=1, gpuDevice=0):
#         super(SpatialCrossMapLRN_temp, self).__init__()
#
#         self.size = size
#         self.alpha = alpha
#         self.beta = beta
#         self.k = k
#         self.scale = None
#         self.paddedRatio = None
#         self.accumRatio = None
#         self.gpuDevice = gpuDevice
#
#     def updateOutput(self, input):
#         assert input.dim() == 4
#
#         if self.scale is None:
#             self.scale = input.new()
#
#         if self.output is None:
#             self.output = input.new()
#
#         batchSize = input.size(0)
#         channels = input.size(1)
#         inputHeight = input.size(2)
#         inputWidth = input.size(3)
#
#         if input.is_cuda:
#             self.output = self.output.cuda(self.gpuDevice)
#             self.scale = self.scale.cuda(self.gpuDevice)
#
#         self.output.resize_as_(input)
#         self.scale.resize_as_(input)
#
#         # use output storage as temporary buffer
#         inputSquare = self.output
#         torch.pow(input, 2, out=inputSquare)
#
#         prePad = int((self.size - 1) / 2 + 1)
#         prePadCrop = channels if prePad > channels else prePad
#
#         scaleFirst = self.scale.select(1, 0)
#         scaleFirst.zero_()
#         # compute first feature map normalization
#         for c in range(prePadCrop):
#             scaleFirst.add_(inputSquare.select(1, c))
#
#         # reuse computations for next feature maps normalization
#         # by adding the next feature map and removing the previous
#         for c in range(1, channels):
#             scalePrevious = self.scale.select(1, c - 1)
#             scaleCurrent = self.scale.select(1, c)
#             scaleCurrent.copy_(scalePrevious)
#             if c < channels - prePad + 1:
#                 squareNext = inputSquare.select(1, c + prePad - 1)
#                 scaleCurrent.add_(1, squareNext)
#
#             if c > prePad:
#                 squarePrevious = inputSquare.select(1, c - prePad)
#                 scaleCurrent.add_(-1, squarePrevious)
#
#         self.scale.mul_(self.alpha / self.size).add_(self.k)
#
#         torch.pow(self.scale, -self.beta, out=self.output)
#         self.output.mul_(input)
#
#         return self.output
#
#     def updateGradInput(self, input, gradOutput):
#         assert input.dim() == 4
#
#         batchSize = input.size(0)
#         channels = input.size(1)
#         inputHeight = input.size(2)
#         inputWidth = input.size(3)
#
#         if self.paddedRatio is None:
#             self.paddedRatio = input.new()
#         if self.accumRatio is None:
#             self.accumRatio = input.new()
#         self.paddedRatio.resize_(channels + self.size - 1, inputHeight, inputWidth)
#         self.accumRatio.resize_(inputHeight, inputWidth)
#
#         cacheRatioValue = 2 * self.alpha * self.beta / self.size
#         inversePrePad = int(self.size - (self.size - 1) / 2)
#
#         self.gradInput.resize_as_(input)
#         torch.pow(self.scale, -self.beta, out=self.gradInput).mul_(gradOutput)
#
#         self.paddedRatio.zero_()
#         paddedRatioCenter = self.paddedRatio.narrow(0, inversePrePad, channels)
#         for n in range(batchSize):
#             torch.mul(gradOutput[n], self.output[n], out=paddedRatioCenter)
#             paddedRatioCenter.div_(self.scale[n])
#             torch.sum(self.paddedRatio.narrow(0, 0, self.size - 1), 0, out=self.accumRatio)
#             for c in range(channels):
#                 self.accumRatio.add_(self.paddedRatio[c + self.size - 1])
#                 self.gradInput[n][c].addcmul_(-cacheRatioValue, input[n][c], self.accumRatio)
#                 self.accumRatio.add_(-1, self.paddedRatio[c])
#
#         return self.gradInput
#
#     def clearState(self):
#         clear(self, 'scale', 'paddedRatio', 'accumRatio')
#         return super(SpatialCrossMapLRN_temp, self).clearState()


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


def Conv2d(in_dim, out_dim, kernel, stride, padding):
    l = torch.nn.Conv2d(in_dim, out_dim, kernel, stride=stride, padding=padding)
    return l


def BatchNorm(dim):
    bn = torch.nn.BatchNorm2d(dim)
    return bn


def CrossMapLRN(size, alpha, beta, k=1.0):
    if SpatialCrossMapLRN_temp is not None:
        n = SpatialCrossMapLRN_temp(size, alpha, beta, k)  # gpuDevice=gpu_device)
        # n = Lambda(lambda x, lrn=lrn: Variable(lrn.forward(x.data).cuda(gpu_device)) if x.data.is_cuda else Variable(lrn.forward(x.data)))
    else:
        n = nn.LocalResponseNorm(size, alpha, beta, k)  # .cuda(gpu_device)
    return n


def Linear(in_dim, out_dim):
    ln = torch.nn.Linear(in_dim, out_dim)
    return ln


class Inception(nn.Module):
    def __init__(self, input_size, kernel_size, kernel_stride, output_size, reduce_size, pool, use_batch_norm,
                 reduce_stride=None, padding=True):
        super(Inception, self).__init__()
        #
        self.seq_list = []
        self.outputSize = output_size

        # 1x1 conv (reduce) -> 3x3 conv
        # 1x1 conv (reduce) -> 5x5 conv
        # ...
        for i in range(len(kernel_size)):
            od = OrderedDict()
            # 1x1 conv
            od['1_conv'] = Conv2d(input_size, reduce_size[i], (1, 1),
                                  reduce_stride[i] if reduce_stride is not None else 1, (0, 0))
            if use_batch_norm:
                od['2_bn'] = BatchNorm(reduce_size[i])
            od['3_relu'] = nn.ReLU()
            # nxn conv
            pad = int(numpy.floor(kernel_size[i] / 2)) if padding else 0
            od['4_conv'] = Conv2d(reduce_size[i], output_size[i], kernel_size[i], kernel_stride[i], pad)
            if use_batch_norm:
                od['5_bn'] = BatchNorm(output_size[i])
            od['6_relu'] = nn.ReLU()
            #
            self.seq_list.append(nn.Sequential(od))

        ii = len(kernel_size)
        # pool -> 1x1 conv
        od = OrderedDict()
        od['1_pool'] = pool
        if ii < len(reduce_size) and reduce_size[ii] is not None:
            i = ii
            od['2_conv'] = Conv2d(input_size, reduce_size[i], (1, 1),
                                  reduce_stride[i] if reduce_stride is not None else 1, (0, 0))
            if use_batch_norm:
                od['3_bn'] = BatchNorm(reduce_size[i])
            od['4_relu'] = nn.ReLU()
        #
        self.seq_list.append(nn.Sequential(od))
        ii += 1

        # reduce: 1x1 conv (channel-wise pooling)
        if ii < len(reduce_size) and reduce_size[ii] is not None:
            i = ii
            od = OrderedDict()
            od['1_conv'] = Conv2d(input_size, reduce_size[i], (1, 1),
                                  reduce_stride[i] if reduce_stride is not None else 1, (0, 0))
            if use_batch_norm:
                od['2_bn'] = BatchNorm(reduce_size[i])
            od['3_relu'] = nn.ReLU()
            self.seq_list.append(nn.Sequential(od))

        self.seq_list = nn.ModuleList(self.seq_list)

    def forward(self, input):
        x = input

        ys = []
        target_size = None
        depth_dim = 0
        for seq in self.seq_list:
            # print(seq)
            # print(self.outputSize)
            # print('x_size:', x.size())
            y = seq(x)
            y_size = y.size()
            # print('y_size:', y_size)
            ys.append(y)
            #
            if target_size is None:
                target_size = [0] * len(y_size)
            #
            for i in range(len(target_size)):
                target_size[i] = max(target_size[i], y_size[i])
            depth_dim += y_size[1]

        target_size[1] = depth_dim
        # print('target_size:', target_size)

        for i in range(len(ys)):
            y_size = ys[i].size()
            pad_l = int((target_size[3] - y_size[3]) // 2)
            pad_t = int((target_size[2] - y_size[2]) // 2)
            pad_r = target_size[3] - y_size[3] - pad_l
            pad_b = target_size[2] - y_size[2] - pad_t
            ys[i] = F.pad(ys[i], (pad_l, pad_r, pad_t, pad_b))

        output = torch.cat(ys, 1)

        return output


class OpenFaceModel(nn.Module):
    def __init__(self):
        super(OpenFaceModel, self).__init__()

        # self.gpuDevice = gpu_device

        self.layer1 = Conv2d(3, 64, (7, 7), (2, 2), (3, 3))
        self.layer2 = BatchNorm(64)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1))
        self.layer5 = CrossMapLRN(5, 0.0001, 0.75)
        self.layer6 = Conv2d(64, 64, (1, 1), (1, 1), (0, 0))
        self.layer7 = BatchNorm(64)
        self.layer8 = nn.ReLU()
        self.layer9 = Conv2d(64, 192, (3, 3), (1, 1), (1, 1))
        self.layer10 = BatchNorm(192)
        self.layer11 = nn.ReLU()
        self.layer12 = CrossMapLRN(5, 0.0001, 0.75)
        self.layer13 = nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1))
        self.layer14 = Inception(192, (3, 5), (1, 1), (128, 32), (96, 16, 32, 64),
                                 nn.MaxPool2d((3, 3), stride=(2, 2), padding=(0, 0)), True)
        self.layer15 = Inception(256, (3, 5), (1, 1), (128, 64), (96, 32, 64, 64),
                                 nn.LPPool2d(2, (3, 3), stride=(3, 3)), True)
        self.layer16 = Inception(320, (3, 5), (2, 2), (256, 64), (128, 32, None, None),
                                 nn.MaxPool2d((3, 3), stride=(2, 2), padding=(0, 0)), True)
        self.layer17 = Inception(640, (3, 5), (1, 1), (192, 64), (96, 32, 128, 256),
                                 nn.LPPool2d(2, (3, 3), stride=(3, 3)), True)
        self.layer18 = Inception(640, (3, 5), (2, 2), (256, 128), (160, 64, None, None),
                                 nn.MaxPool2d((3, 3), stride=(2, 2), padding=(0, 0)), True)
        self.layer19 = Inception(1024, (3,), (1,), (384,), (96, 96, 256), nn.LPPool2d(2, (3, 3), stride=(3, 3)), True)
        self.layer21 = Inception(736, (3,), (1,), (384,), (96, 96, 256),
                                 nn.MaxPool2d((3, 3), stride=(2, 2), padding=(0, 0)), True)
        self.layer22 = nn.AvgPool2d((3, 3), stride=(1, 1), padding=(0, 0))
        self.layer25 = Linear(736, 128)

        self.resize1 = nn.UpsamplingNearest2d(scale_factor=3)
        self.resize2 = nn.AvgPool2d(4)

        # self.eval()

        # if use_cuda:
        #     self.cuda(gpu_device)

    def forward(self, input):
        x = input

        # if x.data.is_cuda and self.gpuDevice != 0:
        #     x = x.cuda(self.gpuDevice)

        if x.size()[-1] == 128:
            x = self.resize2(self.resize1(x))

        x = self.layer8(self.layer7(self.layer6(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x))))))))
        x = self.layer13(self.layer12(self.layer11(self.layer10(self.layer9(x)))))
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer21(x)
        x = self.layer22(x)
        x = x.view((-1, 736))

        x_736 = x

        x = self.layer25(x)
        x_norm = torch.sqrt(torch.sum(x ** 2, 1) + 1e-6)
        x = torch.div(x, x_norm.view(-1, 1).expand_as(x))

        return x  # , x_736


def load_open_face(use_cuda=True, gpu_device=0, use_multi_gpu=False):
    model = OpenFaceModel(use_cuda, gpu_device)
    model.load_state_dict(torch.load(os.path.join(containing_dir, 'openface.pth')))

    if use_multi_gpu:
        model = nn.DataParallel(model)

    return model


#
if __name__ == '__main__':
    #
    useCuda = True
    if useCuda:
        assert torch.cuda.is_available()
    else:
        assert False, 'Sorry, .pth file contains CUDA version of the network only.'

    nof = load_open_face()
    nof = nof.eval()

    # test
    #
    I = numpy.reshape(numpy.array(range(96 * 96), dtype=numpy.float32) * 0.01, (1, 96, 96))
    I = numpy.concatenate([I, I, I], axis=0)
    I_ = torch.from_numpy(I).unsqueeze(0)

    if useCuda:
        I_ = I_.cuda()

    print(nof)
    I_ = Variable(I_)
    print(nof(I_))

    import cv2

    def ReadImage(pathname):
        img = cv2.imread(pathname)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR)
        img = numpy.transpose(img, (2, 0, 1))
        img = img.astype(numpy.float32) / 255.0
        print(numpy.min(img), numpy.max(img))
        print(numpy.sum(img[0]), numpy.sum(img[1]), numpy.sum(img[2]))
        I_ = torch.from_numpy(img).unsqueeze(0)
        if useCuda:
            I_ = I_.cuda()
        return I_

    img_paths = [
        '/home/polphit/Downloads/face_images/lennon-1.jpg_aligned.png',
        '/home/polphit/Downloads/face_images/lennon-2.jpg_aligned.png',
        '/home/polphit/Downloads/face_images/clapton-1.jpg_aligned.png',
        '/home/polphit/Downloads/face_images/clapton-2.jpg_aligned.png',
        ]

    imgs = []
    for img_path in img_paths:
        imgs.append(ReadImage(img_path))

    I_ = torch.cat(imgs, 0)
    I_ = Variable(I_, requires_grad=False)
    start = time.time()
    f, f_736 = nof(I_)
    print("  + Forward pass took {} seconds.".format(time.time() - start))
    print(f)
    for i in range(f_736.size(0) - 1):
        for j in range(i + 1, f_736.size(0)):
            df = f_736[i] - f_736[j]
            print(img_paths[i].split('/')[-1], img_paths[j].split('/')[-1], torch.dot(df, df))

    # in OpenFace's sample code, cosine distance is usually used for f (128d).
