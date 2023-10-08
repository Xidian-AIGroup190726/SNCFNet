from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms

from pycontourlet4d.pycontourlet import batch_multi_channel_pdfbdec
from dwtmodel.DWT_IDWT.DWT_IDWT_layer import *


def get_inplanes():
    return [16, 32, 64, 128]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=3,
                 conv1_t_stride=1,
                 no_max_pool=True,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=11):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 3, 3),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 1, 1),
                               bias=False)
        self.conv1ms = nn.Conv3d(4,
                                 16,
                                 kernel_size=(conv1_t_size, 3, 3),
                                 stride=(conv1_t_stride, 2, 2),
                                 padding=(conv1_t_size // 2, 1, 1),
                                 bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.bn1ms = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer1ms = self._make_layer(block, block_inplanes[0], layers[0],
                                         shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.in_planes = int(self.in_planes / 2)
        self.layer2ms = self._make_layer(block,
                                         block_inplanes[1],
                                         layers[1],
                                         shortcut_type,
                                         stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.in_planes = int(self.in_planes / 2)
        self.layer3ms = self._make_layer(block,
                                         block_inplanes[2],
                                         layers[2],
                                         shortcut_type,
                                         stride=2)
        self.in_planes = int(self.in_planes / 2)
        self.layer3x4ms = self._make_layer(block,
                                           block_inplanes[2],
                                           layers[2],
                                           shortcut_type,
                                           stride=(1,2,2))
        self.in_planes = int(self.in_planes / 2)
        self.layer3ms4x = self._make_layer(block,
                                           block_inplanes[2],
                                           layers[2],
                                           shortcut_type,
                                           stride=(1,2,2))
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        self.in_planes = int(self.in_planes / 2)
        self.layer4ms = self._make_layer(block,
                                         block_inplanes[3],
                                         layers[3],
                                         shortcut_type,
                                         stride=2)
        self.in_planes = int(self.in_planes / 2)
        self.layer4and = self._make_layer(block,
                                          block_inplanes[3],
                                          layers[3],
                                          shortcut_type,
                                          stride=2)
        self.layerinput1 = nn.Sequential(conv1x1x1(1, 16, 1),
                                         nn.BatchNorm3d(16),
                                         nn.ReLU(inplace=True)
                                         )
        self.layerinput1ms = nn.Sequential(conv1x1x1(4, 16, 1),
                                           nn.BatchNorm3d(16),
                                           nn.ReLU(inplace=True)
                                           )
        self.layerinput2 = nn.Sequential(conv1x1x1(1, 32, 1),
                                         nn.BatchNorm3d(32),
                                         nn.ReLU(inplace=True)
                                         )
        self.layerinput2ms = nn.Sequential(conv1x1x1(4, 32, 1),
                                           nn.BatchNorm3d(32),
                                           nn.ReLU(inplace=True)
                                           )
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(384, n_classes)

        self.dwt = DWT_3D(wavename='haar')
        self.dwtMS = DWT_3D(wavename='haar')
        self.kernelConv = nn.Conv3d(64 * 8, 32, kernel_size=1, stride=1, bias=False, groups=8)
        self.kernelConvMs = nn.Conv3d(64 * 8, 32, kernel_size=1, stride=1, bias=False, groups=8)

        self.kernelLayer = nn.Sequential(nn.Conv3d(16, 16, kernel_size=3, stride=(1, 2, 2,), padding=1, bias=False),
                                         nn.BatchNorm3d(16),
                                         nn.ReLU(inplace=True)
                                         )
        self.kernelBn = nn.BatchNorm3d(32)
        self.kernelLayerMS = nn.Sequential(nn.Conv3d(16, 16, kernel_size=3, stride=(1, 2, 2,), padding=1, bias=False),
                                           nn.BatchNorm3d(16),
                                           nn.ReLU(inplace=True)
                                           )
        self.kernelBnMS = nn.BatchNorm3d(32)

        self.gauKernelSize = 5
        self.blurFactor = 1.6
        #  预处理
        self.preBlur1 = transforms.GaussianBlur(5, 1.6)
        self.transCont = transforms.Resize((128, 128))
        self.transGD1 = transforms.Resize((64, 64))
        self.transGD2 = transforms.Resize((32, 32))
        self.transGD3 = transforms.Resize((16, 16))
        self.bn0 = nn.BatchNorm3d(1)
        self.bn0Ms = nn.BatchNorm3d(4)

        '''
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
'''

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, xMs, x):
        # 多尺度生成
        # m, m1, m2, m3 = self.msMultileScaleGener(xMs)
        x, x1, x2, m, m1, m2 = self.singleChannelMultileScaleGner(x, xMs)
        '''
        x = self.bn0(x)
        x1 = self.bn0(x1)
        x2 = self.bn0(x2)
        x3 = self.bn0(x3)
        m = self.bn0Ms(m)
        m1 = self.bn0Ms(m1)
        m2 = self.bn0Ms(m2)
        m3 = self.bn0Ms(m3)
'''
        # pan进网络
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        # ms进网络
        m = self.conv1ms(m)
        m = self.bn1ms(m)
        m = self.relu(m)

        x = self.layer1(x)
        x4Ms = x  # 用于交互滤波
        x1 = self.layerinput1(x1)
        x = torch.cat([x, x1], 2)

        m = self.layer1ms(m)
        m4Pan = m  # 用于交互滤波
        m1 = self.layerinput1ms(m1)
        m = torch.cat([m, m1], 2)

        x = self.layer2(x)
        x2 = self.layerinput2(x2)
        x = torch.cat([x, x2], 2)

        m = self.layer2ms(m)
        m2 = self.layerinput2ms(m2)
        m = torch.cat([m, m2], 2)

        x = self.layer3(x)
        m = self.layer3ms(m)

        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.dwt(x)
        kernel = torch.cat((LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH), 1)
        kernel = self.kernelGen(kernel, modal='pan')

        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.dwtMS(m)
        kernelMs = torch.cat((LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH), 1)
        kernelMs = self.kernelGen(kernelMs, modal='ms')

        # 交互滤波
        x4Ms = self.kernelLayer(x4Ms)
        x4Ms = F.pad(x4Ms, pad=(1, 2, 1, 2, 1, 2), mode='constant', value=0)
        x4Ms = F.conv3d(x4Ms, kernelMs, stride=1)
        x4Ms = self.kernelBn(x4Ms)
        x4Ms = self.layer3x4ms(x4Ms)

        m4Pan = self.kernelLayerMS(m4Pan)
        m4Pan = F.pad(m4Pan, pad=(1, 2, 1, 2, 1, 2), mode='constant', value=0)
        m4Pan = F.conv3d(m4Pan, kernel, stride=1)
        m4Pan = self.kernelBnMS(m4Pan)
        m4Pan = self.layer3ms4x(m4Pan)

        # Unique fearure
        x = x - x4Ms
        m = m - m4Pan
        # Shared feature
        xAndMS = (x4Ms + m4Pan) / 2

        x = self.layer4(x)
        m = self.layer4ms(m)
        xAndMS = self.layer4and(xAndMS)

        x = self.avgpool(x)
        m = self.avgpool(m)
        xAndMS = self.avgpool(xAndMS)

        m = m.view(m.size(0), -1)
        x = x.view(x.size(0), -1)
        xAndMS = xAndMS.view(x.size(0), -1)

        x = torch.cat([m, x, xAndMS], 1)
        x = self.fc(x)

        return x

    def kernelGen(self, x, modal='pan'):
        if modal == 'pan':
            x = self.kernelConv(x)
        else:
            x = self.kernelConvMs(x)
        kernel = torch.zeros([16, 32, 4, 4, 4])
        for i in range(0, kernel.size(0)):
            kernel[i] = torch.sum(
                x[int(x.size(0) / kernel.size(0) * i):int(x.size(0) / kernel.size(0) * (i + 1)), :, :, :, :], 0)
            # print(kernel.size())
        kernel = kernel.transpose(0, 1).cuda()

        return kernel

    def singleChannelMultileScaleGner(self, x, ms):
        # 轮廓波变换提取多级边缘特征
        coefs = self.contourletTrans(x)
        coefs = self.__pdfbdec(coefs)

        level1 = torch.sum(coefs[0], 1, keepdim=True).cuda()
        level2 = torch.sum(coefs[1], 1, keepdim=True).cuda()
        level3 = torch.sum(coefs[2][:, 1:4, :, :], 1, keepdim=True).cuda()

        # 边缘、纹理特征增强后高斯滤波
        baseImg = self.preBlur(x) + level1
        baseImgMs = self.preBlur(ms) + level1.repeat(1, 4, 1, 1)
        inputPan0, baseImg, inputMs0, baseImgMs = self.gaussianBlur(baseImg, baseImgMs, self.blurFactor)
        baseImg = self.gaussianDown(baseImg)
        baseImgMs = self.gaussianDown(baseImgMs)
        inputPan0 = inputPan0.unsqueeze(1)  # (N,C,S,H,W)
        inputMs0 = self.channelResize(inputMs0)

        baseImg += level2
        baseImgMs += level2.repeat(1, 4, 1, 1)
        inputPan1, baseImg, inputMs1, baseImgMs = self.gaussianBlur(baseImg, baseImgMs, self.blurFactor)
        baseImg = self.gaussianDown(baseImg)
        baseImgMs = self.gaussianDown(baseImgMs)
        inputPan1 = inputPan1.unsqueeze(1)  # (N,C,S,H,W)
        inputMs1 = self.channelResize(inputMs1)

        baseImg += level3
        baseImgMs += level3.repeat(1, 4, 1, 1)
        inputPan2, baseImg, inputMs2, baseImgMs = self.gaussianBlur(baseImg, baseImgMs, self.blurFactor)
        inputPan2 = inputPan2.unsqueeze(1)  # (N,C,S,H,W)
        inputMs2 = self.channelResize(inputMs2)

        return inputPan0, inputPan1, inputPan2, inputMs0, inputMs1, inputMs2

    def gaussianBlur(self, x, ms, sigma):
        # 多层高斯滤波
        k = 2 ** (1.0 / 8)
        output = x
        outputMs = ms
        for i in range(1, 9):
            theta = k ** i * sigma
            transGau = transforms.GaussianBlur(self.gauKernelSize, theta)
            oct = transGau(x)
            octMs = transGau(ms)

            if i < 8:
                output = torch.cat([output, oct], 1)
                outputMs = torch.cat([outputMs, octMs], 1)
            else:
                imgBaseNext = oct
                imgBaseNextMs = octMs
        return output, imgBaseNext, outputMs, imgBaseNextMs

    def channelResize(self, ms):
        ms = torch.stack([ms[:, 0:4, :, :], ms[:, 4:8, :, :], ms[:, 8:12, :, :], ms[:, 12:16, :, :], ms[:, 16:20, :, :],
                          ms[:, 20:24, :, :], ms[:, 24:28, :, :], ms[:, 28:32, :, :]], dim=2)

        return ms

    def preBlur(self, x):
        x = self.preBlur1(x)
        return x

    def contourletTrans(self, x):
        # 适应轮廓波变换尺寸
        x = self.transCont(x)

        return x

    def gaussianDown(self, x):
        # 高斯金字塔间降采样
        if x.size(2) == 128:
            x = self.transGD1(x)
        elif x.size(2) == 64:
            x = self.transGD2(x)
        else:
            x = self.transGD3(x)

        return x

    def __pdfbdec(self, x, method="resize"):
        """Pyramidal directional filter bank decomposition for a batch of
        images.
        Returns a list of 4D numpy array.
        Here's an example with an image with 3 channels, and batch_size=2:
            >>> self.n_levs = [0, 3, 3, 3]
            >>> coefs, sfs = self.__pdfbdec(x)
        This will yield:
            >>> coefs[0].shape
            (2, 24, 112, 112)
            >>> coefs[1].shape
            (2, 24, 56, 56)
            >>> coefs[2].shape
            (2, 24, 28, 28)
            >>> coefs[3].shape
            (2, 12, 14, 14)
            >>> sfs.shape
            (2, 168)
        """
        # Convert to from N-D channels to single channel by averaging
        '''
        if self.spec_type == 'avg':
            imgs = []
            # Iterate each image in a batch
            for i in range(x.shape[0]):
                # Convert to PIL and image and to grayscale image
                img = transforms.ToPILImage()(x[i])
                img = to_grayscale(img)
                imgs.append(img)
            # Restack and convert back to PyTorch tensor
            x = torch.from_numpy((np.expand_dims(np.stack(imgs, axis=0), axis=1)))
'''
        # Obtain coefficients
        # coefs = batch_multi_channel_pdfbdec(x=x, pfilt="maxflat", dfilt="dmaxflat7", nlevs=[0, 3, 3, 3],
        #                                    device=self.device)
        coefs = batch_multi_channel_pdfbdec(x=x, pfilt="maxflat", dfilt="dmaxflat7", nlevs=[0, 3, 3])

        # Stack channels with same image dimension
        coefs = self.stack_same_dim(coefs)

        # Resize or splice
        if method == "resize":
            for k in coefs.keys():
                # Resize if image is not square
                if k[2] != k[3]:
                    # Get maximum dimension (height or width)
                    max_dim = int(np.max((k[2], k[3])))
                    # Resize the channels
                    trans = transforms.Compose([transforms.Resize((max_dim, max_dim))])
                    coefs[k] = trans(coefs[k])
        else:
            for k in coefs.keys():
                # Resize if image is not square
                if k[2] != k[3]:
                    # Get minimum dimension (height or width)
                    min_dim = int(np.argmin((k[2], k[3]))) + 2
                    # Splice alternate channels (always even number of channels exist)
                    coefs[k] = torch.cat((coefs[k][:, ::2, :, :], coefs[k][:, 1::2, :, :]), dim=min_dim)

        # Stack channels with same image dimension
        coefs = self.stack_same_dim(coefs)

        # Change coefs's key to number (n-1 to 0), instead of dimension
        for i, k in enumerate(coefs.copy()):
            idx = len(coefs.keys()) - i - 1
            coefs[idx] = coefs.pop(k)
        '''
        # Get statistical features (mean and std) for each image
        sfs = []
        for k in coefs.keys():
            sfs.append(coefs[k].mean(dim=[2, 3]))
            sfs.append(coefs[k].std(dim=[2, 3]))
        sfs = torch.cat(sfs, dim=1)
'''
        return coefs

    def stack_same_dim(self, x):
        """Stack a list/dict of 4D tensors of same img dimension together."""
        # Collect tensor with same dimension into a dict of list
        output = {}

        # Input is list
        if isinstance(x, list):
            for i in range(len(x)):
                if isinstance(x[i], list):
                    for j in range(len(x[i])):
                        shape = tuple(x[i][j].shape)
                        if shape in output.keys():
                            output[shape].append(x[i][j])
                        else:
                            output[shape] = [x[i][j]]
                else:
                    shape = tuple(x[i].shape)
                    if shape in output.keys():
                        output[shape].append(x[i])
                    else:
                        output[shape] = [x[i]]
        else:
            for k in x.keys():
                shape = tuple(x[k].shape[2:4])
                if shape in output.keys():
                    output[shape].append(x[k])
                else:
                    output[shape] = [x[k]]

        # Concat the list of tensors into single tensor
        for k in output.keys():
            output[k] = torch.cat(output[k], dim=1)

        return output




def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model