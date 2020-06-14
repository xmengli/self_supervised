import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from lib.normalize import Normalize
import torch
from models.cbam import CBAM
import torch.nn.functional as F
from lib.utils import showfeature, showimage
import numpy as np
import random
import torch.backends.cudnn as cudnn
import os
my_whole_seed = 111
random.seed(my_whole_seed)
np.random.seed(my_whole_seed)
torch.manual_seed(my_whole_seed)
torch.cuda.manual_seed_all(my_whole_seed)
torch.cuda.manual_seed(my_whole_seed)
np.random.seed(my_whole_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(my_whole_seed)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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

    def __init__(self, block, layers, low_dim=128, multitask=False, showfeature=False, finetune=False, domain=False,args=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
        #                        bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)    # 7 if input is 224 !!!!!!!!!!
        self.fc = nn.Linear(512 * block.expansion, low_dim)
        self.l2norm = Normalize(2)
        self.saveembed = args.saveembed


        self.showfeature = showfeature
        self.multitask = multitask
        self.finetune = finetune
        self.domain = domain
        if self.finetune:
            self.finetune_layer = nn.Sequential(
                Flatten(),
                nn.Linear(128, 128, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2, bias=False),
            )

        if self.multitask and self.domain:
            self.domain_classifier = nn.Linear(128, 2)
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
            self.fc_block = nn.Sequential(
                Flatten(),
                # 3*3 if input is 224
                nn.Linear(512 * 3 * 3, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
            )


        if self.multitask:
            self.rotation_classifier = nn.Linear(128, 4)
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
            self.fc_block = nn.Sequential(
                Flatten(),
                # 3*3 if input is 224
                nn.Linear(512 * 3 * 3, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):


        # dataX_90 = torch.flip(torch.transpose(x, 2, 3), [2])
        # dataX_180 = torch.flip(torch.flip(x, [2]), [3])
        # dataX_270 = torch.transpose(torch.flip(x, [2]), 2, 3)
        # x = torch.stack([x, dataX_90, dataX_180, dataX_270], dim=1)
        # x = x.view([3 * 4, 3,224,224])
        #

        # print (x.shape)
        # exit(0)
        # for b in range(0, 8):
        # showimage(x[0], "batch-0-image.png")
        # showimage(x[75], "batch-75-image.png")
        # showimage(x[150], "batch-150-image.png")
        # showimage(x[225], "batch-225-image.png")
        # exit(0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # if self.showfeature:
        #     showfeature(x[4,:,:,:], "feature_1rot.png")
        # if self.showfeature:
        #     showfeature(x[5,:,:,:], "feature_101rot.png")
        # if self.showfeature:
        #     showfeature(x[6,:,:,:], "feature_201rot.png")
        # if self.showfeature:
        #     showfeature(x[7,:,:,:], "feature_301rot.png")
        # print (x.shape)
        # exit(0)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.l2norm(x)

        # if self.saveembed != "":
        #     print (x.shape)
        #     print ("save to ", self.saveembed)
        #     np.savetxt("embed/"+self.saveembed, x.cpu().data.numpy())
        #     exit(0)
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
