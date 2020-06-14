from lib.utils import showfeature, showimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.normalize import Normalize
import sys
sys.path.insert(0, "../")
from torch.autograd import Variable
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = P4MConvP4M(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, batch_norm=True)
        # self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = P4MConvP4M(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, batch_norm=True)
        # self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                P4MConvP4M(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, batch_norm=True),
                # nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu((self.conv1(x)))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = P4MConvP4M(in_planes, planes, kernel_size=1, bias=False, batch_norm=True)
        # self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = P4MConvP4M(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, batch_norm=True)
        # self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = P4MConvP4M(planes, self.expansion*planes, kernel_size=1, bias=False, batch_norm=True)
        # self.bn3 = nn.BatchNorm3d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                P4MConvP4M(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, batch_norm=True),
                # nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu((self.conv1(x)))
        out = F.relu((self.conv2(out)))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, low_dim=128, multitask=False):
        super(ResNet, self).__init__()
        self.multitask = multitask

        self.in_planes = 23
        self.conv1 = P4MConvZ2(3, 23, kernel_size=7, stride=2, padding=3, bias=False, batch_norm=True, max_pool=True)

        self.layer1 = self._make_layer(block, 23, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 45, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 91, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 181, num_blocks[3], stride=2)
        self.linear = nn.Linear(181*8*block.expansion, low_dim)

        self.l2norm = Normalize(2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        # dataX_90 = torch.flip(torch.transpose(x, 2, 3), [2])
        # dataX_180 = torch.flip(torch.flip(x, [2]), [3])
        # dataX_270 = torch.transpose(torch.flip(x, [2]), 2, 3)
        # x = torch.stack([x, dataX_90, dataX_180, dataX_270], dim=1)
        # x = x.view([3 * 4, 3,224,224])
        #
        #
        #
        # # print (x.shape)
        # for b in range(4, 8):
        #     showimage(x[b], "batch-"+str(b)+"-image.png")


        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        outs = out.size()
        out = out.view(outs[0], outs[1]*outs[2], outs[3], outs[4])


        out = F.avg_pool2d(out, 7)  # 12 if input is 96
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.l2norm(out)

        return out


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], **kwargs)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


# def test():
#     net = ResNet18()
#     y = net(Variable(torch.randn(1,3,32,32)))
#     print(y.size())

