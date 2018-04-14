from __future__ import absolute_import

'''ShpereNet
'''
import torch.nn as nn
import math
from ..SphereConv import Sphere_Conv2d


__all__ = ['sphere_resnet','sphere_resnet110','sphere_resnet56','sphere_resnet44',
           'sphere_resnet32','sphere_resnet20','sphere_wresnet20','sphere_wresnet44',
           'sphere_wresnet110']

def sphere_conv3x3(in_planes, out_planes, stride=1, doini=2):
    "sphere 3x3 convolution with padding"
    return Sphere_Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, doini=doini)


class SphereBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SphereBasicBlock, self).__init__()
        self.conv1 = sphere_conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = sphere_conv3x3(planes, planes)
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


class SphereBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, doini=2):
        super(SphereBottleneck, self).__init__()
        self.conv1 = Sphere_Conv2d(inplanes, planes, kernel_size=1, bias=False, doini=doini)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Sphere_Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, doini=doini)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Sphere_Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, doini=doini)
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

    def __init__(self, depth, nfilter = [16,32,64], num_classes=10, rescale_bn=False):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6
        self.depth = n*3

        block = SphereBottleneck if depth >=44 else SphereBasicBlock

        self.inplanes = nfilter[0]
        self.conv1 = Sphere_Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False, doini=1)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, nfilter[0], n)
        self.layer2 = self._make_layer(block, nfilter[1], n, stride=2)
        self.layer3 = self._make_layer(block, nfilter[2], n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(nfilter[2] * block.expansion, num_classes)

        self.features = nn.Sequential(
                self.conv1,
                self.bn1,
                self.relu,    # 32x32
                self.layer1,  # 32x32
                self.layer2,  # 16x16
                self.layer3,  # 8x8
                self.avgpool
        )

        for m_name, m in self.named_modules():
            if 'bn' in m_name:
                if rescale_bn and 'bn2' in m_name and 'layer' in m_name:
                    m.weight.data.fill_(1/self.depth)
                else:
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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    # I dont think we should prune batch normalization and nerver tried
    def prune(self, sparsity, create_mask = True, prune_bias = False, prune_bn = False):
        first_layer = True
        if create_mask:
            self.masks = []
        masksid = -1
        for m in self.modules():
            if isinstance(m, Sphere_Conv2d) or isinstance(m, nn.Linear) or (isinstance(m,nn.BatchNorm2d) and prune_bn):
                if first_layer:
                    first_layer = False
                else:
                    if create_mask:
                        prune_k = int(m.weight.numel()*(sparsity))
                        val, _ = m.weight.view(-1).abs().topk(prune_k, largest = False)
                        prune_threshold = val[-1]
                        self.masks.append((m.weight.abs()<=prune_threshold).data)
                    masksid += 1
                    m.weight.data.masked_fill_(self.masks[masksid], 0)
                    if prune_bias:
                        if create_mask:
                            prune_k = int(m.bias.numel()*(sparsity))
                            val, _ = m.bias.view(-1).abs().topk(prune_k, largest = False)
                            prune_threshold = val[-1]
                            self.masks.append((m.bias.abs()<=prune_threshold).data)
                        masksid += 1
                        m.bias.data.masked_fill_(self.masks[masksid], 0)


def sphere_resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)

def sphere_resnet110(**kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(110, **kwargs)
    return model

def sphere_resnet56(**kwargs):
    """Constructs a ResNet-56 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(56, **kwargs)
    return model

def sphere_resnet44(**kwargs):
    """Constructs a ResNet-44 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(44, **kwargs)
    return model

def sphere_resnet32(**kwargs):
    """Constructs a ResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(32, **kwargs)
    return model

def sphere_resnet20(**kwargs):
    """Constructs a ResNet-20 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(20, **kwargs)
    return model

def sphere_wresnet20(**kwargs):
    """Constructs a Wide ResNet-20 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(20, nfilter = [64,128,256], **kwargs)
    return model

def sphere_wresnet44(**kwargs):
    """Constructs a Wide ResNet-44 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(44, nfilter = [64,128,256], **kwargs)
    return model

def sphere_wresnet110(**kwargs):
    """Constructs a Wide ResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(110, nfilter = [64,128,256], **kwargs)
    return model
