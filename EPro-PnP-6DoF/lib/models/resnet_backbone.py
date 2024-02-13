"""
This file is modified from
https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi
"""

import torch.nn as nn
import torch
from pylab import *#可视化
import matplotlib.pyplot as plt

class ResNetBackboneNet(nn.Module):
    def __init__(self, block, layers, in_channel=3, freeze=False):
        self.freeze = freeze
        self.inplanes = 64
        super(ResNetBackboneNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x): # x.shape [32, 3, 256, 256]
        #特征图可视化
        # outputs = []
        # processed = []
        if self.freeze:
            with torch.no_grad():
                x = self.conv1(x)   # x.shape [32, 64, 128, 128]
                x = self.bn1(x)
                x = self.relu(x)
                x_low_feature = self.maxpool(x)  # x.shape [32, 64, 64, 64]
                x = self.layer1(x_low_feature)   # x.shape [32, 256, 64, 64]
                x = self.layer2(x)  # x.shape [32, 512, 32, 32]
                x = self.layer3(x)  # x.shape [32, 1024, 16, 16]
                x_high_feature = self.layer4(x)  # x.shape [32, 2048, 8, 8]
                return x_high_feature.detach()
        else:
            # x_cpu = x.cpu()
            # outputs.append(x_cpu)
            #b, _, f, p = x.shape
            #print("输入的形状:", b, f, p)
            x = self.conv1(x)   # x.shape [32, 64, 128, 128]
            x = self.bn1(x)
            x = self.relu(x)
            #可视化
            # x_cpu = x.cpu()
            # outputs.append(x_cpu)

            x_low_feature = self.maxpool(x) # x.shape [32, 64, 64, 64]
            x = self.layer1(x_low_feature)  # x.shape [16, 64, 64, 64]
            # print('第一层', x.shape)
            # x_cpu = x.cpu()
            # outputs.append(x_cpu)

            x = self.layer2(x)  # x.shape [16, 128, 32, 32]
            # print('第二层', x.shape)
            # x_cpu = x.cpu()
            # outputs.append(x_cpu)

            x = self.layer3(x)  # x.shape [16, 256, 16, 16]
            # print('第三层', x.shape)
            # x_cpu = x.cpu()
            # outputs.append(x_cpu)

            x_high_feature = self.layer4(x)  # x.shape [16, 256, 16, 16]
            # print('第四层', x.shape)
            # x_cpu2 = x_high_feature.cpu()
            # outputs.append(x_cpu2)
            # print("")
            # for feature_map in outputs:
            #     feature_map = feature_map.squeeze(0)
            #     gray_scale = torch.sum(feature_map, 0)
            #     gray_scale = gray_scale / feature_map.shape[0]
            #     processed.append(gray_scale.data.cpu().numpy())
            # fig = plt.figure(figsize=(30, 50))
            # for i in range(len(processed)):
            #     a = fig.add_subplot(5, 7, i + 1)  # 不能小于网络深度
            #     imgplot = plt.imshow(processed[i])
            #     a.axis("off")
            #     # if not names[i].split('(')[0]:
            #     #     a.set_title(names[i].split('(')[0], fontsize=30)
            #     # else:
            #     #     a.set_title(names[i] + str(i % 3).split('(')[0], fontsize=30)
            #     # a.set_title(names[i].split('(')[0], fontsize=30)
            # plt.savefig(str('/home/ivclab/path/EPro-PnP-main/EPro-PnP-6DoF/resnet-34.jpg'), bbox_inches='tight')
            # print("保存")
            return x_high_feature
