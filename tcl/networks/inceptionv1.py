from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import os
import sys

CM = 16

class BasicConv1d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dilation=1, hasBN=False):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False, dilation=dilation) # verify bias false
        self.bn = nn.BatchNorm1d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.hasBN = hasBN 

    def forward(self, x):
        x = self.conv(x)
        if self.hasBN:
            x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool1d(3, stride=2)
        self.conv = BasicConv1d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_Single(nn.Module):

    def __init__(self, inf, c1, c2, c3, c4, dilation=2, hasBN=False):
        super(Inception_Single, self).__init__()
        self.branch1 = BasicConv1d(inf, c1, kernel_size=1, stride=1, dilation=dilation, hasBN=hasBN)

        self.branch2 = nn.Sequential(
            BasicConv1d(inf, CM, kernel_size=1, stride=1, dilation=dilation, hasBN=hasBN),
            BasicConv1d(CM, c2, kernel_size=5, stride=1, padding=2*dilation, dilation=dilation, hasBN=hasBN)
        )

        self.branch3 = nn.Sequential(
            BasicConv1d(inf, CM, kernel_size=1, stride=1, dilation=dilation, hasBN=hasBN),
            BasicConv1d(CM, c3, kernel_size=7, stride=1, padding=3*dilation, dilation=dilation, hasBN=hasBN)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool1d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv1d(inf, c4, kernel_size=1, stride=1, dilation=dilation, hasBN=hasBN)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        #print('Inception_Single: x1 {}, x2 {}, x3 {}, x4 {}'.format(x1.shape, x2.shape, x3.shape, x4.shape))
        out = torch.cat((x1, x2, x3, x4), 1)
        return out


class InceptionV1(nn.Module):

    def __init__(self, num_classes=5, hasBN=False, dilation=1):
        super(InceptionV1, self).__init__()
        # Modules
        self.features = nn.Sequential(
            BasicConv1d(1, 64, kernel_size=3, stride=1, dilation=1),
            nn.MaxPool1d(3, stride=3),
            BasicConv1d(64, 128, kernel_size=3, stride=1, dilation=1),
            nn.MaxPool1d(3, stride=3),

            Inception_Single(inf=128, c1=64, c2=64, c3=32, c4=32, dilation=dilation),  # Inception*2
            Inception_Single(inf=192, c1=64, c2=64, c3=32, c4=32, dilation=dilation),  
            nn.MaxPool1d(3, stride=3),

            Inception_Single(inf=192, c1=64, c2=64, c3=32, c4=32, dilation=dilation),  # Inception*2
            Inception_Single(inf=192, c1=64, c2=64, c3=32, c4=32, dilation=dilation),  
            Inception_Single(inf=192, c1=48, c2=48, c3=32, c4=32, dilation=dilation),  # Inception*3
            Inception_Single(inf=160, c1=48, c2=48, c3=32, c4=32, dilation=dilation),
            Inception_Single(inf=160, c1=48, c2=48, c3=32, c4=32, dilation=dilation),
            nn.MaxPool1d(3, stride=3),


            Inception_Single(inf=160, c1=64, c2=96, c3=64, c4=32, dilation=dilation), # Inception*3
            Inception_Single(inf=256, c1=64, c2=96, c3=64, c4=32, dilation=dilation),
            Inception_Single(inf=256, c1=64, c2=96, c3=64, c4=32, dilation=dilation),
            nn.MaxPool1d(3, stride=3),

            Inception_Single(inf=256, c1=128, c2=128, c3=256, c4=96, dilation=dilation),  # Inception*2
            nn.MaxPool1d(2, stride=2),

            Inception_Single(inf=608, c1=96, c2=96, c3=128, c4=64, dilation=dilation), # Inception5a
            Inception_Single(inf=384, c1=96, c2=96, c3=128, c4=64, hasBN=False, dilation=dilation), # Inception5b
            nn.MaxPool1d(2, stride=2)
        )
        self.dropout = nn.Dropout(0.2) # in original paper
        self.last_linear = nn.Linear(1536, num_classes)

    def logits(self, features):
        #Allows image of any size to be processed
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool1d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        batch_size = x.size()[0]
        input_size = x.size()[-1]
        x = x.view(batch_size, -1, input_size)
        #print('x shape {}'.format(x.shape))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        #print('x shape2 {}'.format(x.shape))
        return x


# slim版本，裁剪一部分Inception Layer
class InceptionV1s(nn.Module):

    def __init__(self, num_classes=5, hasBN=False, dilation=1):
        super(InceptionV1s, self).__init__()
        # Modules
        self.features = nn.Sequential(
            BasicConv1d(1, 64, kernel_size=3, stride=1, dilation=1),
            nn.MaxPool1d(3, stride=3),
            BasicConv1d(64, 128, kernel_size=3, stride=1, dilation=1),
            nn.MaxPool1d(3, stride=3),

            Inception_Single(inf=128, c1=64, c2=64, c3=32, c4=32, dilation=dilation),  # Inception*2
            Inception_Single(inf=192, c1=64, c2=64, c3=32, c4=32, dilation=dilation),  
            nn.MaxPool1d(3, stride=3),

            Inception_Single(inf=192, c1=64, c2=64, c3=32, c4=32, dilation=dilation),  # Inception*2
            #Inception_Single(inf=192, c1=64, c2=64, c3=32, c4=32, dilation=dilation),  
            Inception_Single(inf=192, c1=48, c2=48, c3=32, c4=32, dilation=dilation),  # Inception*3
            #Inception_Single(inf=160, c1=48, c2=48, c3=32, c4=32, dilation=dilation),
            #Inception_Single(inf=160, c1=48, c2=48, c3=32, c4=32, dilation=dilation),
            nn.MaxPool1d(3, stride=3),


            Inception_Single(inf=160, c1=64, c2=96, c3=64, c4=32, dilation=dilation), # Inception*3
            #Inception_Single(inf=256, c1=64, c2=96, c3=64, c4=32, dilation=dilation),
            #Inception_Single(inf=256, c1=64, c2=96, c3=64, c4=32, dilation=dilation),
            nn.MaxPool1d(3, stride=3),

            Inception_Single(inf=256, c1=128, c2=128, c3=256, c4=96, dilation=dilation),  # Inception*2
            nn.MaxPool1d(2, stride=2),

            Inception_Single(inf=608, c1=96, c2=96, c3=128, c4=64, dilation=dilation), # Inception5a
            #Inception_Single(inf=384, c1=96, c2=96, c3=128, c4=64, hasBN=False, dilation=dilation), # Inception5b
            nn.MaxPool1d(2, stride=2)
        )
        self.dropout = nn.Dropout(0.2) # in original paper
        self.last_linear = nn.Linear(1536, num_classes)

    def logits(self, features):
        #Allows image of any size to be processed
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool1d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        batch_size = x.size()[0]
        input_size = x.size()[-1]
        x = x.view(batch_size, -1, input_size)
        #print('x shape {}'.format(x.shape))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        #print('x shape2 {}'.format(x.shape))
        return x

# if __name__ == "__main__":
#     I4 = InceptionV4(num_classes=16)
#     test = torch.rand(16,2,500)
#     print(I4(test))
