from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import os
import sys

CM = 16

class BasicConv1d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dilation=2):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False, dilation=1) # verify bias false
        self.bn = nn.BatchNorm1d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        #x = self.bn(x)
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


class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv1d(160, 64, kernel_size=1, stride=1),
            BasicConv1d(64, 96, kernel_size=3, stride=1)
        )

        self.branch1 = nn.Sequential(
            BasicConv1d(160, 64, kernel_size=1, stride=1),
            BasicConv1d(64, 64, kernel_size=7, stride=1, padding=3),
            BasicConv1d(64, 96, kernel_size=3, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv1d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool1d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):

    def __init__(self, inf, c0, c1, c2, c3):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv1d(inf, c0, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv1d(inf, CM, kernel_size=1, stride=1),
            BasicConv1d(CM, c1, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv1d(inf, CM, kernel_size=1, stride=1),
            BasicConv1d(CM, CM, kernel_size=3, stride=1, padding=1),
            BasicConv1d(CM, c2, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool1d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv1d(inf, c3, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        print('InceptionA: x0 {}, x1 {}, x2 {}, x3 {}'.format(x0.shape, x1.shape, x2.shape, x3.shape))
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv1d(384, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv1d(384, 192, kernel_size=1, stride=1),
            BasicConv1d(192, 224, kernel_size=3, stride=1, padding=1),
            BasicConv1d(224, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool1d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):

    def __init__(self, inf, c0, c1, c2, c3):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv1d(inf, c0, kernel_size=1, stride=1, dilation=1)

        self.branch1 = nn.Sequential(
            BasicConv1d(inf, CM, kernel_size=1, stride=1, dilation=1),
            BasicConv1d(CM, CM, kernel_size=7, stride=1, padding=3, dilation=1),
            BasicConv1d(CM, c1, kernel_size=7, stride=1, padding=3, dilation=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv1d(inf, CM, kernel_size=1, stride=1, dilation=1),
            BasicConv1d(CM, CM, kernel_size=7, stride=1, padding=3, dilation=1),
            BasicConv1d(CM, c2, kernel_size=7, stride=1, padding=3, dilation=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool1d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv1d(inf, c3, kernel_size=1, stride=1, dilation=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        print('InceptionB: x0 {}, x1 {}, x2 {}, x3 {}'.format(x0.shape, x1.shape, x2.shape, x3.shape))
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv1d(1024, 192, kernel_size=1, stride=1),
            BasicConv1d(192, 192, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv1d(1024, 256, kernel_size=1, stride=1),
            BasicConv1d(256, 320, kernel_size=7, stride=1, padding=3),
            BasicConv1d(320, 320, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool1d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):

    def __init__(self, inf, c0, c1, c2, c3):
        super(Inception_C, self).__init__()

        self.branch0 = BasicConv1d(inf, c0, kernel_size=1, stride=1)

        self.branch1_0 = BasicConv1d(inf, CM, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv1d(CM, c1, kernel_size=3, stride=1, padding=1)
        self.branch1_1b = BasicConv1d(CM, c1, kernel_size=3, stride=1, padding=1)

        self.branch2_0 = BasicConv1d(inf, CM, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv1d(CM, CM, kernel_size=3, stride=1, padding=1)
        self.branch2_2 = BasicConv1d(CM, CM, kernel_size=3, stride=1, padding=1)
        self.branch2_3a = BasicConv1d(CM, c2, kernel_size=3, stride=1, padding=1)
        self.branch2_3b = BasicConv1d(CM, c2, kernel_size=3, stride=1, padding=1)

        self.branch3 = nn.Sequential(
            nn.AvgPool1d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv1d(inf, c3, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        print('InceptionC: x0 {}, x1 {}, x2 {}, x3 {}'.format(x0.shape, x1.shape, x2.shape, x3.shape))
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionV4(nn.Module):

    def __init__(self, num_classes=5):
        super(InceptionV4, self).__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (20, 25, 1)
        self.mean = None
        self.std = None
        # Modules
        self.features = nn.Sequential(
            BasicConv1d(1, 64, kernel_size=3, stride=1),
            nn.MaxPool1d(3, stride=3),
            BasicConv1d(64, 128, kernel_size=3, stride=1),
            nn.MaxPool1d(3, stride=3),

            Inception_C(inf=128, c0=64, c1=64, c2=32, c3=32),  # Inception*2
            nn.MaxPool1d(3, stride=3),

            Inception_C(inf=192, c0=64, c1=64, c2=32, c3=32),  # Inception*2
            Inception_A(inf=192, c0=48, c1=48, c2=32, c3=32),  # Inception*3
            nn.MaxPool1d(3, stride=3),


            Inception_A(inf=160, c0=64, c1=96, c2=64, c3=32),  # Inception*3
            nn.MaxPool1d(3, stride=3),

            Inception_C(inf=256, c0=128, c1=128, c2=256, c3=96),  # Inception*2
            nn.MaxPool1d(2, stride=2),

            Inception_B(inf=608, c0=96, c1=96, c2=128, c3=64),
            nn.BatchNorm1d(384, eps=0.001, momentum=0.1, affine=True),
            nn.MaxPool1d(2, stride=2)

            # Inception_A(), # originally 4 layers
            # Inception_A(),
            # Inception_A(),
            # Inception_A(),

            # Reduction_A(), # Mixed_6a

            # Inception_B(), # originally 7 layers
            # Inception_B(),
            # Inception_B(),
            # Inception_B(),
            # Inception_B(),
            # Inception_B(),
            # Inception_B(),

            # Reduction_B(), # Mixed_7a
            
            # Inception_C(), # originally 3 layers
            # Inception_C(),
            # Inception_C()
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
        # x = self.logits(x)
        return x


class InceptionV4Aux(nn.Module):

    def __init__(self, num_classes=5):
        super(InceptionV4Aux, self).__init__()

        # Modules
        self.features1 = nn.Sequential(
            BasicConv1d(1, 64, kernel_size=3, stride=1, dilation=1),
            nn.MaxPool1d(3, stride=3),
            BasicConv1d(64, 128, kernel_size=3, stride=1, dilation=1),
            nn.MaxPool1d(3, stride=3),

            Inception_C(inf=128, c0=64, c1=64, c2=32, c3=32),  # Inception*2
            nn.MaxPool1d(3, stride=3),

            Inception_C(inf=192, c0=64, c1=64, c2=32, c3=32),  # Inception*2
            Inception_A(inf=192, c0=64, c1=64, c2=32, c3=32),  # Inception*3
            nn.MaxPool1d(3, stride=3)
        )

        self.features2 = nn.Sequential(
            Inception_A(inf=160, c0=64, c1=64, c2=32, c3=32),  # Inception*3
            nn.MaxPool1d(3, stride=3),

            Inception_C(inf=256, c0=64, c1=64, c2=32, c3=32),  # Inception*2
            nn.MaxPool1d(2, stride=2)
        )

        self.features3 = nn.Sequential(
            Inception_B(inf=608, c0=64, c1=64, c2=32, c3=32),
            nn.BatchNorm1d(384, eps=0.001, momentum=0.1, affine=True),
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
        # print('x shape {}'.format(x.shape))
        x1 = self.features1(x)
        # print('x1 shape {}'.format(x1.shape))
        x2 = self.features2(x1)
        # print('x2 shape {}'.format(x2.shape))
        x3 = self.features3(x2)
        # print('x3 shape {}'.format(x3.shape))

        # x = self.logits(x)
        return x1, x2, x3

if __name__ == "__main__":
    I4 = InceptionV4(num_classes=16)
    test = torch.rand(16,2,500)
    print(I4(test))
