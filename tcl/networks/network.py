import numpy as np
import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.autograd import Function

from utils.functions import ReverseLayerF

class Extractor(nn.Module):
    def __init__(self, n_flattens, n_hiddens, bn=False):
        super(Extractor, self).__init__()
        # feature之后的维度
        self.n_flattens = n_flattens
        # 全连接层的hidden size
        self.n_hiddens = n_hiddens

        self.feature = nn.Sequential()

        # features 使用1-D CNN提取特征
        self.feature.add_module('f_conv1', nn.Conv1d(1, 8, kernel_size=64, stride=2))
        if bn:
            self.feature.add_module('f_bn1', nn.BatchNorm1d(8))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_pool1', nn.MaxPool1d(kernel_size=2, stride=2))
        self.feature.add_module('f_drop1', nn.Dropout(0.5))

        self.feature.add_module('f_conv2', nn.Conv1d(8, 16, kernel_size=16, stride=2))
        if bn:
            self.feature.add_module('f_bn2', nn.BatchNorm1d(16))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_pool2', nn.MaxPool1d(kernel_size=2, stride=2))
        self.feature.add_module('f_drop2', nn.Dropout(0.5))

        self.feature.add_module('f_conv3', nn.Conv1d(16, 32, kernel_size=8, stride=2))
        if bn:
            self.feature.add_module('f_bn3', nn.BatchNorm1d(32))
        self.feature.add_module('f_relu3', nn.ReLU(True))
        self.feature.add_module('f_pool3', nn.MaxPool1d(kernel_size=2, stride=2))
        self.feature.add_module('f_drop3', nn.Dropout(0.5))

        self.feature.add_module('f_conv4', nn.Conv1d(32, 64, kernel_size=3, stride=2))
        if bn:
            self.feature.add_module('f_bn4', nn.BatchNorm1d(64))
        self.feature.add_module('f_relu4', nn.ReLU(True))
        self.feature.add_module('f_pool4', nn.MaxPool1d(kernel_size=2, stride=2))
        self.feature.add_module('f_drop4', nn.Dropout(0.5)) 

        self.feature.add_module('f_conv5', nn.Conv1d(64, 128, kernel_size=3, stride=2))
        if bn:
            self.feature.add_module('f_bn5', nn.BatchNorm1d(128))
        self.feature.add_module('f_relu5', nn.ReLU(True))
        self.feature.add_module('f_pool5', nn.MaxPool1d(kernel_size=2, stride=2))
        self.feature.add_module('f_drop5', nn.Dropout(0.5))

        #self.feature.add_module('f_conv6', nn.Conv1d(128, 256, kernel_size=3, stride=2))
        #self.feature.add_module('f_bn6', nn.BatchNorm1d(32))
        #self.feature.add_module('f_relu6', nn.ReLU(True))
        #self.feature.add_module('f_pool6', nn.MaxPool1d(kernel_size=2, stride=2))
        #self.feature.add_module('f_drop6', nn.Dropout(0.2))


    def extract_feature(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return x

    def dann(self, input_data, alpha):
        feature = self.feature(input_data)
        feature = feature.view(feature.size(0), -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output, feature

    def forward(self, x):
        x = self.feature(x)
        output = x.view(x.size(0), -1)
        return output


class Classifier(nn.Module):
    def __init__(self, n_flattens, n_hiddens, n_class, bn=False):
        super(Classifier, self).__init__()
        # feature之后的维度
        self.n_flattens = n_flattens
        # 全连接层的hidden size
        self.n_hiddens = n_hiddens
        # 分类器输出的类别个数
        self.n_class = n_class

        self.class_classifier = nn.Sequential()

        # class_classifier 使用全连接层进行分类
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.n_flattens, self.n_hiddens))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout(0.5))
        self.class_classifier.add_module('c_fc2', nn.Linear(self.n_hiddens, self.n_hiddens))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_drop2', nn.Dropout(0.5))
        self.class_classifier.add_module('c_fc3', nn.Linear(self.n_hiddens, self.n_class))

    def forward(self, x):
        if isinstance(x, tuple):
            x1, x2, x3 = x
            x = x3
        x = x.view(x.size(0), -1)
        output = self.class_classifier(x)
        return output



# class ClassifierAux(nn.Module):
#     def __init__(self, n_flattens, n_hiddens, n_class, bn=False):
#         super(ClassifierAux, self).__init__()
#         self.n_flattens = n_flattens
#         self.n_hiddens = n_hiddens
#         self.n_class = n_class

#         self.features1 = nn.Sequential(
#             nn.Linear(10080, 1000),
#             nn.ReLU(True),
#             nn.Dropout(0.5)
#         )
#         self.features2 = nn.Sequential(
#             nn.Linear(6080, 1000),
#             nn.ReLU(True),
#             nn.Dropout(0.5)
#         )
#         self.features3 = nn.Sequential(
#             nn.Linear(1920, 1000),
#             nn.ReLU(True),
#             nn.Dropout(0.5)
#         )
#         self.features = nn.Sequential(
#             nn.Linear(3000, 500),
#             nn.ReLU(True),
#             nn.Dropout(0.5),
#             nn.Linear(500, self.n_class)
#         )

#     # def forward(self, x1, x2, x3):
#     def forward(self, sets):
#         x1, x2, x3 = sets
#         x1 = x1.view(x1.size(0), -1)
#         x2 = x2.view(x2.size(0), -1)
#         x3 = x3.view(x3.size(0), -1)
#         output1 = self.features1(x1)
#         output2 = self.features2(x2)
#         output3 = self.features3(x3)
#         output_all = torch.cat((output1, output2, output3), 1)
#         output = self.features(output_all)
#         return output



# class ClassifierAux(nn.Module):
#     def __init__(self, n_flattens, n_hiddens, n_class, bn=False):
#         super(ClassifierAux, self).__init__()
#         self.n_flattens = n_flattens
#         self.n_hiddens = n_hiddens
#         self.n_class = n_class

#         # self.features1 = nn.Sequential(
#         #     nn.Linear(10080, 1000),
#         #     nn.ReLU(True),
#         #     nn.Dropout(0.5)
#         # )
#         # self.features2 = nn.Sequential(
#         #     nn.Linear(6080, 1000),
#         #     nn.ReLU(True),
#         #     nn.Dropout(0.5)
#         # )
#         # self.features3 = nn.Sequential(
#         #     nn.Linear(1920, 1000),
#         #     nn.ReLU(True),
#         #     nn.Dropout(0.5)
#         # )
#         # self.features = nn.Sequential(
#         #     nn.Linear(3000, 500),
#         #     nn.ReLU(True),
#         #     nn.Dropout(0.5),
#         #     nn.Linear(500, self.n_class)
#         # )
#         self.features = nn.Sequential(
#             nn.Linear(18080, 3000),
#             nn.ReLU(True),
#             nn.Dropout(0.5),
#             nn.Linear(3000, 500),
#             nn.ReLU(True),
#             nn.Dropout(0.5),
#             nn.Linear(500, self.n_class)
#         )

#     # def forward(self, x1, x2, x3):
#     def forward(self, sets):
#         x1, x2, x3 = sets
#         x1 = x1.view(x1.size(0), -1)
#         x2 = x2.view(x2.size(0), -1)
#         x3 = x3.view(x3.size(0), -1)
#         x = torch.cat((x1, x2, x3), 1)
#         output = self.features(x)

#         # output1 = self.features1(x1)
#         # output2 = self.features2(x2)
#         # output3 = self.features3(x3)
#         # output_all = torch.cat((output1, output2, output3), 1)
#         # output = self.features(output_all)
#         return output



class ClassifierAux(nn.Module):
    def __init__(self, n_flattens, n_hiddens, n_class, bn=False):
        super(ClassifierAux, self).__init__()
        self.n_flattens = n_flattens
        self.n_hiddens = n_hiddens
        self.n_class = n_class

        self.features1 = nn.Sequential(
            nn.Linear(10080, 1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000, 100),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(100, self.n_class)            
        )
        self.features2 = nn.Sequential(
            nn.Linear(6080, 1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000, 100),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(100, self.n_class)   
        )
        self.features3 = nn.Sequential(
            nn.Linear(1920, 1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000, 100),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(100, self.n_class)   
        )
        # self.features = nn.Sequential(
        #     nn.Linear(3000, 500),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Linear(500, self.n_class)
        # )
        # self.features = nn.Sequential(
        #     nn.Linear(18080, 3000),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Linear(3000, 500),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Linear(500, self.n_class)
        # )

    # def forward(self, x1, x2, x3):
    def forward(self, sets):
        x1, x2, x3 = sets
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        output1 = self.features1(x1)
        output2 = self.features2(x2)
        output3 = self.features3(x3)
        # output = torch.cat((output1, output2, output3), 1)
        output = output1 + output2 + output3
        return output

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

class Predictor(nn.Module):
    def __init__(self, n_flattens, n_hiddens, num_class, temp=0.05):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(n_flattens, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        return x_out

class Predictor_deep(nn.Module):
    def __init__(self, n_flattens, n_hiddens, num_class, temp=0.05):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Linear(n_flattens, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x_out


class ClassifierSep(nn.Module):
    def __init__(self, n_flattens, n_hiddens, n_class, bn=False):
        super(ClassifierSep, self).__init__()
        # feature之后的维度
        self.n_flattens = n_flattens
        # 全连接层的hidden size
        self.n_hiddens = n_hiddens
        # 分类器输出的类别个数
        self.n_class = n_class

        # self.class_classifier = nn.Sequential()

        # class_classifier 使用全连接层进行分类
        # self.class_classifier = nn.Sequential()
        # self.class_classifier.add_module('c_fc1', nn.Linear(self.n_flattens, self.n_hiddens))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout(0.2))
        # self.class_classifier.add_module('c_fc2', nn.Linear(self.n_hiddens, self.n_hiddens))
        # self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop2', nn.Dropout(0.2))
        # self.class_classifier.add_module('c_fc3', nn.Linear(self.n_hiddens, self.n_class))

        self.layer1 = nn.Linear(self.n_flattens, self.n_hiddens)
        self.layer2 = nn.Linear(self.n_hiddens, self.n_hiddens)
        self.layer3 = nn.Linear(self.n_hiddens, self.n_class)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # output = self.class_classifier(x)

        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.layer3(x)
        return y

    def get_layer1(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        return x

    def get_layer2(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return x

    def get_layer3(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.layer3(x)
        return y

class Critic(nn.Module):
    def __init__(self, n_flattens, n_hiddens, bn=False):
        super(Critic, self).__init__()
        self.n_flattens = n_flattens
        self.n_hiddens = n_hiddens

        self.domain_critic = nn.Sequential()
        # 用于w距离
        self.domain_critic = nn.Sequential()
        self.domain_critic.add_module('dc_fc1', nn.Linear(self.n_flattens, self.n_hiddens))
        self.domain_critic.add_module('dc_relu1', nn.ReLU(True))
        self.domain_critic.add_module('dc_drop1', nn.Dropout(0.2))
        self.domain_critic.add_module('dc_fc2', nn.Linear(self.n_hiddens, self.n_hiddens))
        self.domain_critic.add_module('dc_relu2', nn.ReLU(True))
        self.domain_critic.add_module('dc_drop2', nn.Dropout(0.2))
        self.domain_critic.add_module('dc_fc3', nn.Linear(self.n_hiddens, 1))

    def forward(self, x):
        output = self.domain_critic(x)
        return output


class Critic2(nn.Module):
    def __init__(self, n_flattens, n_hiddens, bn=False):
        super(Critic2, self).__init__()
        self.n_flattens = n_flattens
        self.n_hiddens = n_hiddens

        self.domain_critic = nn.Sequential()
        # 用于w距离
        self.domain_critic = nn.Sequential()
        self.domain_critic.add_module('dc_fc1', nn.Linear(self.n_flattens, self.n_hiddens))
        self.domain_critic.add_module('dc_relu1', nn.ReLU(True))
        self.domain_critic.add_module('dc_drop1', nn.Dropout(0.2))
        self.domain_critic.add_module('dc_fc2', nn.Linear(self.n_hiddens, self.n_hiddens))
        self.domain_critic.add_module('dc_relu2', nn.ReLU(True))
        self.domain_critic.add_module('dc_drop2', nn.Dropout(0.2))
        self.domain_critic.add_module('dc_fc3', nn.Linear(self.n_hiddens, 2))

    def forward(self, x):
        output = self.domain_critic(x)
        return output

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1    

class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    # self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0

  def forward(self, x, gamma=1, training=True):
    if self.training:
        self.iter_num += 1
    # coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    coeff = gamma
    x = x * 1.0
    # print('coeff {}, x {}'.format(coeff, x[:2]))
    if training:
        x.register_hook(grl_hook(coeff))
    # print('after hook: coeff {},  x {}'.format(coeff, x[:2]))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    # print('y {}'.format(y[:2]))
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]
