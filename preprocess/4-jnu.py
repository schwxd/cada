# Common imports
import os
import sys
import re
import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from numpy.random import seed
from tqdm import tqdm

from itertools import islice

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=False, default='D:/fault/cdan-machines/matdata/jnu-bearingset', help='where the data folder is')
parser.add_argument('--outputpath', required=False, default='ims', help='where the output folder is')
parser.add_argument('--framesize', type=int, required=False, default=2048, help='frame size')
parser.add_argument('--trainnumber', type=int, required=False, default=1000, help='how many samples each class for training dataset')
parser.add_argument('--normal', required=False, type=int, default=0, help='normal or not')
parser.add_argument('--fft', required=False, type=int, default=0, help='use fft or not')
args = parser.parse_args()


"""
故障情况：
    1st 的轴承3：IR
    1st 的轴承4：Ball
    2nd 的轴承1：OR
    3rd 的轴承3：OR
    其他的是正常
采样频率：20kHz
转速：2000rpm
最小的frame length：60/2000 * 2 * 20000 = 1200
"""

"""
    标签：
    Healthy：0
    IR：1
    OR：2
    Ball：3
"""

def load_txtfile(txtfile, dataname, label, signal_size, samplenumber):
    feature = np.loadtxt(txtfile)
    print("dataname {}, shape {}".format(dataname, feature.shape))

    data = [] 
    lab = []
    all_length = feature.shape[0]
    for _ in range(samplenumber):
        random_start = np.random.randint(low=0 * signal_size, high=all_length - signal_size)
        sample = feature[random_start:random_start + signal_size]
        data.append(sample)
        lab.append(label)

    return data, lab


def load_nju(filelist, labellist, samplenumber):
    #path_to_data = 'D:/fault/cdan-machines/matdata/jnu-bearingset'
    path_to_data = '/nas/data/jnu'
    data = []
    lab = []

    for i in range(len(filelist)):
        csvpath = os.path.join(path_to_data, filelist[i])
        print('load file {}'.format(csvpath))
        data1, lab1 = load_txtfile(csvpath, dataname=filelist[i], label=labellist[i], signal_size=args.framesize, samplenumber=samplenumber)
        data += data1
        lab += lab1

    return data, lab



def shuffle_data(features, labels):
    rand_index = np.arange(len(features))
    np.random.shuffle(rand_index)
    features_shuffled = features[rand_index]
    labels_shuffled = labels[rand_index]
    return features_shuffled, labels_shuffled

if __name__ == "__main__":
    filelist6 = ['ib600_2.csv', 'n600_3_2.csv', 'ob600_2.csv', 'tb600_2.csv']
    filelist8 = ['ib800_2.csv', 'n800_3_2.csv', 'ob800_2.csv', 'tb800_2.csv']
    filelist10 = ['ib1000_2.csv', 'n1000_3_2.csv', 'ob1000_2.csv', 'tb1000_2.csv']
    labellist = [1, 2, 3, 0]

    
    # 1
    features, labels = load_nju(filelist6, labellist, samplenumber=2000)
    features = np.array(features)
    labels = np.array(labels)
    print("feature {}, label {}".format(features.shape, labels.shape))
    res_dir = 'nju6_{}'.format(args.framesize)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    if args.normal > 0:
        scaler = preprocessing.MinMaxScaler()
        features = scaler.fit_transform(features)
    features_shuffled, labels_shuffled = shuffle_data(features, labels)
    np.save("{}/data_features_train.npy".format(res_dir), features_shuffled)
    np.save("{}/data_labels_train.npy".format(res_dir), labels_shuffled)
    print("files save to {}/data_features_train.npy".format(res_dir))


    # 2
    features, labels = load_nju(filelist8, labellist, samplenumber=args.trainnumber)
    features = np.array(features)
    labels = np.array(labels)
    print("feature {}, label {}".format(features.shape, labels.shape))
    res_dir = 'nju8_{}'.format(args.framesize)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    if args.normal > 0:
        scaler = preprocessing.MinMaxScaler()
        features = scaler.fit_transform(features)
    features_shuffled, labels_shuffled = shuffle_data(features, labels)
    np.save("{}/data_features_train.npy".format(res_dir), features_shuffled)
    np.save("{}/data_labels_train.npy".format(res_dir), labels_shuffled)
    print("files save to {}/data_features_train.npy".format(res_dir))


    # 3
    features, labels = load_nju(filelist10, labellist, samplenumber=args.trainnumber)
    features = np.array(features)
    labels = np.array(labels)
    print("feature {}, label {}".format(features.shape, labels.shape))
    res_dir = 'nju10_{}'.format(args.framesize)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    if args.normal > 0:
        scaler = preprocessing.MinMaxScaler()
        features = scaler.fit_transform(features)
    features_shuffled, labels_shuffled = shuffle_data(features, labels)
    np.save("{}/data_features_train.npy".format(res_dir), features_shuffled)
    np.save("{}/data_labels_train.npy".format(res_dir), labels_shuffled)
    print("files save to {}/data_features_train.npy".format(res_dir))
