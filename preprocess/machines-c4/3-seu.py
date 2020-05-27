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
parser.add_argument('--dataroot', required=False, default='D:/fault/cdan-machines/matdata/seu-bearingset', help='where the data folder is')
parser.add_argument('--outputpath', required=False, default='seu', help='where the output folder is')
parser.add_argument('--framesize', type=int, required=False, default=2048, help='frame size')
parser.add_argument('--trainnumber', type=int, required=False, default=1000, help='how many samples each class for training dataset')
parser.add_argument('--normal', required=False, type=int, default=0, help='normal or not')
parser.add_argument('--fft', required=False, type=int, default=0, help='use fft or not')
args = parser.parse_args()

"""
    东南大学邵思雨轴承实验台
        https://github.com/cathysiyu/Mechanical-datasets
    工况：
        两种工况，20-0和30-2（转速-负载）
    故障类型和标签：
        Healthy：0
        IR：1
        OR：2
        Ball：3
        Combo：暂没用到
    故障深度：不区分故障深度
"""

def load_csvfile(filename, dataname, label, signal_size, samplenumber):
    f = open(filename, "r", encoding='gb18030', errors='ignore')
    fl = []
    if dataname == "ball_20_0.csv":
        for line in islice(f, 16, None):  #Skip the first 16 lines
            line = line.rstrip()
            word = line.split(",",8)   #Separated by commas
            fl.append(eval(word[1]))   # Take a vibration signal in the x direction as input
    else:
        for line in islice(f, 16, None):  #Skip the first 16 lines
            line = line.rstrip()
            word = line.split("\t",8)   #Separated by \t
            fl.append(eval(word[1]))   # Take a vibration signal in the x direction as input
    fl = np.array(fl)
    print("dataname {}, shape {}".format(dataname, fl.shape))
    # fl = fl.reshape(-1, 1)
    # print("dataname {}, shape {}".format(dataname, fl.shape))

    data=[] 
    lab=[]
    start, end = 0, signal_size
    # while end <= fl.shape[0] / 10:
    # while end <= fl.shape[0]:
    #     data.append(fl[start:end])
    #     lab.append(label)
    #     start += signal_size
    #     end += signal_size

    all_length = fl.shape[0]
    for index in range(samplenumber):
        random_start = np.random.randint(low=0 * signal_size, high=all_length - signal_size)
        sample = fl[random_start:random_start + signal_size]
        data.append(sample)
        lab.append(label)

    return data, lab

def load_seu_noaug(filelist, labellist):
    #path_to_data = 'D:/fault/cdan-machines/matdata/seu-bearingset'
    # path_to_data = '/nas/data/seu/bearingset2'
    path_to_data = args.dataroot

    data = []
    lab = []
    # for i in tqdm(range(len(filelist))):
    for i in range(len(filelist)):
        csvpath = os.path.join(path_to_data, filelist[i])
        print('load file {}'.format(csvpath))
        data1, lab1 = load_csvfile(csvpath, dataname=filelist[i], label=labellist[i], signal_size=args.framesize)
        data += data1
        lab += lab1

    return data, lab

def load_seu(filelist, labellist, samplenumber):
    #path_to_data = 'D:/fault/cdan-machines/matdata/seu-bearingset'
    # path_to_data = '/nas/data/seu/bearingset2'
    path_to_data = args.dataroot

    data = []
    lab = []
    # for i in tqdm(range(len(filelist))):
    for i in range(len(filelist)):
        csvpath = os.path.join(path_to_data, filelist[i])
        print('load file {}'.format(csvpath))
        data1, lab1 = load_csvfile(csvpath, dataname=filelist[i], label=labellist[i], signal_size=args.framesize, samplenumber=samplenumber)
        data += data1
        lab += lab1

    return data, lab

def shuffle_data(features, labels):
    rand_index = np.arange(len(features))
    np.random.shuffle(rand_index)
    features_shuffled = features[rand_index]
    labels_shuffled = labels[rand_index]
    return features_shuffled, labels_shuffled

def to_fft(data, axes):
    """ 对序列进行fft处理
    """
    dim_coef = int(np.floor(args.framesize / 2))
    print('dimension of fft coeffient to keep: {}'.format(dim_coef))
    re_fft_data = np.fft.fftn(data, axes=axes) 
    return abs(re_fft_data)[:, :dim_coef]

def draw_fig(data, labels, count, prefix, res_dir):
    data = np.array(data)
    labels = np.array(labels)
    for _label in np.unique(labels):
        # _index = labels[labels==_label]
        # _index = np.where(labels==_label)
        _feauture = data[labels == _label]
        for i in range(count):
            plt.plot(_feauture[i])
            plt.title('{}-index{}-label{}'.format(prefix, i, _label))
            plt.savefig('{}/{}-index{}-label{}.png'.format(res_dir, prefix, i, _label))
            plt.close()

if __name__ == "__main__":
    #Data names of 5 bearing fault types under two working conditions
    # Bdata = ["ball_20_0.csv","comb_20_0.csv","health_20_0.csv","inner_20_0.csv","outer_20_0.csv","ball_30_2.csv","comb_30_2.csv","health_30_2.csv","inner_30_2.csv","outer_30_2.csv"]

    filelist20 = ["ball_20_0.csv", "health_20_0.csv", "inner_20_0.csv", "outer_20_0.csv"]
    filelist30 = ["ball_30_2.csv", "health_30_2.csv", "inner_30_2.csv", "outer_30_2.csv"]
    # labellist = [0, 2, 1, 3]
    labellist = [2, 0, 1, 3]
    
    # HEALTH = 0
    # IR = 1
    # BALL = 2
    # OR = 3

    res_dir = 'seu20-fft{}-fs{}-num{}'.format(args.fft, args.framesize, args.trainnumber)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    features, labels = load_seu(filelist20, labellist, samplenumber=args.trainnumber)
    features = np.array(features)
    labels = np.array(labels)
    print("feature {}, label {}".format(features.shape, labels.shape))
    draw_fig(features, labels, count=10, prefix='fft0-norm0', res_dir=res_dir)

    if args.normal > 0:
        scaler = preprocessing.StandardScaler()
        features = scaler.fit_transform(features)
        draw_fig(features, labels, count=10, prefix='fft0-norm1', res_dir=res_dir)

    if args.fft == 1:
        features = to_fft(features, axes=(1,))
        print("after fft: feature {}, label {}".format(features.shape, labels.shape))
        draw_fig(features, labels, count=10, prefix='fft1-norm0', res_dir=res_dir)

    # scaler = preprocessing.MinMaxScaler()
    features_shuffled, labels_shuffled = shuffle_data(features, labels)

    np.save("{}/data_features_train.npy".format(res_dir), features_shuffled)
    np.save("{}/data_labels_train.npy".format(res_dir), labels_shuffled)
    print("files save to {}/data_features_train.npy".format(res_dir))


    #
    # seu30
    #
    res_dir = 'seu30-fft{}-fs{}-num{}'.format(args.fft, args.framesize, args.trainnumber)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    features, labels = load_seu(filelist30, labellist, samplenumber=args.trainnumber)
    features = np.array(features)
    labels = np.array(labels)
    print("feature {}, label {}".format(features.shape, labels.shape))
    draw_fig(features, labels, count=10, prefix='fft0-norm0', res_dir=res_dir)

    if args.normal > 0:
        scaler = preprocessing.StandardScaler()
        features = scaler.fit_transform(features)
        draw_fig(features, labels, count=10, prefix='fft0-norm1', res_dir=res_dir)

    if args.fft == 1:
        features = to_fft(features, axes=(1,))
        print("after fft: feature {}, label {}".format(features.shape, labels.shape))
        draw_fig(features, labels, count=10, prefix='fft1-norm0', res_dir=res_dir)

    features_shuffled, labels_shuffled = shuffle_data(features, labels)
    np.save("{}/data_features_train.npy".format(res_dir), features_shuffled)
    np.save("{}/data_labels_train.npy".format(res_dir), labels_shuffled)
    print("files save to {}/data_features_train.npy".format(res_dir))