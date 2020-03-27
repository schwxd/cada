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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=False, default='F:/ds/IMS', help='where the data folder is')
parser.add_argument('--outputpath', required=False, default='ims', help='where the output folder is')
parser.add_argument('--framesize', type=int, required=False, default=2048, help='frame size')
parser.add_argument('--trainnumber', type=int, required=False, default=2000, help='number of training samples')
parser.add_argument('--fft', required=False, type=int, default=0, help='use fft or not')
parser.add_argument('--normal', required=False, type=int, default=0, help='normal or not')
parser.add_argument('--severity', required=False, default='incipient', help='1: incipient, 2: medium, 3: severe')

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

# 跟cwru保持一致
HEALTH = 2
IR = 1
OR = 3
BALL = 0

# labels1 = {'B1a':-1,'B1b':-1,'B2a':-1,'B2b':-1, 'B3a':IR,'B3b':-1,'B4a':BALL,'B4b':-1}
# labels1 = [-1, -1, -1, -1, IR, -1, BALL, -1]
labels1 = [-1, -1, -1, -1, -1, IR, -1, BALL]
# 总文件个数2156个
count_begin1 = {'incipient':1500, 'medium':1700, 'severe':1900}

# labels2 = {'B1':OR,'B2':-1,'B3':-1,'B4':HEALTH}
labels2 = [OR, -1, -1, HEALTH]
# 总文件个数985个
count_begin2 = {'incipient':300, 'medium':500, 'severe':700}

# labels3 = {'B1':-1,'B2':-1,'B3':OR,'B4':-1}
labels3 = [-1, -1, OR, HEALTH]
# 总文件个数6325个
count_begin3 = {'incipient':5700, 'medium':5900, 'severe':6100}

FILES_PER_CLASS = 200

def get_data2(path_to_data, data_dict, severity):
    labels = labels1
    if  '1st_test' in path_to_data:
        labels = labels1
        count_begin = count_begin1[severity]
        txtname = '1st_test'

    elif '2nd_test' in path_to_data:
        labels = labels2
        count_begin = count_begin2[severity]
        txtname = '2nd_test'

    elif '3rd_test' in path_to_data:
        labels = labels3
        count_begin = count_begin3[severity]
        txtname = '3rd_test'

    print('path {}, labels {}, severity {}, count_begin {}'.format(path_to_data, labels, severity, count_begin))

    files_loaded = []
    merged_data = []
    count = 0
    for filename in os.listdir(path_to_data):
        if count < count_begin:
            count += 1
            continue
        if count > count_begin + FILES_PER_CLASS:
            break

        # dataset = np.loadtxt(os.path.join(path_to_data, filename), dtype=int, delimiter='\t')
        dataset = np.loadtxt(os.path.join(path_to_data, filename), delimiter='\t')
        files_loaded.append(filename)
        # print('dataset type {}, shape {}'.format(type(dataset), dataset.shape))
        merged_data.append(dataset)
        # if count % 10 == 0:
            # print('{} files loaded'.format(count))
        # if count >= COUNT_END:
        #     break
        count += 1

    merged_data = np.concatenate(merged_data, axis=0)
    print('merged_data type {}, shape {}'.format(type(merged_data), merged_data.shape))
    for column in range(merged_data.shape[1]):
        if labels[column] >= 0:
            data_dict[labels[column]] = merged_data[:, column]
            print('take column {} as label {}'.format(column, labels[column]))

    print(data_dict.keys())
    # 保存加载了哪些文件，用于校验
    #with open('{}-{}.txt'.format(txtname, count_begin),"w") as f:
    #    for content in files_loaded:
    #        f.write(content)
    #        f.write('\n')


def segment(data_dict, framesize, trainnumber):

    features = []
    labels = []

    for key in data_dict.keys():

        signal_begin = 0
        signal_len = len(data_dict[key])
        
        # print('key {}, signal_len {}'.format(key, signal_len))
        feature = []
        label = []
        '''
        while signal_begin + frame_size < signal_len:
            feature.append(data_dict[key][signal_begin:signal_begin+frame_size])
            signal_begin += step_size
            label.append(key)
        '''
        for j in range(trainnumber):
            random_start = np.random.randint(low=0 * framesize, high=signal_len-framesize)
            sample = data_dict[key][random_start:random_start + framesize]
            feature.append(sample)
            label.append(key)


        print('key {}, total length {}, feature {}, label {}'.format(key, signal_len, len(feature), len(label)))
        features.append(feature)
        labels.append(label)

    # features = np.array(features).astype('float32')
    # labels = np.array(labels)
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    print("features {}, labels {}".format(features.shape, labels.shape))
    return features, labels


def split_data(data, train_fraction=0.5):
    split_index = str(data.index[int(len(data)*train_fraction)])
    X_train = data[:split_index]
    X_test = data[split_index:]
    
    scaler = preprocessing.MinMaxScaler()

    X_train = pd.DataFrame(scaler.fit_transform(X_train), 
                                  columns=X_train.columns, 
                                  index=X_train.index)

    X_train.sample(frac=1)

    X_test = pd.DataFrame(scaler.transform(X_test), 
                                 columns=X_test.columns, 
                                 index=X_test.index)
    return X_train, X_test


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

def do_scalar(features, normal):
    print('do_scalar normal type {}'.format(normal))
    if normal == 1:
        scalar = preprocessing.StandardScaler()
    elif normal == 2:
        scalar = preprocessing.MinMaxScaler()
    features_scaled = scalar.fit_transform(features)
    return features_scaled 

def get_ims(dataroot, severity):
    res_dir = 'ims-{}-fft{}-normal{}-fs{}-num{}'.format(severity, args.fft, args.normal, args.framesize, args.trainnumber)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    data_dict = {}

    # 加载文件
    get_data2(path_to_data=os.path.join(dataroot, '1st_test'), data_dict=data_dict, severity=severity)
    # get_data2(path_to_data='F:/ds/IMS/2nd_test', data_dict=data_dict, count_begin=600, count_end=800)
    get_data2(path_to_data=os.path.join(dataroot, '3rd_test'), data_dict=data_dict, severity=severity)

    # 按照framesize切分
    features, labels = segment(data_dict=data_dict, framesize=args.framesize, trainnumber=args.trainnumber)
    draw_fig(features, labels, count=1, prefix='raw', res_dir=res_dir)
    if args.normal > 0:
        features = do_scalar(features, args.normal)
        draw_fig(features, labels, count=1, prefix='normal{}'.format(args.normal), res_dir=res_dir)
    if args.fft == 1:
        features = to_fft(features, axes=(1,))
        draw_fig(features, labels, count=1, prefix='normal{}-fft1'.format(args.normal), res_dir=res_dir)


    features_shuffled, labels_shuffled = shuffle_data(features, labels)

    np.save("{}/data_features_train.npy".format(res_dir), features_shuffled)
    np.save("{}/data_labels_train.npy".format(res_dir), labels_shuffled)
    print("files save to {}/data_features_train.npy".format(res_dir))

if __name__ == "__main__":

    get_ims(dataroot=args.dataroot, severity='incipient')
    get_ims(dataroot=args.dataroot, severity='medium')
    get_ims(dataroot=args.dataroot, severity='severe')

