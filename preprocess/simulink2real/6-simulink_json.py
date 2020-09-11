"""
"""

import os
import re
import sys
import math
import json
import argparse
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
from sklearn import preprocessing
import configparser

parser = argparse.ArgumentParser()
parser.add_argument('--configfile', required=False, default='default_cfg.json', help='the config file')
# parser.add_argument('--dataroot', required=False, default='D:\\code\\3rcgan\\RACGAN-upload\\matdata\\simulink_6205_0626_processed\\Fr1000_Ax', help='where the data folder is')
parser.add_argument('--framesize', type=int, required=False, default=1200, help='Frame Size of sliding windows')
parser.add_argument('--fft', required=False, type=int, default=0, help='use fft or not')
parser.add_argument('--normal', required=False, type=int, default=0, help='normal or not')
args = parser.parse_args()

normalize_factor = {
    'IR007': 10.0,
    'IR014': 10.0,
    'IR021': 10.0,
    'OR007': 20.0,
    'OR014': 20.0,
    'OR021': 20.0,
    'Normal007': 1.0,
    'Normal014': 1.0,
    'Normal021': 1.0}

debug = True
def PRINT_DEBUG(infos):
    if debug == True:
        print(infos)

def load_matfile(filepath, filename):
    matlab_file = scipy.io.loadmat(filepath)
    if 'axout' in matlab_file.keys():
        features = matlab_file['axout']
    elif 'azout' in matlab_file.keys():
        features = matlab_file['azout']
    else:
        print('cannot read matfile')

    features = features[:, 0]
    PRINT_DEBUG('Load file {}, feature shape {}'.format(filename, features.shape))
    return features

def feature_segment(filename, matdata, framesize, samplenumber):
    signal_begin = 0
    signal_len = matdata.shape[0]
    
    samples = []
    if samplenumber > 0:
        for i in range(samplenumber):
            random_start = np.random.randint(low=1 * framesize, high=signal_len-framesize)
            sample = matdata[random_start:random_start + framesize]
            samples.append(sample)
    else:
        while signal_begin + framesize < signal_len:
            samples.append(matdata[signal_begin:signal_begin+framesize])
            signal_begin += framesize

    sample_tensor = np.array(samples).astype('float32')
    if debug:
        print("Segment file {} into shape {}".format(filename, sample_tensor.shape))
    return sample_tensor

def to_fft(data, axes):
    """ 对序列进行fft处理
    """
    dim_coef = int(np.floor(args.framesize / 2))
    print('dimension of fft coeffient to keep: {}'.format(dim_coef))
    re_fft_data = np.fft.fftn(data, axes=axes) 
    return abs(re_fft_data)[:, :dim_coef]

def do_scalar(features, normal):
    if normal == 1:
        scalar = preprocessing.StandardScaler()
    elif normal == 2:
        scalar = preprocessing.MinMaxScaler()
    Train_X = scalar.fit_transform(features)
    return Train_X

def draw_fig(data, labels, count, prefix, res_dir):
    data = np.array(data)
    labels = np.array(labels)
    for _label in np.unique(labels):
        _feauture = data[labels == _label]
        for i in range(count):
            plt.plot(_feauture[i])
            plt.title('{}-index{}-label{}'.format(prefix, i, _label))
            plt.savefig('{}/{}-index{}-label{}.png'.format(res_dir, prefix, i, _label))
            plt.close()

def read_files(dataset_path, files_match, labels_dict, frames_dict, framesize):
    total_features = []
    total_labels = []

    for filename in files_match:
        if filename not in labels_dict:
            print('filename {} does not has label information'.format(filename))
            raise Exception('filename {} does not has label information'.format(filename))

        # 加载mat文件
        matdata = load_matfile(os.path.join(dataset_path, filename), filename)
        # 查询该文件应该生成多少个样本
        n_samples = int(frames_dict[filename])
        # 生成长度为framesize，总数为n_samples的样本特征
        features = feature_segment(filename, matdata, framesize, samplenumber=n_samples)
        # 生成对应数量的标签
        labels = np.ones(features.shape[0], dtype=np.int8) * int(labels_dict[filename])
        PRINT_DEBUG('file {} with label {}, features {}, labels {}'.format(filename, labels_dict[filename], features.shape, labels.shape))

        total_features.append(features)
        total_labels.append(labels)

    total_features = np.concatenate(total_features)
    total_labels = np.concatenate(total_labels)
    PRINT_DEBUG('total_features {}, total_labels {}'.format(total_features.shape, total_labels.shape))
    return total_features, total_labels


if __name__ == '__main__':
    cf = configparser.ConfigParser()
    # config_file = os.path.join(args.dataroot, args.configfile)
    print('read config file {}'.format(args.configfile))
    cf.read(args.configfile)

    dataset_name = cf['datasets']['name']
    dataset_path = cf['datasets']['path']
    labels_dict = dict({label[0] : label[1] for label in cf.items('labels')})
    loads_dict = dict({load[0] : load[1] for load in cf.items('loads')})
    frames_dict = dict({frame[0] : frame[1] for frame in cf.items('frames')})
    
    for load in np.unique(list(loads_dict.values())):
        files_match = []
        for filename in loads_dict:
            if loads_dict[filename] == load:
                files_match.append(filename)
        print('load {} has {} files {}'.format(load, len(files_match), files_match))

        features, labels = read_files(dataset_path, files_match, labels_dict, frames_dict, framesize=args.framesize)

        if args.fft == 1:
            features_prune = to_fft(features, axes=(1,))

        res_dir = "{}_load{}_fft{}_frame{}_normal{}".format(dataset_name, load, args.fft, args.framesize, args.normal)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        # 乱序
        rand_index = np.arange(len(features))
        np.random.shuffle(rand_index)
        if args.fft == 1:
            features = features_prune[rand_index]
        else:
            features = features[rand_index]
        labels = labels[rand_index]

        # 保存
        np.save("{}/data_features_train.npy".format(res_dir), features)
        np.save("{}/data_labels_train.npy".format(res_dir), labels)
        print('load {} saved to {}, features {}, labels {}'.format(load, res_dir, features.shape, labels.shape))