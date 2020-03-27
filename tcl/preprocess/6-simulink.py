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

FRAME_SIZE = 5120           # 默认滑动窗口大小
STEP_SIZE = 5100            # 默认滑动窗口移动步长

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=False, default='/nas/data/paderborn', help='where the data folder is')
parser.add_argument('--framesize', type=int, required=False, default=FRAME_SIZE, help='Frame Size of sliding windows')
parser.add_argument('--stepsize', type=int, required=False, default=STEP_SIZE, help='Step Size of Sliding window')
parser.add_argument('--load', type=int, required=False, default=-1, help='Specify one of the load conditions (0/1/2/3). -1 = all')
parser.add_argument('--fft', required=False, type=int, default=0, help='use fft or not')
parser.add_argument('--normal', required=False, type=int, default=0, help='normal or not')
args = parser.parse_args()

def load_matfile(filepath, filename):
    """ 加载mat文件，按照FRAME_SIZE划分成序列
    """
    matlab_file = scipy.io.loadmat(filepath)

    # features = matlab_file[0][0][2][0][6][2][0]  #Take out the data
    features = matlab_file['axout']

    features = features[:, 0]

    # print('filename {}, feature shape {}'.format(filename, features.shape))
    return features


def feature_segment(filename, matdata, framesize, samplenumber):
    signal_begin = 0
    signal_len = matdata.shape[0]
    
    samples = []
    if samplenumber > 0:
        for i in range(samplenumber):
            random_start = np.random.randint(low=0 * framesize, high=signal_len-framesize)
            sample = matdata[random_start:random_start + framesize]
            samples.append(sample)
    else:
        while signal_begin + framesize < signal_len:
            samples.append(matdata[signal_begin:signal_begin+framesize])
            signal_begin += framesize

    sample_tensor = np.array(samples).astype('float32')
    # print("Load file {} into shape {}".format(filename, sample_tensor.shape))
    return sample_tensor

labels_map = {
    'IR007': 0,
    'IR014': 0,
    'IR021': 0,
    'OR007': 1,
    'OR014': 1,
    'OR021': 1,
    'Normal': 2}

loads_map = {
    'IR007': 0,
    'IR014': 0,
    'IR021': 0,
    'OR007': 0,
    'OR014': 0,
    'OR021': 0,
    'Normal': 0}

frames_per_class = {
    'IR007': 1000,
    'IR014': 1000,
    'IR021': 1000,
    'OR007': 1000,
    'OR014': 1000,
    'OR021': 1000,
    'Normal': 3000}


def get_labels_and_loads(filelist, labels_map, loads_map):
    """
    将filelist中的每个文件名，提取标签Label和工况Load，分别存放到列表中
    labels_map: 文件名与标签的对应关系
    loads_map:  文件名与工况的对应关系
    """

    labels_list = []
    loads_list = []
    for filename in filelist:
        labels_list.append(labels_map[filename])
        loads_list.append(loads_map[filename])
    return loads_list, labels_list

def calculate_frames_per_class(filename):
    n_samples = 0
    for key in frames_per_class.keys():
        if key in filename:
            n_samples = frames_per_class[key]
    return n_samples
        
def load_files_from_list(filepaths, filenames, labels_map, loads_map, framesize):
    """ 读取指定文件夹下所有数据文件

    dirpath: mat数据文件路径，忽略子文件夹
    filenames: mat数据的文件名
    labels_map: 类标的编号关系
    loads_map: 工况的编号关系
    framesize：样本的frame大小
    samplenumber：如果为0，则用滑动窗口方法生成样本，滑动窗口的大小等同于framesize。如果不为0，则随机采样samplenum条样本
    """

    total_features = {}
    total_labels = {}

    # 获取文件名对应的类标和工况
    loads_list, labels_list = get_labels_and_loads(filenames, labels_map, loads_map)

    for filepath, filename, load, label in zip(filepaths, filenames, loads_list, labels_list):
        # 加载mat文件
        matdata = load_matfile(filepath, filename)

        # 查询该文件应该生成多少个样本
        n_samples = calculate_frames_per_class(filename)

        # 生成长度为framesize，总数为n_samples的样本特征
        features = feature_segment(filename, matdata, framesize, samplenumber=n_samples)

        # 生成对应数量的标签
        labels = np.ones(features.shape[0], dtype=np.int8) * label

        # 将相同工况的样本合并
        if load in total_features.keys():
            total_features[load] = np.concatenate((total_features[load], features))
            total_labels[load] = np.concatenate((total_labels[load], labels))
        else:
            total_features[load] = features
            total_labels[load] = labels

    return total_features, total_labels, loads_list, labels_list


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

if __name__ == '__main__':
    filepaths = []  # 文件路径
    filenames = []  # 去掉后缀的文件名
    subdirs = []

    for filename in os.listdir(args.dataroot):
        if not filename.endswith('.mat'):
            continue
        filenames.append(filename[:-4])
        filepaths.append(os.path.join(args.dataroot, filename))

    print('filepaths len {}'.format(len(filepaths)))
    total_features, total_labels, labels_dict, labels_list = load_files_from_list(filepaths, filenames, labels_map, loads_map, framesize=args.framesize)

    #print('total_features {}'.format(total_features.keys()))
    #print('total_labels {}'.format(total_labels.keys()))
    #print('labels_dict {}'.format(labels_dict))
    #print('labels_list {}'.format(labels_list))
    
    features_prune = {}
    labels_prune = {}
    features_prune_test = {}
    labels_prune_test = {}
    for load in total_features.keys():
        res_dir = "./data/simulink_fft{}_frame{}/load{}".format(args.fft, args.framesize, load)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        if args.fft == 1:
            features_prune[load] = to_fft(total_features[load], axes=(1,))
            labels_prune[load] = total_labels[load]
        else:
            features_prune[load] = total_features[load]
            labels_prune[load] = total_labels[load]

        # 归一化
        # 按照论文ACDIN，归一化不是全局进行的，是在每个batch里做的
        #if args.normal > 0:
        #    features_prune[load] = do_scalar(features_prune[load], args.normal)
        #    draw_fig(features_prune[load], labels_prune[load], count=1, prefix='fft0-norm1', res_dir=res_dir)

        # 乱序
        rand_index = np.arange(len(features_prune[load]))
        np.random.shuffle(rand_index)
        features = features_prune[load][rand_index]
        labels = labels_prune[load][rand_index]

        # 保存
        np.save("{}/data_features_train.npy".format(res_dir), features)
        np.save("{}/data_labels_train.npy".format(res_dir), labels)
        print('load {} saved to {}, features {}, labels {}'.format(load, res_dir, features.shape, labels.shape))
