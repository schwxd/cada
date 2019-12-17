"""
http://groups.uni-paderborn.de/kat/BearingDataCenter/
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

FRAME_SIZE = 5120           # 滑动窗口大小
STEP_SIZE = 5100            # 滑动窗口移动步长

NUM_PER_CLASS_TRAIN = 4000 
NUM_PER_CLASS_TEST = 1000
 
parser = argparse.ArgumentParser()
parser.add_argument('--framesize', type=int, required=False, default=FRAME_SIZE, help='Frame Size of sliding windows')
parser.add_argument('--stepsize', type=int, required=False, default=STEP_SIZE, help='Step Size of Sliding window')
parser.add_argument('--methods', required=False, default='raw1d', help='preprocess method, raw1d or fft')
parser.add_argument('--normalize', type=int, required=False, default=0, help='normalize (1) or not (0)')
parser.add_argument('--randn', type=int, required=False, default=0, help='random number')

args = parser.parse_args()


# 类编号
labels_map = {
    "K001": 0,
    "K002": 0,
    "K003": 0,
    "K004": 0,
    "K005": 0,
    "KA04": 1,
    "KA15": 1,
    "KA16": 1,
    "KA22": 1,
    "KA30": 1,
    "KI04": 2,
    "KI14": 2,
    "KI16": 2,
    "KI18": 2,
    "KI21": 2}

# 工况编号
loads_map = {
    'N09_M07_F10' : 0,
    'N15_M01_F10' : 1,
    'N15_M07_F04' : 2,
    'N15_M07_F10' : 3
}

def parse_paderborn_matdata(filepath):
    """ 读取paderborn数据集mat文件，将振动信号及关键参数保存至dict，方便后续使用
        filepath: mat文件的绝对路径
        参考实现: https://github.com/ddrrrr/bearing-fault-diagnosis
    """

    a = scipy.io.loadmat(filepath)
    filedir, file_name = os.path.split(filepath)
    file_name = file_name.replace('.mat', '')
    a = a[file_name]
    a = a['Y']
    for _ in range(3):
        a = a[0]
    file_data = a[6]
    '''
        0 for force 16005
        1 for phase_current_1 256070
        2 for phase_current_2 256070
        3 for speed 16005
        4 for temp_2_bearing_module 5
        5 for torque 16005
        6 for vibration signal 256070
    '''
    file_data = file_data[2]
    file_data = file_data[0]
    dict_file_data = {}
    dict_file_data['name'] = file_name
    dict_file_data['speed'] = file_name[0:3]
    dict_file_data['bearing'] = file_name[12:16]
    dict_file_data['sample_rate'] = '64'
    dict_file_data['load'] = file_name[4:11]
    dict_file_data['data'] = file_data

    return dict_file_data


def load_matfile(filename, frame_size=args.framesize, step_size=args.stepsize):
    """ 加载mat文件，按照FRAME_SIZE划分成序列

    """
    matlab_file = parse_paderborn_matdata(filename)

    signal_begin = 0
    signal_len = len(matlab_file['data'])
    
    DE_samples = []
    while signal_begin + frame_size < signal_len:
        DE_samples.append(matlab_file['data'][signal_begin:signal_begin+frame_size])
        signal_begin += step_size

    sample_tensor = np.array(DE_samples).astype('float32')
    #print("Load file {} into shape {}".format(filename, sample_tensor.shape))
    return sample_tensor

def get_labels_and_loads(filelist, ignore_load = True):
    """
    生成工况列表和标签列表（与filelist一一对应）
    """

    labels_list = []        # 每个文件的原始label，用于绘制图形时标注文件名
    loads_list = []
    for filepath in filelist:
        filedir, filename = os.path.split(filepath)
        # print('filedir {}, filename {}'.format(filedir, filename))

        load = filename[:11]
        label = filename[12:16]
        ##print('filename {}, load {}, label {}'.format(filename, load, label))

        labels_list.append(labels_map[label])
        loads_list.append(loads_map[load])
    return loads_list, labels_list


def read_dir_all(dirpath):
    """ 读取指定文件夹下所有mat数据文件
        return:
            features: 包含所有特征的dict，以工况load为索引
            labels: 标签
            loads_list: 每个文件对应的工况编号list
            labels_list：每个文件对应的标签编号list
    """

    features = {}
    labels = {}

    filepaths = []
    for (root, _, files) in os.walk(dirpath):
        for filename in files:
            if filename.endswith('.mat'):
                filepaths.append(os.path.join(root, filename))

    # 读取filepaths中的文件，将其拼接成ndarray
    loads_list, labels_list = get_labels_and_loads(filepaths, ignore_load=True)
    for filepath, load, label in zip(filepaths, loads_list, labels_list):
        print('load file {} with label {}, load {}'.format(filepath, label, load))
        feature_tensor = load_matfile(filepath)
        feature_label = np.ones(feature_tensor.shape[0], dtype=np.int8) * label

        if load in features.keys():
            features[load] = np.concatenate((features[load], feature_tensor))
            labels[load] = np.concatenate((labels[load], feature_label))
        else:
            features[load] = feature_tensor
            labels[load] = feature_label

    return features, labels, loads_list, labels_list


def to_fft(data, axes):
    """ 对序列进行fft处理
    """
    dim_coef = int(np.floor(args.framesize / 2))
    print('dimension of fft coeffient to keep: {}'.format(dim_coef))
    re_fft_data = np.fft.fftn(data, axes=axes) 
    return abs(re_fft_data)[:, :dim_coef]

def post_process(features, labels):
    per_class_count = {}    # 每个类的数量（裁剪前）

    features_prune = []     # 裁剪后的features
    labels_prune = []       # 裁剪后的labels
    features_prune_test = []     # 裁剪后的features
    labels_prune_test = []       # 裁剪后的labels



    for label in np.unique(labels):
        # 计算该class的样本数
        per_class_count[label] = np.sum(np.array(labels == label))

        # 根据key筛选出该class的feature，并乱序
        feature = features[labels == label]
        rand_index = np.arange(len(feature))
        np.random.seed(args.randn)
        np.random.shuffle(rand_index)
        feature = feature[rand_index]

        # 根据per_class_num中定义的每个类要保留的样本数，对该类样本进行裁减，并保存到dict中
        # 将裁减后的feature保存到列表中
        features_prune.append(np.array(feature[:NUM_PER_CLASS_TRAIN]))
        features_prune_test.append(np.array(feature[NUM_PER_CLASS_TRAIN : NUM_PER_CLASS_TRAIN + NUM_PER_CLASS_TEST]))

        # 为裁减后的feature生成标签
        labels_prune.append(np.full(NUM_PER_CLASS_TRAIN, label))
        labels_prune_test.append(np.full(NUM_PER_CLASS_TEST, label))

    print("instance count of each classes:")
    print(per_class_count)

    features_prune = np.concatenate(features_prune, axis=0)
    labels_prune = np.concatenate(labels_prune, axis=0)
    features_prune_test = np.concatenate(features_prune_test, axis=0)
    labels_prune_test = np.concatenate(labels_prune_test, axis=0)

    return features_prune, labels_prune, features_prune_test, labels_prune_test

if __name__ == '__main__':

    print(args)
    data_dir = "D:/fault/paderborn_dataset"
    res_dir = "./data/paderborn_{}_c3_randn{}".format(args.methods.lower(), args.randn)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    total_features, total_labels, labels_dict, labels_list = read_dir_all(data_dir)
    
    features_prune = {}
    labels_prune = {}
    features_prune_test = {}
    labels_prune_test = {}
    for load in total_features.keys():
        # 后处理，裁剪数量，归一化等
        features, labels, features_test, labels_test = post_process(total_features[load], total_labels[load])
        print("Load {}: features: {}, labels: {}".format(load, total_features[load].shape, total_labels[load].shape))
        print("Load {} after prune: features {}, labels {}, features_test {}, labels_test {}".format(load, features.shape, labels.shape, features_test.shape, labels_test.shape))

        if args.methods.lower() == 'fft':
            labels_prune[load] = labels
            labels_prune_test[load] = labels_test

            features_prune[load] = to_fft(features, axes=(1,))
            features_prune_test[load] = to_fft(features_test, axes=(1,))
        elif args.methods.lower() == 'raw1d':
            features_prune[load] = features
            labels_prune[load] = labels
            features_prune_test[load] = features_test
            labels_prune_test[load] = labels_test
    
    # # 保存处理后的features
    # features = []

    if args.normalize == 1:
        from sklearn.preprocessing import StandardScaler
        std = StandardScaler()
        std_out = std.fit(features_prune[0])

    for load in features_prune.keys():
        if args.normalize == 1:
            std.transform(features_prune[load])
        
        # 再次乱序
        rand_index = np.arange(len(features_prune[load]))
        np.random.seed(args.randn)
        np.random.shuffle(rand_index)
        features = features_prune[load][rand_index]
        labels = labels_prune[load][rand_index]

        rand_index = np.arange(len(features_prune_test[load]))
        np.random.seed(args.randn)
        np.random.shuffle(rand_index)
        features_test = features_prune_test[load][rand_index]
        labels_test = labels_prune_test[load][rand_index]

        # save to disk
        if not os.path.exists('{}/load_{}'.format(res_dir, load)):
            os.makedirs('{}/load_{}'.format(res_dir, load))
        np.save("{}/load_{}/data_features_train.npy".format(res_dir, load), features)
        np.save("{}/load_{}/data_labels_train.npy".format(res_dir, load), labels)
        np.save("{}/load_{}/data_features_test.npy".format(res_dir, load), features_test)
        np.save("{}/load_{}/data_labels_test.npy".format(res_dir, load), labels_test)

