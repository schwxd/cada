"""
python 5-pb-c3.py --dataroot=D:\fault\paderborn_dataset --fft=0 --train=0 --normal=2 
python 5-pb-c3.py --dataroot=D:\fault\paderborn_dataset --fft=0 --train=1 --normal=2 

python 5-pb-c3.py --dataroot=/nas/data/paderborn-raw --fft=0 --train=0 --normal=2 
python 5-pb-c3.py --dataroot=/nas/data/paderborn-raw --fft=0 --train=1 --normal=2 
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

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=False, default='/nas/data/paderborn', help='where the data folder is')
parser.add_argument('--framesize', type=int, required=False, default=FRAME_SIZE, help='Frame Size of sliding windows')
parser.add_argument('--stepsize', type=int, required=False, default=STEP_SIZE, help='Step Size of Sliding window')
parser.add_argument('--load', type=int, required=False, default=-1, help='Specify one of the load conditions (0/1/2/3). -1 = all')
# parser.add_argument('--methods', required=False, default='fft', help='preprocess method')
parser.add_argument('--fft', required=False, type=int, default=0, help='use fft or not')
parser.add_argument('--train', type=int, required=False, default=1, help='training dataset or testing dataset')
parser.add_argument('--normal', required=False, type=int, default=0, help='normal or not')
args = parser.parse_args()

# def load_matfile(filepath, filename, frame_size=args.framesize, step_size=args.stepsize):
def load_matfile(filepath, filename):
    """ 加载mat文件，按照FRAME_SIZE划分成序列
    """
    matlab_file = scipy.io.loadmat(filepath)[filename]

    features = matlab_file[0][0][2][0][6][2][0]  #Take out the data

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

# def get_labels_list(filelist, ignore_load = True):
#     """
#     将去除后缀的文件名作为label，按顺序保存至字典中

#     假设文件名为‘OR007@3-0.mat’，则label为‘OR007@3-0’
#     OR代表外圈故障，
#     007代表故障直径，
#     @3代表故障位置在3点钟方位，
#     0代表负载为0、速度为1797rpm
#     """

#     labels_dict = {}        # label：value对应关系表
#     labels_list = []        # 每个文件的原始label，用于绘制图形时标注文件名
#     label_value = 0
#     for filename in filelist:
#         label1 = filename.split('.')[0]  #去掉文件后缀
#         if ignore_load:
#             label2 = label1.split('-')       #去掉工况，如果有的话
#             if len(label2) > 0:
#                 label = label2[0]
#             else:
#                 label = label1
#         else:
#             label = label1

#         if not label in labels_dict:
#             labels_dict[label] = label_value
#             label_value += 1

#         labels_list.append(label)
#     return labels_dict, labels_list

# labels_map = {
#     "K001": 0,
#     "K002": 1,
#     "K003": 2,
#     "K004": 3,
#     "K005": 4,
#     "KA04": 5,
#     "KA15": 6,
#     "KA16": 7,
#     "KA22": 8,
#     "KA30": 9,
#     "KI04": 10,
#     "KI14": 11,
#     "KI16": 12,
#     "KI18": 13,
#     "KI21": 14}


labels_map_train = {
    "K002": 0,
    "KA01": 1,
    "KA05": 1,
    "KA07": 1,
    "KI01": 2,
    "KI03": 2,
    "KI05": 2}
    # "KI07": 2}

labels_map_test = {
    "K001": 0,
    "KA04": 1,
    "KA15": 1,
    # "KA16": 1,
    "KA22": 1,
    # "KA30": 1,
    "KI14": 2,
    # "KI16": 2,
    "KI17": 2,
    # "KI18": 2,
    "KI21": 2}

loads_map = {
    'N15_M07_F10' : 0,
    'N09_M07_F10' : 1,
    'N15_M01_F10' : 2,
    'N15_M07_F04' : 3
}

frames_per_class = {
    "K002": 300,
    "KA01": 100,
    "KA05": 100,
    "KA07": 100,
    "KI01": 100,
    "KI03": 100,
    "KI05": 100,
    # "KI07": 30,
    
    "K001": 300,
    "KA04": 100,
    "KA15": 100,
    # "KA16": 24,
    "KA22": 100,
    # "KA30": 24,
    "KI14": 100,
    # "KI16": 24,
    "KI17": 100,
    # "KI18": 24,
    "KI21": 100}


def get_labels_and_loads(filelist, labels_map, loads_map):
    """
    将filelist中的每个文件名，提取标签Label和工况Load，分别存放到列表中
    labels_map: 文件名与标签的对应关系
    loads_map:  文件名与工况的对应关系
    """

    labels_list = []
    loads_list = []
    for filename in filelist:
        load = filename[:11]
        label = filename[12:16]
        # print('filename {}, load {}, label {}'.format(filename, load, label))

        labels_list.append(labels_map[label])
        loads_list.append(loads_map[load])
    return loads_list, labels_list


def concatenate_datasets(xd, yd, xo, yo):
    """ 将所有单独的文件合并成完整的数据集
    """

    if xd is None or yd is None:
        xd = xo
        yd = yo
    else:
        xd = np.concatenate((xd, xo))
        yd = np.concatenate((yd, yo))
    return xd, yd

def calculate_frames_per_class(filename):
    n_samples = 0
    for key in frames_per_class.keys():
        if key in filename:
            n_samples = frames_per_class[key]
    return n_samples
        

# def read_dir_all(dirpath, labels_map, loads_map, framesize, samplenum=0):
#     """ 读取指定文件夹下所有数据文件

#     dirpath: mat数据文件路径，忽略子文件夹
#     labels_map: 类标的编号关系
#     loads_map: 工况的编号关系
#     framesize：样本的frame大小
#     samplenumber：如果为0，则用滑动窗口方法生成样本，滑动窗口的大小等同于framesize。如果不为0，则随机采样samplenum条样本
#     """

#     features = {}
#     labels = {}

#     # 读取当前文件夹下，以mat后缀结尾的所有文件名
#     filelist = []
#     dirlist = os.listdir(dirpath)
#     for filename in dirlist:
#         if os.path.isdir(os.path.join(dirpath, filename)):
#             continue
#         if not filename.endswith('.mat'):
#             continue
#         filelist.append(filename)

#     # 获取文件名对应的类标和工况
#     loads_list, labels_list = get_labels_and_loads(filelist, labels_map, loads_map)

#     for filename, load, label in zip(filelist, loads_list, labels_list):
#         # 加载mat文件
#         matdata = load_matfile(os.path.join(dirpath, filename), filename)
#         # 生成长度为framesize，总数为frames_per_class[filename]的样本特征
#         features = feature_segment(matdata, framesize, samplenumber=frames_per_class[filename])
#         # 生成对应数量的标签
#         labels = np.ones(features.shape[0], dtype=np.int8) * label

#         # features[load], labels[load] = concatenate_datasets(features[load], labels[load], feature_tensor, feature_label)
#         # 将相同工况的样本合并
#         if load in features.keys():
#             features[load] = np.concatenate((features[load], features))
#             labels[load] = np.concatenate((labels[load], labels))
#         else:
#             features[load] = features
#             labels[load] = labels

#     return features, labels, loads_list, labels_list


def load_files_from_list(filepaths, filenames, labels_map, loads_map, framesize):
    """ 读取指定文件夹下所有数据文件

    dirpath: mat数据文件路径，忽略子文件夹
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

        n_samples = calculate_frames_per_class(filename)

        # 生成长度为framesize，总数为frames_per_class[filename]的样本特征
        features = feature_segment(filename, matdata, framesize, samplenumber=n_samples)
        # 生成对应数量的标签
        labels = np.ones(features.shape[0], dtype=np.int8) * label

        # features[load], labels[load] = concatenate_datasets(features[load], labels[load], feature_tensor, feature_label)
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


def post_process(features, labels):
    per_class_count = {}    # 每个类的数量（裁剪前）

    features_prune = []     # 裁剪后的features
    labels_prune = []       # 裁剪后的labels
    features_prune_test = []     # 裁剪后的features
    labels_prune_test = []       # 裁剪后的labels

    NUM_PER_CLASS = 4000 
    NUM_PER_CLASS_TEST = 1000 

    for label in np.unique(labels):
        # 计算该class的样本数
        per_class_count[label] = np.sum(np.array(labels == label))

        # 根据key筛选出该class的feature，并乱序
        feature = features[labels == label]
        rand_index = np.arange(len(feature))
        # np.random.seed(0)
        np.random.shuffle(rand_index)
        feature = feature[rand_index]

        # 根据per_class_num中定义的每个类要保留的样本数，对该类样本进行裁减，并保存到dict中
        # 将裁减后的feature保存到列表中
        features_prune.append(np.array(feature[:NUM_PER_CLASS]))
        features_prune_test.append(np.array(feature[NUM_PER_CLASS : NUM_PER_CLASS + NUM_PER_CLASS_TEST]))

        # 为裁减后的feature生成标签
        labels_prune.append(np.full(NUM_PER_CLASS, label))
        labels_prune_test.append(np.full(NUM_PER_CLASS_TEST, label))

    print("instance count of each classes:")
    print(per_class_count)

    features_prune = np.concatenate(features_prune, axis=0)
    labels_prune = np.concatenate(labels_prune, axis=0)
    features_prune_test = np.concatenate(features_prune_test, axis=0)
    labels_prune_test = np.concatenate(labels_prune_test, axis=0)
    # print(features_prune.shape)
    # print(labels_prune.shape)
    # print(features_prune_test.shape)
    # print(labels_prune_test.shape)

    return features_prune, labels_prune, features_prune_test, labels_prune_test

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
    if args.train == True:
        labels_map = labels_map_train
    else:
        labels_map = labels_map_test

    # 不适用于每个类一个文件夹的情况
    # subdirs = os.listdir(data_dir)
    # for subdir in subdirs:
    #     # 读取mat文件并生成样本，
    #     total_features, total_labels, labels_dict, labels_list = read_dir_all(subdir, labels_map, loads_map, framesize=args.framesize, samplenum=frames_per_class[subdir])

    filepaths = []
    filenames = []
    subdirs = []
    for roots, dirs, files in os.walk(args.dataroot):
        # 遍历文件夹，看是否需要读取该文件夹
        hits = False
        for key in labels_map.keys():
            if key in roots:
                hits = True
        if not hits:
            print('roots {} not in the list {}'.format(roots, labels_map.keys()))
            continue

        for filename in files:
            if not filename.endswith('.mat'):
                continue
            filenames.append(filename[:-4])
            filepaths.append(os.path.join(roots, filename))
            subdirs.append(roots)

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
        # 后处理，裁剪数量，归一化等
        # features, labels, features_test, labels_test = post_process(total_features[load], total_labels[load])
        # print("Load {}: features: {}, labels: {}".format(load, total_features[load].shape, total_labels[load].shape))
        # print("Load {} after prune: features {}, labels {}, features_test {}, labels_test {}".format(load, features.shape, labels.shape, features_test.shape, labels_test.shape))

        res_dir = "./data/paderborn_train{}_fft{}_normal{}/load{}".format(args.train, args.fft, args.normal, load)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        if args.fft == 1:
            features_prune[load] = to_fft(total_features[load], axes=(1,))
            labels_prune[load] = total_labels[load]
        else:
            features_prune[load] = total_features[load]
            labels_prune[load] = total_labels[load]

        # 归一化
        if args.normal > 0:
            features_prune[load] = do_scalar(features_prune[load], args.normal)
            draw_fig(features_prune[load], labels_prune[load], count=1, prefix='fft0-norm1', res_dir=res_dir)

        # 再次乱序
        rand_index = np.arange(len(features_prune[load]))
        np.random.shuffle(rand_index)
        features = features_prune[load][rand_index]
        labels = labels_prune[load][rand_index]

        # 保存
        np.save("{}/data_features_train.npy".format(res_dir), features)
        np.save("{}/data_labels_train.npy".format(res_dir), labels)
        print('load {} saved to {}, features {}, labels {}'.format(load, res_dir, features.shape, labels.shape))
