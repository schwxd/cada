"""
    读取CWRU数据，按指定算法将振动信号转化为Image信号
    
    支持的振动信号转图像的算法有：
    - RAW：直接堆积
    - FFT：快速傅里叶变换
    - GAF：
    - RP：Recurrence Plot
    - STrans：Stockwell Transform
    - DOST：Discrete Orthonormal Stockwell Transform

    流程：
    1. 读取数据文件，并对数据进行滑动窗口处理
    2. 从文件名中获取标签信息，为该文件下的所有数据条目设置该标签
    3. 将所有数据窗口转化为图像格式。保存npy数据文件供后续分析，保存jpg图像文件供查看。
    4. 保存文件的标签信息


Credits：
    cwru-conv1d-master
    MCHE485---Mechanical-Vibrations-Spring2019
"""

import os
import re
import sys
import math
import json
import argparse
import scipy.io
# import threading
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
from sklearn import preprocessing
import multiprocessing
from multiprocessing import Process

FRAME_SIZE = 5120           # 滑动窗口大小
STEP_SIZE = 5100            # 滑动窗口移动步长
IMG_SIZE = 128
PREP_METHODS = ['Raw', 'FFT', 'GAF', 'FFT_GAF', 'STrans', 'DOST', 'EMD', 'RP', 'NSP', 'Wavelet']

parser = argparse.ArgumentParser()
parser.add_argument('--framesize', type=int, required=False, default=FRAME_SIZE, help='Frame Size of sliding windows')
parser.add_argument('--stepsize', type=int, required=False, default=STEP_SIZE, help='Step Size of Sliding window')
parser.add_argument('--stepsize_normal', type=int, required=False, default=-1, help='Step Size of Sliding window for normal work condition')
parser.add_argument('--imgsize', type=int, required=False, default=IMG_SIZE, help='number of img rows')
parser.add_argument('--load', type=int, required=False, default=-1, help='Specify one of the load conditions (0/1/2/3). -1 = all')
parser.add_argument('--methods', required=False, default='fft', help='preprocess method')

args = parser.parse_args()

def load_matfile(filename, frame_size=args.framesize, step_size=args.stepsize):
    """ 加载mat文件，按照FRAME_SIZE划分成序列

    """

    matlab_file = scipy.io.loadmat(filename)

    signal_begin = 0
    signal_len = len(matlab_file['data'][0])
    
    DE_samples = []
    while signal_begin + frame_size < signal_len:
        DE_samples.append(matlab_file['data'][0][signal_begin:signal_begin+frame_size])
        signal_begin += step_size

    sample_tensor = np.array(DE_samples).astype('float32')
    #print("Load file {} into shape {}".format(filename, sample_tensor.shape))
    return sample_tensor


def get_labels_list(filelist, ignore_load = True):
    """
    将去除后缀的文件名作为label，按顺序保存至字典中

    假设文件名为‘OR007@3-0.mat’，则label为‘OR007@3-0’
    OR代表外圈故障，
    007代表故障直径，
    @3代表故障位置在3点钟方位，
    0代表负载为0、速度为1797rpm
    """

    labels_dict = {}        # label：value对应关系表
    labels_list = []        # 每个文件的原始label，用于绘制图形时标注文件名
    label_value = 0
    for filename in filelist:
        label1 = filename.split('.')[0]  #去掉文件后缀
        if ignore_load:
            label2 = label1.split('-')       #去掉工况，如果有的话
            if len(label2) > 0:
                label = label2[0]
            else:
                label = label1
        else:
            label = label1

        if not label in labels_dict:
            labels_dict[label] = label_value
            label_value += 1

        labels_list.append(label)
    return labels_dict, labels_list

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

loads_map = {
    'N09_M07_F10' : 0,
    'N15_M01_F10' : 1,
    'N15_M07_F04' : 2,
    'N15_M07_F10' : 3
}

def get_labels_and_loads(filelist, ignore_load = True):
    """
    生成工况列表和标签列表（与filelist一一对应）
    """

    labels_list = []        # 每个文件的原始label，用于绘制图形时标注文件名
    loads_list = []
    for filename in filelist:
        load = filename[:11]
        label = filename[12:16]
        ##print('filename {}, load {}, label {}'.format(filename, load, label))

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


def read_dir_all(dirpath):
    """ 读取指定文件夹下所有数据文件
    """

    features = {}
    labels = {}

    dirlist = os.listdir(dirpath)
    filelist = []
    for filename in dirlist:
        if os.path.isdir(os.path.join(dirpath, filename)):
            continue

        if not filename.endswith('.mat'):
            continue

        filelist.append(filename)

    # 读取filelist中的文件，将其拼接成ndarray
    loads_list, labels_list = get_labels_and_loads(filelist, ignore_load=True)
    # print("labels_dict for dir {} : {}".format(dirpath, labels_dict))
    for filename, load, label in zip(filelist, loads_list, labels_list):
        feature_tensor = load_matfile(os.path.join(dirpath, filename))
        feature_label = np.ones(feature_tensor.shape[0], dtype=np.int8) * label

        # features[load], labels[load] = concatenate_datasets(features[load], labels[load], feature_tensor, feature_label)
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


from pyts.image import GASF, GADF
from PIL import Image
import cv2
def to_gaf(features, labels, fft=False):
    """
    """
    features_gaf = []

    if fft:
        features = to_fft(features, axes=(1,))

    for i in range(features.shape[0]):
        X = features[i].reshape(1, -1)
        gasf = GASF(args.imgsize)
        X_gasf = gasf.fit_transform(X)

        img = X_gasf[0]
        img = (1+img)*255/2.0
        img = img.astype('uint8').reshape(args.imgsize, args.imgsize, 1)
        cv2.imwrite("data/imgs/{}-{}.jpg".format(labels[i], i), img)
        
        features_gaf.append(img)
    return features_gaf


from utils.my_fdost import fdost
from scipy.misc import imresize
# https://github.com/Gsonggit/stockwell_transform
def to_st(features, labels):
    """
    Stockwell Transform to image
    """

    features_st = []
    results = []
    for i in range(features.shape[0]):
        X = features[i].reshape(-1)
        result = fdost(X, origion=True)
        results.append(np.abs(result))

        if i % 10 == 0:
            print("processed {} / {}".format(i, features.shape[0]))

    results = np.array(results)
    shape0=results.shape[0]
    shape1=results.shape[1]
    results = results.reshape(shape0, -1)
    MinMaxScaler = preprocessing.MinMaxScaler()
    results = MinMaxScaler.fit_transform(results)
    results = results.reshape(shape0, shape1, -1)
    results = results*255
    for i in range(results.shape[0]):

        # save imgs
        img = imresize(results[i], (args.imgsize, args.imgsize))
        img = img.astype('uint8').reshape(args.imgsize, args.imgsize, 1)
        cv2.imwrite("{}/{}-{}.jpg".format(res_dir, labels[i], i), img)
        features_st.append(img)

    return features_st

# 多线程进行strans转换，将转换后的features保存在各自的npy文件中
# 不进行缩放，不进行转图像操作，所有进程转换完成后统一进行
def to_st_mthread(features, labels, res_dir):
    """
    Stockwell Transform to image
    """

    # print('thread {} is running...'.format(threading.current_thread().name))
    # print("thread {} has #{} instances to deal with".format(threading.current_thread().name, features.shape))
    print('thread {} is running...'.format(multiprocessing.current_process().name))
    print("thread {} has #{} instances to deal with".format(multiprocessing.current_process().name, features.shape))

    features_st = []
    for i in range(features.shape[0]):
        X = features[i].reshape(-1)
        result = fdost(X, origion=True)
        features_st.append(np.abs(result))

        if i % 10 == 0:
            print("{} processed {} / {}".format(multiprocessing.current_process().name, i, features.shape[0]))

    features_st = np.array(features_st).astype('float32')
    np.save('{}/features_{}.npy'.format(res_dir, multiprocessing.current_process().name), features_st)

# 读取所有线程的转换结果，进行统一缩放和转图像操作
def to_st_post(labels):
    dirlist = os.listdir(res_dir)
    print(dirlist)

    filelist = []
    for filename in dirlist:
        if os.path.isdir(os.path.join(res_dir, filename)):
            continue
        
        if filename.startswith('features_Thread'):
            filelist.append(filename)
    print(filelist)

    results = []
    filelist.sort(key= lambda x:int(x[-6:-4]))
    print(filelist)
    for filename in filelist:
        data = np.load('{}/{}'.format(res_dir, filename))
        print("read file {} with shape {}".format(filename, data.shape))
        results.append(data)

    results = np.concatenate(results)
    print("all features shape {}".format(results.shape))

    features_st = []
    shape0=results.shape[0]
    shape1=results.shape[1]
    results = results.reshape(shape0, -1)
    MinMaxScaler = preprocessing.MinMaxScaler()
    results = MinMaxScaler.fit_transform(results)
    results = results.reshape(shape0, shape1, -1)
    results = results*255
    for i in range(results.shape[0]):

        # save imgs
        img = imresize(results[i], (args.imgsize, args.imgsize))
        img = img.astype('uint8').reshape(args.imgsize, args.imgsize, 1)
        cv2.imwrite("{}/imgs/{}-{}.jpg".format(res_dir, labels[i], i), img)
        features_st.append(img)

    return features_st


def to_raw(features, labels):
    """
        直接将振动信号拼接成图片
        取每个窗口长度的平方根作为图片的长宽
    """
    features_raw = []

    # 全局scale
    features = (features - np.min(features)) / (np.max(features) - np.min(features))
    features *= 255

    steps = np.floor(np.sqrt(args.framesize)).astype(np.int)

    for i in range(features.shape[0]):
        start = 0
        img = []
        while start + steps <= len(features[i]):
            img.append(features[i, start : start+steps])
            start += steps

        img = np.array(img).reshape(steps, steps)
        img = imresize(img, (args.imgsize, args.imgsize))
        img = np.expand_dims(img, -1)
        imgfolder = "data/imgs/{}".format(labels[i])
        if not os.path.exists(imgfolder):
            os.makedirs(imgfolder)
        cv2.imwrite(os.path.join(imgfolder, "{}.jpg".format(i)), img)

        features_raw.append(img)

    return features_raw

def to_nsp():
    pass

from PyEMD import EMD
def to_emd(features, labels):

    features_emd = []
    for i in range(features.shape[0]):
        img = EMD().emd(features[i])

        # 画EMD结果图
        # for i in range(img.shape[0]):
        #     plt.subplot(img.shape[0], 1, i+1)
        #     plt.plot(img[i])
        # plt.show()

        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img *= 255
        img = imresize(img, (args.imgsize, args.imgsize, 1))
        cv2.imwrite("data/imgs/{}-{}.jpg".format(labels[i], i), img)

        features_emd.append(img)

    return features_emd


from utils.plot_recurrence import rec_plot
def to_rp(features, labels):
    eps = 0.05
    steps = 20

    features_rp = []
    for i in range(features.shape[0]):
        img = rec_plot(features[i], eps, steps)

        img = imresize(img, (args.imgsize, args.imgsize))
        img = np.array(img)
        img = np.expand_dims(img, -1)
        cv2.imwrite("data/imgs/{}-{}.jpg".format(labels[i], i), img)

        features_rp.append(img)

    return features_rp


def to_dost():
    pass

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

if __name__ == '__main__':

    print("Vibration To Image using method {}".format(args.methods))
    data_dir = "/nas/data/paderborn"
    res_dir = "./data/paderborn_{}".format(args.methods.lower())
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    if not os.path.exists('{}/imgs'.format(res_dir)):
        os.makedirs('{}/imgs'.format(res_dir))

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
    
    # 保存处理后的features
    features = []

    if args.methods.lower() == 'fft':
        pass
        # features = to_fft(features_prune, axes=(1,))
        # features_test = to_fft(features_prune_test, axes=(1,))

    elif args.methods.lower() == 'gaf':
        features = to_gaf(features_prune, labels_prune, fft=False)

    elif args.methods.lower() == 'fft_gaf':
        features = to_gaf(features_prune, labels_prune, fft=True)

    elif args.methods.lower() == 'strans':
        threads = []
        NUM_THREAD = 3
        per_thread_count = int(np.ceil(len(features_prune) / NUM_THREAD))
        for i in range(NUM_THREAD):
            per_thread_feature = features_prune[i*per_thread_count : (i+1)*per_thread_count]
            per_thread_label = labels_prune[i*per_thread_count : (i+1)*per_thread_count]
            t = Process(target=to_st_mthread, args=(per_thread_feature, per_thread_label, res_dir), name='Thread{:0>2d}'.format(i))
            t.start()
            threads.append(t)

        for i in range(1, NUM_THREAD):
            threads[i].join()

        features = to_st_post(labels_prune)

    elif args.methods.lower() == 'raw':
        features = to_raw(features_prune, labels_prune)

    elif args.methods.lower() == 'raw1d':
        # 不需要处理，直接保存即可
        # features = features_prune
        # features_test = features_prune_test
        pass
        
    elif args.methods.lower() == 'rp':
        features = to_rp(features_prune, labels_prune)

    elif args.methods.lower() == 'emd':
        features = to_emd(features_prune, labels_prune)

    elif args.methods.lower() == 'nsp':
        print("Methods not supported yet: " + args.methods)
        exit()

    elif args.methods.lower() == 'dost':
        print("Methods not supported yet: " + args.methods)
        exit()

    else:
        print("Methods not supported yet: " + args.methods)
        exit()

    from sklearn.preprocessing import StandardScaler
    std = StandardScaler()
    std_out = std.fit(features_prune[0])

    for load in features_prune.keys():
        std.transform(features_prune[load])
        
        # 再次乱序
        rand_index = np.arange(len(features_prune[load]))
        np.random.shuffle(rand_index)
        features = features_prune[load][rand_index]
        labels = labels_prune[load][rand_index]

        rand_index = np.arange(len(features_prune_test[load]))
        np.random.shuffle(rand_index)
        features_test = features_prune_test[load][rand_index]
        labels_test = labels_prune_test[load][rand_index]

        # 输出类标及其对应关系
        if not os.path.exists('{}/load_{}'.format(res_dir, load)):
            os.makedirs('{}/load_{}'.format(res_dir, load))
        np.save("{}/load_{}/data_features_train.npy".format(res_dir, load), features)
        np.save("{}/load_{}/data_labels_train.npy".format(res_dir, load), labels)
        np.save("{}/load_{}/data_features_test.npy".format(res_dir, load), features_test)
        np.save("{}/load_{}/data_labels_test.npy".format(res_dir, load), labels_test)

        # jsObj = json.dumps(labels_list)
        # fileObject = open('{}/labels_list.json'.format(res_dir), 'w')
        # fileObject.write(jsObj)
        # fileObject.close()

        # jsObj = json.dumps(labels_dict)
        # fileObject = open('{}/labels_dict.json'.format(res_dir), 'w')
        # fileObject.write(jsObj)
        # fileObject.close()
