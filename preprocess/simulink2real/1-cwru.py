from scipy.io import loadmat
import numpy as np
import os
import json
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同
import matplotlib.pyplot as plt


labels_map = {
    'IR007': 0,
    'IR014': 0,
    'IR021': 0,
    'OR007@6': 1,
    'OR014@6': 1,
    'OR021@6': 1,
    'Normal1': 2,
    'Normal2': 2,
    'Normal3': 2}


def capture(original_path):
    """读取mat文件，返回字典

    :param original_path: 读取路径
    :return: 数据字典
    """

    files = {}
    filenames = os.listdir(original_path)
    for i in filenames:
        # 文件路径
        file_path = os.path.join(original_path, i)
        file = loadmat(file_path)
        file_keys = file.keys()
        for key in file_keys:
            if args.defe in key:
                files[i] = file[key].ravel()
    return files


def slice_enc(data, trainnumber, framesize):
    """数据切片

    data: 数据字典，filename:feature
    trainnumber: 生成多少条训练样本
    framesize: 每个样本的长度

    return: 数据字典，filename:指定数量指定framesize的feature
    """

    keys = data.keys()
    all_samples = {}
    for i in keys:
        slice_data = data[i]
        all_length = len(slice_data)
        samples = []

        # 抓取训练数据，使用数据增强
        for j in range(trainnumber):
            random_start = np.random.randint(low=0 * framesize, high=all_length-framesize)
            sample = slice_data[random_start:random_start + framesize]
            samples.append(sample)

        all_samples[i] = samples
    return all_samples


def get_labels_list(d_path, labels_map, ignore_load = True):
    """
    根据文件名，获得所对应的标签

    d_path: matlab文件所在的文件夹
    ignore_load: 分配label时是否考虑工况。如B007-2.mat，考虑工况时label为B007-2，不考虑工况时label为B007

    return:
    label_dict: 类名：类标，比如B007:1, OR007:2
    label_list: 文件名：类标，比如B007-2.mat: 2, B007-3.mat: 2
    """

    filelist = os.listdir(d_path)
    labels_dict = {}        # label：value对应关系表
    labels_list = {}        # 每个文件的原始label，用于绘制图形时标注文件名
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

        # if not label in labels_list:
        #     labels_list[filename] = labels_map[label]
        if label in labels_map.keys():
            labels_list[filename] = labels_map[label]

    return labels_dict, labels_list 


# 对每个样本增加对应的标签
def add_labels(train_test, labels_list):
    X = []
    Y = []
    for i in labels_list:
        x = train_test[i]
        X += x
        lenx = len(x)
        Y += [labels_list[i]] * lenx
    return X, Y


# 将数据手动乱序
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


def draw_fig(data, labels, count, prefix):
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


def prepro(d_path, framesize, trainnumber, res_dir, normal):
    """对数据进行预处理,返回train_X, train_Y, test_X, test_Y样本.

    :param d_path: 源数据地址
    :param framesize: 信号长度，一般不小于2个信号周期
    :param trainnumber: 训练样本中，每个类的样本数
    :param res_dir: 输出目录
    :param normal: 是否标准化.True,Fales.默认True
    :return: Train_X, Train_Y, Test_X, Test_Y

    """
    # 获得该文件夹下所有.mat文件名
    filenames = os.listdir(d_path)

    # 从所有.mat文件中读取出数据的字典
    data = capture(original_path=d_path)
    print(data.keys())
    # 将数据切分为训练集、测试集
    data_sliced = slice_enc(data, trainnumber, framesize)
    print('after slice_enc, data_sliced.keys {}'.format(data_sliced.keys()))

    labels_dict, labels_list = get_labels_list(d_path, labels_map=labels_map)
    print('labels_dict {}'.format(labels_dict))
    print('labels_list {}'.format(labels_list))

    jsObj = json.dumps(labels_map)
    fileObject = open('{}/labels_map.json'.format(res_dir), 'w')
    fileObject.write(jsObj)
    fileObject.close()

    jsObj = json.dumps(labels_list)
    fileObject = open('{}/labels_list.json'.format(res_dir), 'w')
    fileObject.write(jsObj)
    fileObject.close()

    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(data_sliced, labels_list)
    Train_X = np.asarray(Train_X)
    Train_Y = np.asarray(Train_Y)
    Train_X, Train_Y = shuffle_data(Train_X, Train_Y)

    if args.fft == 1:
        Train_X = to_fft(Train_X, axes=(1,))

    return Train_X, Train_Y


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=False, default='D:\\fault\cwru-dataset\cross-machine', help='where the data folder is')
parser.add_argument('--framesize', type=int, required=False, default=1200, help='frame size')
parser.add_argument('--trainnumber', type=int, required=False, default=1000, help='how many samples each class for training dataset')
parser.add_argument('--defe', required=False, default='DE', help='DE or FE')
parser.add_argument('--fft', required=False, type=int, default=0, help='use fft or not')
parser.add_argument('--normal', required=False, type=int, default=0, help='normal or not')
args = parser.parse_args()
from sklearn import preprocessing

if __name__ == "__main__":
    subdirs = os.listdir(args.dataroot)
    print('args: {}'.format(args))
    print('subdirs: {}'.format(subdirs))
    for subdir in subdirs:
        print('process subdir {}'.format(subdir))
        res_dir = 'CWRU-{}{}-fft{}-norm{}-fs{}-num{}'.format(args.defe, subdir, args.fft, args.normal, args.framesize, args.trainnumber)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        datapath = os.path.join(args.dataroot, subdir)
        train_X, train_Y = prepro(d_path=datapath,
                                    framesize=args.framesize,
                                    trainnumber=args.trainnumber,
                                    res_dir=res_dir,
                                    normal=args.normal) 

        draw_fig(train_X, train_Y, 10, 'subdir{}'.format(subdir))

        np.save("{}/data_features_train.npy".format(res_dir), train_X)
        np.save("{}/data_labels_train.npy".format(res_dir), train_Y)
        print(" train features {}, train labels {}".format(train_X.shape, train_Y.shape))
        print('save result to {}\n'.format(res_dir))
