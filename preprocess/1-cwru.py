from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同
import matplotlib.pyplot as plt

np.random.seed(0)

def prepro(d_path, framesize, trainnumber, testnumber, res_dir, normal):
    """对数据进行预处理,返回train_X, train_Y, test_X, test_Y样本.

    :param d_path: 源数据地址
    :param length: 信号长度，默认2个信号周期，864
    :param number: 每种信号个数,总共10类,默认每个类别1000个数据
    :param normal: 是否标准化.True,Fales.默认True
    :return: Train_X, Train_Y, Test_X, Test_Y

    """
    # 获得该文件夹下所有.mat文件名
    filenames = os.listdir(d_path)

    def capture(original_path):
        """读取mat文件，返回字典

        :param original_path: 读取路径
        :return: 数据字典
        """
        files = {}
        filenames = os.listdir(original_path)
        for i in filenames:
            # 文件路径
            file_path = os.path.join(d_path, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                if args.defe in key:
                    files[i] = file[key].ravel()
        return files

    def slice_enc(data, trainnumber, testnumber, framesize):
        """将数据切分为前面多少比例，后面多少比例.

        data: 数据字典，filename:feature
        trainnumber: 生成多少条训练样本
        testnumber: 生成多少条测试样本
        framesize: 每个样本的长度

        return: 数据字典，filename:指定数量指定framesize的feature
        """
        keys = data.keys()
        Train_Samples = {}
        Test_Samples = {}
        for i in keys:
            slice_data = data[i]
            all_length = len(slice_data)
            Train_sample = []
            Test_Sample = []
            #train_start = np.ceil((testnumber / (testnumber+trainnumber))*all_length)
            #print('key {}, len {}, train_start {}'.format(i, all_length, #train_start))
            # 抓取测试数据，使用数据增强
            for h in range(testnumber):
                # sample = slice_data[h*framesize:(h+1)*framesize]
                # Test_Sample.append(sample)
                random_start = np.random.randint(low=0 * framesize, high=all_length - framesize)
                sample = slice_data[random_start:random_start + framesize]
                Test_Sample.append(sample)


            # 抓取训练数据，使用数据增强
            for j in range(trainnumber):
                # random_start = np.random.randint(low=testnumber * framesize, high=all_lenght-framesize)
                random_start = np.random.randint(low=0 * framesize, high=all_length-framesize)
                sample = slice_data[random_start:random_start + framesize]
                Train_sample.append(sample)

            Train_Samples[i] = Train_sample
            Test_Samples[i] = Test_Sample
        return Train_Samples, Test_Samples

    def get_labels_list(d_path, ignore_load = True):
        """
        d_path: matlab文件所在的文件夹
        ignore_load: 分配label时是否考虑工况。如B007-2.mat，考虑工况时label为B007-2，不考虑工况时label为B007

        return:
        label_dict: 类名：类标，比如B007:1, OR007:2
        label_list: 文件名：类标，比如B007-2.mat: 2, B007-3.mat: 2
        """

        filelist = os.listdir(d_path)
        # labels_dict = {}        # label：value对应关系表
        labels_dict = {'Normal': 0, 
                        'IR007': 1, 
                        'IR014': 2, 
                        'IR021': 3, 
                        'B007': 4, 
                        'B014': 5, 
                        'B021': 6, 
                        'OR007@6': 7, 
                        'OR014@6': 8, 
                        'OR021@6': 9}
        labels_list = {}        # 每个文件的原始label，用于绘制图形时标注文件名
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

            labels_list[filename] = labels_dict[label]

            # if not label in labels_dict:
            #     labels_dict[label] = label_value
            #     label_value += 1
            # if not label in labels_list:
            #     # labels_list[filename] = label_value
            #     labels_list[filename] = labels_dict[label]


        return labels_dict, labels_list 


    # 仅抽样完成，打标签
    def add_labels(train_test, labels_list):
        X = []
        Y = []
        label = 0
        labels_dict = {}
        for i in filenames:
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [labels_list[i]] * lenx
        return X, Y

    # one-hot编码
    def one_hot(Train_Y, Test_Y):
        Train_Y = np.array(Train_Y).reshape([-1, 1])
        Test_Y = np.array(Test_Y).reshape([-1, 1])
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(Train_Y)
        Train_Y = Encoder.transform(Train_Y).toarray()
        Test_Y = Encoder.transform(Test_Y).toarray()
        Train_Y = np.asarray(Train_Y, dtype=np.int32)
        Test_Y = np.asarray(Test_Y, dtype=np.int32)
        return Train_Y, Test_Y

    def scalar_stand(Train_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Test_X

    def do_scalar(Train_X, Test_X, normal):
        if normal == 1:
            scalar = preprocessing.StandardScaler()
        elif normal == 2:
            scalar = preprocessing.MinMaxScaler()
        Train_X = scalar.fit_transform(Train_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Test_X

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

    # 从所有.mat文件中读取出数据的字典
    data = capture(original_path=d_path)
    print(data.keys())
    # 将数据切分为训练集、测试集
    train, test = slice_enc(data, trainnumber, testnumber, framesize)
    print('after slice_enc, train.keys {}'.format(train.keys()))

    labels_dict, labels_list = get_labels_list(d_path)
    print('labels_dict {}'.format(labels_dict))
    print('labels_list {}'.format(labels_list))

    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train, labels_list)
    # 为测试集制作标签，返回X，Y
    Test_X, Test_Y = add_labels(test, labels_list)

    # 为训练集Y/测试集One-hot标签
    # Train_Y, Test_Y = one_hot(Train_Y, Test_Y)

    # 训练数据/测试数据 是否标准化.
    draw_fig(Train_X, Train_Y, count=1, prefix='fft0-norm0')
    if normal > 0:
        Train_X, Test_X = do_scalar(Train_X, Test_X, normal)
        draw_fig(Train_X, Train_Y, count=1, prefix='fft0-norm1')

    else:
        # 需要做一个数据转换，转换成np格式.
        Train_X = np.asarray(Train_X)
        Test_X = np.asarray(Test_X)

    if args.fft == 1:
        Train_X = to_fft(Train_X, axes=(1,))
        Test_X = to_fft(Test_X, axes=(1,))
        draw_fig(Train_X, Train_Y, count=1, prefix='fft1')

    Train_Y = np.asarray(Train_Y)
    Test_Y = np.asarray(Test_Y)
    Train_X, Train_Y = shuffle_data(Train_X, Train_Y)
    Test_X, Test_Y = shuffle_data(Test_X, Test_Y)
    return Train_X, Train_Y, Test_X, Test_Y

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=False, default='/nas/code/iw/matdata/de', help='where the data folder is')
parser.add_argument('--framesize', type=int, required=False, default=2048, help='frame size')
parser.add_argument('--trainnumber', type=int, required=False, default=660, help='how many samples each class for training dataset')
parser.add_argument('--testnumber', type=int, required=False, default=25, help='how many samples each class for test dataset')
parser.add_argument('--defe', required=False, default='DE', help='DE or FE')
parser.add_argument('--fft', required=False, type=int, default=0, help='use fft or not')
parser.add_argument('--normal', required=False, type=int, default=0, help='normal or not')
parser.add_argument('--outputpath', required=False, default='output/defe-noload', help='where the output folder is')
args = parser.parse_args()
from sklearn import preprocessing

if __name__ == "__main__":
    subdirs = os.listdir(args.dataroot)
    print('args: {}'.format(args))
    print('subdirs: {}'.format(subdirs))
    for subdir in subdirs:
        print('process subdir {}'.format(subdir))
        res_dir = 'cwru-fft{}-fs{}-num{}/{}{}'.format(args.fft, args.framesize, args.trainnumber, args.defe, subdir)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        datapath = os.path.join(args.dataroot, subdir)
        train_X, train_Y, test_X, test_Y = prepro(d_path=datapath,
                                                    framesize=args.framesize,
                                                    trainnumber=args.trainnumber,
                                                    testnumber=args.testnumber,
                                                    res_dir=res_dir,
                                                    normal=args.normal) 
             
        np.save("{}/data_features_train.npy".format(res_dir), train_X)
        np.save("{}/data_labels_train.npy".format(res_dir), train_Y)
        np.save("{}/data_features_test.npy".format(res_dir), test_X)
        np.save("{}/data_labels_test.npy".format(res_dir), test_Y)
        print(" train features {}, test features {}".format(train_X.shape, test_X.shape))
        print(" train labels {}, test labels {}".format(train_Y.shape, test_Y.shape))
        print('save result to {}\n'.format(res_dir))
