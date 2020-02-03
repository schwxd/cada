import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from torchvision import datasets, transforms
import torch.utils.data as data

# white gaussian noise
def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


class BearingDataset(data.Dataset):
    def __init__(self, features, labels, transform, snr=0):
        self.features = features
        self.labels = labels
        self.transform = transform

        # if snr != 0:
        #     print('apply snr {} to signal'.format(snr))
        #     for i in range(len(self.data)):
        #         noise = wgn(self.data[i], snr)
        #         self.data[i] += noise

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature, label = self.features[idx], int(self.labels[idx])
        if self.transform != None:
            feature = self.transform(feature)

        return feature, label


class Normalize(object):
    def __init__(self, type = "mean-std"): # "0-1","-1-1","mean-std"
        self.type = type

    def __call__(self, seq):
        if  self.type == "0-1":
            seq =(seq-seq.min())/(seq.max()-seq.min())
            # seq =(seq-self.min)/(self.max-self.min)
        elif  self.type == "-1-1":
            seq = 2*(seq-seq.min())/(seq.max()-seq.min()) + -1
            #seq = 2*(seq-self.min)/(self.max-self.min) + -1
        elif self.type == "mean-std" :
            #seq = (seq-self.mean)/self.std
            seq = (seq-seq.mean())/seq.std()
        else:
            raise NameError('This normalization is not included!')

        return seq


def get_raw_1d(rootdir, batch_size, trainonly=False, split=0.5, snr=0, normal=0):

    data = np.load(os.path.join(rootdir, 'data_features_train.npy')).astype(np.float32)
    labels = np.load(os.path.join(rootdir, 'data_labels_train.npy')).astype(np.uint8)
    classes = np.unique(labels)
    total_len = len(data)
    print('load data from {} into shape {}'.format(os.path.join(rootdir, 'data_features_train.npy'), data.shape))
    print('load labels from {} into shape {}'.format(os.path.join(rootdir, 'data_labels_train.npy'), labels.shape))

    train_features = None
    train_labels = None
    test_features = None
    test_labels = None

    train_dataset = None
    test_dataset = None

    pre_process = None
    if normal == 1:
        pre_process = transforms.Compose([Normalize(type="0-1")])
    elif normal == 2:
        pre_process = transforms.Compose([Normalize(type="-1-1")])
    elif normal == 3:
        pre_process = transforms.Compose([Normalize(type="mean-std")])
    else:
        pre_process = None


    if trainonly:
        # 全部数据用于训练集
        train_features = data[:, np.newaxis, :]
        train_labels = labels
        train_dataset = BearingDataset(train_features, train_labels, transform=pre_process)

    else:
        # 将数据切分为训练集和测试集，切分比例按照参数split
        # 可以用train_test_split，也可以手动切割。
        # 测试比较一下哪个更合适
        train_features, test_features, train_labels, test_labels = train_test_split(data[:, np.newaxis, :], labels, test_size=1-split)

        #train_num = int(np.floor(total_len * split))
        #train_features = data[:train_num]
        #train_features = train_features[:, np.newaxis, :]

        #test_features = data[train_num:]
        #test_features = test_features[:, np.newaxis, :]

        #train_labels = labels[:train_num]
        #test_labels = labels[train_num:]

        train_dataset = BearingDataset(train_features, train_labels, transform=pre_process)
        test_dataset = BearingDataset(test_features, test_labels, transform=pre_process)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    if test_dataset != None:
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True)
    else:
        test_loader = None

    return train_loader, test_loader, classes 
