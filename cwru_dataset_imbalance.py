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
    def __init__(self, features, labels, transform, snr=0, snrp=0):
        self.features = features
        self.labels = labels
        self.transform = transform

        if snr != 0 and snrp != 0:
            print('apply noise with probability of {} and level {} to signal'.format(snrp, snr))
            idxs = int(np.floor(len(features) * snrp))
            with_noise = np.ones(idxs)
            without_noise = np.zeros(len(features)-idxs)
            snr_weights = np.concatenate((with_noise, without_noise))
            np.random.shuffle(snr_weights)
            for i in range(len(self.features)):
                if snr_weights[i] == 1:
                    noise = wgn(self.features[i], snr)
                    self.features[i] += noise

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


def get_raw_1d(rootdir, batch_size, trainonly=False, split=0.5, snr=0, snrp=0, normal=0, slim=0, target_labeling=0):

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
    train_dataset_semi = None
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
        train_dataset = BearingDataset(train_features, train_labels, transform=pre_process, snr=snr, snrp=snrp)

    else:
        # 将数据切分为训练集和测试集，切分比例按照参数split
        train_features, test_features, train_labels, test_labels = train_test_split(data[:, np.newaxis, :], labels, test_size=1-split)
        train_features_semi = []
        train_labels_semi = []

        if slim > 0:
            for class_label in np.unique(train_labels):
                label = train_labels[train_labels == class_label]
                feature = train_features[train_labels == class_label]
                train_features_semi.append(feature[:slim])
                train_labels_semi.append(label[:slim])
            train_features_semi = np.concatenate(train_features_semi, axis=0)
            train_labels_semi = np.concatenate(train_labels_semi, axis=0)
            shuffle_array = np.arange(len(train_features_semi))
            np.random.shuffle(shuffle_array)
            train_features_semi = train_features_semi[shuffle_array]
            train_labels_semi = train_labels_semi[shuffle_array]
            print('train_features_semi {}, train_labels_semi {}'.format(train_features_semi.shape, train_labels_semi.shape))
            print('np.bincount(train_labels_semi) {}'.format(np.bincount(train_labels_semi)))
            train_dataset_semi = BearingDataset(train_features_semi, train_labels_semi, transform=pre_process, snr=snr, snrp=snrp)

            if target_labeling == 1:
                # 只保留healthy
                selected = 0

                train_labels_normal = train_labels[train_labels == selected]
                train_features_normal = train_features[train_labels == selected]
                print('selected class: {}'.format(selected))
                print('train_features_normal {}, train_labels_normal {}'.format(train_features_normal.shape, train_labels_normal.shape))
                print('np.bincount(train_labels_normal) {}'.format(np.bincount(train_labels_normal)))
                train_dataset = BearingDataset(train_features_normal, train_labels_normal, transform=pre_process, snr=snr, snrp=snrp)
            elif target_labeling == 2:
                # 按照比例裁剪                
                percents = {0:1, 1:0.5, 2:0.5, 3:0.5}

                classes = np.unique(train_labels)
                train_features_imbalance = []
                train_labels_imbalance = []
                for class_i in classes:
                    train_labels_i = train_labels[train_labels == class_i]
                    train_features_i = train_features[train_labels == class_i]
                    length_io = len(train_labels_i)
                    length_i = int(length_io * percents[class_i])
                    train_labels_i = train_labels_i[:length_i]
                    train_features_i = train_features_i[:length_i]
                    train_features_imbalance.append(train_features_i)
                    train_labels_imbalance.append(train_labels_i)
                    print('selected class: {}, length {}, percent {}, after prone {}'.format(class_i, length_io, percents[class_i], length_i))

                train_features_imbalance = np.concatenate(train_features_imbalance, axis=0)
                train_labels_imbalance = np.concatenate(train_labels_imbalance, axis=0)
                shuffle_array = np.arange(len(train_features_imbalance))
                np.random.shuffle(shuffle_array)
                train_features_imbalance = train_features_imbalance[shuffle_array]
                train_labels_imbalance = train_labels_imbalance[shuffle_array]
                print('train_features_imbalance {}, train_labels_imbalance {}'.format(train_features_imbalance.shape, train_labels_imbalance.shape))
                print('np.bincount(train_labels_imbalance) {}'.format(np.bincount(train_labels_imbalance)))
                train_dataset = BearingDataset(train_features_imbalance, train_labels_imbalance, transform=pre_process, snr=snr, snrp=snrp)

            else:
                print('train_features {}, train_labels {}'.format(train_features.shape, train_labels.shape))
                print('np.bincount(train_labels) {}'.format(np.bincount(train_labels)))
                train_dataset = BearingDataset(train_features, train_labels, transform=pre_process, snr=snr, snrp=snrp)

        else:

            if target_labeling == 1:
                # 只保留healthy，类标是6
                # labels_dict {'B007': 0, 'B014': 1, 'B021': 2, 'IR007': 3, 'IR014': 4, 'IR021': 5, 'Normal': 6, 'OR007@6': 7, 'OR014@6': 8, 'OR021@6': 9}
                selected = 0

                train_labels_normal = train_labels[train_labels == selected]
                train_features_normal = train_features[train_labels == selected]
                print('selected class: {}'.format(selected))
                print('train_features_normal {}, train_labels_normal {}'.format(train_features_normal.shape, train_labels_normal.shape))
                print('np.bincount(train_labels_normal) {}'.format(np.bincount(train_labels_normal)))
                train_dataset = BearingDataset(train_features_normal, train_labels_normal, transform=pre_process, snr=snr, snrp=snrp)

            elif target_labeling == 2:
                # 按照比例裁剪                
                percents = {0:1, 1:0.5, 2:0.5, 3:0.5}

                classes = np.unique(train_labels)
                train_features_imbalance = []
                train_labels_imbalance = []
                for class_i in classes:
                    train_labels_i = train_labels[train_labels == class_i]
                    train_features_i = train_features[train_labels == class_i]
                    length_io = len(train_labels_i)
                    length_i = int(length_io * percents[class_i])
                    train_labels_i = train_labels_i[:length_i]
                    train_features_i = train_features_i[:length_i]
                    train_features_imbalance.append(train_features_i)
                    train_labels_imbalance.append(train_labels_i)
                    print('selected class: {}, length {}, percent {}, after prone {}'.format(class_i, length_io, percents[class_i], length_i))

                train_features_imbalance = np.concatenate(train_features_imbalance, axis=0)
                train_labels_imbalance = np.concatenate(train_labels_imbalance, axis=0)
                shuffle_array = np.arange(len(train_features_imbalance))
                np.random.shuffle(shuffle_array)
                train_features_imbalance = train_features_imbalance[shuffle_array]
                train_labels_imbalance = train_labels_imbalance[shuffle_array]
                print('train_features_imbalance {}, train_labels_imbalance {}'.format(train_features_imbalance.shape, train_labels_imbalance.shape))
                print('np.bincount(train_labels_imbalance) {}'.format(np.bincount(train_labels_imbalance)))
                train_dataset = BearingDataset(train_features_imbalance, train_labels_imbalance, transform=pre_process, snr=snr, snrp=snrp)

            else:
                print('train_features {}, train_labels {}'.format(train_features.shape, train_labels.shape))
                print('np.bincount(train_labels) {}'.format(np.bincount(train_labels)))
                train_dataset = BearingDataset(train_features, train_labels, transform=pre_process, snr=snr, snrp=snrp)

        # # pada
        # # train_dataset移除一部分class，test_dataset不做处理
        # if slim == 1:
        #     n_class = len(np.unique(train_labels))
        #     classes = np.arange(n_class)
        #     np.random.shuffle(classes)
        #     half = n_class // 2
        #     selected = classes[:half]

        #     train_features_slim = []
        #     train_labels_slim = []
        #     for class_label in selected:
        #         label = train_labels[train_labels == class_label]
        #         feature = train_features[train_labels == class_label]
        #         train_features_slim.append(feature)
        #         train_labels_slim.append(label)
        #     train_features_slim = np.concatenate(train_features_slim, axis=0)
        #     train_labels_slim = np.concatenate(train_labels_slim, axis=0)
        #     print('selected class: {}'.format(selected))
        #     print('train_features_slim {}, train_labels_slim {}'.format(train_features_slim.shape, train_labels_slim.shape))
        #     train_dataset = BearingDataset(train_features_slim, train_labels_slim, transform=pre_process, snr=snr, snrp=snrp)

        # elif slim == 2:
        #     # 只保留healthy，类标是6
        #     # labels_dict {'B007': 0, 'B014': 1, 'B021': 2, 'IR007': 3, 'IR014': 4, 'IR021': 5, 'Normal': 6, 'OR007@6': 7, 'OR014@6': 8, 'OR021@6': 9}
        #     selected = 6

        #     train_labels_slim = train_labels[train_labels == selected]
        #     train_features_slim = train_features[train_labels == selected]
        #     print('selected class: {}'.format(selected))
        #     print('train_features_slim {}, train_labels_slim {}'.format(train_features_slim.shape, train_labels_slim.shape))
        #     train_dataset = BearingDataset(train_features_slim, train_labels_slim, transform=pre_process, snr=snr, snrp=snrp)

        # elif slim == 9:
        #     n_class = len(np.unique(train_labels))
        #     labels_len = len(train_labels)
        #     shuffle_n = int(np.floor(labels_len * 0.1))
        #     remain_n = labels_len - shuffle_n
        #     shuffle_array = np.random.randint(low=1, high=n_class, size=shuffle_n).astype(np.uint8)
        #     print("random shuffle 10'%' labels in training set. shuffle_array len {}".format(len(shuffle_array)))
        #     print('shuffle_array {}'.format(shuffle_array[:20]))
        #     remain_array = np.zeros(remain_n)
        #     noise_array = np.concatenate((shuffle_array, remain_array))
        #     np.random.shuffle(noise_array)
        #     print('original labels {}'.format(train_labels[:20]))
        #     print('noise_array {}'.format(noise_array[:20]))
        #     labels_with_noise = (train_labels + noise_array) % n_class
        #     print('labels_with_noise {}'.format(labels_with_noise[:20]))
        #     # print(labels_with_noise[:20]-labels[:20])
        #     labels_with_noise = np.uint8(labels_with_noise)
        #     train_dataset = BearingDataset(train_features, labels_with_noise, transform=pre_process, snr=snr, snrp=snrp)
        # else:
        #     train_dataset = BearingDataset(train_features, train_labels, transform=pre_process, snr=snr, snrp=snrp)

        print('test_features {}, test_labels {}'.format(test_features.shape, test_labels.shape))
        print('np.bincount(test_labels) {}'.format(np.bincount(test_labels)))
        test_dataset = BearingDataset(test_features, test_labels, transform=pre_process, snr=snr, snrp=snrp)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True)

    if test_dataset != None:
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True)
    else:
        test_loader = None

    if train_dataset_semi != None:
        train_semi_loader = torch.utils.data.DataLoader(
            dataset=train_dataset_semi,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True)
    else:
        train_semi_loader = None

    return train_loader, test_loader, train_semi_loader, classes 
