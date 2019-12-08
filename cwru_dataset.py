import os
import torch
import numpy as np

from torchvision import datasets, transforms
import torch.utils.data as data

# white gaussian noise
def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

class BearingDataset1D(data.Dataset):

    def __init__(self, root, train, transform=None, snr=0):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root
        self.transform = transform
        self.train = train

        if train:
            self.data = np.load(os.path.join(self.root_dir, 'data_features_train.npy')).astype(np.float32)
            self.targets = np.load(os.path.join(self.root_dir, 'data_labels_train.npy')).astype(np.uint8)
            print('data shape {}'.format(self.data.shape))
            if snr != 0:
                print('apply snr {} to signal'.format(snr))
                for i in range(len(self.data)):
                    noise = wgn(self.data[i], snr)
                    self.data[i] += noise

            self.data = self.data[:, np.newaxis, :]
            # self.data = self.data.transpose((0,3,1,2))
            print('load data from {} into shape {}'.format(os.path.join(self.root_dir, 'data_features_train.npy'), self.data.shape))
            print('load targets from {} into shape {}'.format(os.path.join(self.root_dir, 'data_labels_train.npy'), self.targets.shape))
        else:
            self.data = np.load(os.path.join(self.root_dir, 'data_features_test.npy')).astype(np.float32)
            self.targets = np.load(os.path.join(self.root_dir, 'data_labels_test.npy')).astype(np.uint8)
            if snr != 0:
                print('apply snr {} to signal'.format(snr))
                for i in range(len(self.data)):
                    noise = wgn(self.data[i], snr)
                    self.data[i] += noise

            self.data = self.data[:, np.newaxis, :]
            # self.data = self.data.transpose((0,3,1,2))
            print('load data from {} into shape {}'.format(os.path.join(self.root_dir, 'data_features_test.npy'), self.data.shape))
            print('load targets from {} into shape {}'.format(os.path.join(self.root_dir, 'data_labels_test.npy'), self.targets.shape))
        self.classes = np.unique(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])

        if self.transform is not None:
            img = self.transform(img)

        return img, target

def get_raw_1d(rootdir, train, batch_size, snr=0):
    # image pre-processing
    # pre_process = transforms.Compose([transforms.ToTensor(),
    #                                     transforms.Normalize(
    #                                         mean=params.dataset_mean,
    #                                         std=params.dataset_std)])

    # dataset and data loader
    bearing_dataset = BearingDataset1D(root=rootdir,
                                   train=train,
                                   transform=None,
                                   snr=snr)
    bearing_data_loader = torch.utils.data.DataLoader(
        dataset=bearing_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)

    return bearing_data_loader, bearing_dataset.classes
