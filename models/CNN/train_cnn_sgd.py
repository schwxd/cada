import os
import math
import logging
import numpy as np
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import torch.nn.functional as F

from utils.vis import draw_tsne, draw_confusion_matrix
from utils.functions import test, set_log_config
from networks.network import Extractor, Classifier, Critic, Critic2, RandomLayer, AdversarialNetwork, ClassifierAux
from networks.inceptionv1 import InceptionV1, InceptionV1s


def inv_lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = lr * (1 + gamma * iter_num) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print('epoch {}, lr {}'.format(iter_num, lr))
    return optimizer

def train_cnn(config):
    if config['network'] == 'inceptionv1':
        extractor = InceptionV1(num_classes=32, dilation=config['dilation'])
    elif config['network'] == 'inceptionv1s':
        extractor = InceptionV1s(num_classes=32, dilation=config['dilation'])
    else:
        extractor = Extractor(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'])
    classifier = Classifier(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], n_class=config['n_class'])

    if torch.cuda.is_available():
        extractor = extractor.cuda()
        classifier = classifier.cuda()

    res_dir = os.path.join(config['res_dir'], 'normal{}-{}-lr{}'.format(config['normal'], 
                                                                        config['network'],
                                                                        config['lr']))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    set_log_config(res_dir)
    logging.debug(extractor)
    logging.debug(classifier)
    logging.debug(config)

    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = optim.Adam(
    #     list(extractor.parameters()) + list(classifier.parameters()),
    #     lr = config['lr'])
    opts = optim.SGD(
        list(extractor.parameters()) + list(classifier.parameters()),
        lr = config['lr'], nesterov=True, momentum=0.9)

    def train(extractor, classifier, config, epoch):
        extractor.train()
        classifier.train()

        optimizer = inv_lr_scheduler(opts, epoch, gamma=0.01, power=0.75, lr=config['lr'], weight_decay=0.0005)

        for step, (features, labels) in enumerate(config['source_train_loader']):
            if torch.cuda.is_available():
                features, labels = features.cuda(), labels.cuda()

            optimizer.zero_grad()
            # if config['aux_classifier'] == 1:
            #     x1, x2, x3 = extractor(features)
            #     preds = classifier(x1, x2, x3)
            preds = classifier(extractor(features))
            # print('preds {}, labels {}'.format(preds.shape, labels.shape))
            # print(preds[0])
            # preds_l = F.softmax(preds, dim=1)
            # print('preds_l {}'.format(preds_l.shape))
            # print(preds_l[0])
            # print('------')

            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

    for epoch in range(1, config['n_epochs'] + 1):
        train(extractor, classifier, config, epoch)
        if epoch % config['TEST_INTERVAL'] == 0:
            print('test on source_test_loader')
            test(extractor, classifier, config['source_test_loader'], epoch)
            print('test on target_test_loader')
            test(extractor, classifier, config['target_test_loader'], epoch)
        # if epoch % config['VIS_INTERVAL'] == 0:
        #     draw_confusion_matrix(extractor, classifier, config['target_test_loader'], res_dir, epoch, config['models'])
            # draw_tsne(extractor, classifier, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, config['models'], separate=True)
            # draw_tsne(extractor, classifier, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, config['models'], separate=False)

