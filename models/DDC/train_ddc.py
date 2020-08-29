import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.functions import test, set_log_config
from utils.vis import draw_tsne, draw_confusion_matrix
from networks.network import Extractor, Classifier, Critic, Critic2, RandomLayer, AdversarialNetwork, Classifier2
from networks.inceptionv1 import InceptionV1, InceptionV1s


# import models.DDC.mmd
from models.DDC.mmd import mmd_linear

def train_ddc(config):
    if config['network'] == 'inceptionv1':
        extractor = InceptionV1(num_classes=32, dilation=config['dilation'])
    elif config['network'] == 'inceptionv1s':
        extractor = InceptionV1s(num_classes=32, dilation=config['dilation'])
    else:
        extractor = Extractor(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'])
    classifier = Classifier2(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], n_class=config['n_class'])

    if torch.cuda.is_available():
        extractor = extractor.cuda()
        classifier = classifier.cuda()

    res_dir = os.path.join(config['res_dir'], 'slim{}-targetLabel{}-normal{}-{}-dilation{}-lr{}-mmdgamma{}'.format(config['normal'], 
                                                                                    config['slim'],
                                                                                    config['target_labeling'],
                                                                                    config['network'], 
                                                                                    config['dilation'], 
                                                                                    config['lr'], 
                                                                                    config['mmd_gamma']))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    criterion = torch.nn.CrossEntropyLoss()

    set_log_config(res_dir)
    logging.debug('train_ddc')
    logging.debug(extractor)
    logging.debug(classifier)
    logging.debug(config)

    optimizer = optim.Adam(
        list(extractor.parameters()) + list(classifier.parameters()),
        lr = config['lr'])

    def train(extractor, classifier, config, epoch):
        extractor.train()
        classifier.train()

        iter_source = iter(config['source_train_loader'])
        iter_target = iter(config['target_train_loader'])
        len_source_loader = len(config['source_train_loader'])
        len_target_loader = len(config['target_train_loader'])
        if config['slim'] > 0:
            iter_target_semi = iter(config['target_train_semi_loader'])
            len_target_semi_loader = len(config['target_train_semi_loader'])

        num_iter = len_source_loader
        for i in range(1, num_iter+1):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            if config['slim'] > 0:
                data_target_semi, label_target_semi = iter_target_semi.next()
                if i % len_target_semi_loader == 0:
                    iter_target_semi = iter(config['target_train_semi_loader'])

            if i % len_target_loader == 0:
                iter_target = iter(config['target_train_loader'])
            if torch.cuda.is_available():
                data_source, label_source = data_source.cuda(), label_source.cuda()
                data_target = data_target.cuda()
                if config['slim'] > 0:
                    data_target_semi, label_target_semi = data_target_semi.cuda(), label_target_semi.cuda()

            optimizer.zero_grad()

            source = extractor(data_source)
            source = source.view(source.size(0), -1)
            target = extractor(data_target)
            target = target.view(target.size(0), -1)

            preds, _ = classifier(source)
            loss_cls = criterion(preds, label_source)

            loss_mmd = mmd_linear(source, target)

            #gamma = 2 / (1 + math.exp(-10 * (epoch) / config['n_epochs'])) - 1
            loss = loss_cls + config['mmd_gamma'] * loss_mmd

            if config['slim'] > 0:
                feature_target_semi = extractor(data_target_semi)
                feature_target_semi = feature_target_semi.view(feature_target_semi.size(0), -1)
                preds_target_semi, _ = classifier(feature_target_semi)
                loss += criterion(preds_target_semi, label_target_semi)

            if i % 50 == 0:
                print('loss_cls {}, loss_mmd {}, gamma {}, total loss {}'.format(loss_cls.item(), loss_mmd.item(), config['mmd_gamma'], loss.item()))
            loss.backward()
            optimizer.step()

    for epoch in range(1, config['n_epochs'] + 1):
        train(extractor, classifier, config, epoch)
        if epoch % config['TEST_INTERVAL'] == 0:
            print('test on source_test_loader')
            test(extractor, classifier, config['source_test_loader'], epoch)
            print('test on target_test_loader')
            test(extractor, classifier, config['target_test_loader'], epoch)
        if epoch % config['VIS_INTERVAL'] == 0:
            draw_confusion_matrix(extractor, classifier, config['target_test_loader'], res_dir, epoch, config['models'])
            draw_tsne(extractor, classifier, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, config['models'], separate=True)
            draw_tsne(extractor, classifier, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, config['models'], separate=False)
