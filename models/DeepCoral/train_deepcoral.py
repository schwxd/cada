import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from functions import test, set_log_config
from network import Extractor, Classifier, Critic, Critic2, RandomLayer, AdversarialNetwork
from vis import draw_tsne, draw_confusion_matrix

from models.DeepCoral.Coral import CORAL

def train_deepcoral(config):
    extractor = Extractor(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'])
    classifier = Classifier(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], n_class=config['n_class'])
    if torch.cuda.is_available():
        extractor = extractor.cuda()
        classifier = classifier.cuda()

    res_dir = os.path.join(config['res_dir'], 'lr{}-mmdgamma{}'.format(config['lr'], config['mmd_gamma']))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    criterion = torch.nn.CrossEntropyLoss()

    set_log_config(res_dir)
    logging.debug('train_deepcoral')
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
        num_iter = len_source_loader
        for i in range(1, num_iter+1):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            if i % len_target_loader == 0:
                iter_target = iter(config['target_train_loader'])
            if torch.cuda.is_available():
                data_source, label_source = data_source.cuda(), label_source.cuda()
                data_target = data_target.cuda()

            optimizer.zero_grad()

            source = extractor(data_source)
            source = source.view(source.size(0), -1)
            target = extractor(data_target)
            target = target.view(target.size(0), -1)

            preds = classifier(source)
            loss_cls = criterion(preds, label_source)

            loss_coral = CORAL(source, target)

            # gamma = 2 / (1 + math.exp(-10 * (epoch) / config['n_epochs'])) - 1
            # loss = loss_cls + gamma * loss_coral
            loss = loss_cls + config['mmd_gamma'] * loss_coral
            if i % 50 == 0:
                print('loss_cls {}, loss_coral {}, gamma {}, total loss {}'.format(loss_cls.item(), loss_coral.item(), config['mmd_gamma'], loss.item()))
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
            # draw_tsne(extractor, classifier, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, config['models'], separate=False)
