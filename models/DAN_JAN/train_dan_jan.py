import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.functions import test, set_log_config, ReverseLayerF
from utils.vis import draw_tsne, draw_confusion_matrix
from network import Extractor, Classifier, Critic, Critic2, RandomLayer, AdversarialNetwork
import models.DAN_JAN.dan_jan_loss as loss

def train_dan_jan(config):
    extractor = Extractor(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'])
    classifier = Classifier(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], n_class=config['n_class'])
    if torch.cuda.is_available():
        extractor = extractor.cuda()
        classifier = classifier.cuda()

    criterion = nn.CrossEntropyLoss()
    transfer_criterion = loss.loss_dict[config['models']]

    ## add additional network for some methods
    if config['models'] == "JAN" or config['models'] == "JAN_Linear":
        softmax_layer = nn.Softmax(dim=1).cuda()

    l2_decay = 5e-4
    momentum = 0.9

    res_dir = os.path.join(config['res_dir'], 'lr{}-mmdgamma{}'.format(config['lr'], config['mmd_gamma']))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    set_log_config(res_dir)

    logging.debug('train_dan_jan {}'.format(config['models']))
    logging.debug(extractor)
    logging.debug(classifier)
    logging.debug(config)

    def train(extractor, classifier, config, epoch):
        extractor.train()
        classifier.train()

        LEARNING_RATE = config['lr'] / math.pow((1 + 10 * (epoch - 1) / config['n_epochs']), 0.75)
        print('epoch {}, learning rate{: .4f}'.format(epoch, LEARNING_RATE) )
        optimizer = torch.optim.SGD([
            {'params': extractor.parameters()},
            {'params': classifier.parameters(), 'lr': LEARNING_RATE},
            ], lr= LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

        # optimizer = optim.Adam(model.parameters(), lr=lr)
        gamma = 2 / (1 + math.exp(-10 * (epoch) / config['n_epochs'])) - 1

        iter_source = iter(config['source_train_loader'])
        iter_target = iter(config['target_train_loader'])
        len_source_loader = len(config['source_train_loader'])
        len_target_loader = len(config['target_train_loader'])
        num_iter = len_source_loader
        for i in range(1, num_iter):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            if i % len_target_loader == 0:
                iter_target = iter(config['target_train_loader'])
            if torch.cuda.is_available():
                data_source, label_source = data_source.cuda(), label_source.cuda()
                data_target = data_target.cuda()

            optimizer.zero_grad()

            feature_source = extractor(data_source)
            feature_source = feature_source.view(feature_source.size(0), -1)
            preds_source = classifier(feature_source)
            classifier_loss = criterion(preds_source, label_source)

            feature_target = extractor(data_target)
            feature_target = feature_target.view(feature_target.size(0), -1)
            preds_target = classifier(feature_target)


            if config['models'] == "DAN" or config['models'] == "DAN_Linear":
                transfer_loss = transfer_criterion(feature_source, feature_target)
            elif config['models'] == "JAN" or config['models'] == "JAN_Linear":
                softmax_source = softmax_layer(preds_source)
                softmax_target = softmax_layer(preds_target)
                transfer_loss = transfer_criterion([feature_source, softmax_source], [feature_target, softmax_target])

            total_loss = config['mmd_gamma'] * transfer_loss + classifier_loss
            if i % 50 == 0:
                print('transfer_loss: {}, classifier_loss: {}, total_loss: {}'.format(transfer_loss.item(), classifier_loss.item(), total_loss.item()))

            total_loss.backward()
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

