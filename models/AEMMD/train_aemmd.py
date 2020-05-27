import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.functions import test, set_log_config, ReverseLayerF
from utils.vis import draw_tsne, draw_confusion_matrix
from models.DDC.mmd import mmd_linear
from networks.network import Extractor, Classifier, Critic, Critic2, RandomLayer, AdversarialNetwork, Decoder
torch.set_num_threads(2)


def train_aemmd(config):
    extractor = Extractor(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], bn=config['bn'])
    decoder = Decoder(n_flattens=config['n_flattens'], bn=config['bn'])
    classifier = Classifier(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], n_class=config['n_class'])
    if torch.cuda.is_available():
        extractor = extractor.cuda()
        classifier = classifier.cuda()
        decoder = decoder.cuda()

    criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()

    l2_decay = 5e-4
    momentum = 0.9

    res_dir = os.path.join(config['res_dir'], 'slim{}-lr{}-mmdgamma{}'.format(config['slim'], config['lr'], config['mmd_gamma']))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    set_log_config(res_dir)

    logging.debug('train_aemmd {}'.format(config['models']))
    logging.debug(extractor)
    print(extractor)
    print(decoder)
    logging.debug(classifier)
    logging.debug(config)

    def train(extractor, classifier, decoder, config, epoch):
        extractor.train()
        classifier.train()
        decoder.train()

        # LEARNING_RATE = config['lr'] / math.pow((1 + 10 * (epoch - 1) / config['n_epochs']), 0.75)
        # print('epoch {}, learning rate{: .4f}'.format(epoch, LEARNING_RATE) )
        # optimizer = torch.optim.SGD([
        #     {'params': extractor.parameters()},
        #     {'params': classifier.parameters()},
        #     {'params': decoder.parameters()},
        #     ], lr= config['lr'], momentum=momentum, weight_decay=l2_decay)

        optimizer = optim.Adam([{'params': extractor.parameters()},
            {'params': classifier.parameters()},
            {'params': decoder.parameters()}],
            lr=config['lr'])

        iter_source = iter(config['source_train_loader'])
        iter_target = iter(config['target_train_loader'])
        len_source_loader = len(config['source_train_loader'])
        len_target_loader = len(config['target_train_loader'])
        if config['slim'] > 0:
            iter_target_semi = iter(config['target_train_semi_loader'])
            len_target_semi_loader = len(config['target_train_semi_loader'])

        num_iter = len_source_loader
        for i in range(1, num_iter):
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

            feature_source = extractor(data_source)
            feature_source = feature_source.view(feature_source.size(0), -1)
            feature_target = extractor(data_target)
            feature_target = feature_target.view(feature_target.size(0), -1)
            preds_source = classifier(feature_source)

            reconstruct_source = decoder(feature_source)
            reconstruct_target = decoder(feature_target)

            mmd_loss = mmd_linear(feature_source, feature_target)
            classifier_loss = criterion(preds_source, label_source)
            if config['slim'] > 0:
                feature_target_semi = extractor(data_target_semi)
                feature_target_semi = feature_target_semi.view(feature_target_semi.size(0), -1)
                preds_target_semi = classifier(feature_target_semi)
                classifier_loss += criterion(preds_target_semi, label_target_semi)

            recons_loss = 0.5*mse_criterion(reconstruct_source, data_source) + 0.5*mse_criterion(reconstruct_target, data_target)
            total_loss = config['mmd_gamma'] * mmd_loss + classifier_loss + 0.01*recons_loss
            if i % 50 == 0:
                print('mmd_loss {:.2f} classifier_loss {:.2f} recons_loss {:.2f} total_loss {:.2f}'.format(mmd_loss.item(), 
                                    classifier_loss.item(), 
                                    recons_loss.item(), 
                                    total_loss.item()))
            # total_loss = config['mmd_gamma'] * mmd_loss + classifier_loss
            # if i % 50 == 0:
            #     print('mmd_loss {:.2f} classifier_loss {:.2f} total_loss {:.2f}'.format(mmd_loss.item(), 
            #                         classifier_loss.item(), 
            #                         total_loss.item()))
            total_loss.backward()
            optimizer.step()

    for epoch in range(1, config['n_epochs'] + 1):
        train(extractor, classifier, decoder, config, epoch)
        if epoch % config['TEST_INTERVAL'] == 0:
            print('test on source_test_loader')
            test(extractor, classifier, config['source_test_loader'], epoch)
            print('test on target_test_loader')
            test(extractor, classifier, config['target_test_loader'], epoch)
        if epoch % config['VIS_INTERVAL'] == 0:
            draw_confusion_matrix(extractor, classifier, config['target_test_loader'], res_dir, epoch, config['models'])
            draw_tsne(extractor, classifier, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, config['models'], separate=True)
            # draw_tsne(extractor, classifier, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, config['models'], separate=False)

