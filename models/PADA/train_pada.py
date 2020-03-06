"""
python .\main.py --models=PADA --dataroot=data --src=DEload0 --dest=DEload3 --network=cnn --n_flattens=128
"""

import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import models.CDAN.cdan_loss as loss_func

from networks.network import Extractor, Classifier, Critic, Critic2, RandomLayer, AdversarialNetwork
from networks.resnet18_1d import resnet18_features
from networks.inceptionv4 import InceptionV4
from networks.inceptionv1 import InceptionV1, InceptionV1s

from torchsummary import summary
from utils.functions import test, set_log_config
from utils.vis import draw_tsne, draw_confusion_matrix


def train_pada(config):
    if config['network'] == 'inceptionv1':
        extractor_s = InceptionV1(num_classes=32)
        extractor_t = InceptionV1(num_classes=32)
    elif config['network'] == 'inceptionv1s':
        extractor_s = InceptionV1s(num_classes=32)
        extractor_t = InceptionV1s(num_classes=32)
    else:
        extractor_s = Extractor(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'])
        extractor_t = Extractor(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'])

    classifier_s = Classifier(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], n_class=config['n_class'])
    classifier_t = Classifier(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], n_class=config['n_class'])

    if torch.cuda.is_available():
        extractor_s = extractor_s.cuda()
        classifier_s = classifier_s.cuda()

        extractor_t = extractor_t.cuda()
        classifier_t = classifier_t.cuda()

    cdan_random = config['random_layer'] 
    res_dir = os.path.join(config['res_dir'], 'normal{}-{}-cons{}-lr{}'.format(config['normal'], 
                                                                        config['network'], 
                                                                        config['pada_cons_w'], 
                                                                        config['lr']))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    print('train_pada')
    print(config)

    set_log_config(res_dir)
    logging.debug('train_pada')
    # logging.debug(extractor)
    # logging.debug(classifier)
    logging.debug(config)

    if config['models'] == 'PADA':
        random_layer = None
        ad_net = AdversarialNetwork(config['n_flattens'], config['n_hiddens'])
    elif cdan_random:
        random_layer = RandomLayer([config['n_flattens'], config['n_class']], config['n_hiddens'])
        ad_net = AdversarialNetwork(config['n_hiddens'], config['n_hiddens'])
        random_layer.cuda()
    else:
        random_layer = None
        ad_net = AdversarialNetwork(config['n_flattens'] * config['n_class'], config['n_hiddens'])
    ad_net = ad_net.cuda()
    optimizer_s = torch.optim.Adam([
        {'params': extractor_s.parameters(), 'lr': config['lr']},
        {'params': classifier_s.parameters(), 'lr': config['lr']}
        ])
    optimizer_t = torch.optim.Adam([
        {'params': extractor_t.parameters(), 'lr': config['lr']},
        {'params': classifier_t.parameters(), 'lr': config['lr']}
        ])

    optimizer_ad = torch.optim.Adam(ad_net.parameters(), lr=config['lr'])

    def train_stage1(extractor_s, classifier_s, config, epoch):
        extractor_s.train()
        classifier_s.train()

        # STAGE 1: 
        # 在labeled source上训练extractor_s和classifier_s
        # 训练完成后freeze这两个model 
        iter_source = iter(config['source_train_loader'])
        len_source_loader = len(config['source_train_loader'])
        for step in range(1, len_source_loader + 1):
            data_source, label_source = iter_source.next()
            if torch.cuda.is_available():
                data_source, label_source = data_source.cuda(), label_source.cuda()

            optimizer_s.zero_grad()

            h_s = extractor_s(data_source)
            h_s = h_s.view(h_s.size(0), -1)
            source_preds = classifier_s(h_s)
            cls_loss = nn.CrossEntropyLoss()(source_preds, label_source)

            cls_loss.backward()
            optimizer_s.step()


    def train(extractor_s, classifier_s, extractor_t, classifier_t, ad_net, config, epoch):
        start_epoch = 0

        # extractor_s.train()
        # classifier_s.train()
        # ad_net.train()

        # # STAGE 1: 
        # # 在labeled source上训练extractor_s和classifier_s
        # # 训练完成后freeze这两个model 
        # iter_source = iter(config['source_train_loader'])
        # len_source_loader = len(config['source_train_loader'])
        # for step in range(1, len_source_loader + 1):
        #     data_source, label_source = iter_source.next()
        #     if torch.cuda.is_available():
        #         data_source, label_source = data_source.cuda(), label_source.cuda()

        #     optimizer_s.zero_grad()

        #     h_s = extractor_s(data_source)
        #     h_s = h_s.view(h_s.size(0), -1)
        #     source_preds = classifier_s(h_s)
        #     cls_loss = nn.CrossEntropyLoss()(source_preds, label_source)

        #     cls_loss.backward()
        #     optimizer_s.step()

        # for param in extractor_s.parameters():
        #     param.requires_grad = False
        # for param in classifier_s.parameters():
        #     param.requires_grad = False

        # STAGE 2: 
        # 使用新的extractor和classifier进行DANN训练
        # 不同的地方是，每个target 同时使用extractor_s和extractor_t

        extractor_t.train()
        classifier_t.train()
        ad_net.train()

        iter_source = iter(config['source_train_loader'])
        iter_target = iter(config['target_train_loader'])
        len_source_loader = len(config['source_train_loader'])
        len_target_loader = len(config['target_train_loader'])
        num_iter = len_source_loader
        for step in range(1, num_iter + 1):
            data_source, label_source = iter_source.next()
            data_target, label_target = iter_target.next()
            if step % len_target_loader == 0:
                iter_target = iter(config['target_train_loader'])
            if torch.cuda.is_available():
                data_source, label_source = data_source.cuda(), label_source.cuda()
                data_target, label_target = data_target.cuda(), label_target.cuda()

            optimizer_t.zero_grad()
            optimizer_ad.zero_grad()

            h_s = extractor_t(data_source)
            h_s = h_s.view(h_s.size(0), -1)
            h_t = extractor_t(data_target)
            h_t = h_t.view(h_t.size(0), -1)

            source_preds = classifier_t(h_s)
            cls_loss = nn.CrossEntropyLoss()(source_preds, label_source)
            softmax_output_s = nn.Softmax(dim=1)(source_preds)

            target_preds = classifier_t(h_t)
            softmax_output_t = nn.Softmax(dim=1)(target_preds)
            if config['target_labeling'] == 1:
                cls_loss += nn.CrossEntropyLoss()(target_preds, label_target)

            feature = torch.cat((h_s, h_t), 0)
            softmax_output = torch.cat((softmax_output_s, softmax_output_t), 0)

            if epoch > start_epoch:
                gamma = 2 / (1 + math.exp(-10 * (epoch) / config['n_epochs'])) - 1
                if config['models'] == 'CDAN-E':
                    entropy = loss_func.Entropy(softmax_output)
                    d_loss = loss_func.CDAN([feature, softmax_output], ad_net, gamma, entropy, loss_func.calc_coeff(num_iter*(epoch-start_epoch)+step), random_layer)
                elif config['models'] == 'CDAN':
                    d_loss = loss_func.CDAN([feature, softmax_output], ad_net, gamma, None, None, random_layer)
                elif config['models'] == 'PADA':
                    d_loss = loss_func.DANN(feature, ad_net, gamma)
                else:
                    raise ValueError('Method cannot be recognized.')
            else:
                d_loss = 0

            # constraints loss
            h_s_prev = extractor_s(data_source)
            cons_loss = nn.L1Loss()(h_s, h_s_prev)

            loss = cls_loss + d_loss + config['pada_cons_w'] * cons_loss
            loss.backward()
            optimizer_t.step()
            if epoch > start_epoch:
                optimizer_ad.step()
            if (step) % 20 == 0:
                print('Train Epoch {} closs {:.6f}, dloss {:.6f}, cons_loss {:.6f}, Loss {:.6f}'.format(epoch, cls_loss.item(), d_loss.item(), cons_loss.item(), loss.item()))

    for epoch in range(1, config['n_epochs'] + 1):
        train_stage1(extractor_s, classifier_s, config, epoch)
        if epoch % config['TEST_INTERVAL'] == 0:
            print('test on source_test_loader')
            test(extractor_s, classifier_s, config['source_test_loader'], epoch)
            # print('test on target_test_loader')
            # accuracy = test(extractor_s, classifier_s, config['target_test_loader'], epoch)

    extractor_t.load_state_dict(extractor_s.state_dict())
    classifier_t.load_state_dict(classifier_s.state_dict())

    for param in extractor_s.parameters():
        param.requires_grad = False
    for param in classifier_s.parameters():
        param.requires_grad = False


    for epoch in range(1, config['n_epochs'] + 1):
        train(extractor_s, classifier_s, extractor_t, classifier_t, ad_net, config, epoch)
        if epoch % config['TEST_INTERVAL'] == 0:
            # print('test on source_test_loader')
            # test(extractor, classifier, config['source_test_loader'], epoch)
            print('test on target_test_loader')
            accuracy = test(extractor_t, classifier_t, config['target_test_loader'], epoch)

        if epoch % config['VIS_INTERVAL'] == 0:
            title = config['models']
            draw_confusion_matrix(extractor_t, classifier_t, config['target_test_loader'], res_dir, epoch, title)
            draw_tsne(extractor_t, classifier_t, config['source_train_loader'], config['target_test_loader'], res_dir, epoch, title, separate=True)
            # draw_tsne(extractor, classifier, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, title, separate=False)
