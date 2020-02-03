import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.functions import test, set_log_config, ReverseLayerF
from network import Extractor, Classifier, Critic, Critic2, RandomLayer, AdversarialNetwork
from network import Predictor, Predictor_deep
from utils.vis import draw_tsne, draw_confusion_matrix
from models.inceptionv4 import InceptionV4
from torchsummary import summary


def train_dann_mm(config):
    if config['inception'] == 1:
        extractor = InceptionV4(num_classes=32)
    else:
        extractor = Extractor(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'])
    classifier = Predictor_deep(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], num_class=config['n_class'])
    if torch.cuda.is_available():
        extractor = extractor.cuda()
        classifier = classifier.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    res_dir = os.path.join(config['res_dir'], 'normal{}-lr{}'.format(config['normal'], config['lr']))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    set_log_config(res_dir)
    logging.debug('train_dann')
    logging.debug(extractor)
    logging.debug(classifier)
    logging.debug(config)

    optimizer = optim.Adam([{'params': extractor.parameters()},
        {'params': classifier.parameters()}
        ],
        lr=config['lr'])


    # def dann(input_data, alpha):
    #     feature = extractor(input_data)
    #     feature = feature.view(feature.size(0), -1)
    #     reverse_feature = ReverseLayerF.apply(feature, alpha)
    #     class_output = classifier(feature)
    #     domain_output = critic(reverse_feature)

    #     return class_output, domain_output, feature


    def adentropy(F1, feat, lamda, eta=1.0):
        out_t1 = F1(feat, reverse=True, eta=eta)
        out_t1 = F.softmax(out_t1, dim=1)
        loss_adent = lamda * torch.mean(torch.sum(out_t1 *
                                                (torch.log(out_t1 + 1e-5)), 1))
        return loss_adent

    def train(extractor, classifier, critic, config, epoch):
        extractor.train()
        classifier.train()

        gamma = 2 / (1 + math.exp(-10 * (epoch) / config['n_epochs'])) - 1

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


            # 在source上训练
            optimizer.zero_grad()
            feature_s = extractor(data_source)
            feature_s = feature_s.view(feature_s.size(0), -1)
            class_output_s = classifier(feature_s)
            class_loss = criterion(class_output_s, label_source)
            class_loss.backward(retain_graph=True)
            optimizer.step()


            # 在target上训练
            optimizer.zero_grad()
            feature_t = extractor(data_target)
            entropy_loss = adentropy(classifier, feature_t, 1)
            entropy_loss.backward()
            optimizer.step()
        
            # err_s_label = loss_class(class_output_s, label_source)
            # domain_label = torch.zeros(data_source.size(0)).long().cuda()
            # err_s_domain = loss_domain(domain_output, domain_label)

            # # Training model using target data
            # domain_label = torch.ones(data_target.size(0)).long().cuda()
            # class_output_t, domain_output, _ = dann(input_data=data_target, alpha=gamma)
            # #class_output_t, domain_output, _ = dann(input_data=data_target, alpha=0.5)
            # err_t_domain = loss_domain(domain_output, domain_label)
            # err = err_s_label + err_s_domain + err_t_domain

            if i % 20 == 0:
                print('epoch {}, class_loss {}, entropy_loss {}'.format(epoch, class_loss.item(), entropy_loss.item()))

            # err.backward()
            # optimizer.step()


    for epoch in range(1, config['n_epochs'] + 1):
        train(extractor, classifier, None, config, epoch)
        if epoch % config['TEST_INTERVAL'] == 0:
            # print('test on source_test_loader')
            # test(extractor, classifier, config['source_test_loader'], epoch)
            print('test on target_test_loader')
            test(extractor, classifier, config['target_test_loader'], epoch)
        if epoch % config['VIS_INTERVAL'] == 0:
            draw_confusion_matrix(extractor, classifier, config['target_test_loader'], res_dir, epoch, config['models'])
            draw_tsne(extractor, classifier, config['source_train_loader'], config['target_test_loader'], res_dir, epoch, config['models'], separate=True)
            draw_tsne(extractor, classifier, config['source_train_loader'], config['target_test_loader'], res_dir, epoch, config['models'], separate=False)
