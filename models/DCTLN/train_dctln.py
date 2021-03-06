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
from networks.network import Extractor, Classifier, Critic, Critic2, RandomLayer, AdversarialNetwork, Classifier2
from networks.inceptionv1 import InceptionV1, InceptionV1s
torch.set_num_threads(2)

def train_dctln(config):
    if config['network'] == 'inceptionv1':
        extractor = InceptionV1(num_classes=32, dilation=config['dilation'])
    elif config['network'] == 'inceptionv1s':
        extractor = InceptionV1s(num_classes=32, dilation=config['dilation'])
    else:
        extractor = Extractor(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'])
    classifier = Classifier2(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], n_class=config['n_class'])

    critic = Critic2(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'])
    if torch.cuda.is_available():
        extractor = extractor.cuda()
        classifier = classifier.cuda()
        critic = critic.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()

    res_dir = os.path.join(config['res_dir'], 'slim{}-snr{}-lr{}'.format(config['slim'], 
                                                                        config['snr'],
                                                                        config['lr']))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    set_log_config(res_dir)
    logging.debug('train_dann')
    logging.debug(extractor)
    logging.debug(classifier)
    logging.debug(critic)
    logging.debug(config)

    optimizer = optim.Adam([{'params': extractor.parameters()},
        {'params': classifier.parameters()},
        {'params': critic.parameters()}],
        lr=config['lr'])


    def dann(input_data, alpha):
        feature = extractor(input_data)
        feature = feature.view(feature.size(0), -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output, _ = classifier(feature)
        domain_output = critic(reverse_feature)

        return class_output, domain_output, feature


    def train(extractor, classifier, critic, config, epoch):
        extractor.train()
        classifier.train()
        critic.train()

        gamma = 2 / (1 + math.exp(-10 * (epoch) / config['n_epochs'])) - 1

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

            if i % len_target_loader == 0:
                iter_target = iter(config['target_train_loader'])
            if config['slim'] > 0:
                if i % len_target_semi_loader == 0:
                    iter_target_semi = iter(config['target_train_semi_loader'])

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
            # loss_mmd = mmd_linear(source, target)


            class_output_s, domain_output, _ = dann(input_data=data_source, alpha=gamma)
            # print('domain_output {}'.format(domain_output.size()))
            err_s_label = loss_class(class_output_s, label_source)
            domain_label = torch.zeros(data_source.size(0)).long().cuda()
            err_s_domain = loss_domain(domain_output, domain_label)

            # Training model using target data
            domain_label = torch.ones(data_target.size(0)).long().cuda()
            class_output_t, domain_output, _ = dann(input_data=data_target, alpha=gamma)
            err_t_domain = loss_domain(domain_output, domain_label)

            # err = 1.0*err_s_label + err_s_domain + err_t_domain + 0*loss_mmd + err_t_label
            err = 1.0*err_s_label + err_s_domain + err_t_domain
            
            if config['slim'] > 0:
                class_output_semi_t, _, _ = dann(input_data=data_target_semi, alpha=gamma)
                err_t_label = loss_class(class_output_semi_t, label_target_semi)
                err += err_t_label

            # if i % 200 == 0:
            #     # print('err_s_label {}, err_s_domain {}, gamma {}, err_t_domain {}, loss_mmd {}, total err {}'.format(err_s_label.item(), err_s_domain.item(), gamma, err_t_domain.item(), loss_mmd.item(), err.item()))
            #     print('err_s_label {:.2f}, err_t_label {:.2f}, err_s_domain {:.2f}, gamma {:.2f}, err_t_domain {:.2f}, total err {:.2f}'.format(err_s_label.item(), err_t_label.item(), err_s_domain.item(), gamma, err_t_domain.item(), err.item()))

            err.backward()
            optimizer.step()


    for epoch in range(1, config['n_epochs'] + 1):
        train(extractor, classifier, critic, config, epoch)
        if epoch % config['TEST_INTERVAL'] == 0:
            # print('test on source_test_loader')
            # test(extractor, classifier, config['source_test_loader'], epoch)
            print('test on target_test_loader')
            test(extractor, classifier, config['target_test_loader'], epoch)
        if epoch % config['VIS_INTERVAL'] == 0:
            draw_confusion_matrix(extractor, classifier, config['target_test_loader'], res_dir, epoch, config['models'])
            draw_tsne(extractor, classifier, config['source_train_loader'], config['target_test_loader'], res_dir, epoch, config['models'], separate=True)
            # draw_tsne(extractor, classifier, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, config['models'], separate=False)
