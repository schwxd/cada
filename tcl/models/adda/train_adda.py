import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks.network import Extractor, Classifier, Critic, Critic2, RandomLayer, AdversarialNetwork
from utils.functions import test, set_log_config, set_requires_grad
from utils.vis import draw_tsne, draw_confusion_matrix

def train_adda(config):

    criterion = torch.nn.CrossEntropyLoss()
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()

    TEST_INTERVAL = 10
    lr = config['lr']
    l2_decay = 5e-4
    momentum = 0.9

    res_dir = os.path.join(config['res_dir'], 'lr{}'.format(config['lr']))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    set_log_config(res_dir)

    logging.debug('train_adda')
    logging.debug(model.feature)
    logging.debug(model.class_classifier)
    logging.debug(config)

    def pretrain(model, config, pretrain_epochs):
        model.class_classifier.train()
        model.feature.train()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(pretrain_epochs):
            for step, (features, labels) in enumerate(config['source_train_loader']):
                if torch.cuda.is_available():
                    features, labels = features.cuda(), labels.cuda()

                optimizer.zero_grad()

                preds = model.class_classify(features)
                loss = criterion(preds, labels)

                loss.backward()
                optimizer.step()


    def train(model, config, epoch):
        model.class_classifier.train()
        model.feature.train()

        # LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / config['n_epochs']), 0.75)
        # print('epoch {}, learning rate{: .4f}'.format(epoch, LEARNING_RATE) )
        # optimizer = torch.optim.SGD([
        #     {'params': model.feature.parameters()},
        #     {'params': model.class_classifier.parameters(), 'lr': LEARNING_RATE},
        #     ], lr= LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

        optimizer = optim.Adam(model.parameters(), lr=lr)
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

            class_output_s, domain_output, embeddings_s = model.dann(input_data=data_source, alpha=gamma)
            # print('domain_output {}'.format(domain_output.size()))
            err_s_label = loss_class(class_output_s, label_source)
            domain_label = torch.zeros(data_source.size(0)).long().cuda()
            err_s_domain = loss_domain(domain_output, domain_label)

            # Training model using target data
            domain_label = torch.ones(data_target.size(0)).long().cuda()
            class_output_t, domain_output, embeddings_t = model.dann(input_data=data_target, alpha=gamma)
            err_t_domain = loss_domain(domain_output, domain_label)
            err = err_s_label + err_s_domain + err_t_domain

            if i % 50 == 0:
                print('err_s_label {}, err_s_domain {}, gamma {}, err_t_domain {}, total err {}'.format(err_s_label.item(), err_s_domain.item(), gamma, err_t_domain.item(), err.item()))

            err.backward()
            optimizer.step()

    pretrain(model, config, pretrain_epochs=20)
    for epoch in range(1, config['n_epochs'] + 1):
        train(model, config, epoch)
        if epoch % config['TEST_INTERVAL'] == 0:
            print('test on source_test_loader')
            test(extractor, classifier, config['source_test_loader'], epoch)
            # print('test on target_train_loader')
            # test(model, config['target_train_loader'], epoch)
            print('test on target_test_loader')
            test(extractor, classifier, config['target_test_loader'], epoch)
        if epoch % config['VIS_INTERVAL'] == 0:
            draw_confusion_matrix(extractor, classifier, config['target_test_loader'], res_dir, epoch, config['models'])
            draw_tsne(extractor, classifier, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, config['models'], separate=True)
            # draw_tsne(extractor, classifier, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, config['models'], separate=False)



