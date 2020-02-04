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
from networks.inceptionv1 import InceptionV1

from torchsummary import summary
from utils.functions import test, set_log_config
from utils.vis import draw_tsne, draw_confusion_matrix


def train_cdan(config):
    # if config['inception'] == 1:
    #     #extractor = create_inception(1, 32, depth=4)
    #     extractor = InceptionV4(num_classes=32)
    # else:
    #     extractor = Extractor(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'])
    # #extractor = resnet18_features(False)
    # classifier = Classifier(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], n_class=config['n_class'])
    extractor = InceptionV1(num_classes=32)
    classifier = Classifier(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], n_class=config['n_class'])

    if torch.cuda.is_available():
        extractor = extractor.cuda()
        classifier = classifier.cuda()
        #summary(extractor, (1, 5120))

    cdan_random = config['random_layer'] 
    res_dir = os.path.join(config['res_dir'], 'inception{}-normal{}-lr{}'.format(config['inception'], config['normal'], config['lr']))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    print('train_cdan')
    #print(extractor)
    #print(classifier)
    print(config)

    set_log_config(res_dir)
    logging.debug('train_cdan')
    logging.debug(extractor)
    logging.debug(classifier)
    logging.debug(config)

    if config['models'] == 'DANN':
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
    optimizer = torch.optim.Adam([
        {'params': extractor.parameters(), 'lr': config['lr']},
        {'params': classifier.parameters(), 'lr': config['lr']}
        ])
    optimizer_ad = torch.optim.Adam(ad_net.parameters(), lr=config['lr'])


    extractor_path = os.path.join(res_dir, "extractor.pth")
    classifier_path = os.path.join(res_dir, "classifier.pth")
    adnet_path = os.path.join(res_dir, "adnet.pth")

    def train(extractor, classifier, ad_net, config, epoch):
        start_epoch = 0

        extractor.train()
        classifier.train()
        ad_net.train()

        iter_source = iter(config['source_train_loader'])
        iter_target = iter(config['target_train_loader'])
        len_source_loader = len(config['source_train_loader'])
        len_target_loader = len(config['target_train_loader'])
        num_iter = len_source_loader
        for step in range(1, num_iter + 1):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            if step % len_target_loader == 0:
                iter_target = iter(config['target_train_loader'])
            if torch.cuda.is_available():
                data_source, label_source = data_source.cuda(), label_source.cuda()
                data_target = data_target.cuda()

            optimizer.zero_grad()
            optimizer_ad.zero_grad()

            h_s = extractor(data_source)
            h_s = h_s.view(h_s.size(0), -1)
            h_t = extractor(data_target)
            h_t = h_t.view(h_t.size(0), -1)

            source_preds = classifier(h_s)
            cls_loss = nn.CrossEntropyLoss()(source_preds, label_source)
            softmax_output_s = nn.Softmax(dim=1)(source_preds)

            target_preds = classifier(h_t)
            softmax_output_t = nn.Softmax(dim=1)(target_preds)

            feature = torch.cat((h_s, h_t), 0)
            softmax_output = torch.cat((softmax_output_s, softmax_output_t), 0)

            if epoch > start_epoch:
                gamma = 2 / (1 + math.exp(-10 * (epoch) / config['n_epochs'])) - 1
                if config['models'] == 'CDAN-E':
                    entropy = loss_func.Entropy(softmax_output)
                    d_loss = loss_func.CDAN([feature, softmax_output], ad_net, gamma, entropy, loss_func.calc_coeff(num_iter*(epoch-start_epoch)+step), random_layer)
                elif config['models'] == 'CDAN':
                    d_loss = loss_func.CDAN([feature, softmax_output], ad_net, gamma, None, None, random_layer)
                elif config['models'] == 'DANN':
                    d_loss = loss_func.DANN(feature, ad_net, gamma)
                else:
                    raise ValueError('Method cannot be recognized.')
            else:
                d_loss = 0

            loss = cls_loss + d_loss
            loss.backward()
            optimizer.step()
            if epoch > start_epoch:
                optimizer_ad.step()
            if (step) % 20 == 0:
                print('Train Epoch {} closs {:.6f}, dloss {:.6f}, Loss {:.6f}'.format(epoch, cls_loss.item(), d_loss.item(), loss.item()))

    if config['testonly'] == 0:
        best_accuracy = 0
        best_model_index = -1
        for epoch in range(1, config['n_epochs'] + 1):
            train(extractor, classifier, ad_net, config, epoch)
            if epoch % config['TEST_INTERVAL'] == 0:
                # print('test on source_test_loader')
                # test(extractor, classifier, config['source_test_loader'], epoch)
                print('test on target_test_loader')
                accuracy = test(extractor, classifier, config['target_test_loader'], epoch)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_index = epoch
                    torch.save(extractor.state_dict(), extractor_path)
                    torch.save(classifier.state_dict(), classifier_path)
                    torch.save(ad_net.state_dict(), adnet_path)
                print('epoch {} accuracy: {:.6f}, best accuracy {:.6f} on epoch {}'.format(epoch, accuracy, best_accuracy, best_model_index))


            if epoch % config['VIS_INTERVAL'] == 0:
                title = config['models']
                draw_confusion_matrix(extractor, classifier, config['target_test_loader'], res_dir, epoch, title)
                draw_tsne(extractor, classifier, config['source_train_loader'], config['target_test_loader'], res_dir, epoch, title, separate=True)
                # draw_tsne(extractor, classifier, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, title, separate=False)
    else:
        if os.path.exists(extractor_path) and os.path.exists(classifier_path) and os.path.exists(adnet_path):
            extractor.load_state_dict(torch.load(extractor_path))
            classifier.load_state_dict(torch.load(classifier_path))
            ad_net.load_state_dict(torch.load(adnet_path))
            print('Test only mode, model loaded')

            # print('test on source_test_loader')
            # test(extractor, classifier, config['source_test_loader'], -1)
            print('test on target_test_loader')
            test(extractor, classifier, config['target_test_loader'], -1)

            title = config['models']
            draw_confusion_matrix(extractor, classifier, config['target_test_loader'], res_dir, -1, title)
            # draw_tsne(extractor, classifier, config['source_test_loader'], config['target_test_loader'], res_dir, -1, title, separate=True)
            draw_tsne(extractor, classifier, config['source_train_loader'], config['target_test_loader'], res_dir, -1, title, separate=True)
        else:
            print('no saved model found')

