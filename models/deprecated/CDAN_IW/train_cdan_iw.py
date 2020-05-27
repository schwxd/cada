import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import models.CDAN_IW.cdan_loss as loss_func
from models.CDAN_IW.Coral import CORAL 

from utils.functions import test, set_log_config
from utils.vis import draw_tsne, draw_confusion_matrix
from networks.network import Extractor, Classifier, Critic, Critic2, RandomLayer, AdversarialNetwork
from networks.inceptionv4 import InceptionV4
from networks.inceptionv1 import InceptionV1, InceptionV1s

from torchsummary import summary

def train_cdan_iw(config):
    if config['network'] == 'inceptionv1':
        extractor = InceptionV1(num_classes=32)
    elif config['network'] == 'inceptionv1s':
        extractor = InceptionV1s(num_classes=32)
    else:
        extractor = Extractor(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'])
    classifier = Classifier(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], n_class=config['n_class'])

    if torch.cuda.is_available():
        extractor = extractor.cuda()
        classifier = classifier.cuda()
        #summary(extractor, (1, 5120))

    cdan_random = config['random_layer'] 
    res_dir = os.path.join(config['res_dir'], 'normal{}-{}-dilation{}-iw{}-lr{}'.format(config['normal'],
                                                                        config['network'],
                                                                        config['dilation'],
                                                                        config['iw'],
                                                                        config['lr']))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    print('train_cdan_iw')
    #print(extractor)
    #print(classifier)
    print(config)

    set_log_config(res_dir)
    logging.debug('train_cdan')
    logging.debug(extractor)
    logging.debug(classifier)
    logging.debug(config)

    if config['models'] == 'DANN_IW':
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
        ], weight_decay=0.0001)
    optimizer_ad = torch.optim.Adam(ad_net.parameters(), lr=config['lr'], weight_decay=0.0001)
    print(ad_net)

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


            """
            add code start
            """
            with torch.no_grad():
                if config['models'] == 'CDAN_IW':
                    h_s = extractor(data_source)
                    h_s = h_s.view(h_s.size(0), -1)
                    h_t = extractor(data_target)
                    h_t = h_t.view(h_t.size(0), -1)

                    source_preds = classifier(h_s)
                    softmax_output_s = nn.Softmax(dim=1)(source_preds)
                    # print(softmax_output_s.shape)
                    # print(softmax_output_s.unsqueeze(2).shape)
                    # print(softmax_output_s)


                    # target_preds = classifier(h_t)
                    # softmax_output_t = nn.Softmax(dim=1)(target_preds)

                    # feature = torch.cat((h_s, h_t), 0)
                    # softmax_output = torch.cat((softmax_output_s, softmax_output_t), 0)
                    weights = torch.ones(softmax_output_s.shape).cuda()
                    weights = 1.0*weights
                    weights = weights.unsqueeze(2)

                    # op_out = torch.bmm(softmax_output_s.unsqueeze(2), h_s.unsqueeze(1))
                    op_out = torch.bmm(weights, h_s.unsqueeze(1))
                    # gamma = 2 / (1 + math.exp(-10 * (epoch) / config['n_epochs'])) - 1
                    gamma = 1
                    ad_out = ad_net(op_out.view(-1, softmax_output_s.size(1) * h_s.size(1)), gamma, training=False)
                    # dom_entropy = loss_func.Entropy(ad_out)
                    dom_entropy = 1+(torch.abs(0.5-ad_out))**config['iw']
                    # dom_weight = dom_entropy / torch.sum(dom_entropy)
                    dom_weight = dom_entropy

                elif config['models'] == 'DANN_IW':
                    h_s = extractor(data_source)
                    h_s = h_s.view(h_s.size(0), -1)
                    # gamma = 2 / (1 + math.exp(-10 * (epoch) / config['n_epochs'])) - 1
                    gamma = 1
                    ad_out = ad_net(h_s, gamma, training=False)
                    # dom_entropy = 1-((torch.abs(0.5-ad_out))**config['iw'])
                    # dom_weight = dom_entropy
                    dom_weight = torch.ones(ad_out.shape).cuda()
                    #dom_entropy = loss_func.Entropy(dom_entropy)
                    # dom_weight = dom_entropy / torch.sum(dom_entropy)


            """
            add code end
            """

            optimizer.zero_grad()
            optimizer_ad.zero_grad()

            h_s = extractor(data_source)
            h_s = h_s.view(h_s.size(0), -1)
            h_t = extractor(data_target)
            h_t = h_t.view(h_t.size(0), -1)

            source_preds = classifier(h_s)
            softmax_output_s = nn.Softmax(dim=1)(source_preds)

            target_preds = classifier(h_t)
            softmax_output_t = nn.Softmax(dim=1)(target_preds)

            feature = torch.cat((h_s, h_t), 0)
            softmax_output = torch.cat((softmax_output_s, softmax_output_t), 0)

            cls_loss = nn.CrossEntropyLoss(reduction='none')(source_preds, label_source)
            cls_loss = torch.mean(dom_weight * cls_loss)

            if epoch > start_epoch:
                gamma = 2 / (1 + math.exp(-10 * (epoch) / config['n_epochs'])) - 1
                if config['models'] == 'CDAN_EIW':
                    entropy = loss_func.Entropy(softmax_output)
                    # print('softmax_output {}, entropy {}'.format(softmax_output.size(), entropy.size()))
                    d_loss = loss_func.CDAN([feature, softmax_output], ad_net, gamma, entropy, loss_func.calc_coeff(num_iter*(epoch-start_epoch)+step), random_layer)
                elif config['models'] == 'CDAN_IW':
                    d_loss = loss_func.CDAN([feature, softmax_output], ad_net, gamma, None, None, random_layer)
                elif config['models'] == 'DANN_IW':
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
            draw_tsne(extractor, classifier, config['source_train_loader'], config['target_test_loader'], res_dir, epoch, title, separate=False)

