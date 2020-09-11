import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.functions import test, set_log_config, ReverseLayerF
from utils.vis import draw_tsne, draw_confusion_matrix
from networks.network import Extractor, Classifier2, Critic, Critic2, RandomLayer, AdversarialNetwork
from networks.inceptionv4 import InceptionV4
from networks.inceptionv1 import InceptionV1, InceptionV1s
#from models.DDC.mmd import mmd_linear
from models.DCTLN.mmd import MMD_loss
from models.dann_vat.train_dann_vat import VAT

from torchsummary import summary
torch.set_num_threads(2)

def get_loss_entropy(inputs):
    output = F.softmax(inputs, dim=1)
    entropy_loss = - torch.mean(output * torch.log(output + 1e-6))
    return entropy_loss

def get_loss_bnm(inputs):
    output = F.softmax(inputs, dim=1)
    _, L_BNM, _ = torch.svd(output)
    return -torch.mean(L_BNM)

def train_drcn(config):
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

    res_dir = os.path.join(config['res_dir'], 'VIS-slim{}-targetLabel{}-mmd{}-bnm{}-vat{}-ent{}-ew{}-bn{}-bs{}-lr{}'.format(config['slim'],
                                                                                                    config['target_labeling'], 
                                                                                                    config['mmd'], 
                                                                                                    config['bnm'], 
                                                                                                    config['vat'], 
                                                                                                    config['ent'], 
                                                                                                    config['bnm_ew'], 
                                                                                                    config['bn'], 
                                                                                                    config['batch_size'], 
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

    vat_loss = VAT(extractor, classifier, n_power=1, radius=3.5).cuda()

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
        mmd_loss = MMD_loss()

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
            data_target, label_target = iter_target.next()
            if config['slim'] > 0:
                data_target_semi, label_target_semi = iter_target_semi.next()
                if i % len_target_semi_loader == 0:
                    iter_target_semi = iter(config['target_train_semi_loader'])

            if i % len_target_loader == 0:
                iter_target = iter(config['target_train_loader'])
            if torch.cuda.is_available():
                data_source, label_source = data_source.cuda(), label_source.cuda()
                data_target, label_target = data_target.cuda(), label_target.cuda()
                if config['slim'] > 0:
                    data_target_semi, label_target_semi = data_target_semi.cuda(), label_target_semi.cuda()

            optimizer.zero_grad()

            class_output_s, domain_output, feature_s = dann(input_data=data_source, alpha=gamma)
            # print('domain_output {}'.format(domain_output.size()))
            err_s_label = loss_class(class_output_s, label_source)
            domain_label = torch.zeros(data_source.size(0)).long().cuda()
            err_s_domain = loss_domain(domain_output, domain_label)

            # Training model using target data
            domain_label = torch.ones(data_target.size(0)).long().cuda()
            class_output_t, domain_output, feature_t = dann(input_data=data_target, alpha=gamma)
            #class_output_t, domain_output, _ = dann(input_data=data_target, alpha=0.5)
            err_t_domain = loss_domain(domain_output, domain_label)

            err = err_s_label + err_s_domain + err_t_domain

            # if config['target_labeling'] == 1:
            #     err_t_class_healthy = nn.CrossEntropyLoss()(class_output_t, label_target)
            #     err += err_t_class_healthy
            #     if i % 100 == 0:
            #         print('err_t_class_healthy {:.2f}'.format(err_t_class_healthy.item()))

            if config['mmd'] == 1:
                #err +=  gamma * mmd_linear(feature_s, feature_t)
                err +=  config['bnm_ew'] * mmd_loss(feature_s, feature_t)

            if config['bnm'] == 1 and epoch >= config['startiter']:
                err_t_bnm = config['bnm_ew'] * get_loss_bnm(class_output_t)
                err += err_t_bnm
                if i == 1:
                    print('epoch {}, loss_t_bnm {:.2f}'.format(epoch, err_t_bnm.item()))

            if config['ent'] == 1 and epoch >= config['startiter']:
                err_t_ent = config['bnm_ew'] * get_loss_entropy(class_output_t)
                err += err_t_ent
                if i == 1:
                    print('epoch {}, loss_t_ent {:.2f}'.format(epoch, err_t_ent.item()))

            if config['vat'] == 1 and epoch >= config['startiter']:
                err_t_vat = config['bnm_ew'] * vat_loss(data_target, class_output_t)
                err += err_t_vat
                if i == 1:
                    print('epoch {}, loss_t_vat {:.2f}'.format(epoch, err_t_vat.item()))

            if config['slim'] > 0:
                feature_target_semi = extractor(data_target_semi)
                feature_target_semi = feature_target_semi.view(feature_target_semi.size(0), -1)
                preds_target_semi, _ = classifier(feature_target_semi)
                err_t_class_semi = loss_class(preds_target_semi, label_target_semi)
                err += err_t_class_semi
                if i == 1:
                    print('epoch {}, err_t_class_semi {:.2f}'.format(epoch, err_t_class_semi.item()))

            if i == 1:
                print('epoch {}, err_s_label {:.2f}, err_s_domain {:.2f}, err_t_domain {:.2f}, total err {:.2f}'.format(epoch, err_s_label.item(), err_s_domain.item(), err_t_domain.item(), err.item()))

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
            title = 'DANN'
            if config['bnm'] == 1 and config['vat'] == 1:
                title = '(b) Proposed'
            elif config['bnm'] == 1:
                title = 'BNM'
            elif config['vat'] == 1:
                title = 'VADA'
            elif config['mmd'] == 1:
                title = 'DCTLN'
            elif config['ent'] == 1:
                title = 'EntMin'
            # draw_confusion_matrix(extractor, classifier, config['target_test_loader'], res_dir, epoch, config['models'])
            draw_tsne(extractor, classifier, config['source_train_loader'], config['target_test_loader'], res_dir, epoch, title, separate=True)
            draw_tsne(extractor, classifier, config['source_train_loader'], config['target_test_loader'], res_dir, epoch, title, separate=False)
