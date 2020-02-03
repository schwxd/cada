import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.functions import test, set_log_config, set_requires_grad, ReverseLayerF
from network import Extractor, Classifier, Critic, Critic2, RandomLayer, AdversarialNetwork
from utils.vis import draw_tsne, draw_confusion_matrix
from models.inceptionv4 import InceptionV4
from models.inceptionv1 import InceptionV1
from models.MCD_A.mcd_loss import vat, get_loss_vat, get_loss_entropy

from torchsummary import summary

import torch.nn.functional as F

def ent(output):
    return - torch.mean(output * torch.log(output + 1e-6))

def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))

def train_mcd_a(config):
    if config['inception'] == 1:
        G = InceptionV1(num_classes=32)
    else:
        G = Extractor(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'])
    C1 = Classifier(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], n_class=config['n_class'])
    C2 = Classifier(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], n_class=config['n_class'])
    if torch.cuda.is_available():
        G = G.cuda()
        C1 = C1.cuda()
        C2 = C2.cuda()

    # opt_g = optim.Adam(G.parameters(), lr=config['lr'], weight_decay=0.0005)
    # opt_c1 = optim.Adam(C1.parameters(), lr=config['lr'], weight_decay=0.0005)
    # opt_c2 = optim.Adam(C2.parameters(), lr=config['lr'], weight_decay=0.0005)
    opt_g = optim.Adam(G.parameters(), lr=config['lr'])
    opt_c1 = optim.Adam(C1.parameters(), lr=config['lr'])
    opt_c2 = optim.Adam(C2.parameters(), lr=config['lr'])

    criterion = torch.nn.CrossEntropyLoss()
    res_dir = os.path.join(config['res_dir'], 'snr{}-onestep{}-lr{}'.format(config['snr'], config['one_step'], config['lr']))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    set_log_config(res_dir)
    logging.debug('train_mcd')
    logging.debug(G)
    logging.debug(C1)
    logging.debug(C2)
    logging.debug(config)


    def train(G, C1, C2, config, epoch):
        G.train()
        C1.train()
        C2.train()


        iter_source = iter(config['source_train_loader'])
        iter_target = iter(config['target_train_loader'])
        len_source_loader = len(config['source_train_loader'])
        len_target_loader = len(config['target_train_loader'])
        num_iter = len_source_loader
        for i in range(1, num_iter + 1):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            if i % len_target_loader == 0:
                iter_target = iter(config['target_train_loader'])
            if torch.cuda.is_available():
                data_source, label_source = data_source.cuda(), label_source.cuda()
                data_target = data_target.cuda()

            # step 1.
            #
            opt_g.zero_grad()
            opt_c1.zero_grad()
            opt_c2.zero_grad()

            # 源分类误差
            feat_s = G(data_source)
            feat_t = G(data_target)

            output_s1 = C1(feat_s)
            output_s2 = C2(feat_s)
            loss_s1 = criterion(output_s1, label_source)
            loss_s2 = criterion(output_s2, label_source)

            # vat误差
            # source_vat_c1 = vat(C1, feat_s, 0.5)
            # opt_g.zero_grad()
            # opt_c1.zero_grad()
            # opt_c2.zero_grad()

            target_vat_c1 = vat(C1, feat_t, 0.5)
            opt_g.zero_grad()
            opt_c1.zero_grad()
            opt_c2.zero_grad()

            # source_vat_c2 = vat(C2, feat_s, 0.5)
            # opt_g.zero_grad()
            # opt_c1.zero_grad()
            # opt_c2.zero_grad()

            target_vat_c2 = vat(C2, feat_t, 0.5)
            opt_g.zero_grad()
            opt_c1.zero_grad()
            opt_c2.zero_grad()

            # source_vat_loss1 = get_loss_vat(C1, feat_s, source_vat_c1)
            # source_vat_loss2 = get_loss_vat(C2, feat_s, source_vat_c2)
            target_vat_loss1 = get_loss_vat(C1, feat_t, target_vat_c1)
            target_vat_loss2 = get_loss_vat(C2, feat_t, target_vat_c2)

            entropy_loss_c1 = get_loss_entropy(C1, feat_t)
            entropy_loss_c2 = get_loss_entropy(C2, feat_t)

            # loss_s = loss_s1 + loss_s2 + source_vat_loss1 + source_vat_loss2 + target_vat_loss1 + target_vat_loss2
            loss_s = loss_s1 + loss_s2 + target_vat_loss1 + target_vat_loss2 + entropy_loss_c1 + entropy_loss_c2
            loss_s.backward()
            opt_g.step()
            opt_c1.step()
            opt_c2.step()


            # step 2.
            #
            opt_g.zero_grad()
            opt_c1.zero_grad()
            opt_c2.zero_grad()
            feat_s = G(data_source)
            output_s1 = C1(feat_s)
            output_s2 = C2(feat_s)
            loss_s1 = criterion(output_s1, label_source)
            loss_s2 = criterion(output_s2, label_source)
            loss_dis_s = discrepancy(output_s1, output_s2)

            feat_t = G(data_target)
            output_t1 = C1(feat_t)
            output_t2 = C2(feat_t)
            loss_dis_t = discrepancy(output_t1, output_t2)

            # 源分类误差 - 源和目的特征差异
            loss = loss_s1 + loss_s2 - loss_dis_t - loss_dis_s
            #loss =  - loss_dis
            loss.backward()
            opt_c1.step()
            opt_c2.step()


            # 更新特征提取器
            for _ in range(1):
                opt_g.zero_grad()
                opt_c1.zero_grad()
                opt_c2.zero_grad()
                feat_t = G(data_target)
                output_t1 = C1(feat_t)
                output_t2 = C2(feat_t)
                loss_dis_t = discrepancy(output_t1, output_t2)


                feat_s = G(data_source)
                output_s1 = C1(feat_s)
                output_s2 = C2(feat_s)
                loss_s1 = criterion(output_s1, label_source)
                loss_s2 = criterion(output_s2, label_source)
                loss_dis_s = discrepancy(output_s1, output_s2)


                loss = loss_s1 + loss_s2 + loss_dis_s + loss_dis_t

                loss.backward()

                #loss_dis.backward()
                opt_g.step()

            if i % 20 == 0:
                print('Train Epoch: {} Loss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy S: {:.6f}\t  Discrepancy T: {:.6f}'.format(
                    epoch, loss_s1.item(), loss_s2.item(), loss_dis_s.item(), loss_dis_t.item()))
                # logging.debug('Train Epoch: {} Loss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy: {:.6f}'.format(
                #     epoch, loss_s1.item(), loss_s2.item(), loss_dis.item()))

    def train_onestep(G, C1, C2, config, epoch):
        criterion = nn.CrossEntropyLoss().cuda()
        G.train()
        C1.train()
        C2.train()

        gamma = 2 / (1 + math.exp(-10 * (epoch) / config['n_epochs'])) - 1

        iter_source = iter(config['source_train_loader'])
        iter_target = iter(config['target_train_loader'])
        len_source_loader = len(config['source_train_loader'])
        len_target_loader = len(config['target_train_loader'])
        num_iter = len_source_loader
        for i in range(1, num_iter + 1):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            if i % len_target_loader == 0:
                iter_target = iter(config['target_train_loader'])
            if torch.cuda.is_available():
                data_source, label_source = data_source.cuda(), label_source.cuda()
                data_target = data_target.cuda()

            opt_g.zero_grad()
            opt_c1.zero_grad()
            opt_c2.zero_grad()

            set_requires_grad(G, requires_grad=True)
            set_requires_grad(C1, requires_grad=True)
            set_requires_grad(C2, requires_grad=True)
            feat_s = G(data_source)
            output_s1 = C1(feat_s)
            output_s2 = C2(feat_s)
            loss_s1 = criterion(output_s1, label_source)
            loss_s2 = criterion(output_s2, label_source)
            loss_s = loss_s1 + loss_s2
            # loss_s.backward(retain_variables=True)
            ##loss_s.backward()

            set_requires_grad(G, requires_grad=False)
            set_requires_grad(C1, requires_grad=True)
            set_requires_grad(C2, requires_grad=True)
            with torch.no_grad():
                feat_t = G(data_target)
            reverse_feature_t = ReverseLayerF.apply(feat_t, gamma)
            output_t1 = C1(reverse_feature_t)
            output_t2 = C2(reverse_feature_t)

            loss_dis = -discrepancy(output_t1, output_t2)
            ##loss_dis.backward()
            loss = loss_s + loss_dis
            loss.backward()
            opt_c1.step()
            opt_c2.step()
            opt_g.step()

            if i % 20 == 0:
                print('Train Epoch: {}, Loss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy: {:.6f}'.format(
                    epoch, loss_s1.item(), loss_s2.item(), loss_dis.item()))



    for epoch in range(1, config['n_epochs'] + 1):
        if config['one_step'] == 1:
            train_onestep(G, C1, C2, config, epoch)
        else:
            train(G, C1, C2, config, epoch)

        if epoch % config['TEST_INTERVAL'] == 0:
            #print('C1 on source_test_loader')
            #logging.debug('C1 on source_test_loader')
            #test(G, C1, config['source_test_loader'], epoch)
            #print('C2 on source_test_loader')
            #logging.debug('C2 on source_test_loader')
            #test(G, C2, config['source_test_loader'], epoch)
            print('C1 on target_test_loader')
            logging.debug('C1 on target_test_loader')
            test(G, C1, config['target_test_loader'], epoch)
            print('C2 on target_test_loader')
            logging.debug('C2 on target_test_loader')
            test(G, C2, config['target_test_loader'], epoch)
        if epoch % config['VIS_INTERVAL'] == 0:
            draw_confusion_matrix(G, C1, config['target_test_loader'], res_dir, epoch, config['models'])
            draw_tsne(G, C1, config['source_train_loader'], config['target_test_loader'], res_dir, epoch, config['models'], separate=True)
            draw_tsne(G, C1, config['source_train_loader'], config['target_test_loader'], res_dir, epoch, config['models'], separate=False)
            # draw_tsne(G, C1, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, config['models'], separate=False)
