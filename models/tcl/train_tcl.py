import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import models.tcl.cdan_loss as loss_func

from utils.functions import test, set_log_config
from utils.vis import draw_tsne, draw_confusion_matrix
from networks.network import Extractor, Classifier, Critic, Critic2, RandomLayer, AdversarialNetwork
from networks.inceptionv4 import InceptionV4
from networks.inceptionv1 import InceptionV1, InceptionV1s

from torchsummary import summary


def train_tcl(config):
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

    res_dir = os.path.join(config['res_dir'], 'slim{}-{}-Lythred{}-Ldthred{}-lambdad{}-lr{}'.format(config['slim'],
                                                                        config['network'],
                                                                        config['Lythred'],
                                                                        config['Ldthred'],
                                                                        config['lambdad'],
                                                                        config['lr']))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    print('train_tcl')
    #print(extractor)
    #print(classifier)
    print(config)

    set_log_config(res_dir)
    logging.debug('train_tcl')
    logging.debug(extractor)
    logging.debug(classifier)
    logging.debug(config)

    ad_net = AdversarialNetwork(config['n_flattens'], config['n_hiddens'])
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

    def cal_Ly(source_y_softmax, source_d, label):
        #
        # source_y_softmax, category预测结果带softmax
        # source_d，domain预测结果
        # label: 实际category标签
        #
        agey = - math.log(config['Lythred'])
        aged = - math.log(1.0 - config['Ldthred'])
        age = agey + config['lambdad'] * aged
        # print('agey {}, labmdad {}, aged {}, age {}'.format(agey, config['lambdad'], aged, age))
        y_softmax = source_y_softmax
        the_index = torch.LongTensor(np.array(range(config['batch_size']))).cuda()
        # 这是什么意思？对于每个样本，只取出实际label对应的softmax值
        # 与softmax loss有什么区别？

        y_label = y_softmax[the_index, label]
        # print('y_softmax {}, the_index {}, y_label shape {}'.format(y_softmax.shape, the_index.shape, y_label.shape))
        y_loss = - torch.log(y_label + 1e-8)

        d_loss = - torch.log(1.0 - source_d)
        d_loss = d_loss.view(config['batch_size'])

        weight_loss = y_loss + config['lambdad'] * d_loss
        # print('y_loss {}'.format(torch.mean(y_loss)))
        # print('lambdad {}'.format(config['lambdad']))
        # print('d_loss {}'.format(torch.mean(d_loss)))

        # print('y_loss {}'.format(y_loss.item()))
        # print('lambdad {}'.format(config['lambdad']))
        # print('d_loss {}'.format(d_loss.item()))

        weight_var = (weight_loss < age).float().detach()
        Ly = torch.mean(y_loss * weight_var)

        source_weight = weight_var.data.clone()
        source_num = float((torch.sum(source_weight)))
        return Ly, source_weight, source_num

    def cal_Lt(target_y_softmax):
        # 这是entropy loss吧？
        Gt_var = target_y_softmax
        Gt_en = - torch.sum((Gt_var * torch.log(Gt_var + 1e-8)), 1)
        Lt = torch.mean(Gt_en)
        return Lt

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

            source_domain_label = torch.FloatTensor(config['batch_size'], 1)
            target_domain_label = torch.FloatTensor(config['batch_size'], 1)
            source_domain_label.fill_(1)
            target_domain_label.fill_(0)
            domain_label = torch.cat([source_domain_label, target_domain_label],0)
            domain_label = domain_label.cuda()

            inputs = torch.cat([data_source, data_target],0)
            features = extractor(inputs)
            gamma = 2 / (1 + math.exp(-10 * (epoch) / config['n_epochs'])) - 1
            y_var = classifier(features)
            d_var = ad_net(features, gamma)
            y_softmax_var = nn.Softmax(dim=1)(y_var)
            source_y, target_y = y_var.chunk(2,0)
            source_y_softmax, target_y_softmax = y_softmax_var.chunk(2,0)
            source_d, target_d = d_var.chunk(2,0)
 
            # h_s = extractor(data_source)
            # h_s = h_s.view(h_s.size(0), -1)
            # h_t = extractor(data_target)
            # h_t = h_t.view(h_t.size(0), -1)        

            # source_preds = classifier(h_s)
            # softmax_output_s = nn.Softmax(dim=1)(source_preds)
            # target_preds = classifier(h_t)
            # softmax_output_t = nn.Softmax(dim=1)(target_preds)

            # source_d, d_loss_source = loss_func.DANN_logits(h_s, ad_net, gamma)
            # target_d, d_loss_target = loss_func.DANN_logits(h_t, ad_net, gamma)
            # source_d = ad_net(h_s, gamma)
            # target_d = ad_net(h_t, gamma)

            #calculate Ly 
            if epoch < config['startiter']:
                #也就是cls_loss，不考虑权重
                Ly = nn.CrossEntropyLoss()(source_y, label_source)    
            else:
                Ly, source_weight, source_num = cal_Ly(source_y_softmax, source_d, label_source)
                # print('source_num {}'.format(source_num))
                target_weight = torch.ones(source_weight.size()).cuda()

            #calculate Lt
            # 计算target category的熵
            Lt = cal_Lt(target_y_softmax)

            #calculate Ld
            if epoch < config['startiter']:
                Ld = nn.BCELoss()(d_var, domain_label)
            else:
                domain_weight = torch.cat([source_weight, target_weight], 0)
                domain_weight = domain_weight.view(-1,1)
                # print('domain_weight {}'.format(domain_weight.shape))
                # print('domain_weight {}'.format(domain_weight))
                # print('d_var {}'.format(d_var))

                domain_criterion = nn.BCELoss(weight=domain_weight).cuda()
                # domain_criterion = nn.BCELoss().cuda()

                # print('max {}'.format(torch.max(d_var)))
                # print('min {}'.format(torch.min(d_var)))
                # print(d_var)
                Ld = domain_criterion(d_var, domain_label)

            loss = Ly + config['traded'] * Ld + config['tradet'] * Lt
            
            optimizer.zero_grad()
            optimizer_ad.zero_grad()
            # net.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_ad.step()

            # if (step) % 20 == 0:
                # print('Train Epoch {} closs {:.6f}, dloss {:.6f}, coral_loss {:.6f}, Loss {:.6f}'.format(epoch, cls_loss.item(), d_loss.item(), coral_loss.item(), loss.item()))
                # print('Train Epoch {} closs {:.6f}, dloss {:.6f}, Loss {:.6f}'.format(epoch, cls_loss.item(), d_loss.item(), loss.item()))

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


