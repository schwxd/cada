import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import models.CDAN.cdan_loss as loss_func

from networks.network import Extractor, Classifier, Critic, Critic2, RandomLayer, AdversarialNetwork
from networks.resnet18_1d import resnet18_features
from networks.inceptionv4 import InceptionV4
from networks.inceptionv1 import InceptionV1, InceptionV1s

from torchsummary import summary
from utils.functions import test, set_log_config
from utils.vis import draw_tsne, draw_confusion_matrix


def get_loss_entropy(inputs):
    output = F.softmax(inputs, dim=1)
    entropy_loss = - torch.mean(output * torch.log(output + 1e-6))

    return entropy_loss

class VAT(nn.Module):
    def __init__(self, extractor, classifier, n_power, radius):
        super(VAT, self).__init__()
        self.n_power = n_power
        self.XI = 1e-6
        self.extractor = extractor
        self.classifier = classifier
        self.epsilon = radius

    def forward(self, X, logit):
        vat_loss = self.virtual_adversarial_loss(X, logit)
        return vat_loss

    def generate_virtual_adversarial_perturbation(self, x, logit):
        d = torch.randn_like(x, device='cuda')

        for _ in range(self.n_power):
            d = self.XI * self.get_normalized_vector(d).requires_grad_()
            logit_m = self.classifier(self.extractor(x + d))
            dist = self.kl_divergence_with_logit(logit, logit_m)
            grad = torch.autograd.grad(dist, [d])[0]
            d = grad.detach()

        return self.epsilon * self.get_normalized_vector(d)

    def kl_divergence_with_logit(self, q_logit, p_logit):
        q = F.softmax(q_logit, dim=1)
        qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
        qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
        return qlogq - qlogp

    def get_normalized_vector(self, d):
        return F.normalize(d.view(d.size(0), -1), p=2, dim=1).reshape(d.size())

    def virtual_adversarial_loss(self, x, logit):
        r_vadv = self.generate_virtual_adversarial_perturbation(x, logit)
        logit_p = logit.detach()
        logit_m = self.classifier(self.extractor(x + r_vadv))
        loss = self.kl_divergence_with_logit(logit_p, logit_m)
        return loss


def train_cdan_vat(config):
    if config['network'] == 'inceptionv1':
        extractor = InceptionV1(num_classes=32)
    elif config['network'] == 'inceptionv1s':
        extractor = InceptionV1s(num_classes=32)
    else:
        extractor = Extractor(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], bn=config['bn'])
    classifier = Classifier(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], n_class=config['n_class'])

    if torch.cuda.is_available():
        extractor = extractor.cuda()
        classifier = classifier.cuda()

    cdan_random = config['random_layer'] 
    res_dir = os.path.join(config['res_dir'], 'normal{}-{}-dilation{}-lr{}'.format(config['normal'],
                                                                        config['network'],
                                                                        config['dilation'],
                                                                        config['lr']))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    print('train_cdan_vat')
    #print(extractor)
    #print(classifier)
    print(config)

    set_log_config(res_dir)
    logging.debug('train_cdan_vat')
    logging.debug(extractor)
    logging.debug(classifier)
    logging.debug(config)

    if config['models'] == 'DANN_VAT':
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

    vat_loss = VAT(extractor, classifier, n_power=1, radius=3.5).cuda()


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
            data_target, label_target = iter_target.next()
            if step % len_target_loader == 0:
                iter_target = iter(config['target_train_loader'])
            if torch.cuda.is_available():
                data_source, label_source = data_source.cuda(), label_source.cuda()
                data_target, label_target = data_target.cuda(), label_target.cuda()


            """
            add code start
            """
            with torch.no_grad():
                if 'CDAN' in config['models']:
                    h_s = extractor(data_source)
                    h_s = h_s.view(h_s.size(0), -1)
                    source_preds = classifier(h_s)
                    softmax_output_s = nn.Softmax(dim=1)(source_preds)

                    op_out = torch.bmm(softmax_output_s.unsqueeze(2), h_s.unsqueeze(1))
                    gamma = 2 / (1 + math.exp(-10 * (epoch) / config['n_epochs'])) - 1
                    ad_out = ad_net(op_out.view(-1, softmax_output_s.size(1) * h_s.size(1)), gamma, training=False)
                    dom_entropy = 1-(torch.abs(0.5-ad_out))**config['iw']
                    dom_weight = dom_entropy

                elif 'DANN' in config['models']:
                    h_s = extractor(data_source)
                    h_s = h_s.view(h_s.size(0), -1)
                    gamma = 2 / (1 + math.exp(-10 * (epoch) / config['n_epochs'])) - 1
                    ad_out = ad_net(h_s, gamma, training=False)
                    dom_entropy = 1-(torch.abs(0.5-ad_out))**config['iw']
                    dom_weight = dom_entropy

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

            if config['iw'] > 0:
                cls_loss = nn.CrossEntropyLoss(reduction='none')(source_preds, label_source)
                cls_loss = torch.mean(dom_weight * cls_loss)
                # print('dom_weight mean {}'.format(torch.mean(dom_weight)))
            else:
                cls_loss = nn.CrossEntropyLoss()(source_preds, label_source)

            target_preds = classifier(h_t)
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
                elif config['models'] == 'CDAN_VAT':
                    d_loss = loss_func.CDAN([feature, softmax_output], ad_net, gamma, None, None, random_layer)
                elif config['models'] == 'DANN_VAT':
                    d_loss = loss_func.DANN(feature, ad_net, gamma)
                else:
                    raise ValueError('Method cannot be recognized.')
            else:
                d_loss = 0

            # target entropy loss
            err_t_entropy = get_loss_entropy(softmax_output_t)

            # virtual adversarial loss.
            err_s_vat = vat_loss(data_source, source_preds)
            err_t_vat = vat_loss(data_target, target_preds)

            # loss = cls_loss + d_loss
            loss = cls_loss + d_loss + err_t_entropy + err_s_vat + err_t_vat
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
            print('epoch {} accuracy: {:.6f}, best accuracy {:.6f} on epoch {}'.format(epoch, accuracy, best_accuracy, best_model_index))


        if epoch % config['VIS_INTERVAL'] == 0:
            title = config['models']
            draw_confusion_matrix(extractor, classifier, config['target_test_loader'], res_dir, epoch, title)
            draw_tsne(extractor, classifier, config['source_train_loader'], config['target_test_loader'], res_dir, epoch, title, separate=True)
            draw_tsne(extractor, classifier, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, title, separate=False)

