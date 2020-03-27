import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from networks.network import Extractor, Classifier, Critic, Critic2, RandomLayer, AdversarialNetwork

from utils.functions import test, set_log_config, ReverseLayerF
from utils.vis import draw_tsne, draw_confusion_matrix

# loss params.
dw = 1e-2
cw = 1
sw = 1
tw = 1e-2
bw = 1e-2

class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)

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

def train_dann_vat_iw(config):
    extractor = Extractor(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'])
    classifier = Classifier(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], n_class=config['n_class'])
    critic = Critic2(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'])
    if torch.cuda.is_available():
        extractor = extractor.cuda()
        classifier = classifier.cuda()
        critic = critic.cuda()

    res_dir = os.path.join(config['res_dir'], 'slim{}-target_labeling{}-lr{}'.format(config['slim'], config['target_labeling'], config['lr']))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    criterion = torch.nn.CrossEntropyLoss()
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()
    cent = ConditionalEntropyLoss().cuda()
    vat_loss = VAT(extractor, classifier, n_power=1, radius=3.5).cuda()


    print('train_dann_vat')
    print(extractor)
    print(classifier)
    print(critic)
    print(config)

    set_log_config(res_dir)
    logging.debug('train_dann_vat')
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
        class_output = classifier(feature)
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
        num_iter = len_source_loader
        for i in range(1, num_iter+1):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            if i % len_target_loader == 0:
                iter_target = iter(config['target_train_loader'])
            if torch.cuda.is_available():
                data_source, label_source = data_source.cuda(), label_source.cuda()
                data_target = data_target.cuda()

            optimizer.zero_grad()

            class_output_s, domain_output_s, features_source = dann(input_data=data_source, alpha=gamma)
            err_s_class = loss_class(class_output_s, label_source)
            domain_label_s = torch.zeros(data_source.size(0)).long().cuda()
            err_s_domain = loss_domain(domain_output_s, domain_label_s)

            # Training model using target data
            class_output_t, domain_output_t, features_target = dann(input_data=data_target, alpha=gamma)
            domain_label_t = torch.ones(data_target.size(0)).long().cuda()
            err_t_domain = loss_domain(domain_output_t, domain_label_t)
            # target entropy loss
            err_t_entropy = get_loss_entropy(class_output_t)

            # virtual adversarial loss.
            err_s_vat = vat_loss(data_source, class_output_s)
            err_t_vat = vat_loss(data_target, class_output_t)

            err_domain = 0.5 * (err_s_domain + err_t_domain)

            # combined loss.
            dw = 1
            cw = 1
            sw = 1
            tw = 1
            bw = 1
            err_all = (
                    dw * err_domain +
                    cw * err_s_class +
                    sw * err_s_vat +
                    tw * err_t_vat +
                    tw * err_t_entropy
            )

            if i % 20 == 0:
                print('err_s_class {:.2f}, err_s_domain {:.2f}, gamma {:.2f}, err_t_domain {:.2f}, err_t_vat {:.2f}, err_s_vat {:.2f}, err_all {:.2f}'.format(err_s_class.item(), 
                                                err_s_domain.item(), 
                                                gamma, 
                                                err_t_domain.item(),
                                                err_s_vat.item(),
                                                err_t_vat.item(),
                                                err_all.item()))

            err_all.backward()
            optimizer.step()


    if config['testonly'] == 0:
        best_accuracy = 0
        best_model_index = -1
        for epoch in range(1, config['n_epochs'] + 1):
            train(extractor, classifier, critic, config, epoch)
            if epoch % config['TEST_INTERVAL'] == 0:
                print('test on source_test_loader')
                test(extractor, classifier, config['source_test_loader'], epoch)
                print('test on target_test_loader')
                accuracy = test(extractor, classifier, config['target_test_loader'], epoch)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_index = epoch
                print('epoch {} accuracy: {:.6f}, best accuracy {:.6f} on epoch {}'.format(epoch, accuracy, best_accuracy, best_model_index))


            if epoch % config['VIS_INTERVAL'] == 0:
                title = config['models']
                draw_confusion_matrix(extractor, classifier, config['target_test_loader'], res_dir, epoch, title)
                draw_tsne(extractor, classifier, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, title, separate=True)
                # draw_tsne(extractor, classifier, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, title, separate=False)
