import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.Wasserstein.triplet_loss import triplet_loss

from networks.network import Extractor, Classifier, Critic, Critic2, RandomLayer, AdversarialNetwork, Classifier2
from networks.inceptionv1 import InceptionV1

from utils.functions import test, set_log_config, set_requires_grad, gradient_penalty
from utils.vis import draw_tsne, draw_confusion_matrix

def train_wasserstein(config):
    extractor = Extractor(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'])
    # extractor = InceptionV1(num_classes=32)
    classifier = Classifier2(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], n_class=config['n_class'])
    critic = Critic(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'])
    if torch.cuda.is_available():
        extractor = extractor.cuda()
        classifier = classifier.cuda()
        critic = critic.cuda()

    triplet_type = config['triplet_type']
    gamma = config['w_gamma'] 
    weight_wd = config['w_weight']
    weight_triplet = config['t_weight']
    t_margin = config['t_margin']
    t_confidence = config['t_confidence']
    k_critic = 3
    k_clf = 1
    TRIPLET_START_INDEX = 95 

    if triplet_type == 'none':
        res_dir = os.path.join(config['res_dir'], 'bs{}-lr{}-w{}-gamma{}'.format(config['batch_size'], config['lr'], weight_wd, gamma))
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        extractor_path = os.path.join(res_dir, "extractor.pth")
        classifier_path = os.path.join(res_dir, "classifier.pth")
        critic_path = os.path.join(res_dir, "critic.pth")
        EPOCH_START = 1
        TEST_INTERVAL = 10

    else:
        TEST_INTERVAL = 1
        w_dir = os.path.join(config['res_dir'], 'bs{}-lr{}-w{}-gamma{}'.format(config['batch_size'], config['lr'], weight_wd, gamma))
        if not os.path.exists(w_dir):
            os.makedirs(w_dir)
        res_dir = os.path.join(w_dir, '{}_t_weight{}_margin{}_confidence{}'.format(triplet_type, weight_triplet, t_margin, t_confidence))
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        extractor_path = os.path.join(w_dir, "extractor.pth")
        classifier_path = os.path.join(w_dir, "classifier.pth")
        critic_path = os.path.join(w_dir, "critic.pth")

        if os.path.exists(extractor_path):
            extractor.load_state_dict(torch.load(extractor_path))
            classifier.load_state_dict(torch.load(classifier_path))
            critic.load_state_dict(torch.load(critic_path))
            print('load models')
            EPOCH_START = TRIPLET_START_INDEX
        else:
            EPOCH_START = 1

    set_log_config(res_dir)
    print('start epoch {}'.format(EPOCH_START))
    print('triplet type {}'.format(triplet_type))
    print(config)

    logging.debug('train_wt')
    logging.debug(extractor)
    logging.debug(classifier)
    logging.debug(critic)
    logging.debug(config)

    criterion = torch.nn.CrossEntropyLoss()
    softmax_layer = nn.Softmax(dim=1)

    critic_opt = torch.optim.Adam(critic.parameters(), lr=config['lr'])
    classifier_opt = torch.optim.Adam(classifier.parameters(), lr=config['lr'])
    feature_opt = torch.optim.Adam(extractor.parameters(), lr=config['lr']/10)


    def train(extractor, classifier, critic, config, epoch):
        extractor.train()
        classifier.train()
        critic.train()

        iter_source = iter(config['source_train_loader'])
        iter_target = iter(config['target_train_loader'])
        len_source_loader = len(config['source_train_loader'])
        len_target_loader = len(config['target_train_loader'])
        num_iter = len_source_loader
        for step in range(1, num_iter):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            if step % len_target_loader == 0:
                iter_target = iter(config['target_train_loader'])
            if torch.cuda.is_available():
                data_source, label_source = data_source.cuda(), label_source.cuda()
                data_target = data_target.cuda()

            # 1. train critic
            set_requires_grad(extractor, requires_grad=False)
            set_requires_grad(classifier, requires_grad=False)
            set_requires_grad(critic, requires_grad=True)
            with torch.no_grad():
                h_s = extractor(data_source)
                h_s = h_s.view(h_s.size(0), -1)
                h_t = extractor(data_target)
                h_t = h_t.view(h_t.size(0), -1)

            for j in range(k_critic):
                gp = gradient_penalty(critic, h_s, h_t)
                critic_s = critic(h_s)
                critic_t = critic(h_t)
                wasserstein_distance = critic_s.mean() - critic_t.mean()
                critic_cost = -wasserstein_distance + gamma*gp

                critic_opt.zero_grad()
                critic_cost.backward()
                critic_opt.step()

                if step == 10 and j == 0:
                    print('EPOCH {}, DISCRIMINATOR: wd {}, gp {}, loss {}'.format(epoch, wasserstein_distance.item(), (gamma*gp).item(), critic_cost.item()))
                    logging.debug('EPOCH {}, DISCRIMINATOR: wd {}, gp {}, loss {}'.format(epoch, wasserstein_distance.item(), (gamma*gp).item(), critic_cost.item()))

            # 2. train feature and class_classifier
            set_requires_grad(extractor, requires_grad=True)
            set_requires_grad(classifier, requires_grad=True)
            set_requires_grad(critic, requires_grad=False)
            for _ in range(k_clf):
                h_s = extractor(data_source)
                h_s = h_s.view(h_s.size(0), -1)
                h_t = extractor(data_target)
                h_t = h_t.view(h_t.size(0), -1)

                source_preds, _ = classifier(h_s)
                clf_loss = criterion(source_preds, label_source)
                wasserstein_distance = critic(h_s).mean() - critic(h_t).mean()

                if triplet_type != 'none' and epoch >= TRIPLET_START_INDEX:
                    target_preds, _ = classifier(h_t)
                    target_labels = target_preds.data.max(1)[1]
                    target_logits = softmax_layer(target_preds)
                    if triplet_type == 'all':
                        triplet_index = np.where(target_logits.data.max(1)[0].cpu().numpy() > t_margin)[0]
                        images = torch.cat((h_s, h_t[triplet_index]), 0)
                        labels = torch.cat((label_source, target_labels[triplet_index]), 0)
                    elif triplet_type == 'src':
                        images = h_s
                        labels = label_source
                    elif triplet_type == 'tgt':
                        triplet_index = np.where(target_logits.data.max(1)[0].cpu().numpy() > t_confidence)[0]
                        images = h_t[triplet_index]
                        labels = target_labels[triplet_index]
                    elif triplet_type == 'sep':
                        triplet_index = np.where(target_logits.data.max(1)[0].cpu().numpy() > t_confidence)[0]
                        images = h_t[triplet_index]
                        labels = target_labels[triplet_index]
                        t_loss_sep, _ = triplet_loss(extractor, {"X": images, "y": labels}, t_confidence)
                        images = h_s
                        labels = label_source

                    t_loss, _ = triplet_loss(extractor, {"X": images, "y": labels}, t_margin)
                    loss = clf_loss + \
                        weight_wd * wasserstein_distance + \
                        weight_triplet * t_loss
                    if triplet_type == 'sep':
                        loss += t_loss_sep
                    feature_opt.zero_grad()
                    classifier_opt.zero_grad()
                    loss.backward()
                    feature_opt.step()
                    classifier_opt.step()

                    if step == 10:
                        print('EPOCH {}, CLASSIFIER: clf_loss {}, wd {}, t_loss {}, total loss {}'.format(
                            epoch, clf_loss.item(),
                            weight_wd * wasserstein_distance.item(),
                            weight_triplet * t_loss.item(),
                            loss.item()))
                        logging.debug('EPOCH {}, CLASSIFIER: clf_loss {}, wd {}, t_loss {}, total loss {}'.format(
                            epoch, clf_loss.item(),
                            weight_wd * wasserstein_distance.item(),
                            weight_triplet * t_loss.item(),
                            loss.item()))

                else:
                    loss = clf_loss + weight_wd * wasserstein_distance
                    feature_opt.zero_grad()
                    classifier_opt.zero_grad()
                    loss.backward()
                    feature_opt.step()
                    classifier_opt.step()

                    if step == 10:
                        print('EPOCH {}, CLASSIFIER: clf_loss {}, wd {},  loss {}'.format(
                            epoch, clf_loss.item(),
                            weight_wd * wasserstein_distance.item(),
                            loss.item()))
                        logging.debug('EPOCH {}, CLASSIFIER: clf_loss {}, wd {},  loss {}'.format(
                            epoch, clf_loss.item(),
                            weight_wd * wasserstein_distance.item(),
                            loss.item()))

    # pretrain(model, config, pretrain_epochs=20)
    for epoch in range(EPOCH_START, config['n_epochs'] + 1):
        train(extractor, classifier, critic, config, epoch)
        if epoch % TEST_INTERVAL == 0:
            # print('test on source_test_loader')
            # test(extractor, classifier, config['source_test_loader'], epoch)
            # print('test on target_train_loader')
            # test(model, config['target_train_loader'], epoch)
            print('test on target_test_loader')
            test(extractor, classifier, config['target_test_loader'], epoch)
        if epoch % config['VIS_INTERVAL'] == 0:
            if triplet_type == 'none':
                title = '(a) WDGRL'
            else:
                title = '(b) TLADA'
            draw_confusion_matrix(extractor, classifier, config['target_test_loader'], res_dir, epoch, title)
            draw_tsne(extractor, classifier, config['source_train_loader'], config['target_test_loader'], res_dir, epoch, title, separate=True)
            # draw_tsne(extractor, classifier, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, title, separate=False)
    if triplet_type == 'none':
        torch.save(extractor.state_dict(), extractor_path)
        torch.save(classifier.state_dict(), classifier_path)
        torch.save(critic.state_dict(), critic_path)
