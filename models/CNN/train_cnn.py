import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from functions import test, set_log_config
from network import Extractor, Classifier, Critic, Critic2, RandomLayer, AdversarialNetwork
from vis import draw_tsne, draw_confusion_matrix

from dev import get_dev_risk, get_weight
from sklearn import preprocessing

def validation(extractor, classifier, config, epoch):
    print('validation on epoch {}'.format(epoch))
    extractor.eval()
    classifier.eval()

    source_feature = []
    source_label = []
    source_predict = []

    target_feature = []
    target_label = []
    target_predict = []

    validation_feature = []

    for step, (features, labels) in enumerate(config['source_train_loader']):
        if torch.cuda.is_available():
            features, labels = features.cuda(), labels.cuda()
        embedding = extractor(features)
        predict = classifier(embedding)
        source_feature.append(embedding.detach().cpu().numpy())
        source_label.append(labels.cpu().numpy())
        predict = nn.Softmax(dim=1)(predict)
        predict = predict.data.max(1)[1]
        source_predict.append(predict.detach().cpu().numpy())

    for step, (features, labels) in enumerate(config['target_train_loader']):
        if torch.cuda.is_available():
            features, labels = features.cuda(), labels.cuda()
        embedding = extractor(features)
        predict = classifier(embedding)
        target_feature.append(embedding.detach().cpu().numpy())
        target_label.append(labels.cpu().numpy())
        predict = nn.Softmax(dim=1)(predict)
        predict = predict.data.max(1)[1]
        target_predict.append(predict.detach().cpu().numpy())

    for step, (features, labels) in enumerate(config['source_test_loader']):
        if torch.cuda.is_available():
            features, labels = features.cuda(), labels.cuda()
        embedding = extractor(features).detach()
        validation_feature.append(embedding.cpu().numpy())

    source_feature = np.concatenate(source_feature)
    target_feature = np.concatenate(target_feature)
    validation_feature = np.concatenate(validation_feature)
    print('source_feature {}'.format(source_feature.shape))
    print('target_feature {}'.format(target_feature.shape))
    print('validation_feature {}'.format(validation_feature.shape))

    source_label = np.concatenate(source_label)
    target_label = np.concatenate(target_label)
    source_predict = np.concatenate(source_predict)
    target_predict = np.concatenate(target_predict)

    weight = get_weight(source_feature, target_feature, validation_feature)
    print(weight.shape)
    density_ratio = weight

    test_error = np.mean((target_label - target_predict) ** 2)
    train_error = np.mean((source_label - source_predict) ** 2)

    wl = density_ratio * ((source_label - source_predict) ** 2)
    weighted_val_error = np.mean(wl)

    cov = np.cov(np.concatenate((wl, density_ratio), axis=1), rowvar=False)[0][1]
    var_w = np.var(density_ratio, ddof=1)
    c = - cov / var_w

    dev_error = weighted_val_error + c * np.mean(density_ratio) - c

    return train_error, test_error, weighted_val_error, dev_error


def train_cnn(config):
    extractor = Extractor(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'])
    classifier = Classifier(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], n_class=config['n_class'])

    if torch.cuda.is_available():
        extractor = extractor.cuda()
        classifier = classifier.cuda()

    res_dir = os.path.join(config['res_dir'], 'snr{}-lr{}'.format(config['snr'], config['lr']))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    set_log_config(res_dir)
    logging.debug('train_cnn')
    logging.debug(extractor)
    logging.debug(classifier)
    logging.debug(config)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(extractor.parameters()) + list(classifier.parameters()),
        lr = config['lr'])

    def train(extractor, classifier, config, epoch):
        extractor.train()
        classifier.train()

        for step, (features, labels) in enumerate(config['source_train_loader']):
            if torch.cuda.is_available():
                features, labels = features.cuda(), labels.cuda()

            optimizer.zero_grad()
            preds = classifier(extractor(features))
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

    train_errors = []
    test_errors = []
    iwcv_errors = []
    dev_errors = []
    VALIDATION_START_INDEX = config['n_epochs'] - 5
    for epoch in range(1, config['n_epochs'] + 1):
        train(extractor, classifier, config, epoch)
        if epoch % config['TEST_INTERVAL'] == 0:
            print('test on source_test_loader')
            test(extractor, classifier, config['source_test_loader'], epoch)
            print('test on target_test_loader')
            test(extractor, classifier, config['target_test_loader'], epoch)
        if epoch % config['VIS_INTERVAL'] == 0:
            draw_confusion_matrix(extractor, classifier, config['target_test_loader'], res_dir, epoch, config['models'])
            draw_tsne(extractor, classifier, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, config['models'], separate=True)
            # draw_tsne(extractor, classifier, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, config['models'], separate=False)

        if epoch > VALIDATION_START_INDEX:
            train_error, test_error, iwcv_error, dev_error = validation(extractor, classifier, config, epoch)
            train_errors.append(train_error)
            test_errors.append(test_error)
            iwcv_errors.append(iwcv_error)
            dev_errors.append(dev_error)

    if len(train_errors) > 0:
        print('train_error mean {:.3f}, std {:.3f}'.format(np.mean(train_errors), np.std(train_errors, ddof=1)))
        print('test_errors mean {:.3f}, std {:.3f}'.format(np.mean(test_errors), np.std(test_errors, ddof=1)))
        print('iwcv_errors mean {:.3f}, std {:.3f}'.format(np.mean(iwcv_errors), np.std(iwcv_errors, ddof=1)))
        print('dev_errors mean {:.3f}, std {:.3f}'.format(np.mean(dev_errors), np.std(dev_errors, ddof=1)))


