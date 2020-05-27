import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.functions import test, set_log_config, ReverseLayerF
from utils.vis import draw_tsne, draw_confusion_matrix
from networks.network import Extractor, Classifier, Critic, Critic2, RandomLayer, AdversarialNetwork
from networks.inceptionv4 import InceptionV4
from networks.inceptionv1 import InceptionV1, InceptionV1s

from torchsummary import summary

def train_dann(config):
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

            class_output_s, domain_output, _ = dann(input_data=data_source, alpha=gamma)
            #class_output_s, domain_output, _ = dann(input_data=data_source, alpha=0.5)
            # print('domain_output {}'.format(domain_output.size()))
            err_s_label = loss_class(class_output_s, label_source)
            domain_label = torch.zeros(data_source.size(0)).long().cuda()
            err_s_domain = loss_domain(domain_output, domain_label)

            # Training model using target data
            domain_label = torch.ones(data_target.size(0)).long().cuda()
            class_output_t, domain_output, _ = dann(input_data=data_target, alpha=gamma)
            #class_output_t, domain_output, _ = dann(input_data=data_target, alpha=0.5)
            err_t_domain = loss_domain(domain_output, domain_label)
            err = err_s_label + err_s_domain + err_t_domain

            if i % 20 == 0:
                print('err_s_label {}, err_s_domain {}, gamma {}, err_t_domain {}, total err {}'.format(err_s_label.item(), err_s_domain.item(), gamma, err_t_domain.item(), err.item()))

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
            draw_confusion_matrix(extractor, classifier, config['target_test_loader'], res_dir, epoch, config['models'])
            draw_tsne(extractor, classifier, config['source_train_loader'], config['target_test_loader'], res_dir, epoch, config['models'], separate=True)
            draw_tsne(extractor, classifier, config['source_train_loader'], config['target_test_loader'], res_dir, epoch, config['models'], separate=False)





import dataloader
from models import main_models
# parser.add_argument('--n_epoches_1',type=int,default=10)
# parser.add_argument('--n_epoches_2',type=int,default=100)
# parser.add_argument('--n_epoches_3',type=int,default=100)
# parser.add_argument('--n_target_samples',type=int,default=7)

use_cuda= True if torch.cuda.is_available() else False
device=torch.device('cuda:0') if use_cuda else torch.device('cpu')
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)


def train_fada(config):
    if config['network'] == 'inceptionv1':
        extractor = InceptionV1(num_classes=32, dilation=config['dilation'])
    elif config['network'] == 'inceptionv1s':
        extractor = InceptionV1s(num_classes=32, dilation=config['dilation'])
    else:
        extractor = Extractor(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'])
    classifier = Classifier(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], n_class=config['n_class'])

    critic = Critic2(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'])
    if torch.cuda.is_available():
        extractor = extractor.cuda()
        classifier = classifier.cuda()
        critic = critic.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()

    res_dir = os.path.join(config['res_dir'], 'snr{}-lr{}'.format(config['snr'], config['lr']))
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

    # TODO
    discriminator=main_models.DCD(input_features=128)
    optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=0.001)

    # source samples预训练
    #--------------pretrain g and h for step 1---------------------------------
    for epoch in range(config['n_epoches_1']):
        for data,labels in config['source_train_loader']:
            data=data.to(device)
            labels=labels.to(device)

            optimizer.zero_grad()

            y_pred = classifier(extractor(data))

            loss = loss_class(y_pred, labels)
            loss.backward()

            optimizer.step()

        acc=0
        for data,labels in config['target_test_loader']:
            data=data.to(device)
            labels=labels.to(device)
            y_test_pred=classifier(extractor(data))
            acc+=(torch.max(y_test_pred,1)[1]==labels).float().mean().item()

        accuracy = round(acc / float(len(config['target_test_loader'])), 3)

        print("step1----Epoch %d/%d  accuracy: %.3f "%(epoch+1, config['n_epoches_1'], accuracy))


    #-----------------train DCD for step 2--------------------------------

    # X_s,Y_s=dataloader.sample_data()
    # X_t,Y_t=dataloader.create_target_samples(config['n_target_samples'])

    for epoch in range(config['n_epoches_2']):
        # for data,labels in config['source_train_loader']:
        iter_source = iter(config['source_train_loader'])
        iter_target = iter(config['target_train_loader'])
        len_source_loader = len(config['source_train_loader'])
        len_target_loader = len(config['target_train_loader'])
        num_iter = len_source_loader
        for i in range(1, num_iter+1):
            data_source, label_source = iter_source.next()
            data_target, label_target = iter_target.next()
            if i % len_target_loader == 0:
                iter_target = iter(config['target_train_loader'])
            if torch.cuda.is_available():
                data_source, label_source = data_source.cuda(), label_source.cuda()
                data_target, label_target = data_target.cuda(), label_target.cuda()


        groups, aa = dataloader.sample_groups(data_source, label_source, data_target, label_target,seed=epoch)
        # groups, aa = dataloader.sample_groups(X_s,Y_s,X_t,Y_t,seed=epoch)

        n_iters = 4 * len(groups[1])
        index_list = torch.randperm(n_iters)
        mini_batch_size = 40 #use mini_batch train can be more stable

        loss_mean=[]

        X1=[]; X2=[]; ground_truths=[]
        for index in range(n_iters):

            ground_truth = index_list[index]//len(groups[1])

            x1,x2=groups[ground_truth][index_list[index]-len(groups[1])*ground_truth]
            X1.append(x1)
            X2.append(x2)
            ground_truths.append(ground_truth)

            #select data for a mini-batch to train
            if (index+1) % mini_batch_size==0:
                X1=torch.stack(X1)
                X2=torch.stack(X2)
                ground_truths=torch.LongTensor(ground_truths)
                X1=X1.to(device)
                X2=X2.to(device)
                ground_truths=ground_truths.to(device)

                optimizer_D.zero_grad()
                X_cat=torch.cat([extractor(X1), extractor(X2)],1)
                y_pred=discriminator(X_cat.detach())
                loss=loss_domain(y_pred, ground_truths)
                loss.backward()
                optimizer_D.step()
                loss_mean.append(loss.item())
                X1 = []
                X2 = []
                ground_truths = []

        print("step2----Epoch %d/%d loss:%.3f"%(epoch+1, config['n_epoches_2'], np.mean(loss_mean)))

#-------------------training for step 3-------------------
optimizer_g_h=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()),lr=0.001)
optimizer_d=torch.optim.Adam(discriminator.parameters(),lr=0.001)

test_dataloader=dataloader.svhn_dataloader(train=False,batch_size=config['batch_size'])

for epoch in range(config['n_epoches_3']):
    #---training g and h , DCD is frozen

    groups, groups_y = dataloader.sample_groups(X_s,Y_s,X_t,Y_t,seed=config['n_epoches_2']+epoch)
    G1, G2, G3, G4 = groups
    Y1, Y2, Y3, Y4 = groups_y
    groups_2 = [G2, G4]
    groups_y_2 = [Y2, Y4]

    n_iters = 2 * len(G2)
    index_list = torch.randperm(n_iters)

    n_iters_dcd = 4 * len(G2)
    index_list_dcd = torch.randperm(n_iters_dcd)

    mini_batch_size_g_h = 20 #data only contains G2 and G4 ,so decrease mini_batch
    mini_batch_size_dcd= 40 #data contains G1,G2,G3,G4 so use 40 as mini_batch
    X1 = []
    X2 = []
    ground_truths_y1 = []
    ground_truths_y2 = []
    dcd_labels=[]
    for index in range(n_iters):


        ground_truth=index_list[index]//len(G2)
        x1, x2 = groups_2[ground_truth][index_list[index] - len(G2) * ground_truth]
        y1, y2 = groups_y_2[ground_truth][index_list[index] - len(G2) * ground_truth]
        # y1=torch.LongTensor([y1.item()])
        # y2=torch.LongTensor([y2.item()])
        dcd_label=0 if ground_truth==0 else 2
        X1.append(x1)
        X2.append(x2)
        ground_truths_y1.append(y1)
        ground_truths_y2.append(y2)
        dcd_labels.append(dcd_label)

        if (index+1)%mini_batch_size_g_h==0:

            X1=torch.stack(X1)
            X2=torch.stack(X2)
            ground_truths_y1=torch.LongTensor(ground_truths_y1)
            ground_truths_y2 = torch.LongTensor(ground_truths_y2)
            dcd_labels=torch.LongTensor(dcd_labels)
            X1=X1.to(device)
            X2=X2.to(device)
            ground_truths_y1=ground_truths_y1.to(device)
            ground_truths_y2 = ground_truths_y2.to(device)
            dcd_labels=dcd_labels.to(device)

            optimizer_g_h.zero_grad()

            encoder_X1=encoder(X1)
            encoder_X2=encoder(X2)

            X_cat=torch.cat([encoder_X1,encoder_X2],1)
            y_pred_X1=classifier(encoder_X1)
            y_pred_X2=classifier(encoder_X2)
            y_pred_dcd=discriminator(X_cat)

            loss_X1=loss_fn(y_pred_X1,ground_truths_y1)
            loss_X2=loss_fn(y_pred_X2,ground_truths_y2)
            loss_dcd=loss_fn(y_pred_dcd,dcd_labels)

            loss_sum = loss_X1 + loss_X2 + 0.2 * loss_dcd

            loss_sum.backward()
            optimizer_g_h.step()

            X1 = []
            X2 = []
            ground_truths_y1 = []
            ground_truths_y2 = []
            dcd_labels = []


    #----training dcd ,g and h frozen
    X1 = []
    X2 = []
    ground_truths = []
    for index in range(n_iters_dcd):

        ground_truth=index_list_dcd[index]//len(groups[1])

        x1, x2 = groups[ground_truth][index_list_dcd[index] - len(groups[1]) * ground_truth]
        X1.append(x1)
        X2.append(x2)
        ground_truths.append(ground_truth)

        if (index + 1) % mini_batch_size_dcd == 0:
            X1 = torch.stack(X1)
            X2 = torch.stack(X2)
            ground_truths = torch.LongTensor(ground_truths)
            X1 = X1.to(device)
            X2 = X2.to(device)
            ground_truths = ground_truths.to(device)

            optimizer_d.zero_grad()
            X_cat = torch.cat([encoder(X1), encoder(X2)], 1)
            y_pred = discriminator(X_cat.detach())
            loss = loss_fn(y_pred, ground_truths)
            loss.backward()
            optimizer_d.step()
            # loss_mean.append(loss.item())
            X1 = []
            X2 = []
            ground_truths = []

    #testing
    acc = 0
    for data, labels in test_dataloader:
        data = data.to(device)
        labels = labels.to(device)
        y_test_pred = classifier(encoder(data))
        acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()

    accuracy = round(acc / float(len(test_dataloader)), 3)

    print("step3----Epoch %d/%d  accuracy: %.3f " % (epoch + 1, config['n_epoches_3'], accuracy))
        





















