import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import models.CDAN.cdan_loss as loss_func

from utils.functions import test, set_log_config
from network import Extractor, Classifier, Critic, Critic2, RandomLayer, AdversarialNetwork
from utils.vis import draw_tsne, draw_confusion_matrix

draw_dict = {
    "class_loss_point":[],
    "domain_loss_point":[],
    "target_loss_point":[],
    "source_acc_point":[],
    "domain_acc_point":[],
    "target_acc_point":[],
    "set_len_point":[],
    "confid_threshold_point":[],
    "epoch_point":[],
    "lr_point":[],
    "domain_loss_point_l1":[],
    "domain_loss_point_l2":[],
    "domain_loss_point_l3":[],
    "domain_acc_point_l1":[],
    "domain_acc_point_l2":[],
    "domain_acc_point_l3":[],
}

def set_training_mode(model, training):
    if training == True:
        model.train()
        for param in model.parameters():
            param.requires_grad = True
    else:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

class Contrast_ReLU_activate(nn.Module):

    def __init__(self, initWeightScale, initBias):

        super(Contrast_ReLU_activate, self).__init__()

        self.dom_func_weight = nn.Parameter(torch.ones(1),requires_grad=True)
        self.dom_func_bias = torch.FloatTensor([0]).cuda()

        self.weight_scale = initWeightScale
        self.add_bias = initBias

    def forward(self, dom_res, dom_label):

        w = (self.dom_func_weight * self.weight_scale).cuda()
        b = (self.dom_func_bias + self.add_bias).cuda()

        dom_prob = torch.sigmoid(dom_res).squeeze()
        dom_variance = torch.abs(dom_prob - 0.5)

        act_weight = 0.8 - w * dom_variance**4  + b
        # act_weight = torch.add(0.8, w * dom_variance**4, b)
        
        # Minimise function to zero(target)
        zeros_var = b
        f_weight = torch.max(act_weight, zeros_var)

        final_weight = f_weight
        
        return final_weight, float(w.squeeze()), float(b.squeeze())


def compute_new_loss(logits, target, weights):
    """ logits: Unnormalized probability for each class.
        target: index of the true class(label)
        weights: weights of weighted loss.
    Returns:
        loss: An average weighted loss value
    """
    # print("l: ",logits)
    # print("t: ",target)
    weights = weights.narrow(0,0,len(target))
    # print("w: ",weights)
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size()) * weights
    # losses = losses * weights
    loss = losses.sum() / len(target)
    # length.float().sum()
    return loss

# INI_DISC_WEIGHT_SCALE = (200)**4
# INI_DISC_WEIGHT_SCALE = 200
INI_DISC_WEIGHT_SCALE = 200**3
INI_DISC_BIAS = 0.5

def train_cdan_ican(config):
    BATCH_SIZE = config['batch_size']
    extractor = Extractor(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'])
    classifier = Classifier(n_flattens=config['n_flattens'], n_hiddens=config['n_hiddens'], n_class=config['n_class'])
    disc_activate = Contrast_ReLU_activate(INI_DISC_WEIGHT_SCALE, INI_DISC_BIAS)

    cdan_random = config['random_layer'] 
    if config['models'] == 'DANN':
        random_layer = None
        ad_net = AdversarialNetwork(config['n_flattens'], config['n_hiddens'])
    elif cdan_random:
        random_layer = RandomLayer([config['n_flattens'], config['n_class']], config['n_hiddens'])
        ad_net = AdversarialNetwork(config['n_hiddens'], config['n_hiddens'])
        random_layer.cuda()
    else:
        random_layer = None
        # ad_net = AdversarialNetwork(config['n_flattens'] * config['n_class'], config['n_hiddens'])
        ad_net = AdversarialNetwork(config['n_flattens'], config['n_hiddens'])

    if torch.cuda.is_available():
        extractor = extractor.cuda()
        classifier = classifier.cuda()
        disc_activate = disc_activate.cuda()
        ad_net = ad_net.cuda()

    optimizer = torch.optim.Adam([
        {'params': extractor.parameters(), 'lr': config['lr']},
        {'params': classifier.parameters(), 'lr': config['lr']}
        ])
    optimizer_ad = torch.optim.Adam(ad_net.parameters(), lr=config['lr'])
    pseudo_optimizer = torch.optim.Adam(disc_activate.parameters(), lr=config['lr'])

    class_criterion = nn.CrossEntropyLoss()

    res_dir = os.path.join(config['res_dir'], 'random{}-bs{}-lr{}'.format(cdan_random, config['batch_size'], config['lr']))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    print('train_cdan_ican')
    print(extractor)
    print(classifier)
    print(ad_net)
    print(config)

    set_log_config(res_dir)
    logging.debug('train_cdan_ican')
    logging.debug(extractor)
    logging.debug(classifier)
    logging.debug(ad_net)
    logging.debug(config)


    def select_samples_ican(extractor, classifier, ad_net, disc_activate, config, epoch, epoch_acc_s):
        set_training_mode(extractor, False)
        set_training_mode(classifier, False)
        set_training_mode(ad_net, False)
        set_training_mode(disc_activate, False)

        Pseudo_set = []
        confid_threshold = 1 / (1 + np.exp(-2.4*epoch_acc_s))
        total_pseudo_errors = 0

        # 为什么在target测试集上进行？
        # for target_inputs, target_labels in iter(config['target_test_loader']):
        for target_inputs, target_labels in iter(config['target_train_loader']):
            target_inputs = target_inputs.cuda()
            # 论文中target的domain label是1
            domain_labels_t = torch.FloatTensor([0.]*len(target_inputs)).cuda()

            embeddings = extractor(target_inputs)
            class_t = classifier(embeddings)
            domain_out_t = ad_net(embeddings, training=False)
            disc_weight_t, w_t, b_t = disc_activate(domain_out_t, domain_labels_t)
            top_prob, preds_t = torch.max(class_t, 1)

            for i in range(len(disc_weight_t)):
                if disc_weight_t[i] > b_t and top_prob[i] >= float(confid_threshold):
                    s_tuple = (target_inputs[i].cpu(), (preds_t[i].cpu(), float(disc_weight_t[i])))
                    Pseudo_set.append(s_tuple)
            total_pseudo_errors += preds_t.eq(target_labels.cuda()).cpu().sum()

        # 每个pseudo_set样本中包括[features, category_class_predict, domain_weight_predict], [特征，预测的类标，预测的domain权重]
        # print("Pseudo error/total = {}/{}, confid_threshold: {:.4f}".format(total_pseudo_errors, len(Pseudo_set), 
                                                                            # confid_threshold)) 
        print('Epoch {}, Stage Select_Sample, accuracy {}, confident threshold {}, pseudo number {}, b_t {}'.format(epoch, epoch_acc_s, confid_threshold, len(Pseudo_set), b_t))
        draw_dict['confid_threshold_point'].append(float("%.4f" % confid_threshold))

        return Pseudo_set


    # TODO: 为什么不在上一个函数中直接更新呢？选择pseudo-set之后就更新disc-activate的模型参数，完全可以合并成一步
    def update_ican(extractor, classifier, ad_net, disc_activate, config, Pseudo_set, epoch):
        if len(Pseudo_set) == 0:
            return

        set_training_mode(extractor, False)
        set_training_mode(classifier, False)
        set_training_mode(ad_net, False)
        set_training_mode(disc_activate, True)

        pseudo_batch_count = 0
        pseudo_sample_count = 0
        pseudo_epoch_loss = 0.0
        pseudo_epoch_acc = 0
        pseudo_epoch_corrects = 0
        pseudo_avg_loss = 0.0

        # TODO: 每次从pseudo-set中取半个batch-size
        pseudo_loader = torch.utils.data.DataLoader(Pseudo_set,
                                            batch_size=int(BATCH_SIZE / 2), shuffle=True)

        for pseudo_inputs, pseudo_labels in pseudo_loader:

            pseudo_batch_count += 1
            pseudo_sample_count += len(pseudo_inputs)

            pseudo_labels, pseudo_weights = pseudo_labels[0], pseudo_labels[1]
            pseudo_inputs, pseudo_labels = pseudo_inputs.cuda(), pseudo_labels.cuda()
            domain_labels = torch.FloatTensor([0.]*len(pseudo_inputs)).cuda()

            embeddings = extractor(pseudo_inputs)
            pseudo_class = classifier(embeddings)
            pseudo_domain_out = ad_net(embeddings, training=False)
            pseudo_disc_weight, pseudo_ww, pseudo_bb = disc_activate(pseudo_domain_out, domain_labels)

            pseudo_optimizer.zero_grad()

            # TODO：为什么不用这个pseudo_preds, 而要用上个函数保存的结果呢？
            _, pseudo_preds = torch.max(pseudo_class, 1)

            # pseudo_class：未经过softmax的类分类概率
            # pseudo_labels: 经过softmax的类标签
            # pseudo_disc_weight：样本的领域权重
            # TODO：检查pseudo_disc_weight的形状
            # pseudo_class_loss = compute_new_loss(pseudo_class, pseudo_labels, pseudo_disc_weight)
            pseudo_class_loss = compute_new_loss(pseudo_class, pseudo_preds, pseudo_disc_weight)
            # pseudo_class_loss = class_criterion(pseudo_class, pseudo_preds)

            pseudo_epoch_loss += float(pseudo_class_loss)

            # 这个正确率没有意义
            # pseudo_preds 是pseudo_class的最大值，是target train的预测值
            # pseudo_labels 是上一个函数（选择pseudo-set时）计算出来的，同样的公式
            pseudo_epoch_corrects += int(torch.sum(pseudo_preds.squeeze() == pseudo_labels.squeeze()))

            pseudo_loss = pseudo_class_loss
            pseudo_loss.backward()
            pseudo_optimizer.step()

            epoch_discrim_lambda = 1.0 / (abs(pseudo_ww) ** (1. / 4))
            epoch_discrim_bias = pseudo_bb

        pseudo_avg_loss = pseudo_epoch_loss / pseudo_batch_count
        pseudo_epoch_acc = pseudo_epoch_corrects / pseudo_sample_count

        print('Epoch {}, Phase: {}, Loss: {:.4f} Acc: {:.4f} Disc_Lam: {:.6f} Disc_bias: {:.4f} '.format(
                epoch, 'Pseudo_train', pseudo_avg_loss, pseudo_epoch_acc, epoch_discrim_lambda, epoch_discrim_bias))


    def prepare_dataset(epoch, pseudo_set):
        dset_loaders = {}

        dset_loaders['source'] = config['source_train_loader']
        source_size = len(config['source_train_loader'])
        pseudo_size = len(pseudo_set)
        # source_batches_per_epoch = np.floor(source_size * 2 / BATCH_SIZE).astype(np.int16)
        # total_epochs = config['n_epochs']

        if pseudo_size == 0:
            dset_loaders['pseudo'] = []
            dset_loaders['pseudo_source'] = []
            # source_batchsize = int(BATCH_SIZE / 2)
            source_batchsize = BATCH_SIZE
            pseudo_batchsize = 0
        else:
            # source_batchsize = int(int(BATCH_SIZE / 2) * source_size
            #                             / (source_size + pseudo_size))
            # if source_batchsize == int(BATCH_SIZE / 2):
            #     source_batchsize -= 1
            # if source_batchsize < int(int(BATCH_SIZE / 2) / 2):
            #     source_batchsize = int(int(BATCH_SIZE / 2) / 2)
            # pseudo_batchsize = int(BATCH_SIZE / 2) - source_batchsize
            # print('source_batchsize {}, pseudo_batchsize {}'.format(source_batchsize, pseudo_batchsize))

            # dset_loaders['pseudo'] = torch.utils.data.DataLoader(pseudo_set,
            #                                 batch_size=pseudo_batchsize, shuffle=True)

            # dset_loaders['pseudo_source'] = config['source_train_loader']

            #
            # 重新修改，按照source_train中每个epoch的batch数量，计算pseudo-set的batchsize
            pseudo_batchsize = int(np.floor(pseudo_size / len(config['source_train_loader'])))
            dset_loaders['pseudo'] = torch.utils.data.DataLoader(pseudo_set,
                                            batch_size=pseudo_batchsize, shuffle=True, drop_last=False)
            dset_loaders['pseudo_source'] = config['source_train_loader']
            source_batchsize = BATCH_SIZE


        print('Epoch {}, Stage prepare_dataset, pseudo_size {}, num batch each epoch: {}, pseudo_batchsize {}'.format(epoch, pseudo_size, source_size, pseudo_batchsize))

        target_dict = [(i,j) for (i,j) in config['target_train_loader']]
        if pseudo_size > 0:
            pseudo_dict = [(i,j) for (i,j) in dset_loaders['pseudo']]
            pseudo_source_dict = [(i,j) for (i,j) in dset_loaders['pseudo_source']]
        else:
            pseudo_dict = []
            pseudo_source_dict = []

        # total_iters = source_batches_per_epoch * pre_epochs + \
        #                 source_batches_per_epoch * (total_epochs - pre_epochs) * \
        #                 BATCH_SIZE / (source_batchsize * 2)
        # total_iters = source_batches_per_epoch * (total_epochs) * BATCH_SIZE / (source_batchsize * 2)

        return dset_loaders, target_dict, pseudo_dict, pseudo_source_dict, source_batchsize, pseudo_batchsize

    def train(extractor, classifier, ad_net, disc_activate, config, epoch):
        start_epoch = 0

        extractor.train()
        classifier.train()
        ad_net.train()
        disc_activate.train()

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
                elif config['models'] == 'CDAN_ICAN':
                    d_loss = loss_func.CDAN([feature, softmax_output], ad_net, gamma, None, None, random_layer)
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


    # def do_forward(extractor, classifier, ad_net, disc_activate, src_features, all_features, labels):
    
    #     # 预测source features的class labels
    #     bottle = extractor(src_features)
    #     class_pred = classifier(bottle)

    #     dom_pred = ad_net(bottle)

    #     return class_pred, dom_pred.squeeze(1)


    def do_training(dset_loaders, target_dict, source_batchsize, pseudo_batchsize, pseudo_dict, pseudo_source_dict):

        batch_count = 0
        target_pointer = 0
        target_pointer = 0
        pseudo_pointer = 0
        pseudo_source_pointer = 0
        INI_MAIN_THRESH = -0.8
        # pre_epochs = 10
        pre_epochs = 0

        set_training_mode(extractor, True)
        set_training_mode(classifier, True)
        set_training_mode(ad_net, True)
        set_training_mode(disc_activate, False)


        # class_count = 0
        # epoch_loss = 0.0
        # epoch_corrects = 0
        # domain_epoch_loss = 0.0
        # ini_w_main = torch.FloatTensor([float(INI_MAIN_THRESH)]).cuda()
        # epoch_batch_count = 0
        # total_epoch_loss = 0.0
        # domain_epoch_corrects = 0
        # domain_counts = 0


        for data in dset_loaders['source']:
            inputs, labels = data

            batch_count += 1

            # ---------------- reset exceeded datasets --------------------
            if target_pointer >= len(target_dict) - 1:
                target_pointer = 0
                target_dict = [(i,j) for (i,j) in config['target_train_loader']]

            target_inputs = target_dict[target_pointer][0]

            if epoch <= pre_epochs:
                # 训练CAN，使用source_train和target_train，target_train不经筛选，全部使用
                # -------------------- pretrain model -----------------------
                domain_inputs = torch.cat((inputs, target_inputs),0)
                # domain_labels = torch.FloatTensor([1.]*BATCH_SIZE + [0.]*BATCH_SIZE)
                domain_labels = torch.FloatTensor([1.]*inputs.size(0) + [0.]*target_inputs.size(0))
                domain_inputs, domain_labels = domain_inputs.cuda(), domain_labels.cuda()

                inputs, labels = inputs.cuda(), labels.cuda()

                # print('inputs {}, target_inputs {}, domain_inputs {}, domain_labels {}'.format(inputs.size(), target_inputs.size(), domain_inputs.size(), domain_labels.size()))

                # source数据集上的分类结果
                class_outputs = classifier(extractor(inputs))

                # 在source和target数据集上判断domain分类
                domain_outputs = ad_net(extractor(domain_inputs)).squeeze()

                target_pointer += 1
                # epoch_discrim_bias = 0.5

                # ------------ training classification statistics --------------
                criterion = nn.CrossEntropyLoss()
                class_loss = criterion(class_outputs, labels)
                
            else:
                # -------------- train with pseudo sample model -------------
                # target域使用经过筛选的pseudo-set数据
                pseudo_weights = torch.FloatTensor([])
                pseudo_size = len(pseudo_dict)

                # 重置索引位置
                if (pseudo_pointer >= len(pseudo_dict) - 1) and (len(pseudo_dict) != 0) :
                    pseudo_pointer = 0
                    pseudo_dict = [(i,j) for (i,j) in dset_loaders['pseudo']]

                if (pseudo_source_pointer >= len(pseudo_source_dict) - 1) and (len(pseudo_source_dict) != 0):
                    pseudo_source_pointer = 0
                    pseudo_source_dict = [(i,j) for (i,j) in dset_loaders['pseudo_source']]


                if pseudo_size == 0:
                    # 如果pseudo-set为空，那还是使用全部source_train和target_train

                    domain_inputs = torch.cat((inputs, target_inputs),0)
                    # domain_labels = torch.FloatTensor([1.]*int(BATCH_SIZE / 2)+
                                                        # [0.]*int(BATCH_SIZE / 2))
                    domain_labels = torch.FloatTensor([1.]*inputs.size(0) + [0.]*target_inputs.size(0))

                    fuse_inputs = inputs
                    fuse_labels = labels

                else:
                    pseudo_inputs, pseudo_labels, pseudo_weights = pseudo_dict[pseudo_pointer][0], \
                                    pseudo_dict[pseudo_pointer][1][0], pseudo_dict[pseudo_pointer][1][1]
                    pseudo_source_inputs = pseudo_source_dict[pseudo_source_pointer][0]

                    # TODO: 为什么要这么干？source + pseudo + target + source
                    # domain_inputs = torch.cat((inputs, pseudo_inputs, target_inputs, pseudo_source_inputs),0)
                    # domain_labels = torch.FloatTensor([1.]*inputs.size(0) + [0.]*pseudo_inputs.size(0) + 
                    #                                     [0.]*target_inputs.size(0)+[1.]*pseudo_source_inputs.size(0))
                    domain_inputs = torch.cat((inputs, pseudo_inputs),0)
                    domain_labels = torch.FloatTensor([1.]*inputs.size(0) + [0.]*pseudo_inputs.size(0))

                    fuse_inputs = torch.cat((inputs, pseudo_inputs),0)
                    fuse_labels = torch.cat((labels, pseudo_labels),0)

                    # print('inputs {}, pseudo_inputs {}, target_inputs {}, domain_inputs {}'.format(inputs.size(), pseudo_inputs.size(), target_inputs.size(), domain_inputs.size()))
                    # print('domain_labels {}, fuse_inputs {}, fuse_labels {}'.format(domain_labels.size(), fuse_inputs.size(), fuse_labels.size()))


                inputs, labels = fuse_inputs.cuda(), fuse_labels.cuda()
                domain_inputs, domain_labels = domain_inputs.cuda(), domain_labels.cuda()

                source_weight_tensor = torch.FloatTensor([1.]*source_batchsize)
                pseudo_weights_tensor = pseudo_weights.float()
                class_weights_tensor = torch.cat((source_weight_tensor, pseudo_weights_tensor),0)
                dom_weights_tensor = torch.FloatTensor([0.]*source_batchsize+[1.]*pseudo_batchsize)

                ini_weight = torch.cat((class_weights_tensor, dom_weights_tensor),0).squeeze().cuda()

                class_outputs = classifier(extractor(inputs))
                domain_outputs = ad_net(extractor(domain_inputs)).squeeze()

                # ------------ training classification statistics --------------
                # _, preds = torch.max(class_outputs, 1)
                # class_count += len(preds)
                class_loss = compute_new_loss(class_outputs, labels, ini_weight)


                # epoch_loss += float(class_loss)
                # epoch_corrects += int(torch.sum(preds.squeeze() == labels.squeeze()))

                target_pointer += 1
                pseudo_pointer += 1
                pseudo_source_pointer += 1

            # zero the parameter gradients
            optimizer.zero_grad()
            optimizer_ad.zero_grad()

            # ----------- calculate pred domain labels and losses -----------
            domain_criterion = nn.BCEWithLogitsLoss()
            domain_labels = domain_labels.squeeze()
            domain_loss = domain_criterion(domain_outputs, domain_labels)
            # domain_epoch_loss += float(domain_loss)

            # ------ calculate pseudo predicts and losses with weights and threshold lambda -------
            total_loss = class_loss + 1.0*domain_loss

            # total_epoch_loss += float(total_loss)
            print('class_loss {}, domain_loss {}'.format(class_loss.item(), domain_loss.item()))

            #  -------  backward + optimize in training and Pseudo-training phase -------
            total_loss.backward()
            optimizer.step()
            optimizer_ad.step()

    def train_ican(extractor, classifier, ad_net, disc_activate, config, epoch):
        # start_epoch = 0

        # 1. 计算在source上的准确度，用于选择伪标签
        accuracy_s = test(extractor, classifier, config['source_test_loader'], epoch)

        # 2. 计算伪标签数据集
        pseu_set = select_samples_ican(extractor, classifier, ad_net, disc_activate, config, epoch, accuracy_s)

        # 3. 使用伪数据集训练disc_activate，更新disc threshold
        update_ican(extractor, classifier, ad_net, disc_activate, config, pseu_set, epoch)

        # 4. 准备最终训练ican所用的数据集，将source dataset和pseudo set合并
        dset_loaders, target_dict, pseudo_dict, pseudo_source_dict, source_batchsize, pseudo_batchsize = prepare_dataset(epoch, pseu_set)

        # 5. train
        # do_training()
        do_training(dset_loaders, target_dict, source_batchsize, pseudo_batchsize, pseudo_dict, pseudo_source_dict)


        # iter_source = iter(config['source_train_loader'])
        # iter_target = iter(config['target_train_loader'])
        # len_source_loader = len(config['source_train_loader'])
        # len_target_loader = len(config['target_train_loader'])
        # num_iter = len_source_loader
        # for step in range(1, num_iter + 1):
        #     data_source, label_source = iter_source.next()
        #     data_target, _ = iter_target.next()
        #     if step % len_target_loader == 0:
        #         iter_target = iter(config['target_train_loader'])
        #     if torch.cuda.is_available():
        #         data_source, label_source = data_source.cuda(), label_source.cuda()
        #         data_target = data_target.cuda()

        #     optimizer.zero_grad()
        #     optimizer_ad.zero_grad()

        #     h_s = extractor(data_source)
        #     h_s = h_s.view(h_s.size(0), -1)
        #     h_t = extractor(data_target)
        #     h_t = h_t.view(h_t.size(0), -1)

        #     source_preds = classifier(h_s)
        #     cls_loss = nn.CrossEntropyLoss()(source_preds, label_source)
        #     softmax_output_s = nn.Softmax(dim=1)(source_preds)

        #     target_preds = classifier(h_t)
        #     softmax_output_t = nn.Softmax(dim=1)(target_preds)

        #     feature = torch.cat((h_s, h_t), 0)
        #     softmax_output = torch.cat((softmax_output_s, softmax_output_t), 0)

        #     if epoch > start_epoch:
        #         gamma = 2 / (1 + math.exp(-10 * (epoch) / config['n_epochs'])) - 1
        #         if config['models'] == 'CDAN-E':
        #             entropy = loss_func.Entropy(softmax_output)
        #             d_loss = loss_func.CDAN([feature, softmax_output], ad_net, gamma, entropy, loss_func.calc_coeff(num_iter*(epoch-start_epoch)+step), random_layer)
        #         elif config['models'] == 'CDAN':
        #             d_loss = loss_func.CDAN([feature, softmax_output], ad_net, gamma, None, None, random_layer)
        #         elif config['models'] == 'DANN':
        #             d_loss = loss_func.DANN(feature, ad_net, gamma)
        #         elif config['models'] == 'CDAN_ICAN':
        #             d_loss = loss_func.CDAN([feature, softmax_output], ad_net, gamma, None, None, random_layer)
        #         else:
        #             raise ValueError('Method cannot be recognized.')
        #     else:
        #         d_loss = 0

        #     loss = cls_loss + d_loss
        #     loss.backward()
        #     optimizer.step()
        #     if epoch > start_epoch:
        #         optimizer_ad.step()
        #     if (step) % 20 == 0:
        #         print('Train Epoch {} closs {:.6f}, dloss {:.6f}, Loss {:.6f}'.format(epoch, cls_loss.item(), d_loss.item(), loss.item()))
    # function done


    for epoch in range(1, config['n_epochs'] + 1):
        train_ican(extractor, classifier, ad_net, disc_activate, config, epoch)
        if epoch % config['TEST_INTERVAL'] == 0:
            print('test on source_test_loader')
            test(extractor, classifier, config['source_test_loader'], epoch)
            print('test on target_test_loader')
            accuracy = test(extractor, classifier, config['target_test_loader'], epoch)

        if epoch % config['VIS_INTERVAL'] == 0:
            title = config['models']
            draw_confusion_matrix(extractor, classifier, config['target_test_loader'], res_dir, epoch, title)
            draw_tsne(extractor, classifier, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, title, separate=True)
            # draw_tsne(extractor, classifier, config['source_test_loader'], config['target_test_loader'], res_dir, epoch, title, separate=False)

