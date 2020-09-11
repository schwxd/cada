import os
import math
import logging
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.functions import test, set_log_config, ReverseLayerF
from utils.vis import draw_tsne, draw_confusion_matrix
from networks.network import Extractor, Classifier2, Critic, Critic2, RandomLayer, AdversarialNetwork
from models.GTA import gta_models
from models.GTA import gta_models_unet
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

def train_gta_unet(config):
    nclasses = config['n_class']
    netG = gta_models_unet.GeneratorUNet(in_channels=1, out_channels=1)
    netD = gta_models_unet.Discriminator(in_channels=1, flattens=config['n_flattens'])
    print('netD: {}'.format(netD))
    print('netG: {}'.format(netG))

    optimizerD = optim.Adam(netD.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config['lr'], betas=(0.5, 0.999))

    criterion_c = nn.CrossEntropyLoss()
    criterion_s = nn.BCELoss()
    mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        netD.cuda()
        netG.cuda()
        # netF.cuda()
        # netC.cuda()

        criterion_c.cuda()
        criterion_s.cuda()

    res_dir = os.path.join(config['res_dir'], 'slim{}-targetLabel{}-mmd{}-bnm{}-vat{}-ent{}-ew{}-bn{}-bs{}-lr{}'.format(config['slim'],
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
    logging.debug('train_gta')

    real_label_val = 1
    fake_label_val = 0

    D_tgt_fake = True

    """
    Train function
    """
    def train(epoch):
        netG.train()    
        netD.train()    
        # netF.train()    
        # netC.train()    
        
        reallabel = torch.FloatTensor(config['batch_size'], 1).fill_(real_label_val)
        fakelabel = torch.FloatTensor(config['batch_size'], 1).fill_(fake_label_val)
        reallabel, fakelabel = reallabel.cuda(), fakelabel.cuda()

        source_domain = torch.FloatTensor(config['batch_size']).fill_(real_label_val)
        target_domain = torch.FloatTensor(config['batch_size']).fill_(fake_label_val)
        source_domain, target_domain = source_domain.cuda(), target_domain.cuda()

        iter_source = iter(config['source_train_loader'])
        iter_target = iter(config['target_train_loader'])
        len_source_loader = len(config['source_train_loader'])
        len_target_loader = len(config['target_train_loader'])
        if config['slim'] > 0:
            iter_target_semi = iter(config['target_train_semi_loader'])
            len_target_semi_loader = len(config['target_train_semi_loader'])

        num_iter = len_source_loader
        for i in range(1, num_iter+1):
            src_inputs, src_labels = iter_source.next()
            tgt_inputs, tgt_labels = iter_target.next()
            if config['slim'] > 0:
                data_target_semi, label_target_semi = iter_target_semi.next()
                if i % len_target_semi_loader == 0:
                    iter_target_semi = iter(config['target_train_semi_loader'])

            if i % len_target_loader == 0:
                iter_target = iter(config['target_train_loader'])
            if torch.cuda.is_available():
                src_inputs, src_labels = src_inputs.cuda(), src_labels.cuda()
                tgt_inputs, tgt_labels = tgt_inputs.cuda(), tgt_labels.cuda()
                if config['slim'] > 0:
                    data_target_semi, label_target_semi = data_target_semi.cuda(), label_target_semi.cuda()

            ###########################
            # Forming input variables
            ###########################
            # Creating one hot vector
            labels_onehot = np.zeros((config['batch_size'], nclasses+1), dtype=np.float32)
            for num in range(config['batch_size']):
                labels_onehot[num, src_labels[num]] = 1
            src_labels_onehot = torch.from_numpy(labels_onehot)

            labels_onehot = np.zeros((config['batch_size'], nclasses+1), dtype=np.float32)
            for num in range(config['batch_size']):
                labels_onehot[num, nclasses] = 1
            tgt_labels_onehot = torch.from_numpy(labels_onehot)
            
            src_labels_onehot = src_labels_onehot.cuda()
            tgt_labels_onehot = tgt_labels_onehot.cuda()
            
            ###########################
            # Updating D network
            ###########################
            netD.zero_grad()

            src_fake = netG(src_inputs)
            tgt_fake = netG(tgt_inputs)
            # print('src_fake {}'.format(src_fake.shape))

            src_fakeoutputD_s, src_fakeoutputD_c, src_fake_feature = netD(src_fake)
            # print('src_fakeoutputD_s {}, fakelabel {}'.format(src_fakeoutputD_s.shape, fakelabel.shape))
            errD_src_fake_dloss = criterion_s(src_fakeoutputD_s, fakelabel)
            errD_src_fake_closs = criterion_c(src_fakeoutputD_c, src_labels)
            # print('D src_fake_feature {}'.format(src_fake_feature.shape))

            tgt_fakeoutputD_s, tgt_fakeoutputD_c, tgt_fake_feature = netD(tgt_fake)          
            errD_tgt_fake_dloss = criterion_s(tgt_fakeoutputD_s, fakelabel)

            errD_mmd = mmd_loss(src_fake_feature, tgt_fake_feature)

            errD = (errD_src_fake_dloss + errD_tgt_fake_dloss) + errD_src_fake_closs + config['mmd_gamma'] * errD_mmd
            if i % 100 == 0:
                print('errD {}, (errD_src_fake_dloss {}, errD_tgt_fake_dloss {}, errD_src_fake_closs {}, errD_mmd {})'.format(
                    errD, errD_src_fake_dloss, errD_tgt_fake_dloss, errD_src_fake_closs, errD_mmd))
            errD.backward()    
            optimizerD.step()


            ###########################
            # Updating G network
            ###########################
            netG.zero_grad()

            src_fake = netG(src_inputs)
            tgt_fake = netG(tgt_inputs)

            src_fakeoutputD_s, src_fakeoutputD_c, src_fake_feature = netD(src_fake)
            errG_src_fake_dloss = criterion_s(src_fakeoutputD_s, reallabel)
            errG_src_fake_closs = criterion_c(src_fakeoutputD_c, src_labels)
            # print('D src_fake_feature {}'.format(src_fake_feature.shape))

            tgt_fakeoutputD_s, tgt_fakeoutputD_c, tgt_fake_feature = netD(tgt_fake)          
            errG_tgt_fake_dloss = criterion_s(tgt_fakeoutputD_s, reallabel)

            errG_mmd = mmd_loss(src_fake_feature, tgt_fake_feature)
            errG = (errG_src_fake_dloss + errG_tgt_fake_dloss) + errG_src_fake_closs + config['mmd_gamma'] * errG_mmd
            if i % 100 == 0:
                print('errG {}, (errG_src_fake_dloss {}, errG_tgt_fake_dloss {}, errG_src_fake_closs {}, errG_mmd {})'.format(
                    errG, errG_src_fake_dloss, errG_tgt_fake_dloss, errG_src_fake_closs, errG_mmd
                ))
            errG.backward()    
            optimizerG.step()

            # Visualization
            # if i == 1:
            #     vutils.save_image((src_gen.data/2)+0.5, '%s/visualization/source_gen_%d.png' %(self.opt.outf, epoch))
            #     vutils.save_image((tgt_gen.data/2)+0.5, '%s/visualization/target_gen_%d.png' %(self.opt.outf, epoch))
                
            # Learning rate scheduling
            # if self.opt.lrd:
            #     self.optimizerD = utils.exp_lr_scheduler(self.optimizerD, epoch, self.opt.lr, self.opt.lrd, curr_iter)    
            #     self.optimizerF = utils.exp_lr_scheduler(self.optimizerF, epoch, self.opt.lr, self.opt.lrd, curr_iter)
            #     self.optimizerC = utils.exp_lr_scheduler(self.optimizerC, epoch, self.opt.lr, self.opt.lrd, curr_iter)                  
            #     # optimizerG要不要梯度递减？原始实现没有
            #     self.optimizerG = utils.exp_lr_scheduler(self.optimizerG, epoch, self.opt.lr, self.opt.lrd, curr_iter)                  


    def validate(epoch):
        netG.eval()
        netD.eval()
        total = 0
        correct = 0
    
        # Testing the model
        for i, datas in enumerate(config['target_test_loader']):
            inputs, labels = datas         
            inputs = inputs.cuda()
            with torch.no_grad():
                _, outC, _ = netD(inputs)          
                _, predicted = torch.max(outC.data, 1)        
                total += labels.size(0)
                correct += ((predicted == labels.cuda()).sum())
            
        val_acc = 100*float(correct)/total
        print('Validate1 | Epoch: %d, Val Accuracy: %f %%' % (epoch, val_acc))
        logging.debug('%s| Epoch: %d, Val Accuracy: %f %%' % (datetime.datetime.now(), epoch, val_acc))


    for epoch in range(1, config['n_epochs'] + 1):
        train(epoch)
        if epoch % config['TEST_INTERVAL'] == 0:
            # print('test on source_test_loader')
            # test(extractor, classifier, config['source_test_loader'], epoch)
            print('test on target_test_loader')
            validate(epoch)
            # validate2(epoch)
            # test(extractor, classifier, config['target_test_loader'], epoch)
        # if epoch % config['VIS_INTERVAL'] == 0:
        #     title = 'DANN'
        #     if config['bnm'] == 1 and config['vat'] == 1:
        #         title = '(b) Proposed'
        #     elif config['bnm'] == 1:
        #         title = 'BNM'
        #     elif config['vat'] == 1:
        #         title = 'VADA'
        #     elif config['mmd'] == 1:
        #         title = 'DCTLN'
        #     elif config['ent'] == 1:
        #         title = 'EntMin'
        #     # draw_confusion_matrix(extractor, classifier, config['target_test_loader'], res_dir, epoch, config['models'])
        #     draw_tsne(extractor, classifier, config['source_train_loader'], config['target_test_loader'], res_dir, epoch, title, separate=True)
        #     draw_tsne(extractor, classifier, config['source_train_loader'], config['target_test_loader'], res_dir, epoch, title, separate=False)
