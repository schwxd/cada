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

def train_gta(config):
    nclasses = config['n_class']
    netC = gta_models._netC(nclasses=config['n_class'], flattens=config['n_flattens'])
    netD = gta_models._netD(nclasses=config['n_class'], flattens=config['n_flattens'])
    netF = gta_models._netF()
    netG = gta_models._netG(nclasses=config['n_class'], flattens=config['n_flattens'], nz=config['nz'],)

    print('netC: {}'.format(netC))
    print('netD: {}'.format(netD))
    print('netF: {}'.format(netF))
    print('netG: {}'.format(netG))

    optimizerD = optim.Adam(netD.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    optimizerF = optim.Adam(netF.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    optimizerC = optim.Adam(netC.parameters(), lr=config['lr'], betas=(0.5, 0.999))

    criterion_c = nn.CrossEntropyLoss()
    criterion_s = nn.BCELoss()
    mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        netD.cuda()
        netG.cuda()
        netF.cuda()
        netC.cuda()

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
        netF.train()    
        netC.train()    
        netD.train()    
        
        reallabel = torch.FloatTensor(config['batch_size']).fill_(real_label_val)
        fakelabel = torch.FloatTensor(config['batch_size']).fill_(fake_label_val)
        reallabel, fakelabel = reallabel.cuda(), fakelabel.cuda()

        source_domain = torch.FloatTensor(config['batch_size']).fill_(real_label_val)
        target_domain = torch.FloatTensor(config['batch_size']).fill_(fake_label_val)
        source_domain, target_domain = source_domain.cuda(), target_domain.cuda()

        # list_errD_src_real_c = []
        list_errD_src_real_s = []
        list_errD_src_fake_s = []
        list_errD_tgt_fake_s = []
        list_errG_c = []
        list_errG_s = []
        # list_errC = []
        # list_errF_fromC = []
        list_errF_src_fromD = []
        list_errF_tgt_fromD = []
    
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

            src_emb = netF(src_inputs)
            src_emb_cat = torch.cat((src_labels_onehot, src_emb), 1)
            print('F src_emb {}, src_emb_cat {}'.format(src_emb.shape, src_emb_cat.shape))
            src_gen = netG(src_emb_cat)
            print('src_gen {}'.format(src_gen.shape))

            tgt_emb = netF(tgt_inputs)
            tgt_emb_cat = torch.cat((tgt_labels_onehot, tgt_emb),1)
            tgt_gen = netG(src_emb_cat)

            # 源领域 初始图片的判别损失 -->  reallabel
            src_realoutputD_s, src_realoutputD_c, src_real_feature = netD(src_inputs)   
            # print('src_realoutputD_s {}'.format(src_realoutputD_s.shape))
            # print('src_realoutputD_c {}'.format(src_realoutputD_c.shape))
            # print('src_real_feature {}'.format(src_real_feature.shape))
            errD_src_real_dloss = criterion_s(src_realoutputD_s, reallabel)
            errD_src_real_closs = criterion_c(src_realoutputD_c, src_labels)

            # 目的领域 初始图片的判别损失 --> reallabel
            # diff3，增加了目的领域原始样本，GTA中没有这个分支
            if D_tgt_fake:
                tgt_realoutputD_s, tgt_realoutputD_c, tgt_real_feature = netD(tgt_inputs)   
                errD_tgt_real_dloss = criterion_s(tgt_realoutputD_s, reallabel)

            # 源领域 生成图片的判别损失 --> fakelabel
            # diff2，增加src_gen的分类损失
            src_fakeoutputD_s, src_fakeoutputD_c, src_fake_feature = netD(src_gen)
            # print('src_fakeoutputD_s {}'.format(src_fakeoutputD_s.shape))
            # print('src_fakeoutputD_c {}'.format(src_fakeoutputD_c.shape))
            # print('src_fake_feature {}'.format(src_fake_feature.shape))
            errD_src_fake_dloss = criterion_s(src_fakeoutputD_s, fakelabel)
            errD_src_fake_closs = criterion_c(src_fakeoutputD_c, src_labels)
            # print('D src_fake_feature {}'.format(src_fake_feature.shape))

            # 目的领域 生成图片的判别损失 --> fakelabel
            tgt_fakeoutputD_s, tgt_fakeoutputD_c, tgt_fake_feature = netD(tgt_gen)          
            errD_tgt_fake_dloss = criterion_s(tgt_fakeoutputD_s, fakelabel)

            errD_mmd = mmd_loss(src_fake_feature, tgt_fake_feature)
            
            errD = (errD_src_real_dloss + errD_src_fake_dloss + errD_tgt_fake_dloss) + errD_src_fake_closs + errD_src_real_closs + config['mmd_gamma'] * errD_mmd
            if D_tgt_fake:
                errD += errD_tgt_real_dloss
            if i % 10 == 0:
                print('  D: errD {:.2f}, [errD_src_real_dloss {:.2f}, errD_src_fake_dloss {:.2f}, errD_tgt_fake_dloss {:.2f}], errD_src_fake_closs {:.2f}, errD_src_real_closs {:.2f}, errD_mmd {:.2f}'.format(
                    errD.item(),
                    errD_src_real_dloss.item(),
                    errD_src_fake_dloss.item(),
                    errD_tgt_fake_dloss.item(),
                    errD_src_fake_closs.item(),
                    errD_src_real_closs.item(),
                    errD_mmd.item()
                ))
                logging.debug('  D: errD {:.2f}, [errD_src_real_dloss {:.2f}, errD_src_fake_dloss {:.2f}, errD_tgt_fake_dloss {:.2f}], errD_src_fake_closs {:.2f}, errD_src_real_closs {:.2f}, errD_mmd {:.2f}'.format(
                    errD.item(),
                    errD_src_real_dloss.item(),
                    errD_src_fake_dloss.item(),
                    errD_tgt_fake_dloss.item(),
                    errD_src_fake_closs.item(),
                    errD_src_real_closs.item(),
                    errD_mmd.item()
                ))
            errD.backward()    
            optimizerD.step()
            
            ###########################
            # Updating C network
            # netC的功能由netD替代
            ###########################
            # self.netC.zero_grad()
            # outC = self.netC(src_emb)   
            # errC = self.criterion_c(outC, src_labelsv)
            # # src_fakeoutputD_s, src_fakeoutputD_c, _ = self.netD(src_gen)
            # # errC_src_closs = self.criterion_c(src_fakeoutputD_c, src_labelsv)
            # # errG_src_dloss = self.criterion_s(src_fakeoutputD_s, reallabel)
            # # errC = errG_src_closs
            # if i == 0:
            #     print('C: errC {:.2f}'.format(errC.item()))
            # # errC.backward(retain_graph=True)    
            # errC.backward(retain_graph=True)    
            # self.optimizerC.step()


            ###########################
            # Updating G network
            ###########################
            netG.zero_grad()

            src_emb = netF(src_inputs)
            src_emb_cat = torch.cat((src_labels_onehot, src_emb), 1)
            src_gen = netG(src_emb_cat)

            tgt_emb = netF(tgt_inputs)
            tgt_emb_cat = torch.cat((tgt_labels_onehot, tgt_emb),1)
            tgt_gen = netG(src_emb_cat)

            # # # 源领域 生成图片的判别损失 --> reallabel
            # # #                分类损失
            src_fakeoutputD_s, src_fakeoutputD_c, src_fake_feature = netD(src_gen)
            errG_src_closs = criterion_c(src_fakeoutputD_c, src_labels)
            errG_src_dloss = criterion_s(src_fakeoutputD_s, reallabel)

            # # 目的领域 生成图片的判别损失 --> real
            if D_tgt_fake:
                tgt_fakeoutputD_s, _, tgt_fake_feature = netD(tgt_gen)
                errG_tgt_dloss = criterion_s(tgt_fakeoutputD_s, reallabel)

            # # src_gen / tgt_gen 的MMD
            errG_mmd = mmd_loss(src_fake_feature, tgt_fake_feature)

            errG = errG_src_closs + errG_src_dloss + config['mmd_gamma'] * errG_mmd
            if i % 10 ==0:
                print('  G: errG {:.2f}, [errG_src_closs {:.2f}, errG_src_dloss {:.2f}, errG_mmd {:.2f}]'.format(
                    errG.item(), errG_src_closs.item(), errG_src_dloss.item(), errG_mmd.item()))
                logging.debug('  G: errG {:.2f}, [errG_src_closs {:.2f}, errG_src_dloss {:.2f}, errG_mmd {:.2f}]'.format(
                    errG.item(), errG_src_closs.item(), errG_src_dloss.item(), errG_mmd.item()))
            errG.backward()
            optimizerG.step()
            

            ###########################
            # Updating F network
            ###########################
            netF.zero_grad()

            # errF_fromC = self.criterion_c(outC, src_labelsv)        
            #############################
            # 包括src_gen的分类损失、src_gen的判别损失、tgt_gen的判别损失
            # 增加：src_emd/tgt_emd的MMD
            #############################

            src_emb = netF(src_inputs)
            src_emb_cat = torch.cat((src_labels_onehot, src_emb), 1)
            src_gen = netG(src_emb_cat)

            tgt_emb = netF(tgt_inputs)
            tgt_emb_cat = torch.cat((tgt_labels_onehot, tgt_emb),1)
            tgt_gen = netG(src_emb_cat)

            # errF_fromC = self.criterion_c(outC, src_labelsv)        
            # diff1, 将源域样本的分类损失，放到源域生成样本上了

            # 源领域 生成图片的判别损失 --> reallabel
            src_fakeoutputD_s, src_fakeoutputD_c, src_fake_feature = netD(src_gen)
            errF_srcFake_closs = criterion_c(src_fakeoutputD_c, src_labels)
            errF_srcFake_dloss = criterion_s(src_fakeoutputD_s, reallabel)

            # 目的领域 生成图片的判别损失 --> reallabel
            tgt_fakeoutputD_s, tgt_fakeoutputD_c, tgt_fake_feature = netD(tgt_gen)
            errF_tgtFake_dloss = criterion_s(tgt_fakeoutputD_s, reallabel)

            # errF_mmd = self.mmd_loss(src_fake_feature, tgt_fake_feature)

            errF = errF_srcFake_dloss + errF_tgtFake_dloss + errF_srcFake_closs
            if i % 10 == 0:
                print('  F: errF {:.2f}, [errF_srcFake_dloss {:.2f}, errF_tgtFake_dloss {:.2f}], errF_srcFake_closs {:.2f}'.format(
                    errF.item(), errF_srcFake_dloss.item(), errF_tgtFake_dloss.item(),  errF_srcFake_closs.item()))
                logging.debug('  F: errF {:.2f}, [errF_srcFake_dloss {:.2f}, errF_tgtFake_dloss {:.2f}], errF_srcFake_closs {:.2f}'.format(
                    errF.item(), errF_srcFake_dloss.item(), errF_tgtFake_dloss.item(),  errF_srcFake_closs.item()))
            errF.backward()
            optimizerF.step()        
                    
            # # list_errD_src_real_c.append(errD_src_real_c.item())
            # list_errD_src_real_s.append(errD_src_real_s.item())
            # list_errD_src_fake_s.append(errD_src_fake_s.item())
            # list_errD_tgt_fake_s.append(errD_tgt_fake_s.item())
            # list_errG_c.append(errG_c.item())
            # list_errG_s.append(errG_s.item())
            # # list_errC.append(errC.item())
            # # list_errF_fromC.append(errF_fromC.item())
            # list_errF_src_fromD.append(errF_src_fromD.item())
            # list_errF_tgt_fromD.append(errF_tgt_fromD.item())

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
        netF.eval()
        netC.eval()
        total = 0
        correct = 0
    
        # Testing the model
        for i, datas in enumerate(config['target_test_loader']):
            inputs, labels = datas         
            inputs = inputs.cuda()
            with torch.no_grad():
                outC = netC(netF(inputs))
                _, predicted = torch.max(outC.data, 1)        
                total += labels.size(0)
                correct += ((predicted == labels.cuda()).sum())
            
        val_acc = 100*float(correct)/total
        print('Validate1 | Epoch: %d, Val Accuracy: %f %%' % (epoch, val_acc))
        logging.debug('%s| Epoch: %d, Val Accuracy: %f %%' % (datetime.datetime.now(), epoch, val_acc))


    def validate2(epoch):
        netD.eval()
        total = 0
        correct = 0
    
        # Testing the model
        for i, datas in enumerate(config['target_test_loader']):
            inputs, labels = datas         
            inputs = inputs.cuda()
            with torch.no_grad():
                tgt_s, tgt_c, _ = netD(inputs)
                _, predicted = torch.max(tgt_c, 1)        
                total += labels.size(0)
                correct += ((predicted == labels.cuda()).sum())
                
        val_acc = 100*float(correct)/total
        print('Validate2 | Epoch: %d, Test Accuracy: %f %%' % (epoch, val_acc))
        logging.debug('Validate2 | Epoch: %d, Test Accuracy: %f %%' % (epoch, val_acc))
    

    for epoch in range(1, config['n_epochs'] + 1):
        train(epoch)
        if epoch % config['TEST_INTERVAL'] == 0:
            # print('test on source_test_loader')
            # test(extractor, classifier, config['source_test_loader'], epoch)
            print('test on target_test_loader')
            validate(epoch)
            validate2(epoch)
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
