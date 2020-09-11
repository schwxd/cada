import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.optim as optim
import torchvision.utils as vutils
import itertools, datetime
import numpy as np
import models
import utils
import matplotlib.pyplot as plt
import logging
import math

from visual_utils import draw_tsne_together
from losses import MMD_loss
from losses import get_loss_bnm


class GTA(object):

    def __init__(self, opt, nclasses, mean, std, source_trainloader, source_valloader, target_trainloader, target_valloader, res_dir):

        self.source_trainloader = source_trainloader
        self.source_valloader = source_valloader
        self.target_trainloader = target_trainloader
        self.target_valloader = target_valloader
        self.opt = opt
        self.best_val = 0
        
        # Defining networks and optimizers
        self.nclasses = nclasses
        self.netG = models._netG(opt, nclasses, flattens=opt.flattens)
        self.netD = models._netD(opt, nclasses)
        self.netF = models._netF(opt)
        self.netC = models._netC(opt, nclasses, flattens=opt.flattens)

        # Weight initialization
        self.netG.apply(utils.weights_init)
        self.netD.apply(utils.weights_init)
        self.netF.apply(utils.weights_init)
        self.netC.apply(utils.weights_init)

        logging.basicConfig(
                filename='{}/app.log'.format(res_dir),
                level=logging.DEBUG,
                format='%(asctime)s:%(levelname)s:%(message)s'
        )

        if True:
            print('netG<<')
            print(self.netG)
            logging.debug(self.netG)
            print('>>\n')
            print('netD<<')
            print(self.netD)
            logging.debug(self.netD)
            print('>>\n')
            print('netF<<')
            print(self.netF)
            logging.debug(self.netF)
            print('>>\n')
            print('netC<<')
            print(self.netC)
            logging.debug(self.netC)
            print('>>')

        # Defining loss criterions
        self.criterion_c = nn.CrossEntropyLoss()
        self.criterion_s = nn.BCELoss()

        self.mmd_loss = MMD_loss()
        self.mse_loss = nn.MSELoss()

        if opt.gpu>=0:
            self.netD.cuda()
            self.netG.cuda()
            self.netF.cuda()
            self.netC.cuda()
            self.criterion_c.cuda()
            self.criterion_s.cuda()

        # Defining optimizers
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerF = optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerC = optim.Adam(self.netC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        # Other variables
        self.real_label_val = 1
        self.fake_label_val = 0

    """
    Validation function
    """
    def validate(self, epoch):
        
        self.netF.eval()
        self.netC.eval()
        total = 0
        correct = 0
    
        # Testing the model
        for i, datas in enumerate(self.target_valloader):
            inputs, labels = datas         
            with torch.no_grad():
                inputv = Variable(inputs.cuda())

            outC = self.netC(self.netF(inputv))
            _, predicted = torch.max(outC.data, 1)        
            total += labels.size(0)
            correct += ((predicted == labels.cuda()).sum())
            
        val_acc = 100*float(correct)/total
        print('Validate1 | Epoch: %d, Val Accuracy: %f %%' % (epoch, val_acc))
        logging.debug('%s| Epoch: %d, Val Accuracy: %f %%' % (datetime.datetime.now(), epoch, val_acc))


    """
    Validation function
    """
    def validate2(self, epoch):
        
        self.netD.eval()
        total = 0
        correct = 0
    
        # Testing the model
        for i, datas in enumerate(self.target_valloader):
            inputs, labels = datas         
            with torch.no_grad():
                inputv = Variable(inputs.cuda())

            tgt_s, tgt_c, _ = self.netD(inputv)
            _, predicted = torch.max(tgt_c, 1)        
            total += labels.size(0)
            correct += ((predicted == labels.cuda()).sum())
            
        val_acc = 100*float(correct)/total
        print('Validate2 | Epoch: %d, Test Accuracy: %f %%' % (epoch, val_acc))
        logging.debug('Validate2 | Epoch: %d, Test Accuracy: %f %%' % (epoch, val_acc))
    
    """
    Validation function
    """
    def validate3(self, epoch):
        
        self.netF.eval()
        self.netG.eval()
        self.netD.eval()
        total = 0
        correct = 0
    
        # Testing the model
        for i, datas in enumerate(self.target_valloader):
            inputs, labels = datas         
            with torch.no_grad():
                inputv = Variable(inputs.cuda())

            labels_onehot = np.zeros((self.opt.batchSize, self.nclasses+1), dtype=np.float32)
            for num in range(self.opt.batchSize):
                labels_onehot[num, self.nclasses] = 1
            tgt_labels_onehot = torch.from_numpy(labels_onehot).cuda()

            tgt_emb = self.netF(inputv)
            tgt_emb_cat = torch.cat((tgt_labels_onehot, tgt_emb),1)
            tgt_gen = self.netG(tgt_emb_cat)

            tgt_s, tgt_c, _ = self.netD(tgt_gen)
            _, predicted = torch.max(tgt_c, 1)        
            total += labels.size(0)
            correct += ((predicted == labels.cuda()).sum())
            
        val_acc = 100*float(correct)/total
        print('Validate3 | Epoch: %d, Test Accuracy: %f %%' % (epoch, val_acc))
        logging.debug('Validate3 | Epoch: %d, Test Accuracy: %f %%' % (epoch, val_acc))


    """
    Validation function
    F提取特征 c分类
    """
    def validate4(self, epoch):
        
        self.netD.eval()
        total = 0
        correct = 0
    
        # Testing the model
        for i, datas in enumerate(self.target_valloader):
            inputs, labels = datas         
            with torch.no_grad():
                inputv = Variable(inputs.cuda())

            emb = self.netF(inputv)
            tgt_c = self.netD.forward_cls(emb)
            _, predicted = torch.max(tgt_c, 1)        
            total += labels.size(0)
            correct += ((predicted == labels.cuda()).sum())
            
        val_acc = 100*float(correct)/total
        print('Validate4 | Epoch: %d, Test Accuracy: %f %%' % (epoch, val_acc))
        logging.debug('Validate4 | Epoch: %d, Test Accuracy: %f %%' % (epoch, val_acc))


    """
    Train function
    """
    def train(self):
        
        curr_iter = 0
        
        reallabel = torch.FloatTensor(self.opt.batchSize).fill_(self.real_label_val)
        fakelabel = torch.FloatTensor(self.opt.batchSize).fill_(self.fake_label_val)
        if self.opt.gpu>=0:
            reallabel, fakelabel = reallabel.cuda(), fakelabel.cuda()

        source_domain = torch.FloatTensor(self.opt.batchSize).fill_(self.real_label_val)
        target_domain = torch.FloatTensor(self.opt.batchSize).fill_(self.fake_label_val)
        if self.opt.gpu>=0:
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

        for epoch in range(self.opt.nepochs):
            
            self.netG.train()    
            self.netF.train()    
            self.netC.train()    
            self.netD.train()    
        
            for i, (datas, datat) in enumerate(zip(self.source_trainloader, self.target_trainloader)):
                
                ###########################
                # Forming input variables
                ###########################
                
                src_inputs, src_labels = datas
                tgt_inputs, __ = datat       

                # Creating one hot vector
                labels_onehot = np.zeros((self.opt.batchSize, self.nclasses+1), dtype=np.float32)
                for num in range(self.opt.batchSize):
                    labels_onehot[num, src_labels[num]] = 1
                src_labels_onehot = torch.from_numpy(labels_onehot)

                labels_onehot = np.zeros((self.opt.batchSize, self.nclasses+1), dtype=np.float32)
                for num in range(self.opt.batchSize):
                    labels_onehot[num, self.nclasses] = 1
                tgt_labels_onehot = torch.from_numpy(labels_onehot)
                
                if self.opt.gpu>=0:
                    src_inputs, src_labels = src_inputs.cuda(), src_labels.cuda()
                    tgt_inputs = tgt_inputs.cuda()
                    src_labels_onehot = src_labels_onehot.cuda()
                    tgt_labels_onehot = tgt_labels_onehot.cuda()
                
                ###########################
                # Updating D network
                ###########################
                self.netD.zero_grad()

                src_emb = self.netF(src_inputs)
                src_emb_cat = torch.cat((src_labels_onehot, src_emb), 1)
                # print('F src_emb {}, src_emb_cat {}'.format(src_emb.shape, src_emb_cat.shape))
                src_gen = self.netG(src_emb_cat)
                # print('src_gen {}'.format(src_gen.shape))

                tgt_emb = self.netF(tgt_inputs)
                tgt_emb_cat = torch.cat((tgt_labels_onehot, tgt_emb),1)
                tgt_gen = self.netG(src_emb_cat)

                # 源领域 初始图片的判别损失 -->  reallabel
                src_realoutputD_s, src_realoutputD_c, src_real_feature = self.netD(src_inputs)   
                # print('src_realoutputD_s {}'.format(src_realoutputD_s.shape))
                # print('src_realoutputD_c {}'.format(src_realoutputD_c.shape))
                # print('src_real_feature {}'.format(src_real_feature.shape))
                errD_src_real_dloss = self.criterion_s(src_realoutputD_s, reallabel)
                errD_src_real_closs = self.criterion_c(src_realoutputD_c, src_labels)

                # 目的领域 初始图片的判别损失 --> reallabel
                # diff3，增加了目的领域原始样本，GTA中没有这个分支
                # tgt_realoutputD_s, tgt_realoutputD_c, tgt_real_feature = self.netD(tgt_inputs)   
                # errD_tgt_real_dloss = self.criterion_s(tgt_realoutputD_s, reallabel)

                # 源领域 生成图片的判别损失 --> fakelabel
                # diff2，增加src_gen的分类损失
                src_fakeoutputD_s, src_fakeoutputD_c, src_fake_feature = self.netD(src_gen)
                errD_src_fake_dloss = self.criterion_s(src_fakeoutputD_s, fakelabel)
                errD_src_fake_closs = self.criterion_c(src_fakeoutputD_c, src_labels)
                # print('D src_fake_feature {}'.format(src_fake_feature.shape))

                # 目的领域 生成图片的判别损失 --> fakelabel
                tgt_fakeoutputD_s, tgt_fakeoutputD_c, tgt_fake_feature = self.netD(tgt_gen)          
                errD_tgt_fake_dloss = self.criterion_s(tgt_fakeoutputD_s, fakelabel)

                errD_mmd = self.mmd_loss(src_fake_feature, tgt_fake_feature)
                
                errD = (errD_src_real_dloss + errD_src_fake_dloss + errD_tgt_fake_dloss) + errD_src_fake_closs + errD_src_real_closs + self.opt.mmd_weight * errD_mmd
                if i == 0:
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
                self.optimizerD.step()
                
                ###########################
                # Updating C network
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
                self.netG.zero_grad()

                src_emb = self.netF(src_inputs)
                src_emb_cat = torch.cat((src_labels_onehot, src_emb), 1)
                src_gen = self.netG(src_emb_cat)
                tgt_emb = self.netF(tgt_inputs)
                tgt_emb_cat = torch.cat((tgt_labels_onehot, tgt_emb),1)
                tgt_gen = self.netG(src_emb_cat)

                # # # 源领域 生成图片的判别损失 --> reallabel
                # # #                分类损失
                src_fakeoutputD_s, src_fakeoutputD_c, src_fake_feature = self.netD(src_gen)
                errG_src_closs = self.criterion_c(src_fakeoutputD_c, src_labels)
                errG_src_dloss = self.criterion_s(src_fakeoutputD_s, reallabel)

                # # 目的领域 生成图片的判别损失 --> real
                # tgt_fakeoutputD_s, _, tgt_fake_feature = self.netD(tgt_gen)
                # errG_tgt_dloss = self.criterion_s(tgt_fakeoutputD_s, reallabel)

                # # src_gen / tgt_gen 的MMD
                errG_mmd = self.mmd_loss(src_fake_feature, tgt_fake_feature)

                errG = errG_src_closs + errG_src_dloss + errG_mmd
                if i == 0:
                    print('  G: errG {:.2f}, [errG_src_closs {:.2f}, errG_src_dloss {:.2f}, errG_mmd {:.2f}]'.format(
                        errG.item(), errG_src_closs.item(), errG_src_dloss.item(), errG_mmd.item()))
                    logging.debug('  G: errG {:.2f}, [errG_src_closs {:.2f}, errG_src_dloss {:.2f}, errG_mmd {:.2f}]'.format(
                        errG.item(), errG_src_closs.item(), errG_src_dloss.item(), errG_mmd.item()))
                errG.backward()
                self.optimizerG.step()
                

                ###########################
                # Updating F network
                ###########################
                self.netF.zero_grad()

                # errF_fromC = self.criterion_c(outC, src_labelsv)        
                #############################
                # 包括src_gen的分类损失、src_gen的判别损失、tgt_gen的判别损失
                # 增加：src_emd/tgt_emd的MMD
                #############################

                src_emb = self.netF(src_inputs)
                src_emb_cat = torch.cat((src_labels_onehot, src_emb), 1)
                src_gen = self.netG(src_emb_cat)

                tgt_emb = self.netF(tgt_inputs)
                tgt_emb_cat = torch.cat((tgt_labels_onehot, tgt_emb),1)
                tgt_gen = self.netG(src_emb_cat)

                # errF_fromC = self.criterion_c(outC, src_labelsv)        
                # diff1, 将源域样本的分类损失，放到源域生成样本上了

                # 源领域 生成图片的判别损失 --> reallabel
                src_fakeoutputD_s, src_fakeoutputD_c, src_fake_feature = self.netD(src_gen)
                errF_srcFake_closs = self.criterion_c(src_fakeoutputD_c, src_labels)*(self.opt.adv_weight)
                errF_srcFake_dloss = self.criterion_s(src_fakeoutputD_s, reallabel)*(self.opt.adv_weight*self.opt.alpha)

                # 目的领域 生成图片的判别损失 --> reallabel
                tgt_fakeoutputD_s, tgt_fakeoutputD_c, tgt_fake_feature = self.netD(tgt_gen)
                errF_tgtFake_dloss = self.criterion_s(tgt_fakeoutputD_s, reallabel)*(self.opt.adv_weight*self.opt.alpha)

                # errF_mmd = self.mmd_loss(src_fake_feature, tgt_fake_feature)

                errF = errF_srcFake_dloss + errF_tgtFake_dloss + errF_srcFake_closs
                if i == 0:
                    print('  F: errF {:.2f}, [errF_srcFake_dloss {:.2f}, errF_tgtFake_dloss {:.2f}], errF_srcFake_closs {:.2f}'.format(
                        errF.item(), errF_srcFake_dloss.item(), errF_tgtFake_dloss.item(),  errF_srcFake_closs.item()))
                    logging.debug('  F: errF {:.2f}, [errF_srcFake_dloss {:.2f}, errF_tgtFake_dloss {:.2f}], errF_srcFake_closs {:.2f}'.format(
                        errF.item(), errF_srcFake_dloss.item(), errF_tgtFake_dloss.item(),  errF_srcFake_closs.item()))
                errF.backward()
                self.optimizerF.step()        

                curr_iter += 1
                        
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
                if self.opt.lrd:
                    self.optimizerD = utils.exp_lr_scheduler(self.optimizerD, epoch, self.opt.lr, self.opt.lrd, curr_iter)    
                    self.optimizerF = utils.exp_lr_scheduler(self.optimizerF, epoch, self.opt.lr, self.opt.lrd, curr_iter)
                    self.optimizerC = utils.exp_lr_scheduler(self.optimizerC, epoch, self.opt.lr, self.opt.lrd, curr_iter)                  
                    # optimizerG要不要梯度递减？原始实现没有
                    self.optimizerG = utils.exp_lr_scheduler(self.optimizerG, epoch, self.opt.lr, self.opt.lrd, curr_iter)                  
            
            # Validate every epoch
            self.validate(epoch+1)
            self.validate2(epoch+1)
            # self.validate4(epoch+1)
            # self.validate3(epoch+1)
        
        # iters = range(len(list_errD_src_real_s))
        # plt.figure()
        # # plt.plot(iters, list_errD_src_real_c, 'r', label='errD_src_real_c')
        # plt.plot(iters, list_errD_src_real_s, 'g', label='errD_src_real_s')
        # plt.plot(iters, list_errD_src_fake_s, 'b', label='errD_src_fake_s')
        # plt.plot(iters, list_errD_tgt_fake_s, 'k', label='errD_tgt_fake_s')
        # plt.plot(iters, list_errG_c, 'c', label='errG_c')
        # plt.plot(iters, list_errG_s, 'y', label='errG_s')
        # # plt.plot(iters, list_errC, 'Indigo', label='errC')
        # # plt.plot(iters, list_errF_fromC, 'm', label='errF_fromC')
        # plt.plot(iters, list_errF_src_fromD, 'deepskyblue', label='errF_src_fromD')
        # plt.plot(iters, list_errF_tgt_fromD, 'orange', label='errF_tgt_fromD')
        

        # plt.grid(True)
        # plt.xlabel('epochs')
        # plt.ylabel('fscore')
        # plt.legend(loc="upper left")
        # plt.savefig("{}/fscore.png".format(self.opt.outf), dpi=300)
        # plt.close()

