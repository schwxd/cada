import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CDAN(input_list, ad_net, gamma=1, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        # print('softmax_output.size(1): {}'.format(softmax_output.size(1)))
        # print('feature.size(1): {}'.format(feature.size(1)))
        # print('op_out {}'.format(op_out.size()))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)), gamma)
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)), gamma)       
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target) 

def DANN(features, ad_net, gamma=1):
    ad_out = ad_net(features, gamma)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)


from models.CDAN_VAT.consistency_losses import KLDivLossWithLogits
from models.CDAN_VAT.utils import disable_tracking_bn_stats


def VAT(vat, target_inputs, feature_extractor, classifier, target_consistency_criterion):
    # VAT
    vat_adv, clean_vat_logits = vat(target_inputs)
    vat_adv_inputs = target_inputs + vat_adv

    with disable_tracking_bn_stats(feature_extractor):
        with disable_tracking_bn_stats(classifier):
            adv_vat_features = feature_extractor(vat_adv_inputs)
            adv_vat_logits = classifier(adv_vat_features)

    target_vat_loss = target_consistency_criterion(adv_vat_logits, clean_vat_logits)

    return target_vat_loss
    # target_vat_loss.backward()
    # feature_extractor_grads = self.feature_extractor.module.stash_grad(feature_extractor_grads)
    # classifier_grads = self.classifier.module.stash_grad(feature_extractor_grads)