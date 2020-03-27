import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

# import model.backbone as backbone



def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1, dim = 1) - F.softmax(out2, dim = 1)))



def vat(classifier, inputs, radius):
    eps = Variable(torch.randn(inputs.data.size()).cuda())
    eps_norm = 1e-6 *(eps/torch.norm(eps,dim=1,keepdim=True))
    eps = Variable(eps_norm.cuda(),requires_grad=True)
    outputs_classifier1 = classifier(inputs)
    outputs_classifier2 = classifier(inputs + eps)
    loss_p = discrepancy(outputs_classifier1, outputs_classifier2)
    loss_p.backward(retain_graph=True)

    eps_adv = eps.grad
    eps_adv = eps_adv/torch.norm(eps_adv)
    image_adv = inputs + radius * eps_adv

    return image_adv

def get_loss_vat(classifier, inputs, inputs_vat):
    outputs_classifier1 = classifier(inputs)
    outputs_classifier2 = classifier(inputs_vat)

    vat_loss = discrepancy(outputs_classifier1, outputs_classifier2)

    return vat_loss

def get_loss_entropy(classifier, inputs):
    outputs_classifier = classifier(inputs)

    output = F.softmax(outputs_classifier, dim=1)
    entropy_loss = - torch.mean(output * torch.log(output + 1e-6))

    return entropy_loss