import torch
import torch.nn as nn
from torch.autograd import Function, grad
import logging

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def gradient_penalty(critic, h_s, h_t):
    # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
    alpha = torch.rand(h_s.size(0), 1).cuda()
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def set_log_config(res_dir):
    logging.basicConfig(
            filename='{}/app.log'.format(res_dir),
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
    )


def test(extractor, classifier, data_loader, epoch):
    extractor.eval()
    classifier.eval()
    if torch.cuda.is_available():
        extractor = extractor.cuda()
        classifier = classifier.cuda()

    loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for (features, labels) in data_loader:
            if torch.cuda.is_available():
                features, labels = features.cuda(), labels.cuda()

            outputs = extractor(features)
            outputs = outputs.view(outputs.size(0), -1)
            preds, _ = classifier(outputs)
            loss += criterion(preds, labels).item()

            pred_cls = preds.data.max(1)[1]
            correct += pred_cls.eq(labels.data).cpu().sum().item()
            total += features.size(0)

    loss /= len(data_loader)
    accuracy = correct/total

    print("Epoch {}, {}/{}, Loss {}, Accuracy {:.2%}".format(epoch, correct, total, loss, accuracy))
    logging.debug("Epoch {}, {}/{}, Loss {}, Accuracy {:.2%}".format(epoch, correct, total, loss, accuracy))

    return accuracy
