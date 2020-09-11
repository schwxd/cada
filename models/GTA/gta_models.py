import torch
import torch.nn as nn
from torch.autograd import Variable


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv1d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm1d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose1d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm1d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


"""
Generator network
"""
class _netG(nn.Module):
    def __init__(self, nz, nclasses, flattens):
        super(_netG, self).__init__()

        self.nz = nz
        self.nclasses = nclasses
        self.flattens = flattens
        channels = 16
        self.channels = 16

        # self.aligns = nn.Sequential(
        #     nn.Linear()
        # )

        self.main = nn.Sequential(
            nn.ConvTranspose1d(self.nz+self.flattens+self.nclasses+1, channels, 3, 2, 0, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),

            nn.ConvTranspose1d(channels, channels, 3, 2, 0, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),

            nn.ConvTranspose1d(channels, channels, 8, 2, 0, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),

            nn.ConvTranspose1d(channels, channels, 16, 2, 0, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),

            nn.ConvTranspose1d(channels, channels, 32, 2, 0, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.ConvTranspose1d(channels, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):   
        batchSize = input.size()[0]
        # input = input.view(-1, self.channels+self.nclasses+1, 1)
        print('netG input {}'.format(input.shape))
        input = input.view(batchSize, self.flattens+self.nclasses+1, -1)
        print('netG input reshape to {}'.format(input.shape))
        # print('netG batchsize {}, self.nz {}'.format(batchSize, self.nz))
        noise = torch.FloatTensor(batchSize, self.nz, 1).normal_(0, 1)    
        noise = noise.cuda()
        inputs = torch.cat((input, noise),1)
        output = self.main(inputs)
        # print('netG output {}'.format(output.shape))
        output = output.view(output.size(0), 1, -1)
        # print('netG output {}'.format(output.shape))
        return output


"""
Discriminator network
"""
class _netD(nn.Module):
    def __init__(self, flattens, nclasses):
        super(_netD, self).__init__()
        
        channels = 16
        self.flattens = flattens
        self.feature = nn.Sequential(
            nn.Conv1d(1, channels, 32, 2, 0),            
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool1d(2,2),
            nn.Dropout(0.5),

            nn.Conv1d(channels, channels, 16, 2, 0),         
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool1d(2,2),
            nn.Dropout(0.5),

            nn.Conv1d(channels, channels, 8, 2, 0),           
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool1d(2,2),
            nn.Dropout(0.5),
            
            nn.Conv1d(channels, channels, 3, 2, 0),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool1d(2,2),
            nn.Dropout(0.5),

            nn.Conv1d(channels, channels, 3, 2, 0),           
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(inplace=True)
            # nn.MaxPool1d(2,2),
            # nn.Dropout(0.5),

            # nn.Conv1d(channels*2, channels, 3, 2, 1),           
            # nn.BatchNorm1d(channels),
            # nn.LeakyReLU(inplace=True),
            # nn.MaxPool1d(4,4)           
        )

        self.classifier_c = nn.Sequential(nn.Linear(self.flattens, nclasses))              
        self.classifier_s = nn.Sequential(
        						nn.Linear(self.flattens, 1), 
        						nn.Sigmoid())        

    def forward(self, input):       
        output = self.feature(input)
        # print('netD input {}'.format(input.size()))
        # print('netD output {}'.format(output.size()))
        output = output.view(output.size(0), -1)
        # print('netD output {}'.format(output.size()))
        output_s = self.classifier_s(output)
        output_s = output_s.view(-1)
        output_c = self.classifier_c(output)
        return output_s, output_c, output
    
    def forward_cls(self, input):
        input = input.view(input.size(0), -1)
        return self.classifier_c(input)

"""
Feature extraction network
"""
class _netF(nn.Module):
    def __init__(self):
        super(_netF, self).__init__()
        
        channels = 16
        self.feature = nn.Sequential(
            nn.Conv1d(1, channels, 32, 2, 0),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            # nn.MaxPool1d(2, 2),
            
            nn.Conv1d(channels, channels, 16, 2, 0),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            # nn.MaxPool1d(2, 2),

            nn.Conv1d(channels, channels, 8, 2, 0),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            # nn.MaxPool1d(2, 2),

            nn.Conv1d(channels, channels, 3, 2, 0),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            # nn.MaxPool1d(2, 2),

            nn.Conv1d(channels, channels, 3, 2, 0),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):   
        output = self.feature(input)
        print('_netF output size {}'.format(output.size()))
        return output.view(output.size(0), -1)
        # return output

"""
Classifier network
"""
class _netC(nn.Module):
    def __init__(self, nclasses, flattens):
        super(_netC, self).__init__()
        self.main = nn.Sequential(          
            nn.Linear(flattens, flattens),
            nn.ReLU(inplace=True),
            nn.Linear(flattens, nclasses),                         
        )

    def forward(self, input):       
        output = self.main(input)
        return output

