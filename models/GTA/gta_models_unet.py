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


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((0, 1, 0, 0)),
            nn.Conv1d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    # def forward(self, x):
    #     # U-Net generator with skip connections from encoder to decoder
    #     d1 = self.down1(x)
    #     d2 = self.down2(d1)
    #     d3 = self.down3(d2)
    #     d4 = self.down4(d3)
    #     d5 = self.down5(d4)
    #     d6 = self.down6(d5)
    #     d7 = self.down7(d6)
    #     d8 = self.down8(d7)
    #     u1 = self.up1(d8, d7)
    #     u2 = self.up2(u1, d6)
    #     u3 = self.up3(u2, d5)
    #     u4 = self.up4(u3, d4)
    #     u5 = self.up5(u4, d3)
    #     u6 = self.up6(u5, d2)
    #     u7 = self.up7(u6, d1)

    #     return self.final(u7)

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        # d6 = self.down6(d5)
        # d7 = self.down7(d6)
        # d8 = self.down8(d7)

        u4 = self.up1(d5, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)

        # u1 = self.up1(d8, d7)
        # u2 = self.up2(u1, d6)
        # u3 = self.up3(u2, d5)
        # u4 = self.up4(u3, d4)
        # u5 = self.up5(u4, d3)
        # u6 = self.up6(u5, d2)
        # u7 = self.up7(u6, d1)

        return self.final(u7)


class Discriminator(nn.Module):
    def __init__(self, flattens, in_channels=3, nclasses=3):
        super(Discriminator, self).__init__()
        self.nclasses = nclasses
        self.flattens = flattens

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv1d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm1d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 1, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv1d(512, 1, 4, padding=1, bias=False)
        )

        self.classifier_c = nn.Sequential(nn.Linear(self.flattens, self.nclasses))              
        self.classifier_s = nn.Sequential(
        						nn.Linear(self.flattens, 1), 
        						nn.Sigmoid())  

    # def forward(self, img_A, img_B):
    #     # Concatenate image and condition image by channels to produce input
    #     img_input = torch.cat((img_A, img_B), 1)
    #     return self.model(img_input)

    def forward(self, x):
        output = self.model(x)
        output = output.view(output.size(0), -1)
        output_s = self.classifier_s(output)
        # output_s = output_s.view(-1)
        output_c = self.classifier_c(output)
        return output_s, output_c, output

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
        self.main = nn.Sequential(
            nn.ConvTranspose1d(self.nz+self.flattens+self.nclasses+1, channels, 4, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),

            nn.ConvTranspose1d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),

            nn.ConvTranspose1d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),

            nn.ConvTranspose1d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),

            nn.ConvTranspose1d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            # nn.ConvTranspose1d(channels, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):   
        batchSize = input.size()[0]
        input = input.view(-1, self.flattens+self.nclasses+1, 1)
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
        
        channels = 64
        self.flattens = flattens
        self.feature = nn.Sequential(
            nn.Conv1d(1, channels, 32, 2, 0),            
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
        
        channels = 64
        self.feature = nn.Sequential(
            nn.Conv1d(1, 8, 32, 2, 0),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            # nn.MaxPool1d(2, 2),
            
            nn.Conv1d(8, 16, 16, 2, 0),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool1d(2, 2),

            nn.Conv1d(16, 32, 8, 2, 0),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            # nn.MaxPool1d(2, 2),

            nn.Conv1d(32, 32, 3, 2, 0),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            # nn.MaxPool1d(2, 2),

            nn.Conv1d(32, 32, 3, 2, 0),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):   
        output = self.feature(input)
        # print('output size {}'.format(output.size()))
        return output.view(output.size(0), -1)

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

