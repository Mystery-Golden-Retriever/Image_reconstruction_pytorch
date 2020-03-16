import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnCoder(nn.Module):
    
    def __init__(self):
        super(EnCoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (7,7), padding = 3)
        self.pool1 = nn.MaxPool2d((2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(64, 32, (5,5), padding = 2)
        self.pool2 = nn.MaxPool2d((2,2), stride=(2,2))
        self.conv3 = nn.Conv2d(32, 16, (1,1), padding = 0)
        self.conv4 = nn.Conv2d(16, 8, (3,3), padding = 1)
        self.conv5 = nn.Conv2d(8, 4, (3,3), padding = 1)
    
    def forward(self, input):
        #input RGB channel
        x = input
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x
    

class DeCoder(nn.Module):
    
    def __init__(self):
        super(DeCoder, self).__init__()
        self.inv_conv5 = nn.ConvTranspose2d(4, 8, (3,3), stride=1, padding=1)
        self.inv_conv4 = nn.ConvTranspose2d(8, 16, (3,3), stride=1, padding=1)
        self.inv_conv3 = nn.ConvTranspose2d(16, 32, (1,1), stride=1, padding=0)
        self.inv_pool2 = lambda x: F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        # self.inv_pool2 = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True)
        # or can just use deconv with stride=2
        # self.inv_conv3 = nn.ConvTranspose2d(16, 32, (1,1), stride=3, padding=0)
        self.inv_conv2 = nn.ConvTranspose2d(32, 64, (5,5), stride=1, padding=2)
        self.inv_pool1 = lambda x: F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        # self.inv_conv2 = nn.ConvTranspose2d(32, 64, (5,5), stride=2, padding=2)
        self.inv_conv1 = nn.ConvTranspose2d(64, 3, (7,7), stride=1, padding=3)

    def forward(self, input):
        x = self.inv_conv5(input)
        x = self.inv_conv4(x)
        x = self.inv_conv3(x)
        x = self.inv_pool2(x)
        x = self.inv_conv2(x)
        x = self.inv_pool1(x)
        x = self.inv_conv1(x)
        return x


class Compressor(nn.Module):

    # def __init__(self):
    #     super(Compressor, self).__init__()
    #     self.encoder = EnCoder()
    #     self.decoder = DeCoder()
    
    # def forward(self, input):
    #     out_en = self.encoder(input)
    #     out_de = self.decoder(out_en)
    #     return out_de

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, (7,7), padding = 3)
        self.pool1 = nn.MaxPool2d((2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(64, 32, (5,5), padding = 2)
        self.pool2 = nn.MaxPool2d((2,2), stride=(2,2))
        self.conv3 = nn.Conv2d(32, 16, (1,1), padding = 0)
        self.conv4 = nn.Conv2d(16, 8, (3,3), padding = 1)
        self.conv5 = nn.Conv2d(8, 4, (3,3), padding = 1)

        self.inv_conv5 = nn.ConvTranspose2d(4, 8, (3,3), stride=1, padding=1)
        self.inv_conv4 = nn.ConvTranspose2d(8, 16, (3,3), stride=1, padding=1)
        self.inv_conv3 = nn.ConvTranspose2d(16, 32, (1,1), stride=1, padding=0)
        self.inv_pool2 = lambda x: F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        # self.inv_pool2 = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True)
        # or can just use deconv with stride=2
        # self.inv_conv3 = nn.ConvTranspose2d(16, 32, (1,1), stride=3, padding=0)
        self.inv_conv2 = nn.ConvTranspose2d(32, 64, (5,5), stride=1, padding=2)
        self.inv_pool1 = lambda x: F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        self.inv_conv1 = nn.ConvTranspose2d(64, 3, (7,7), stride=1, padding=3)
    
    def forward(self, input):
        x = input
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        enco = self.conv5(x)

        x = self.inv_conv5(enco)
        x = self.inv_conv4(x)
        x = self.inv_conv3(x)
        x = self.inv_pool2(x)
        x = self.inv_conv2(x)
        x = self.inv_pool1(x)
        deco = self.inv_conv1(x)
        return deco
        





if __name__ == '__main__':
    input_decoder = torch.randn(1,4,2,2)
    input_encoder = torch.randn(1,3,8,8)
    encoder = EnCoder()
    decoder = DeCoder()
    out_decoder = decoder(input_decoder)
    out_encoder = encoder(input_encoder)
    print('shape of out_decoder: {}'.format(str(out_decoder.shape)))
    print('shape of out_encoder: {}'.format(str(out_encoder.shape)))
    
    compressor = Compressor()
    out_comp = compressor(input_encoder)
    print('shape of out_comp: {}'.format(str(out_comp.shape)))  