# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from torch.nn import init
from torch.optim import lr_scheduler


def get_scheduler(optimizer):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def one_hot_encode(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            tmp = (input_tensor == i) * torch.ones_like(input_tensor)
            tensor_list.append(tmp)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, target, one_hot):
        x_shape = list(target.shape)
        if (len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N * D, C, H, W]
            target = torch.transpose(target, 1, 2)
            target = torch.reshape(target, new_shape)

        if one_hot:
            target = self.one_hot_encode(target)
        assert inputs.shape == target.shape, 'size must match'
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            diceloss = dice_loss(inputs[:, i, :, :], target[:, i, :, :])
            class_wise_dice.append(diceloss)
            loss += diceloss
        return loss / self.n_classes


class UNetConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""
    def __init__(self,in_channels, out_channels, dropout_p):
        """
        dropout_p: probability to be zeroed
        """
        super(UNetConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
       
    def forward(self, x):
        return self.conv_conv(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, up_mode, dropout_p):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTransposed2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif up_mode=='upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_chans, out_chans, kernel_size=1),
            )
        self.conv_block = UNetConvBlock(in_chans, out_chans, dropout_p)

    def centre_crop(self, layer, target_size):
        _,_,layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.centre_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out


def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x


def dice_loss(predict,target):
    target = target.float()
    smooth = 1e-4
    intersect = torch.sum(predict*target)
    dice = (2 * intersect + smooth)/(torch.sum(target)+torch.sum(predict)+smooth)
    loss = 1.0 - dice

    return loss


class Dice_Loss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def one_hot_encode(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            tmp = (input_tensor == i) * torch.ones_like(input_tensor)
            tensor_list.append(tmp)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, target, one_hot):
        x_shape = list(target.shape)
        if (len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N * D, C, H, W]
            target = torch.transpose(target, 1, 2)
            target = torch.reshape(target, new_shape)

        if one_hot:
            target = self.one_hot_encode(target)
        assert inputs.shape == target.shape, 'size must match'
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            diceloss = dice_loss(inputs[:, i, :, :], target[:, i, :, :])
            class_wise_dice.append(diceloss)
            loss += diceloss
        return loss / self.n_classes


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_chns=1,  num_class_seg=10, num_class_recon=1, class_num=4, pre_train=False):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                'feature_chns': [32, 64, 128, 256, 512],
                'dropout_p': [0.0, 0.0, 0.0, 0.0, 0.0],
                'num_class_seg':num_class_seg,
                'num_class_recon': num_class_recon,
                'class_num': class_num,
                'bilinear': False,
                'acti_func': 'relu',
                'up_mode': 'upsample',
                'pretrain': pre_train
                  }
        self.pretrain = pre_train
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

        self.enc_opt = torch.optim.Adam(self.encoder.parameters(), lr=0.01, betas=(0.9, 0.999))
        self.dec_opt = torch.optim.Adam(self.decoder.parameters(), lr=0.01, betas=(0.5, 0.999))

        self.segloss = DiceLoss(class_num).to(torch.device('cuda'))
        self.enc_opt_sch = get_scheduler(self.enc_opt)
        self.dec_opt_sch = get_scheduler(self.dec_opt)

    def initialize(self):
        init_weights(self.encoder)
        init_weights(self.decoder)

    def update_lr(self):
        self.enc_opt_sch.step()
        self.dec_opt_sch.step()

    def forward(self, x):
        x_shape = list(x.shape)
        if(len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(x, 1, 2)
            x = torch.reshape(x, new_shape)
        blocks, bottleneck = self.encoder(x)
        if self.pretrain:
            recon_img, seg_logits = self.decoder(bottleneck, blocks)
            return bottleneck, recon_img, seg_logits
        else:
            self.seg = self.decoder(bottleneck, blocks)
            self.seg = self.seg.softmax(1)
            return self.seg

    def train_(self, images, labels):
        self.img = images
        self.lab = labels
        self.forward(self.img)
        #update encoder and decoder
        self.enc_opt.zero_grad()
        self.dec_opt.zero_grad()
        seg_loss = self.segloss(self.seg, self.lab, one_hot = True)
        seg_loss.backward()
        loss = seg_loss.item()
        self.enc_opt.step()
        self.dec_opt.step()

        return loss

    def test_(self, images):
        x_shape = list(images.shape)
        if(len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(images, 1, 2)
            x = torch.reshape(x, new_shape)
            images = x
        self.forward(images)
        output = self.seg

        return output

class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.in_chns   = params['in_chns']
        self.ft_chns   = params['feature_chns']
        self.n_class   = params['class_num']
        self.dropout   = params['dropout_p']
        self.down_path = nn.ModuleList()
        self.down_path.append(UNetConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0]))
        self.down_path.append(UNetConvBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[0]))
        self.down_path.append(UNetConvBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[0]))
        self.down_path.append(UNetConvBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[0]))
        self.down_path.append(UNetConvBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[0]))

    def forward(self, x):
        blocks=[]
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x ,2)
        return blocks, x

class Decoder(nn.Module):
    def __init__(self, params):
        ft_chns = params['feature_chns']
        n_class = params['class_num']
        n_class_seg = params['num_class_seg']
        n_class_recon = params['num_class_recon']
        dropout_p = params['dropout_p']
        up_mode = params['up_mode']

        super().__init__()
        self.pretrain = params['pretrain']
        self.up_path = nn.ModuleList()
        self.up_path.append(UNetUpBlock(ft_chns[4], ft_chns[3], up_mode, dropout_p[4]))
        self.up_path.append(UNetUpBlock(ft_chns[3], ft_chns[2], up_mode, dropout_p[3]))
        self.up_path.append(UNetUpBlock(ft_chns[2], ft_chns[1], up_mode, dropout_p[2]))
        self.up_path.append(UNetUpBlock(ft_chns[1], ft_chns[0], up_mode, dropout_p[1]))
        if self.pretrain:
            self.reconstruction_head = nn.Conv2d(ft_chns[0], n_class_recon, kernel_size=1)
            self.segmentation_head = nn.Conv2d(ft_chns[0], n_class_seg, kernel_size=1)
        else:
            self.last = nn.Conv2d(ft_chns[0], n_class, kernel_size=1)

    def forward(self, x, blocks):
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        if self.pretrain:
            recon_img = self.reconstruction_head(x)
            seg_logits = self.segmentation_head(x)
            return recon_img, seg_logits
        else:
            return self.last(x)










