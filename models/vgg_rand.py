#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 12:24:49 2018

@author: Abhinav
"""
 
'''VGG11/13/16/19 in Pytorch with random activations '''
import torch
import torch.nn as nn
from torch.autograd import Variable


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
    
class rand_activation(nn.Module):
    def __init__(self, out_size, input_size, seed):
        super(rand_activation, self).__init__()
        torch.manual_seed(seed)
        self.R = torch.randn(out_size,input_size)       
        if torch.cuda.is_available():
            self.R = self.R.cuda()
            
        self.input_size = input_size
#        print('Size of R',self.R.size())
        
    def forward(self, x, input_data):
#        print('\nInside rand_activation:\n')
        batch_size = input_data.numel()/self.input_size 
#        print(batch_size,list(self.R.shape)[0],list(self.R.shape)[1])
        
        R_exp = self.R.expand(batch_size,-1,-1)
        
#        print('Shape of appended R: ', R_expshape)
#        print('Shape of input_data: ', input_data.data.shape)
#        print('Input size: ', self.input_size)
        
        # Batch multiply the random matrix over the entire mini-batch
        S = torch.bmm(R_exp,input_data.data.view(-1,self.input_size).unsqueeze(2))
        S.squeeze_(2)
        
#        S = torch.clamp(S,min=0)
        S = (S>0).float()
#        print('Size of S: ', S.shape)
#        print('Shape of x (features): ', x.shape)
        
        S = S.expand(list(x.shape)[2],list(x.shape)[3],-1,-1).permute(2,3,0,1)        
        
#        print('Shape of expanded S: ', S.shape)
        
        return Variable(S*x.data, requires_grad=True)

class VGG_rand(nn.Module):
    def __init__(self, vgg_name, input_size):
        super(VGG_rand, self).__init__()
        self.vgg_type = cfg[vgg_name]
        in_channels = 3
        
        # Layer 1
        seed = 1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation1 = rand_activation(64, input_size, seed) 
        in_channels = 64
        
        # Layer 2
        self.MaxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 3
        seed = 3
        self.conv3 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.activation3 = rand_activation(128, input_size, seed) 
        in_channels = 128
        
        # Layer 4
        self.MaxPool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 5
        seed = 5
        self.conv5 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.activation5 = rand_activation(256, input_size, seed) 
        in_channels = 256
        
        # Layer 6
        seed = 6
        self.conv6 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.activation6 = rand_activation(256, input_size, seed) 
        in_channels = 256
        
        # Layer 7
        self.MaxPool7 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 8
        seed = 8
        self.conv8 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.activation8 = rand_activation(512, input_size, seed) 
        in_channels = 512
        
        # Layer 9
        seed = 9
        self.conv9 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.activation9 = rand_activation(512, input_size, seed) 
        in_channels = 512
        
        # Layer 10
        self.MaxPool10 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 11
        seed = 11
        self.conv11 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.activation11 = rand_activation(512, input_size, seed) 
        in_channels = 512
        
        # Layer 12
        seed = 12
        self.conv12 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.activation12 = rand_activation(512, input_size, seed) 
        in_channels = 512
        
        # Layer 13
        self.MaxPool13 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Last Layer
        self.AvgPool2d = nn.AvgPool2d(kernel_size=1, stride=1)
        
#        self.classifier = nn.Linear(32768, 10)
        self.classifier = nn.Linear(512, 10)
        
        self.ReLU11 = nn.ReLU(inplace=True)
        self.ReLU12 = nn.ReLU(inplace=True)
        
    def forward(self, x):        
        out = x
#        print('Inside the forward pass')
        # Layer 1
        out = self.conv1(out)
        out = self.bn1(out)        
        out = self.activation1(out,x)
#        print('Layer 1 done')
        
        # Layer 2 
        out = self.MaxPool2(out)
#        print('Layer 2 done')
        
        # Layer 3
        out = self.conv3(out)
        out = self.bn3(out)        
        out = self.activation3(out,x)
#        print('Layer 3 done')
        
        # Layer 4
        out = self.MaxPool4(out)
#        print('Layer 4 done')
        
        # Layer 5
        out = self.conv5(out)
        out = self.bn5(out)        
        out = self.activation5(out,x)
#        print('Layer 5 done')
        
        # Layer 6
        out = self.conv6(out)
        out = self.bn6(out)        
        out = self.activation6(out,x)
#        print('Layer 6 done')
        
        # Layer 7
        out = self.MaxPool7(out)
#        print('Layer 7 done')
        
        # Layer 8
        out = self.conv8(out)
        out = self.bn8(out)        
        out = self.activation8(out,x)
#        print('Layer 8 done')
        
        # Layer 9
        out = self.conv9(out)
        out = self.bn9(out)        
        out = self.activation9(out,x)
#        print('Layer 9 done')
        
        # Layer 10
        out = self.MaxPool10(out)
#        print('Layer 10 done')
        
        # Layer 11
        out = self.conv11(out)
        out = self.bn11(out)        
        out = self.ReLU11(out)
#        print('Layer 11 done')
        
        # Layer 12
        out = self.conv12(out)
        out = self.bn12(out)        
        out = self.ReLU12(out)
#        print('Layer 12 done')
        
        # Layer 13 
        out = self.MaxPool13(out)
#        print('Layer 13 done')
        
        # Last Layer        
        out = self.AvgPool2d(out)
#        print('Layer 14 done')
        
        out = out.view(out.size(0), -1)        
        out = self.classifier(out)
#        print('passed through the net')        
        return out

