'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms

import time

import os
import argparse

from models import *
#from utils import progress_bar
from torch.autograd import Variable


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
#use_cuda = False
print(use_cuda)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch = 128

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# Reduce the dataset size (only for debugging)
# trainset.train_data = trainset.train_data[0:8000]

#trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testset.test_data = testset.test_data[0:1000]

testloader = torch.utils.data.DataLoader(testset, batch_size=batch, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # net = VGG('VGG11')
    
    input_size = trainset.train_data[0].size
    net = VGG_rand('VGG11',input_size)
    
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        # For debugging 
#        print('train data size')
#        print(inputs.size())
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # print('Epoch %d: done ' % epoch)
    if epoch % 20 == 0 and epoch != 0:
        
        print('++++tr_iteration_%d: Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1), correct, total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    if epoch % 20 == 0 and epoch != 0:
        print('----ts_iteration_%d: Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return test_loss/(batch_idx+1), correct, total

start_time = time.time()
max_epoch = 500
tr_loss = np.zeros(max_epoch)
ts_loss = np.zeros(max_epoch)
tr_correct = np.zeros(max_epoch)
ts_correct = np.zeros(max_epoch)
tr_total = 0
ts_total = 0
lr = args.lr
for epoch in range(start_epoch, start_epoch+max_epoch):
    # print('epoch %d started' %(epoch))
    if epoch%100 == 0 and epoch!=0:
        lr = lr/4
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    tr_loss[epoch], tr_correct[epoch], tr_total = train(epoch)
    ts_loss[epoch], ts_correct[epoch], ts_total = test(epoch)

total_time = time.time() - start_time
print("--- %s seconds per epoch ---" % (total_time/max_epoch))

# Save checkpoint.
print('Saving..')
state = {
#    'net': net.module if use_cuda else net,
    'tr_loss': tr_loss,
    'ts_loss': ts_loss,
    'tr_correct': tr_correct,
    'ts_correct': ts_correct,
    'tr_total': tr_total,
    'ts_total': ts_total,
    'epoch_time': total_time/max_epoch,
}

if not os.path.isdir('results'):
    os.mkdir('results')
    
# change the name for VGG vs VGG_rand    
torch.save(state, './results/vgg_result_act_0.npy')
torch.save(net,'./results/vgg_net_act_0.npy')
# vgg_result_act1: without rand act
# vgg_result_act: with rand act
# vgg_result_act2: with rand act5
