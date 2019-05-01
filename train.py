from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from loss import FocalLoss
from retinanet import RetinaNet
from datagen import ListDataset

from torch.autograd import Variable
import csv2train_txt
import csv2test_txt


parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--train_dataset', default= './data/af2019-cv-training-20190312',help="the path of train data")
# parser.add_argument('--test_dataset', default= './data/af2019-cv-testA-20190318',help="the path of test data")
parser.add_argument('--train_epoch', default=14, type=int, help='the epoch to train the net')
parser.add_argument('--model', default='checkpoint', help='the path to save model')
args = parser.parse_args()



def load_train_data(transform):
    csv2train_txt.convert_csv2train_txt(args.train_dataset)
    trainset = ListDataset(root=args.train_dataset,
                           list_file='./data/find_star_txt_folder/find_star_train_bbx_gt.txt',
                           train=True,
                           transform=transform, 
                           input_size=600)
    trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=2,
            shuffle=True,
            collate_fn=trainset.collate_fn)
    return trainloader

# Training
def train(epoch, transform, net, optimizer, criterion):
    trainloader = load_train_data(transform)
    print('\nEpoch: %d' % epoch)
    net.train()
#     net.module.freeze_bn()
    train_loss = 0
    print('\nall images:', len(trainloader))
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss = train_loss + loss.data.item()
        if (batch_idx+1) % 100 == 0:
            print('batch_idx: %d |train_loss: %.3f | avg_loss: %.3f' % (batch_idx+1, loss.data.item(), train_loss/(batch_idx+1)))

def save_model(epoch, save_model_path, net, optimizer):
    state = {
        'net': net.state_dict(),
        'optimizer':optimizer.state_dict(),
        'epoch':epoch,
    }
    if not os.path.isdir(save_model_path):
        os.mkdir(save_model_path)
    torch.save(state, './{}/ckpt.pth'.format(save_model_path))

def main():
    # assert torch.cuda.is_available(), 'Error: CUDA not found!'
    best_loss = float('inf')  # best test loss
    start_epoch = 0  # start from epoch 0 or last epoch
    save_model_path = args.model

    # Data
    print('==> Preparing data..')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
    
    # Model
    net = RetinaNet()
    net.load_state_dict(torch.load('./model/net.pth'))
    
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.cuda()
    
    criterion = FocalLoss()
#     optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    
    for epoch in range(start_epoch, start_epoch + args.train_epoch):
        train(epoch + 1, transform, net, optimizer, criterion)
        save_model(epoch + 1, save_model_path, net, optimizer)

if __name__ == '__main__':
    main()

