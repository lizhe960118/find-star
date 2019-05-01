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
import choose_from_train

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Testing')
parser.add_argument('--load_model_epoch', default=12, help='the epoch of loading model in checkpoint')
args = parser.parse_args()

def load_test_data(transform):
    train_data_path = './data/af2019-cv-training-20190312'
    choose_from_train.choose(train_data_path)
    testset = ListDataset(root=train_data_path,
                      list_file='./data/find_star_txt_folder/find_star_test_bbx_gt.txt', 
                      train=False, 
                      transform=transform, 
                      input_size=600)
    testloader = torch.utils.data.DataLoader(testset, 
                                         batch_size=1,
                                         shuffle=False, 
                                         collate_fn=testset.collate_fn)
    return testloader

# Test == > has no test epoch
def test(epoch, transform, net, criterion, optimizer):
    with torch.no_grad():
        testloader = load_test_data(transform)
        print('\nTest')
        net.eval()
        test_loss = 0
        count_test_loss = [0 for i in range(15)] # [0.5<, 0.5-0.6, 0.6 -0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0,1.0-1.1, 1.2-1.3, 1.3-1.4, 1.4-1.5, 1.5-1.6, 1.6-1.7, 1.7-1.8, 1.8-1.9, 1.9 -]
        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
            inputs = Variable(inputs.cuda())
            loc_targets = Variable(loc_targets.cuda())
            cls_targets = Variable(cls_targets.cuda())

            loc_preds, cls_preds = net(inputs)
            loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            test_loss = test_loss + loss.data.item()
            
            if loss.data.item() < 0.5:
                count_test_loss[0] += 1
            elif loss.data.item() > 1.9:
                count_test_loss[-1] += 1
            else:
                count_test_loss[int((loss.data.item() - 0.6) * 10) + 1] += 1
                
            if (batch_idx+1) % 100 == 0:
                print('test_loss: %.3f | avg_loss: %.3f' % (loss.data.item(), test_loss/(batch_idx+1)))
        
        print(count_test_loss)
        
def main():
    print('==> chooseing data..')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
    # Model
    net = RetinaNet()
    criterion = FocalLoss()
#     optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    
    load_model_epoch = args.load_model_epoch
    checkpoint = torch.load('./checkpoint/{}_ckpt.pth'.format(load_model_epoch)) # max_epoch
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.cuda()
    net.load_state_dict(checkpoint['net'])
    
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']

    test(start_epoch,transform, net, criterion, optimizer)
    
if __name__ == '__main__':
    main()