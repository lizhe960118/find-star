'''Init RestinaNet50 with pretrained ResNet50 model.

Download pretrained ResNet50 params from:
  https://download.pytorch.org/models/resnet50-19c8e357.pth
'''
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import sys
sys.path.append("..")

from fpn import FPN50
from retinanet import RetinaNet


print('Loading pretrained ResNet50 model..')
d = torch.load('./../model/resnet50.pth')

print('Loading into FPN50..')
fpn = FPN50()
dd = fpn.state_dict()
for k in d.keys():
#     if not k.startswith('fc'):  # skip fc layers
    if k.startswith('layer1') or k.startswith('layer2') or k.startswith('layer3'): # only layers1 to 3
        dd[k] = d[k]
  

print('Saving RetinaNet..')
net = RetinaNet()
for m in net.modules():
    if isinstance(m, nn.Conv2d):
        init.normal_(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

pi = 0.01
init.constant_(net.cls_head[-1].bias, -math.log((1-pi)/pi))

net.fpn.load_state_dict(dd)
torch.save(net.state_dict(), 'net.pth')
print('Done!')
