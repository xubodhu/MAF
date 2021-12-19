import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class myResnet(nn.Module):
    def __init__(self, resnet, if_fine_tune, device):
        super(myResnet, self).__init__()
        self.resnet = resnet
        self.if_fine_tune = if_fine_tune
        self.device = device

    def forward(self, x, att_size=7):
        # x shape batch_size * channels * 224 * 224
        # 32 * 3 * 224 * 224

        # batch_size * channels * 112 * 112
        # 32 * 64 * 112 * 112
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        # 32 * 256 * 56 * 56
        x = self.resnet.maxpool(x)

        # 32 * 512 * 56 * 56
        x = self.resnet.layer1(x)
        # 32 * 512 * 28 * 28
        x = self.resnet.layer2(x)
        # 32 * 1024 * 14 * 14
        x = self.resnet.layer3(x)
        # 32 * 2048 * 7 * 7
        x = self.resnet.layer4(x)

        # 32 * 2048
        fc = x.mean(3).mean(2)
        # 32 * 2048 * 7 * 7
        att = F.adaptive_avg_pool2d(x,[att_size,att_size])

        # 32 * 2048 * 1 * 1
        x = self.resnet.avgpool(x)
        # 32 * 2048 * 2048
        x = x.view(x.size(0), -1)

        if not self.if_fine_tune:
            
            x= Variable(x.data)
            fc = Variable(fc.data)
            att = Variable(att.data)

        return x, fc, att


