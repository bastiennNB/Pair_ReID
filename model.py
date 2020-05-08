# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:45:49 2020

@author: Nathan
"""

import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True):
        super(ClassBlock, self).__init__()
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)   
        x = self.classifier(x)
        return x
        
class TripletTop(nn.Module):
    def __init__(self,n_features_in = 2048, n_features_out=512):
        super(TripletTop, self).__init__()
        self.fc1 = nn.Linear(n_features_in,1024)
        self.bn = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(inplace = True)
        self.fc2 = nn.Linear(1024, n_features_out)
        self.apply(weights_init_kaiming)


    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, class_num):
        super(SimpleClassifier, self).__init__()

        classifier = []
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier
    def forward(self, x):
        x = self.classifier(x)
        return x

# Define the ResNet50-based Model
class ReID_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=1,simple=False, head = "classification"):
        super(ReID_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # reduce stride
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        self.backbone = model_ft
        
        if head == "classification":
            if simple:
                self.head = SimpleClassifier(2048, class_num)
            else:
                self.head = ClassBlock(2048, class_num, droprate)
        elif head == "clustering":
            self.head = TripletTop(n_features_in = 2048, n_features_out = 512)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.head(x)
        return x