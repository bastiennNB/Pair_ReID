# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:08:09 2020

@author: Nathan
"""

import torch
import torch.nn as nn
from functools import partial

#Metric Learning loss associated functions

def build_triplets(embeddings,labels,device):
    """
        Builds Batch Hard triplets
    """
    batch_size = len(labels)
    n_dim = embeddings.shape[1]
    pos = torch.FloatTensor(batch_size,n_dim).to(device)
    neg = torch.FloatTensor(batch_size,n_dim).to(device)
    for i,emb in enumerate(embeddings):
        pos_idx = (labels==labels[i]).nonzero().flatten()
        same_dist = torch.norm(embeddings[pos_idx,:]-emb,p=2,dim=1)
        pos[i,:] = embeddings[pos_idx[torch.argmax(same_dist)]]
        neg_idx =  torch.ones(len(labels),dtype=torch.bool)
        neg_idx[pos_idx] = False
        other_dist = torch.norm(embeddings[neg_idx,:]-emb,p=2,dim=1)
        neg[i,:] = embeddings[(torch.arange(batch_size)[neg_idx])[torch.argmin(other_dist)]]
    return embeddings, pos, neg

def TripletLoss(xt,xp,xq,soft_margin = False, m=1):
    """
        triplet loss based on Hinge loss with soft margin implementation
    """
    diff = torch.norm(xp-xt,p=2,dim=1) - torch.norm(xq-xt,p=2,dim=1)
    if soft_margin:
        f = torch.log(m + torch.exp(diff))
    else:
        f = m + diff
        f[f<0] = 0
    return torch.mean(f)

#Classification loss associated functions
    
def CrossEntropyLS(outputs, labels, label_smoothing=0.15):
    """ 
        Cross entropy that accepts soft targets
    """
    #Conversion to one-hot encoding
    one_hot_labels = torch.FloatTensor(outputs.shape).cuda()
    one_hot_labels.zero_()
    one_hot_labels.scatter_(1, labels.view(-1,1), 1)
    # Label Smoothing
    LogSoftmax = nn.LogSoftmax(dim=1)
    if label_smoothing!=0:
        n_classes = outputs.shape[1]
        one_hot_labels = torch.mul(1-label_smoothing,one_hot_labels)
        one_hot_labels = torch.add((label_smoothing/n_classes),one_hot_labels)
    return 2*(torch.mean(torch.sum(torch.mul(-one_hot_labels, LogSoftmax(outputs)), dim=1)))

#CL+ML loss

def TripletAndCrossEntropyLoss(outputs,anchors,pos,neg,labels,alpha = 1, beta = 2):
    triplet_l = partial(TripletLoss,soft_margin=True, m =1.0)
    LS_l = partial(CrossEntropyLS,label_smoothing=0.1)  
    return alpha*(LS_l(outputs,labels)) + beta*(triplet_l(anchors,pos,neg))
    