# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:19:33 2020

@author: Nathan Bastien - EPL (31171500) - master thesis "Comparative analysis of re-ID models for matching pairs of Identities"
@file_goal: trains clustering based model
@Needs: - triplet_train.py : definition of the loss, training function, Batch Hard Sampler
        - train_classification.py: training helpers
"""
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import pickle
from functools import partial
import matplotlib
import os
import argparse

import config
from train_classification import get_trainable_layers, set_parameter_requires_grad
from triplet_train import triplet_train, triplet_loss, BatchSampler
from model import ReID_net

class TripletDataset(datasets.folder.ImageFolder):   
    def __init__(self,root, transform=None):
        super(TripletDataset, self).__init__(root,transform)
        self.lbs = [x[1] for x in self.imgs]
        self.unique_lbs = list(set(self.lbs))
        self.idx_label_dict = self.build_dict()

    
    def build_dict(self):
        idx_label_dict = {}
        lb = np.array(self.lbs)
        for label in self.unique_lbs:
            idx = np.argwhere(lb==label).flatten()
            idx_label_dict[label] = idx
        return idx_label_dict

def exp_scheduler(optimizer,i,e0,start,end,betas):
    #start and end iter to epoch
    if i == start:
        optimizer.param_groups[0]['betas'] = betas
    if i > start and i<=end:  
        expo = (i-start)/(end-start)
        optimizer.param_groups[0]['lr'] = e0*(0.001**(expo))
    

if __name__ == '__main__':
    
    ##SETUP
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    matplotlib.rcParams.update({'figure.max_open_warning': 0})
    
    ##PREPARE DATA
    dataset_path = config.DATA_PATH
      
    #Data Loaders with Data augmentation: from the EPFL paper
    data_transforms = {
    config.TRAIN: transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(value='random',scale=(0.02, 0.2))
    ]),
    config.VAL: transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    }
    
    image_datasets = {x: TripletDataset(root = os.path.sep.join([dataset_path,x]),transform = data_transforms[x])
                  for x in [config.TRAIN, config.VAL]}
    
    samplers = {x: BatchSampler(image_datasets[x],18,4)
          for x in [config.TRAIN, config.VAL]}
          
    dataloaders = {x: DataLoader(image_datasets[x], batch_sampler = samplers[x])
              for x in [config.TRAIN, config.VAL]}
    
    
    class_names = image_datasets[config.TRAIN].classes
    
    ##MODEL
    model = ReID_net(len(class_names),head = "clustering")
    
    ##TRAINING PART 1  
    ##OUTPUT
    history_plot_path = os.path.sep.join([config.CLUSTERING_PLOT,"plot_1.png"])
    LR_plot_path = os.path.sep.join([config.CLUSTERING_PLOT,"plot_LR_1.png"])
    plot_path = [history_plot_path,LR_plot_path]
            
    #MODEL
    model = model.to(device)
    trainable = 0
    set_parameter_requires_grad(model, trainable)
    params_to_update = get_trainable_layers(trainable,model)
    
    ##OPTIMIZER
    optimizer = optim.Adam(params_to_update, lr=3*1e-4,betas=(0.9, 0.999))
    scheduler = None
    ##LOSS
    criterion = partial(triplet_loss,soft_margin=True, m = 1.0)
    
    N = 10
    model = triplet_train(model,dataloaders,criterion,optimizer,scheduler,device,plot_path,num_epochs=N)
    
    ##SAVING OUTPUT
    torch.save(model.state_dict(),config.CLUSTERING_MODEL_PATH)
    
    ##TRAINING PART 2
    ##OUTPUT
    history_plot_path = os.path.sep.join([config.CLUSTERING_PLOT,"plot_2.png"])
    LR_plot_path = os.path.sep.join([config.CLUSTERING_PLOT,"plot_LR_2.png"])
    plot_path = [history_plot_path,LR_plot_path]
    
    ##MODEL
    model = model.to(device)
    trainable = 1
    set_parameter_requires_grad(model, trainable)
    params_to_update = get_trainable_layers(trainable,model)
    
    ##OPTIMIZER
    optimizer = optim.Adam(params_to_update, lr=3*1e-4,betas=(0.9, 0.999))
    scheduler = None
    scheduler_fun = partial(exp_scheduler,e0=3*1e-4,start=12500,end=20000,betas=(0.5,0.999))
    
    ##LOSS
    criterion = partial(triplet_loss,soft_margin=True, m =1.0)
    N = 250
    model = triplet_train(model,dataloaders,criterion,optimizer,scheduler,device,plot_path,num_epochs=N,scheduler_fun=scheduler_fun)
    torch.save(model.state_dict(),config.CLUSTERING_MODEL_PATH)
