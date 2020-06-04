# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:48:40 2020

@author: Nathan Bastien - EPL (31171500) - master thesis "Comparative analysis of re-ID models for matching pairs of Identities"
@file_goal: train the CNN models.
@Needs: - train_utils.py : helper functions for training
"""

import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
from functools import partial
import matplotlib
import os
import argparse
import time
import copy

import config
from train_utils import get_trainable_layers, set_parameter_requires_grad, PKBatchSampler, plot_LR, plot_progress, MLDataset, ExpScheduler
from model import ReID_net
from loss import CrossEntropyLS, TripletLoss, build_triplets,TripletAndCrossEntropyLoss



def train_CNN(model,dataloaders,criterion,optimizer,scheduler,device,plot_path,N_epoch):
    since = time.time()

    ###Tracking of the progress
    x_epoch = []
    y_loss = {} # loss history
    y_loss["training"] = []
    y_loss["validation"] = []
    LR_history = []
    
    ##Keep best wts in last 5% of epochs
    best_val_loss = 1e6
    
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in [config.TRAIN, config.VAL]}
    
    for epoch in range(N_epoch):
        print('Epoch {}/{}'.format(epoch+1, N_epoch))
        print('-' * 10)
        x_epoch.append(epoch)

        #Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            #Accumulator for the loss
            running_loss = 0.0

            # Iterate over all training data.
            for i,(inputs, labels) in enumerate(dataloaders[phase]):               
                #Transforms data for CPU/GPU
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # Forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'training'):
                    #Forward propagation depends on the model
                    if model.name == "CL":
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    elif model.name == "ML":
                        outputs = model(inputs)
                        anchors,pos,neg = build_triplets(outputs, labels, device)
                        loss = criterion(anchors,pos,neg)
                    elif model.name == "CL+ML":
                        outputs,eML = model(inputs)
                        anchors,pos,neg = build_triplets(eML, labels, device)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs,anchors,pos,neg,labels)

                    # backward + optimize only if in training phase
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            
            #Stats and progress tracking
            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            if phase =="training":
                y_loss["training"].append(epoch_loss)
            elif phase =="validation":
                y_loss["validation"].append(epoch_loss)
                last_model_wts = model.state_dict()
                #Plot the progress
                if (np.mod(epoch,20) or epoch==N_epoch):
                    plot_progress(epoch+1,x_epoch,y_loss,plot_path[0])
                #Save the best weights if in last part of training
                if(epoch+1 > (0.95*N_epoch)):
                    if(epoch_loss < best_val_loss):
                        best_epoch = epoch
                        best_val_loss = epoch_loss
                        best_model_wts = copy.deepcopy(last_model_wts)
                        
            #LR scheduling
            if phase == 'training':
                if scheduler!= None:
                    LR_history.append(optimizer.param_groups[0]['lr'])
                    scheduler.step()
                    if np.mod(epoch,20) or epoch==N_epoch:
                        plot_LR(epoch,x_epoch,LR_history,plot_path[1])
        
        #End of training        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Loss from the best epoch (epoch {}): {:4f}'.format(best_epoch, y_loss["validation"][best_epoch+1]))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    #Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_epoch',dest = "N_epoch",default=250, type=int)
    parser.add_argument('--model',dest = "model",default="ML", type=str)
    args = parser.parse_args()
    
    ##SETUP
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    matplotlib.rcParams.update({'figure.max_open_warning': 0})
    
    ##PREPARE DATA
    dataset_path = config.DATA_PATH
      
    #Data Loaders with Data augmentation
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
    
    ##PREPARE MODELS AND DATALOADERS
    if args.model == "CL":
        
        #Prepare Batch loaders
        image_datasets = {x: datasets.ImageFolder(os.path.sep.join([dataset_path,x]),data_transforms[x])
                  for x in [config.TRAIN, config.VAL]}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0)
                       for x in [config.TRAIN, config.VAL]}        
        #Model
        class_names = image_datasets[config.TRAIN].classes
        model = ReID_net(len(class_names), name = "CL")       
        #Loss 
        criterion = partial(CrossEntropyLS,label_smoothing=0.1)       
        #Output
        history_plot_path = os.path.sep.join([config.CL_PLOT,"plot_LOSS.png"])
        LR_plot_path = os.path.sep.join([config.CL_PLOT,"plot_LR.png"])
        plot_path = [history_plot_path,LR_plot_path]
        
    if args.model == "ML":
        
        #Prepare Batch loaders
        image_datasets = {x: MLDataset(root = os.path.sep.join([dataset_path,x]),transform = data_transforms[x])
                  for x in [config.TRAIN, config.VAL]}
    
        samplers = {x: PKBatchSampler(image_datasets[x],18,4)
              for x in [config.TRAIN, config.VAL]}
              
        dataloaders = {x: DataLoader(image_datasets[x], batch_sampler = samplers[x])
                  for x in [config.TRAIN, config.VAL]}       
        #Model
        model = ReID_net(name = "ML")        
        #Loss
        criterion = partial(TripletLoss,soft_margin=True, m = 1.0)   
        #Output
        history_plot_path = os.path.sep.join([config.ML_PLOT,"plot_LOSS.png"])
        LR_plot_path = os.path.sep.join([config.ML_PLOT,"plot_LR.png"])
        plot_path = [history_plot_path,LR_plot_path]
        
    if args.model == "CL+ML":
        
        #Prepare Batch loaders
        image_datasets = {x: MLDataset(root = os.path.sep.join([dataset_path,x]),transform = data_transforms[x])
                  for x in [config.TRAIN, config.VAL]}
    
        samplers = {x: PKBatchSampler(image_datasets[x],18,4)
              for x in [config.TRAIN, config.VAL]}
              
        dataloaders = {x: DataLoader(image_datasets[x], batch_sampler = samplers[x])
                  for x in [config.TRAIN, config.VAL]}
           
        class_names = image_datasets[config.TRAIN].classes        
        #Model
        model = ReID_net(len(class_names), name = "CL+ML")        
        #Loss
        criterion = TripletAndCrossEntropyLoss   
        #Output
        history_plot_path = os.path.sep.join([config.CL_ML_PLOT,"plot_LOSS.png"])
        LR_plot_path = os.path.sep.join([config.CL_ML_PLOT,"plot_LR.png"])
        plot_path = [history_plot_path,LR_plot_path]
    
    ##TRAINING WARMUP
    
    model.to(device)
    N_epoch = 15
    trainable = 0
    set_parameter_requires_grad(model, trainable)
    params_to_update = get_trainable_layers(trainable,model)   
        
    #Optimiser and LR scheduling -- Model specific
    if args.model == "CL":
        optimizer = optim.Adam(params_to_update, lr=5*1e-4,amsgrad=True)
        scheduler = None
    elif args.model == "ML":
        optimizer = optim.Adam(params_to_update, lr=3*1e-4,betas=(0.9, 0.999))
        scheduler = None
    elif args.model == "CL+ML":
        optimizer = optim.Adam(params_to_update, lr=3*1e-4,betas=(0.9, 0.999))
        scheduler = None
        
    model = train_CNN(model, dataloaders, criterion, optimizer, scheduler, device, plot_path, N_epoch)
    
    ##TRAINING
    
    model.to(device)
    N_epoch = args.N_epoch
    trainable = 1
    set_parameter_requires_grad(model, trainable)
    params_to_update = get_trainable_layers(trainable,model)   
    
    #Optimiser and LR scheduling -- Model specific
    if args.model == "CL":
        optimizer = optim.Adam(params_to_update, lr=5*1e-4,amsgrad=True)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.1)
    elif args.model == "ML":
        optimizer = optim.Adam(params_to_update, lr=3*1e-4,betas=(0.9, 0.999))
        scheduler = ExpScheduler(optimizer,e0=3*1e-4,start=100, end=230,betas=(0.5, 0.999))
    elif args.model == "CL+ML":
        optimizer = optim.Adam(params_to_update, lr=3*1e-4,betas=(0.9, 0.999))
        scheduler = ExpScheduler(optimizer,e0=3*1e-4,start=100, end=230,betas=(0.5, 0.999))
        
    model = train_CNN(model, dataloaders, criterion, optimizer, scheduler, device, plot_path, N_epoch)
    
    ## SAVE MODEL - Model specific path
    if args.model == "CL":
        torch.save(model.state_dict(),config.CL_MODEL_PATH)
    elif args.model == "ML":
        torch.save(model.state_dict(),config.ML_MODEL_PATH)
    elif args.model == "CL+ML":
        torch.save(model.state_dict(),config.CL_ML_MODEL_PATH)
    