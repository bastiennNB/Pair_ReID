# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:19:33 2020

@author: Nathan Bastien - EPL (31171500) - master thesis "Comparative analysis of re-ID models for matching pairs of Identities"
@file_goal: Trains classification based model
@Needs: - model.py = CNN designed for re-ID
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import matplotlib
import time
import os
from functools import partial
import argparse

import config
from model import ReID_net

def imshow(inp, title=None):
    """Imshow for Tensor."""
    plt.ion()
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.ioff()
    
    
def get_trainable_layers(trainable,model):
    params_to_update = model.parameters()
    print("Trained layers:")
    if trainable!=0:
        params_to_update = []
        layers = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                layer = name.split(".")[0]
                if layer not in layers:
                    layers.append(layer)
                    print("\t",layer)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    return params_to_update



def set_parameter_requires_grad(model, trainable=0):
    if trainable==0:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
    else:
        if trainable==4:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.backbone.layer4.parameters():
                param.requires_grad = True
            for param in model.head.parameters():
                param.requires_grad = True
        if trainable==3:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.backbone.layer4.parameters():
                param.requires_grad = True
            for param in model.backbone.layer3.parameters():
                param.requires_grad = True
            for param in model.head.parameters():
                param.requires_grad = True
        if trainable==2:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.backbone.layer4.parameters():
                param.requires_grad = True
            for param in model.backbone.layer3.parameters():
                param.requires_grad = True
            for param in model.backbone.layer2.parameters():
                param.requires_grad = True
            for param in model.head.parameters():
                param.requires_grad = True
        if trainable==1:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.backbone.layer4.parameters():
                param.requires_grad = True
            for param in model.backbone.layer3.parameters():
                param.requires_grad = True
            for param in model.backbone.layer2.parameters():
                param.requires_grad = True
            for param in model.backbone.layer1.parameters():
                param.requires_grad = True
            for param in model.head.parameters():
                param.requires_grad = True
            
    


def plot_progress(current_epoch,x_epoch,y_loss,y_acc,fig_path):
    matplotlib.use('Agg')
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="accuracy")
    ax0.plot(x_epoch, y_loss['training'], 'r', label='train')
    ax0.plot(x_epoch, y_loss['validation'], 'b', label='val')
    ax1.plot(x_epoch, y_acc['training'], 'r', label='train')
    ax1.plot(x_epoch, y_acc['validation'], 'b', label='val')
    ax0.set_xlabel('# of epoch')
    ax1.set_xlabel('# of epoch')
    ax0.legend()
    ax1.legend()
    fig.savefig(fig_path)
    
def plot_LR(current_epoch,x_epoch,LR,fig_path):
    matplotlib.use('Agg')
    fig,ax = plt.subplots()
    ax.set_yscale('log')
    ax.step(x_epoch,LR)
    ax.set_xlabel('# of epoch')
    ax.set_title("evolution of the LR")
    fig.savefig(fig_path)


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
    


def train_model(model, criterion, optimizer, scheduler,device,dataloaders,fig_path, num_epochs=30):
    since = time.time()

    ###Tracking of the progress
    x_epoch = []
    y_loss = {} # loss history
    y_loss[config.TRAIN] = []
    y_loss[config.VAL] = []
    y_acc = {}
    y_acc[config.TRAIN] = []
    y_acc[config.VAL] = []
    LR_history = []
    
    ##Keep best wts in last 5% of epochs
    best_val_loss = 10
    
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in [config.TRAIN, config.VAL]}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        x_epoch.append(epoch)

        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                
                #Transforms data for CPU/GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'training' and scheduler!= None:
                LR_history.append(scheduler.get_last_lr())
                scheduler.step()
                if (np.mod(epoch,20) or epoch==num_epochs):
                    plot_LR(epoch,x_epoch,LR_history,fig_path[1])
                

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase =="training":
                y_loss[config.TRAIN].append(epoch_loss)
                y_acc[config.TRAIN].append(epoch_acc)

            # save the weights and save the model every 10 epoch
            # If long training keeps best epoch result in last 5% of epochs
            if phase == 'validation':
                last_model_wts = model.state_dict()
                y_loss[config.VAL].append(epoch_loss)
                y_acc[config.VAL].append(epoch_acc)
                if (np.mod(epoch,20) or epoch==num_epochs):
                    plot_progress(epoch+1,x_epoch,y_loss,y_acc,fig_path[0])
                if(epoch > (0.95*num_epochs)):
                    if(epoch_loss < best_val_loss):
                        best_epoch = epoch
                        best_val_loss = epoch_loss
                        best_model_wts = last_model_wts
                        

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Accuracy from the best epoch (epoch {}): {:4f}'.format(best_epoch,y_acc[config.VAL][epoch]))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


#####BEGINNING OF THE SCRIPT
if __name__ == '__main__':
    
    #Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_epoch',dest = "N_epoch",default=250, type=int, help = "number of training epochs")
    parser.add_argument('--LS',dest = "LS",default=0.1, type=float, help = "amount of smoothing on labels")
    args = parser.parse_args()
    
    #Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    matplotlib.rcParams.update({'figure.max_open_warning': 0})
    dataset_path = config.DATA_PATH
      
    #Data Loaders with data augmentation from the EPFL paper
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
    
    image_datasets = {x: datasets.ImageFolder(os.path.sep.join([dataset_path,x]),data_transforms[x])
                  for x in [config.TRAIN, config.VAL]}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=0)
              for x in [config.TRAIN, config.VAL]}
    
    
    # Model
    class_names = image_datasets[config.TRAIN].classes
    model = ReID_net(len(class_names),head = "classification")
    
    ##TRAINING PART 1 - Warmup of the new head
    model = model.to(device)
    trainable = 0
    set_parameter_requires_grad(model, trainable)
    params_to_update = get_trainable_layers(trainable,model)
    
    N_epoch = 30
    
    
    optimizer = optim.Adam(params_to_update, lr=5*1e-4,amsgrad=True)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1)
    criterion = partial(CrossEntropyLS,label_smoothing=args.LS)
    #Output plot
    history_plot_path = os.path.sep.join([config.CLASSIFICATION_PLOT,"plot_1.png"])
    LR_plot_path = os.path.sep.join([config.CLASSIFICATION_PLOT,"plot_LR_1.png"])
    plot_path = [history_plot_path,LR_plot_path]
    if (not os.path.exists(config.TRAINING_PLOT)):
            os.makedirs(config.TRAINING_PLOT)
    model = train_model(model, criterion, optimizer, scheduler, device, dataloaders,
                        plot_path,num_epochs=N_epoch)
    torch.save(model.state_dict(),config.CLASSIFICATION_MODEL_PATH)
    
    
    ##TRAINING PART 2 -- fine tuning the whole model
    model = model.to(device)
    trainable = 1
    set_parameter_requires_grad(model, trainable)
    params_to_update = get_trainable_layers(trainable,model)
    N_epoch = args.N_epoch
    
    history_plot_path = os.path.sep.join([config.CLASSIFICATION_PLOT,"plot_2.png"])
    LR_plot_path = os.path.sep.join([config.CLASSIFICATION_PLOT,"plot_LR_2.png"])
    plot_path = [history_plot_path,LR_plot_path]
    
    optimizer = optim.Adam(params_to_update, lr=5*1e-4,amsgrad=True)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.1)
    criterion = partial(CrossEntropyLS,label_smoothing=args.LS)
    model = train_model(model, criterion, optimizer, scheduler, device, dataloaders, 
                        plot_path,num_epochs=N_epoch)
    torch.save(model.state_dict(),config.CLASSIFICATION_MODEL_PATH)
