# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:55:50 2020

@author: Nathan
"""
import numpy as np
from torch.utils.data import Sampler
from torchvision import datasets
import matplotlib
import matplotlib.pyplot as plt


class ExpScheduler:
    def __init__(self,optimizer,start,end,e0,betas):
        self.optimizer = optimizer
        self.start = start
        self.end = end
        self.e0 = e0
        self.betas = betas
        self.epoch = 0
        
    def step(self):
        if self.epoch == self.start:
            self.optimizer.param_groups[0]['betas'] = self.betas
        if self.epoch > self.start and self.epoch<=self.end:  
            expo = (self.epoch-self.start)/(self.end-self.start)
            self.optimizer.param_groups[0]['lr'] = self.e0*(0.001**(expo))
        self.epoch+=1

class MLDataset(datasets.folder.ImageFolder):   
    def __init__(self,root, transform=None):
        super(MLDataset, self).__init__(root,transform)
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


#Batch Sampler for Metric Learning Training
class PKBatchSampler(Sampler):
    '''
    sampler used in dataloader. method __iter__ should output the indices each time it is called
    '''
    def __init__(self, dataset, n_class, n_num, *args, **kwargs):
        super(PKBatchSampler, self).__init__(dataset, *args, **kwargs)
        self.n_class = n_class
        self.n_num = n_num
        self.batch_size = n_class * n_num
        self.dataset = dataset
        self.labels = np.array(dataset.lbs)
        self.labels_uniq = np.array(dataset.unique_lbs)
        self.idx_label_dict = dataset.idx_label_dict
        self.iter_num = len(self.labels_uniq) // self.n_class
        self.length = len(self.labels) // self.batch_size

    def __iter__(self):
        curr_p = 0
        np.random.shuffle(self.labels_uniq)
        for k, v in self.idx_label_dict.items():
            np.random.shuffle(self.idx_label_dict[k])
        for i in range(self.iter_num):
            label_batch = self.labels_uniq[curr_p: curr_p + self.n_class]
            curr_p = np.mod(curr_p+self.n_class,len(self.labels_uniq)-1)
            idx = []
            for lb in label_batch:
                if len(self.idx_label_dict[lb]) > self.n_num:
                    idx_smp = np.random.choice(self.idx_label_dict[lb],
                            self.n_num, replace = False)
                else:
                    idx_smp = np.random.choice(self.idx_label_dict[lb],
                            self.n_num, replace = True)
                idx.extend(idx_smp.tolist())
            #np.random.shuffle(idx)
            yield idx

    def __len__(self):
        return self.iter_num


## Functions to plot training progress
        
def plot_progress(current_epoch,x_epoch,y_loss,fig_path):
    matplotlib.use('Agg')
    fig, ax = plt.subplots()
    ax.plot(x_epoch, y_loss['training'], 'bo-', label='train')
    ax.plot(x_epoch, y_loss['validation'], 'ro-', label='val')
    ax.set_title("training_history")
    ax.legend()
    fig.savefig(fig_path)

def plot_LR(current_epoch,x_epoch,LR,fig_path):
    matplotlib.use('Agg')
    fig,ax = plt.subplots()
    ax.set_yscale('log')
    ax.step(x_epoch,LR)
    ax.set_xlabel('# of epoch')
    ax.set_title("evolution of the LR")
    fig.savefig(fig_path)


##Set requires gradient parameters 

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