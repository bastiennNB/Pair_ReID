# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:19:33 2020

@author: Nathan Bastien - EPL (31171500) - master thesis "Comparative analysis of re-ID models for matching pairs of Identities"
@file_goal: Trains M matrix for optimal Mahalanobis distance for clustering
@Needs: - triplet_train.py = Dataset of Embeddings and Batch Hard Generations
        - train_features.pkl = training set features
"""

import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pickle
import time
import numpy as np
import os
from functools import partial
import argparse

from triplet_train import EmbeddingDataset, BatchSampler, build_triplets, plot_progress
import config



class DeepMOML(nn.Module):
    """
        Work in progress
    """
    def __init__(self,device,n_layers =3,n_ft = 512, M_list = None):
        super(DeepMOML, self).__init__()
        layers = []
        for i in range(n_layers):
            if M_list == None:
                layers += [MOML(device)]
            else:
                layers += [MOML(device,M=M_list[i])]
            if i<n_layers-1:
                layers += [nn.ReLU()]
        layers = nn.Sequential(*layers)
        self.layers = layers
        self.n_layers = n_layers
    
    def forward(self,x):
        return self.layers(x)
        
    def update(self,g,A,phase, m = 1):
        sum_loss = 0
        for layer in self.layers.children():
            if layer.__class__.__name__ == 'MOML':
                local_loss = layer.update(g,A,phase,m=m)
                sum_loss += (1/self.n_layers)*(local_loss)
        return sum_loss
        
    def get_M(self):
        M_list = []
        for layer in self.layers.children():
            if layer.__class__.__name__ == 'MOML':
                M_list.append(layer.get_M())
        return M_list
    
    def get_distance(self,x1,x2):
        emb1 = self.forward(x1)
        emb2 = self.forward(x2)
        return torch.norm(emb2-emb1,p=2,dim=1)

class MOML(nn.Module):
    """
        Represent matrix objects of MOML problem
    """
    def __init__(self,device,n_ft = 512, M = None):
        super(MOML, self).__init__()
        if M==None:
            self.M = torch.eye(n_ft, requires_grad = False).to(device)
            self.Mprev = torch.eye(n_ft,requires_grad=False).to(device)
            self.L = torch.eye(n_ft,requires_grad = False).to(device)
        else:
            self.M = M
            self.Mprev = torch.eye(n_ft,requires_grad=False).to(device)
            self.L = self.get_sqrt(num_iters=100)
        
    def forward(self, x):
        if len(x.shape)==1:
            x = x.view(1,x.shape[0])
        x = torch.mm(x,self.L.t())
        return x
        
    def get_sqrt(self,num_iters):
        dim = self.M.shape[0]
        normM = self.M.mul(self.M).sum().sqrt()
        with torch.no_grad():
            Y = self.M.div(normM.view(1,1).expand_as(self.M)).cuda()
            Z = torch.eye(dim, requires_grad = False).cuda()
            I = torch.eye(dim, requires_grad = False).cuda()
            for i in range(num_iters):
                T = 0.5*(3.0*I-Z.mm(Y))
                Y = Y.mm(T)
                Z = T.mm(Z)
            sqrt_M = torch.mul(Y,(torch.sqrt(normM).view(1,1).expand_as(self.M)))
        return sqrt_M
    
    def grad_update(self,g,A,z):
        if z>0:
            with torch.no_grad():
                self.Mprev = self.M.clone()
            self.M -= g*A
            self.L = self.get_sqrt(num_iters=10)

    def loss(self,g,A,m=1):
        z = m + torch.trace(self.M.mm(A))
        z[z<0] = 0
        reg = 0.5*(torch.norm(self.M-self.Mprev,p="fro"))**2
        loss = reg + g*z
        return loss,z
    
    def update(self,g,A,phase,m = 1):
        loss, z = self.loss(g,A,m)
        if phase == 'training':
            self.grad_update(g,A,z)
        return loss
    
    def get_M(self):
        return self.M

def build_triplet_matrix(xt,xp,xq):
    n_dim = xt.shape[0]
    x1 = xt-xp
    x2 = xt - xq
    A1 = x1.view(n_dim,1).mm(x1.view(1,n_dim))
    A2 = x2.view(n_dim,1).mm(x2.view(1,n_dim))
    return A1 - A2


def plot_iter_loss(x,loss,fig_path):
    matplotlib.use('Agg')
    fig, ax = plt.subplots()
    ax.plot(x, loss)
    ax.set_title("Evolution of the loss with every iteration")
    fig.savefig(fig_path)

def moml_train(dataloaders,model,g,margin,device,fig_path,num_epochs = 5):   
    ##Tracking of the progress
    x_epoch = []
    y_loss = {} # loss history
    y_loss[config.TRAIN] = []
    y_loss[config.VAL] = []
    
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in [config.TRAIN, config.VAL]}
    
    since = time.time()
    ##Actual training
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        x_epoch.append(epoch)
        # Each epoch has a training and validation phase
        for phase in [config.TRAIN, config.VAL]:
            running_loss = 0.0
            for features, labels in dataloaders[phase]:
                #Transforms data for CPU/GPU
                features = features.to(device)
                labels = labels.to(device) 
                # forward
                with torch.set_grad_enabled(phase == 'training'):
                    embeddings = model(features)
                    anchors,pos,neg = build_triplets(embeddings, labels, device)
                    for i,xt in enumerate(anchors):
                        #print("{}/{}".format(i,anchors.shape[0]))
                        with torch.no_grad():
                            xp = pos[i]
                            xq = neg[i]
                            A = build_triplet_matrix(xt,xp,xq)
                        loss = model.update(g, A,phase, m = margin)  
                        running_loss += loss.item()
            # statistics          
            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            if phase =="training":
                y_loss[config.TRAIN].append(epoch_loss)
            if phase == 'validation':
                y_loss[config.VAL].append(epoch_loss)
                plot_progress(epoch+1,x_epoch,y_loss,fig_path)
                        
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    return model.get_M()


##BEGININNG OF SCRIPT
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type',dest = "type",default="single",type=str)
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #GET TRAINING SET FEATURES AND METRICS DICT
    with open(config.TRAIN_FEATURES,"rb") as f:
        extract = pickle.load(f)
    with open(config.DIST_METRICS, 'rb') as f:
        metrics_param = pickle.load(f)
    
    ##PREPARE OUTPUT
    matplotlib.rcParams.update({'figure.max_open_warning': 0})
    fig_path_deep = os.path.sep.join([config.MOML_DIR,"train_plot_deep.png"])
    fig_path_simple = os.path.sep.join([config.MOML_DIR,"train_plot_simple.png"])
    
    ##TRAINING
    ##Datasets and loaders
    train_dataset = EmbeddingDataset(extract["train_features"],extract["train_labels"],extract["train_cameras"])
    val_dataset = EmbeddingDataset(extract["val_features"],extract["val_labels"],extract["val_cameras"])
    datasets = {config.TRAIN: train_dataset,
                config.VAL: val_dataset}
    samplers = {x: BatchSampler(datasets[x],8,4)
          for x in [config.TRAIN, config.VAL]}
    
    dataloaders = {x: DataLoader(datasets[x], batch_sampler = samplers[x], num_workers = 0)
          for x in [config.TRAIN, config.VAL]}
    
    ##Model and loss
    if args.type == 'deep':
        model = DeepMOML(device)
        margin = 3
        T = samplers["training"].batch_size*samplers["training"].iter_num
        gamma = 1/(np.sqrt(T))
        N = 8
        save_flag = 'DeepMOML'
        fig_path = fig_path_deep
    else:
        model = MOML(device)
        margin = 2
        T = samplers["training"].batch_size*samplers["training"].iter_num
        gamma = 1/(np.sqrt(T))
        N = 5
        save_flag = 'MOML'
        fig_path = fig_path_simple
    M = moml_train(dataloaders,model,gamma,margin,device,fig_path,num_epochs=N)
    ##SAVE OUTPUT
    metrics_param[save_flag] = M       
    with open(config.DIST_METRICS, 'wb') as handle:
        pickle.dump(metrics_param, handle, protocol=pickle.HIGHEST_PROTOCOL)