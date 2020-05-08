# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:19:33 2020

@author: Nathan Bastien - EPL (31171500) - master thesis "Comparative analysis of re-ID models for matching pairs of Identities"
@file_goal: Trains a multi layer perceptron for clustering based on Triplet Loss
@Needs:  - train_features.pkl = training set features
"""

import numpy as np
from torch.utils.data import Dataset,Sampler, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import pickle
import time
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import os
import argparse

import config
from model import weights_init_kaiming
from train_classification import plot_LR


class EmbeddingDataset(Dataset):   
    def __init__(self,features,labels,cameras):
        self.fts = features
        self.lbs = labels
        self.unique_lbs = list(set(labels))
        self.cms = cameras
        self.idx_label_dict = self.build_dict()
    def __len__(self):
        return len(self.lbs)
    
    def __getitem__(self, i):
        return (self.fts[i,:],self.lbs[i])
    
    def build_dict(self):
        idx_label_dict = {}
        lb = np.array(self.lbs)
        for label in self.unique_lbs:
            idx = np.argwhere(lb==label).flatten()
            idx_label_dict[label] = idx
        return idx_label_dict
        

class BatchSampler(Sampler):
    '''
    sampler used in dataloader. method __iter__ should output the indices each time it is called
    '''
    def __init__(self, dataset, n_class, n_num, *args, **kwargs):
        super(BatchSampler, self).__init__(dataset, *args, **kwargs)
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

def triplet_loss(xt,xp,xq,soft_margin = False, m=1):
    """
        triplet loss based on Hinge loss with soft margin implementation
    """
    diff = torch.norm(xp-xt,p=2,dim=1) - torch.norm(xq-xt,p=2,dim=1)
    if soft_margin:
        f = torch.log(m + torch.exp(diff))
    else:
        f = m + diff
        f[f<0] = 0
    return torch.sum(f)

class EmbeddingNet(nn.Module):
    def __init__(self,n_features_in = 512, n_features_out=256):
        super(EmbeddingNet, self).__init__()
        self.fc1   = nn.Linear(n_features_in, 512)
        self.dp1 = nn.Dropout(p=0.4)
        self.relu1 = nn.ReLU()
#        self.fc2 = nn.Linear(256, 256)
#        self.dp2 = nn.Dropout(p=0.15)
#        self.relu2 = nn.ReLU()
        self.fc3  = nn.Linear(512, n_features_out)
        self.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dp1(x)
        x = self.relu1(x)
#        x = self.fc2(x)
#        x = self.dp2(x)
#        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
    def get_distance(self,x1,x2):
        emb1 = self.forward(x1)
        emb2 = self.forward(x2)
        return torch.norm(emb2-emb1,p=2,dim=1)

class TripletNet(nn.Module):
    def __init__(self, embedding_net = EmbeddingNet()):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
    
    def get_distance(self,x1,x2):
        emb1 = self.embedding_net(x1)
        emb2 = self.embedding_net(x2)
        return torch.norm(emb2-emb1,p=2,dim=1)


def plot_progress(current_epoch,x_epoch,y_loss,fig_path):
    matplotlib.use('Agg')
    fig, ax = plt.subplots()
    ax.plot(x_epoch, y_loss['training'], 'bo-', label='train')
    ax.plot(x_epoch, y_loss['validation'], 'ro-', label='val')
    ax.set_title("training_history")
    ax.legend()
    fig.savefig(fig_path)

def triplet_train(model,dataloaders,criterion,optimizer,scheduler,device,fig_path,num_epochs,scheduler_fun = None):
    
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in [config.TRAIN, config.VAL]}
    
    ##Tracking of the progress
    x_epoch = []
    y_loss = {} # loss history
    y_loss[config.TRAIN] = []
    y_loss[config.VAL] = []
    LR_history = []
    
    ##Keep best wts in last 5% of epochs
    best_val_loss = 1e9
    
    since = time.time()
    ##Actual training
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        x_epoch.append(epoch+1)
    
        # Each epoch has a training and validation phase
        for phase in [config.TRAIN, config.VAL]:
            if phase == 'training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
    
            running_loss = 0.0
            for i,(features, labels) in enumerate(dataloaders[phase]):
                #Transforms data for CPU/GPU
                features = features.to(device)
                labels = labels.to(device)
    
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'training'):
                    embeddings = model(features)
                    anchors,pos,neg = build_triplets(embeddings, labels, device)
                    loss = criterion(anchors,pos,neg)

                # backward + optimize only if in training phase
                if phase == 'training':
                    loss.backward()
                    optimizer.step()
                if scheduler_fun != None:
                    scheduler_fun(optimizer,i)
                    
                running_loss += loss.item() * features.size(0)
            # statistics
            if phase == 'training' and scheduler!= None:
                LR_history.append(scheduler.get_last_lr()[0])
                scheduler.step()
                if (np.mod(epoch,5) or epoch==num_epochs):
                    plot_LR(epoch,x_epoch,LR_history,fig_path[1])
                
            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            if phase =="training":
                y_loss[config.TRAIN].append(epoch_loss)

            # save the weights and save the model every 10 epoch
            # If long training keeps best epoch result in last 5% of epochs
            if phase == 'validation':
                last_model_wts = model.state_dict()
                y_loss[config.VAL].append(epoch_loss)
                if (np.mod(epoch,3) or epoch==num_epochs):
                    plot_progress(epoch+1,x_epoch,y_loss,fig_path[0])
                if(epoch >= (0.9*num_epochs)):
                    if(epoch_loss < best_val_loss):
                        best_epoch = epoch
                        best_val_loss = epoch_loss
                        best_model_wts = last_model_wts
                        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("Weight from epoch {} saved as best weights".format(best_epoch+1))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

##BEGINNING OF THE SCRIPT
if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #GET TRAINING SET FEATURES AND METRICS DICT
    with open(config.TRAIN_FEATURES,"rb") as f:
        extract = pickle.load(f)
    with open(config.DIST_METRICS, 'rb') as f:
        metrics_param = pickle.load(f)
    
    ##PREPARE OUTPUT
    matplotlib.rcParams.update({'figure.max_open_warning': 0})
    history_plot_path = os.path.sep.join([config.TRIPLET_DIR,"plot_loss.png"])
    LR_plot_path = os.path.sep.join([config.TRIPLET_DIR,"plot_LR.png"])
    fig_path = [history_plot_path,LR_plot_path]
    
    ##TRAINING
    ##Datasets and loaders
    train_dataset = EmbeddingDataset(extract["train_features"],extract["train_labels"],extract["train_cameras"])
    val_dataset = EmbeddingDataset(extract["val_features"],extract["val_labels"],extract["val_cameras"])
    datasets = {config.TRAIN: train_dataset,
                config.VAL: val_dataset}
    samplers = {x: BatchSampler(datasets[x],16,4)
          for x in [config.TRAIN, config.VAL]}
    
    dataloaders = {x: DataLoader(datasets[x], batch_sampler = samplers[x], num_workers = 0)
          for x in [config.TRAIN, config.VAL]}
          
    ##Model and Loss
    model = EmbeddingNet().to(device)
    criterion = partial(triplet_loss,soft_margin=True, m =1.2)
    
    ##Optimizer
    N = 50
    optimizer = optim.Adam(model.parameters(), lr=2e-3, amsgrad=False)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.3)
          
    model = triplet_train(model,dataloaders,criterion,optimizer,scheduler,device,fig_path,num_epochs=N)
    
    ##SAVING OUTPUT
    triplet_model_path = os.path.sep.join([config.TRIPLET_DIR,"triplet_MLP.pth"])
    torch.save(model.state_dict(),triplet_model_path)
    metrics_param["triplet_dist"] = triplet_model_path
    with open(config.DIST_METRICS, 'wb') as handle:
        pickle.dump(metrics_param, handle, protocol=pickle.HIGHEST_PROTOCOL)


    