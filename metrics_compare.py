"""
Created on Fri Mar  6 15:19:33 2020

@author: Nathan Bastien - EPL (31171500) - master thesis "Comparative analysis of re-ID models for matching pairs of Identities"
@file_goal: Compares performance of distance metrics on training and validation set
@Needs: - metric_param.pkl = stores the trained clustering metrics
"""

import torch
import numpy as np
import time
import os
import pickle
from functools import partial
import matplotlib.pyplot as plt
import argparse

import config
from train_mlp import EmbeddingNet
from train_moml import DeepMOML


def minkowski_dist(x1,x2,p=2):
    return torch.norm(x2-x1,p=p,dim=1)
    
def mahalanobis_dist(x1,x2,M):
    x = x2 - x1
    xm = torch.mm(x,M)
    return torch.sqrt(torch.sum(torch.mul(xm,x),dim=1))

def normalize(vector,mini=None,maxi=None):
    if mini==None:
        mini = np.amin(vector)
    if maxi==None:
        maxi = np.amax(vector)
    return (vector-mini)/(maxi-mini)

def get_SSMD(features,labels,cameras,metric,p,plot_path=None):
    features = features.cuda()
    same_dist = torch.FloatTensor().cuda()
    other_dist = torch.FloatTensor().cuda()
    for i,query in enumerate(features):
        same_i, other_i = separate_index(labels[i],cameras[i],labels,cameras)
        if i==0:
            same_dist = metric(features[same_i,:],query)
            other_dist = metric(features[other_i,:],query)
        else:
            same_dist = torch.cat((same_dist,metric(features[same_i,:],query)),0)
            other_dist = torch.cat((other_dist,metric(features[other_i,:],query)),0)
    SSMD = torch.abs(same_dist.mean()-other_dist.mean())/torch.sqrt(other_dist.std()**2+same_dist.std()**2)
    SSMD = SSMD.cpu().numpy()
    if(plot_path!=None):
        if(p>=1):
            title = "Minkowski distance with p={:.1f} : SSMD={:.4f}".format(p,SSMD)
        elif(p==0):
            title = "MOML distance: SSMD={:.4f}".format(SSMD)
        elif(p==-1):
            title = "Deep learned distance: SSMD={:.4f}".format(SSMD)
        s_ID = normalize(same_dist,mini=torch.min(same_dist),maxi = torch.max(other_dist)).cpu().numpy()
        o_ID = normalize(other_dist,mini=torch.min(same_dist),maxi = torch.max(other_dist)).cpu().numpy()
        plot_distributions(s_ID,o_ID,plot_path,title=title)
    return SSMD

def separate_index(ql,qc,labels,cameras):
    labels = np.array(labels)
    cameras = np.array(cameras)
    query_index = np.argwhere(labels==ql).flatten()
    camera_index = np.argwhere(cameras==qc).flatten()
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    #Discard distactors and same camera for same ID
    junk_index1 = np.argwhere(labels==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1).flatten()  
    mask = np.ones(labels.shape,dtype=bool)
    mask[junk_index] = False
    mask[good_index] = False
    other_index = np.argwhere(mask==True).flatten()
    return good_index, other_index

def plot_distributions(same_ID,other_ID,plot_path,title):
    fig,ax = plt.subplots()
    ax.hist(same_ID,bins = 100,label="same ID", density = True, alpha = 0.6)
    ax.hist(other_ID,bins=100,label="different ID", density = True, alpha = 0.6)
    ax.set_xlabel("distance")
    ax.set_title(title)
    ax.legend()
    fig.savefig(plot_path)
    
    
if __name__ == '__main__':
    
    #SELECTION OF PARAMETERS
    parser = argparse.ArgumentParser()
    parser.add_argument('--type',dest = "type",default='L2',type=str)
    parser.add_argument('--learning',dest = "learning",default=False, type=bool)
    
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #GET TRAINING SET FEATURES
    with open(config.TRAIN_FEATURES, 'rb') as f:
        extract = pickle.load(f)           
        
    ## PREPARING CROSS TYPE VARIABLES
    #Dimensions
    N_dim = extract["train_features"].shape[1]
    N_img_train = extract["train_features"].shape[0]
    N_img_val = extract["val_features"].shape[0]
    #Output
    with open(config.DIST_METRICS, 'rb') as f:
        metrics_param = pickle.load(f)
    
    
    ###STANDARD L2
    if args.type=='L2':
        L2 = 2
        plot_path = os.path.sep.join([config.CLASSIFICATION_DIR,"train_distribution_L2.png"])
        metric = partial(minkowski_dist,p=L2)
        since = time.time()
        SSMD = get_SSMD(extract["train_features"],extract["train_labels"],extract["train_cameras"],
                      metric,p=L2,plot_path=plot_path)
        time_elapsed = time.time() - since
        print('distribution of training set L2 distances, outputed in {:.0f}m {:.0f}s'.format(L2,
                    time_elapsed // 60, time_elapsed % 60))
        ##Validation with L2
        plot_path_val_L2 = os.path.sep.join([config.CLASSIFICATION_DIR,"val_distribution_L2.png"])
        metric = minkowski_dist
        since = time.time()
        val_L2_SSMD = get_SSMD(extract["val_features"],extract["val_labels"],extract["val_cameras"],
                              metric,p=2,plot_path=plot_path_val_L2)
        time_elapsed = time.time() - since
        print('distribution of validation set L2 distances, outputed in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
    
    ###MINKOWSKI DIST
    if args.type=='minkowski':
        N_img = N_img_train
        features = extract["train_features"]
        labels = extract["train_labels"]
        cameras = extract["train_cameras"]
        if(args.learning == True):
            p_list = np.linspace(1.6,1.8,11)
            SSMD = np.zeros(p_list.shape)
            #index = random.sample(range(N_img),7000)
            for i,p in enumerate(p_list):
                since = time.time()
                metric = partial(minkowski_dist,p=p)
                SSMD[i] = get_SSMD(features,labels,cameras,metric,p)
                time_elapsed = time.time() - since
                print('Minkowski with p = {:.2f}, tested in {:.0f}m {:.0f}s'.format(p,
                        time_elapsed // 60, time_elapsed % 60))
                
            plot_path = os.path.sep.join([config.MINKOWSKI_DIR,"p_value_zoom.png"])
            fig,ax = plt.subplots()
            ax.plot(p_list,SSMD,'o-')
            ax.set_title("Tuning of Minkowski p parameter")
            ax.set_xlabel("p value")
            ax.set_ylabel("SSMD")
            fig.savefig(plot_path)
            best_p = p_list[np.argmax(SSMD)]
            print("best value for p: {:.2f}".format(best_p))
            metrics_param['minkowski'] = best_p
            with open(config.DIST_METRICS, 'wb') as handle:
                pickle.dump(metrics_param, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else: ##No learning
            with open(config.DIST_METRICS, 'rb') as f:
                metrics_param = pickle.load(f)
            best_p = metrics_param['minkowski']
            
            plot_path = os.path.sep.join([config.MINKOWSKI_DIR,"train_distribution_Minkowski.png"])
            metric = partial(minkowski_dist,p=best_p)
            since = time.time()
            SSMD = get_SSMD(features,labels,cameras,metric,p=best_p,plot_path=plot_path)
            time_elapsed = time.time() - since
            print('distribution of training set Minkowski distances with p = {:.2f}, outputed in {:.0f}m {:.0f}s'.format(best_p,
                        time_elapsed // 60, time_elapsed % 60))
            plot_path = os.path.sep.join([config.MINKOWSKI_DIR,"val_distribution_Minkowski.png"])
            since = time.time()
            SSMD = get_SSMD(extract["val_features"],extract["val_labels"],extract["val_cameras"],
                                  metric,p=best_p,plot_path=plot_path)
            time_elapsed = time.time() - since
            print('distribution of validation set Minkowski distances with p = {:.2f}, outputed in {:.0f}m {:.0f}s'.format(best_p,
                        time_elapsed // 60, time_elapsed % 60))
    
    ###MAHALANOBIS DIST    
    elif args.type == 'MOML':
        metric_matrix = metrics_param["MOML"]
        metric = partial(mahalanobis_dist, M=metric_matrix)
        plot_path_train = os.path.sep.join([config.MOML_DIR,"train_distribution_MOML.png"])
        plot_path_val = os.path.sep.join([config.MOML_DIR,"val_distribution_MOML.png"])
        ##Training set
        since = time.time()
        with torch.no_grad():
            train_SSMD = get_SSMD(extract["train_features"],extract["train_labels"],extract["train_cameras"],
                                  metric,p=0,plot_path=plot_path_train)
        time_elapsed = time.time() - since
        print('distribution of training set distribution, outputed in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
        ##Validation set
        since = time.time()
        with torch.no_grad():
            val_SSMD = get_SSMD(extract["val_features"],extract["val_labels"],extract["val_cameras"],
                              metric,p=0,plot_path=plot_path_val)
        time_elapsed = time.time() - since
        print('distribution of validation set distribution, outputed in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
                    
    elif args.type == 'DeepMOML':
        ##Loading metric
        M_list = metrics_param['DeepMOML']
        model = DeepMOML(device, M_list = M_list)
        model.eval()
        metric = model.get_distance
        plot_path_train = os.path.sep.join([config.MOML_DIR,"train_distribution_DeepMOML.png"])
        plot_path_val = os.path.sep.join([config.MOML_DIR,"val_distribution_DeepMOML.png"])
        ##Training set
        since = time.time()
        with torch.no_grad():
            train_SSMD = get_SSMD(extract["train_features"],extract["train_labels"],extract["train_cameras"],
                                  metric,p=0,plot_path=plot_path_train)
        time_elapsed = time.time() - since
        print('distribution of training set distribution, outputed in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
        ##Validation set
        since = time.time()
        with torch.no_grad():
            val_SSMD = get_SSMD(extract["val_features"],extract["val_labels"],extract["val_cameras"],
                              metric,p=0,plot_path=plot_path_val)
        time_elapsed = time.time() - since
        print('distribution of validation set distribution, outputed in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
    
    ###TRIPLET NET BASED DIST    
    elif args.type == 'triplet_net':
        ##Loading metric
        model = EmbeddingNet()
        model.load_state_dict(torch.load(metrics_param['triplet_dist']))
        model.to(device)
        model.eval()
        metric = model.get_distance
        ##Preparing output
        plot_path_train = os.path.sep.join([config.TRIPLET_DIR,"train_distribution_3loss.png"])
        plot_path_val = os.path.sep.join([config.TRIPLET_DIR,"val_distribution_3loss.png"])
        ##Training set
        since = time.time()
        with torch.no_grad():
            train_SSMD = get_SSMD(extract["train_features"],extract["train_labels"],extract["train_cameras"],
                                  metric,p=-1,plot_path=plot_path_train)
        time_elapsed = time.time() - since
        print('distribution of training set distribution, outputed in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
        ##Validation set
        since = time.time()
        with torch.no_grad():
            val_SSMD = get_SSMD(extract["val_features"],extract["val_labels"],extract["val_cameras"],
                              metric,p=-1,plot_path=plot_path_val)
        time_elapsed = time.time() - since
        print('distribution of validation set distribution, outputed in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))

            
