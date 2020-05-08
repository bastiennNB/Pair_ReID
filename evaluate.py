"""
Created on Fri Mar  6 15:19:33 2020

@author: Nathan Bastien - EPL (31171500) - master thesis "Comparative analysis of re-ID models for matching pairs of Identities"
@file_goal: - Computes the distances between each pair of image
            - outputs the traditional Re-ID metrics at the same time
@Needs: - metrics_param.pkl = Trained distance metrics from classification based features
"""

import numpy as np
import os
import shutil
import pickle
import time
from metric_learn import NCA
import torch
import torchvision
from torchvision import datasets, transforms
from functools import partial
import argparse

import config
from metrics_compare import mahalanobis_dist, minkowski_dist, normalize
from triplet_train import EmbeddingNet
    
def get_metric(flag,device):
    with open(config.DIST_METRICS, 'rb') as f:
        metrics_param = pickle.load(f)
    if flag=="L2":
        metric = partial(minkowski_dist,p=2)
    elif flag=="MOML":
        metric_matrix = metrics_param[flag].to(device)
        metric = partial(mahalanobis_dist, M=metric_matrix)
    elif flag=="minkowski":
        p = metrics_param[flag]
        metric = partial(minkowski_dist,p=p)
    elif flag=="triplet_dist":
        model = EmbeddingNet()
        model.load_state_dict(torch.load(metrics_param[flag]))
        model.to(device)
        model.eval()
        metric = model.get_distance
    return metric
    
def get_imgs_paths():
    dataset_path = config.DATA_PATH
    
    data_transforms =  transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_datasets = {x: datasets.ImageFolder(os.path.sep.join([dataset_path,x]),data_transforms)
                  for x in [config.QUERY, config.GAL]}
                  
    return image_datasets[config.QUERY].imgs, image_datasets[config.GAL].imgs

def visualize_top(ql,dist,n,i,query_path,gallery_path,good_index,gallery_labels):      
    folder_name = str(i) + '_' + str(ql)
    folder_path = os.path.sep.join([config.RESULT_IMG,folder_name])
    if (not os.path.exists(folder_path)):
          os.makedirs(folder_path)
    query_img = 'q_' + str(ql) + '.jpg' 
    p = os.path.sep.join([folder_path,query_img])
    if (not os.path.exists(p)):
        shutil.copy2(query_path[i][0],p)
    gal_i = torch.argsort(dist).cpu().numpy()
    gal_i = np.intersect1d(gal_i,good_index)
    gal_i = gal_i[:n]
    gal_l = [gallery_labels[j] for j in gal_i]
    for y,idx in enumerate(gal_i):
        gal_img = 'g_' + str(y+1) + '_' + str(gal_l[y]) + '.jpg' 
        p = os.path.sep.join([folder_path,gal_img])
        if (not os.path.exists(p)):
            shutil.copy2(gallery_path[idx][0],p)
            
def evaluate(qf,ql,qc,gf,gl,gc,metric):
    gl = np.array(gl)
    gc = np.array(gc)
    with torch.no_grad():
        dist = metric(qf,gf)
    #Sort gal according to metric
    index = torch.argsort(dist)
    index = index.cpu().numpy()
    #Get indices of same ID and same camera
    
    query_index = np.argwhere(gl==ql).flatten()
    camera_index = np.argwhere(gc==qc).flatten()
    #Get all relevant matches in the gallery
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    #Discard distactors and same camera for same ID
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1).flatten()
    ap_ID, CMC_ID = compute_metrics(index, good_index, junk_index)
    return ap_ID, CMC_ID,dist,good_index

def compute_metrics(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    cmc = cmc.cuda()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index from gal ordering
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    n_TP = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(n_TP):
        d_recall = 1.0/n_TP
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc
    
### BEGINNING OF SCRIPT
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="select the type of distance metric used")
    parser.add_argument('--metric',dest = "metric",default='L2',help = "select the type of distance metric used")
    parser.add_argument('--visualize',dest = "visualize",default= False,type=bool)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with open(config.TEST_FEATURES, 'rb') as f:
        result = pickle.load(f)
    
    metric = get_metric(args.metric,device)
        
    if args.visualize:
        query_path, gallery_path = get_imgs_paths()
        
    query_features = result['query_f'].to(device)
    query_cams = result['query_cam']
    query_labels = result['query_label']
    gallery_features = result['gallery_f'].to(device)
    gallery_cams = result['gallery_cam']
    gallery_labels = result['gallery_label']
        
    since = time.time()
    
    CMC = torch.IntTensor(len(gallery_labels)).zero_()
    CMC = CMC.to(device)
    AP = 0.0
    distances = torch.FloatTensor(len(query_labels),len(gallery_labels)).zero_()
    distances = distances.to(device)
    for i,qf in enumerate(query_features):
        ap_ID, CMC_ID,distances[i,:],good_index_i = evaluate(qf, query_labels[i], query_cams[i], 
                                                             gallery_features, gallery_labels, gallery_cams, metric)
        if CMC_ID[0]==-1:
            continue
        CMC += CMC_ID
        AP += ap_ID
        if args.visualize:
            if np.mod(i,300)==0:
                visualize_top(query_labels[i],distances[i,:],5,i,query_path,gallery_path,good_index_i,gallery_labels)
        
    time_elapsed = time.time() - since
    print('Distances computed in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    
    CMC = CMC.float()
    CMC = CMC/len(query_labels) #average CMC
    AP = AP/len(query_labels)
    #Normalize distances
    distances = normalize(distances,mini=torch.min(distances),maxi=torch.max(distances))
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],AP))
    with open(config.DIST, 'wb') as handle:
        pickle.dump(distances.detach().cpu(), handle, protocol=pickle.HIGHEST_PROTOCOL)