# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:19:33 2020

@author: Nathan Bastien - EPL (31171500) - master thesis "Comparative analysis of re-ID models for matching pairs of Identities"
@file_goal: Outputs the distance distributions of testing set
@Needs: - dist.pkl = distances between testing set pairs of images
        - test_features.pkl = testing set features
"""

import numpy as np
import pickle
import argparse
import torch
import os
import time
from scipy.stats import gaussian_kde
from torchvision import datasets, transforms
from PIL import Image
import random
import matplotlib.pyplot as plt

import config
from metrics_compare import separate_index, plot_distributions


def get_imgs_paths():
    """
        return paths to test images as np arrays
    """
    dataset_path = config.DATA_PATH
    
    data_transforms =  transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_datasets = {x: datasets.ImageFolder(os.path.sep.join([dataset_path,x]),data_transforms)
                  for x in [config.QUERY, config.GAL]}
                  
    query_path = [x[0] for x in image_datasets[config.QUERY].imgs]
    gallery_path =  [x[0] for x in image_datasets[config.GAL].imgs]
    
    query_path = np.array(query_path)
    gallery_path = np.array(gallery_path)
    return query_path, gallery_path


def create_img(img_pair,path):
    """
        makes one image from a pair of images
    """
    widths, heights = zip(*(i.size for i in img_pair))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in img_pair:
      new_im.paste(img, (x_offset,0))
      x_offset += img.size[0]
    new_im.save(path)

def build_imgs_pairs(gallery_path,query_path,s_idx,o_idx,same_ID,other_ID,N_img,same_folder,other_folder):
    """
        Creates image pairs from input indices s_idx and o_idx
    """
    s_query_idx = same_ID[s_idx,1].int().cpu().numpy()
    s_query_img = query_path[s_query_idx]
    s_gal_idx = same_ID[s_idx,2].int().cpu().numpy()
    s_gal_img = gallery_path[s_gal_idx]
    o_query_idx = other_ID[o_idx,1].int().cpu().numpy()
    o_query_img = query_path[o_query_idx]
    o_gal_idx = other_ID[o_idx,2].int().cpu().numpy()
    o_gal_img = gallery_path[o_gal_idx]
    for i in range(N_img):
        s_img_pair = [Image.open(s_query_img[i]), Image.open(s_gal_img[i])]
        filename = s_query_img[i].split(os.path.sep)[-1]
        s_ID = filename.split("_")[0]
        s_img_name = "{}_{}_{:.3f}.jpg".format(i,s_ID,same_ID[s_idx[i],0])
        s_img_path = os.path.sep.join([same_folder,s_img_name])
        create_img(s_img_pair,s_img_path)
        
        o_img_pair = [Image.open(o_query_img[i]), Image.open(o_gal_img[i])]
        query_file = o_query_img[i].split(os.path.sep)[-1]
        gal_file = o_gal_img[i].split(os.path.sep)[-1]
        o_query_ID = query_file.split("_")[0]
        o_gal_ID = gal_file.split("_")[0]
        o_img_name = "{}_{}_{}_{:.3f}.jpg".format(i,o_query_ID,o_gal_ID,other_ID[o_idx[i],0])
        o_img_path = os.path.sep.join([other_folder,o_img_name])
        create_img(o_img_pair,o_img_path)
        

def output_pairs(output_path,same_ID,other_ID,category,query_path,gallery_path,N_img=20):
    """
        Computes relevant indices of image pairs in list of image paths according to the category
    """
    #Prepare Output Folders
    folder = config.IMAGE_PAIRS
    cat_folder = os.path.sep.join([folder,category])
    same_folder = os.path.sep.join([cat_folder,"same_ID"])
    other_folder = os.path.sep.join([cat_folder,"other_ID"])
    if (os.path.exists(same_folder)):
        for filename in os.listdir(same_folder):
            file_path = os.path.join(same_folder, filename)
            try:
                os.unlink(file_path)
            except OSError:
                pass
    else:
        os.makedirs(same_folder)
    if (os.path.exists(other_folder)):
        for filename in os.listdir(other_folder):
            file_path = os.path.join(other_folder, filename)
            try:
                os.unlink(file_path)
            except OSError:
                pass
    else:
        os.makedirs(other_folder)
    #Sorting of the indexes according to the category
    perc = 0.01
    if category == "best_sep":
        s_idx = torch.argsort(same_ID[:,0]).flatten()[:int(perc*len(same_ID[:,0]))]
        s_idx = s_idx[random.sample(range(len(s_idx)),N_img)]
        o_idx = torch.argsort(other_ID[:,0],descending = True).flatten()[:int(perc*len(other_ID[:,0]))]
        o_idx = o_idx[random.sample(range(len(o_idx)),N_img)]
    elif category == "worst_sep":
        s_idx = torch.argsort(same_ID[:,0],descending = True).flatten()[:int(perc*len(same_ID[:,0]))]
        s_idx = s_idx[random.sample(range(len(s_idx)),N_img)]
        o_idx = torch.argsort(other_ID[:,0]).flatten()[:int(perc*len(other_ID[:,0]))]
        o_idx = o_idx[random.sample(range(len(o_idx)),N_img)]
    elif category =="central":
        same_dist = same_ID[0,:]
        other_dist = other_ID[0,:]
        tresh = get_distr_intersection(same_dist,other_dist)
        s_idx = torch.argsort(torch.abs(same_dist-tresh)).flatten()[:N_img]
        o_idx = torch.argsort(torch.abs(other_dist-tresh)).flatten()[:N_img]
    build_imgs_pairs(gallery_path, query_path, s_idx, o_idx, same_ID, other_ID, N_img, same_folder, other_folder)
   
def build_img_grid(output_path):
    for sep_folder in os.listdir(output_path):
        sep_folder_path = os.path.sep.join([output_path,sep_folder])
        for ID_folder in os.listdir(sep_folder_path):
            folder_path = os.path.sep.join([sep_folder_path,ID_folder])
            imgs = os.listdir(folder_path)
            imgs_path = []
            for i,img in enumerate(imgs):   
                imgs_path.append(os.path.sep.join([folder_path,img]))
            fig,axes = plt.subplots(4,4)
            j = 0
            for i in range(len(imgs)):
                ext = imgs[i].split("_")[-1].rsplit(".",1)[-1]
                if ext == 'jpg':
                    iax = axes.flatten()[j]
                    iax.imshow(Image.open(imgs_path[i]))
                    dist = imgs[i].split("_")[-1].rsplit(".",1)[0]
                    dist = str(j) + ": " + dist
                    iax.set_title(dist)
                    iax.axis('off')
                    j += 1
            fig.tight_layout()
            save_path = os.path.sep.join([folder_path,"grid.png"])
            plt.savefig(save_path)

def bhattacharyya_dist(s_ID,o_ID):
    """
        Distribution separation metric
    """
    min_same = min(s_ID)
    max_diff = max(o_ID)
    scope = [min_same,max_diff]
    precision=0.001
    N_bins = int(np.ceil((scope[1]-scope[0])*(1/precision)))
    bins = np.linspace(scope[0],scope[1],N_bins)
    h_same,_ = np.histogram(s_ID,bins=bins,density=False)      
    h_diff,_ = np.histogram(o_ID,bins=bins,density=False)  
    h_same = h_same/h_same.sum()
    h_diff = h_diff/h_diff.sum()
    return np.sqrt(1-(np.sum(np.sqrt(h_same*h_diff))))
    
def get_distr_intersection(same_dist,other_dist):
    same_dist = same_dist.cpu().numpy()
    other_dist = other_dist.cpu().numpy()
    kde_same = gaussian_kde(same_dist)
    kde_diff = gaussian_kde(other_dist)
    min_same = min(same_dist)
    max_diff = max(other_dist)
    scope = [min_same,max_diff]
    tresh = find_intersection(kde_same,kde_diff,init_interval = 0.01, scope=scope,convergence=0.001)
    return tresh

def find_intersection(kde1, kde2, init_interval=0.01, scope =[0,1], convergence=0.0001):
    """
        Finds intersection of two kde functions
    """
    x_left = scope[0]
    x_right = scope[0]+init_interval
    while x_right < scope[1]:
        left = kde1(x_left)[0]-kde2(x_left)[0]
        right = kde1(x_right)[0]-kde2(x_right)[0]
        if left*right < 0: #meaning the functions intersected (an odd number of times) in the interval
            if init_interval <= convergence:
                return x_right
            else: 
                return find_intersection(kde1, kde2, init_interval/10, scope=[x_left, x_right])
        else: #no intersection or an even number of intersections in the interval
            x_left = x_right
            x_right+=init_interval
    return -1 #out of scope means no intersection


if __name__ == '__main__':
    
#        Command line arguments:
#        - extract: Whether you want to sort distances between same ID pairs and other IDs pairs. Needs to be true once. After that result stored with pickle
#        - sep_metric : Choice of distribution separation metric: either bhattacharyya or SSMD (preferred)
#        - overlap: Computes how much of the distributions overlaps
#        - example: Outputs examples in the 1% extremities of the distributions
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract',dest = "extract",default=False, type=bool)
    parser.add_argument('--sep_metric',dest = "sep_metric",default='SSMD')
    parser.add_argument('--overlap',dest = "overlap",default=False, type=bool)
    parser.add_argument('--example',dest = "example",default=False, type=bool)
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.extract == True:
        with open(config.DIST, 'rb') as f:
            dist = pickle.load(f)
        
        with open(config.TEST_FEATURES, 'rb') as f:
            test_ft = pickle.load(f)
        
        #Preparation of the data
        query_features = test_ft['query_f'].to(device)
        query_cameras = test_ft['query_cam']
        query_labels = test_ft['query_label']
        gallery_features = test_ft['gallery_f'].to(device)
        gallery_cameras = test_ft['gallery_cam']
        gallery_labels = test_ft['gallery_label']
        dist = dist.to(device)
        
        counter_same = 0
        counter_other = 0
        same_ID = torch.empty(len(query_labels)*len(gallery_labels),3).fill_(-1).to(device)
        other_ID = torch.empty(len(query_labels)*len(gallery_labels),3).fill_(-1).to(device)
        
        since = time.time()
        
        #Stores distances in 2 Nx3 tensors (N = number of distances)
        #Col 0 = Distance, Col 1 = Query idx in list of query imgs, Col 2 = Gal idx in list of gal imgs
        for i in range(len(query_labels)):
            same_idx, other_idx = separate_index(query_labels[i],query_cameras[i],
                                             gallery_labels,gallery_cameras)
            same_range = range(counter_same,counter_same+len(same_idx))
            other_range = range(counter_other,counter_other+len(other_idx))
            same_ID[same_range,0] = dist[i,same_idx]
            same_ID[same_range,1] = i
            same_ID[same_range,2] = torch.FloatTensor(same_idx).to(device)
            other_ID[other_range,0] = dist[i,other_idx]
            other_ID[other_range,1] = i
            other_ID[other_range,2] = torch.FloatTensor(other_idx).to(device)
            counter_same += len(same_idx)
            counter_other += len(other_idx)
        
        #Cleans up the output
        sorted_dist = {}
        same_ID = same_ID[same_ID[:,0]>-1]
        other_ID = other_ID[other_ID[:,0]>-1]
        sorted_dist["same_ID"] = same_ID.detach().cpu()
        sorted_dist["other_ID"] = other_ID.detach().cpu()
        
        #Separation metric of the distributions
        same_dist = same_ID[:,0]
        other_dist = other_ID[:,0]
        if args.sep_metric == "SSMD":
            SSMD = torch.abs(same_dist.mean()-other_dist.mean())/torch.sqrt(other_dist.std()**2+same_dist.std()**2)
            print("same ID mean: {:.3f}".format(same_dist.mean()))
            print("same ID sigma: {:.3f}".format(same_dist.std()))
            print("diff ID mean: {:.3f}".format(other_dist.mean()))
            print("diff ID sigma: {:.3f}".format(other_dist.std()))
            SSMD = SSMD.cpu().numpy()
            title = "SSMD = {:.4f}".format(SSMD)
            print(title)
        elif args.sep_metric == "bhattacharyya":
            B = bhattacharyya_dist(same_dist.cpu().numpy,other_dist.cpu().numpy)
            title = "bhattacharyya = {}".format(B)
        #Plot the distribution
        plot_path = os.path.sep.join(["output","test_distribution.png"])
        plot_distributions(same_dist.cpu().numpy(),other_dist.cpu().numpy(),plot_path,title)
        
        time_elapsed = time.time() - since
        print('distribution of testing set distances, outputed in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        with open(config.SORTED_DIST, 'wb') as handle:
            pickle.dump(sorted_dist, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    if args.extract == False:
        print("pickle extraction")
        with open(config.SORTED_DIST,"rb") as f:
            sorted_dist = pickle.load(f)
        same_ID = sorted_dist["same_ID"].to(device)
        other_ID = sorted_dist["other_ID"].to(device)
        
    if args.overlap == True:   
        same_dist = same_ID[:,0]
        other_dist = other_ID[:,0]
        tresh = get_distr_intersection(same_dist,other_dist)
        cross_same = same_dist[same_dist>=tresh].size
        cross_diff = other_dist[other_dist<tresh].size      
        print("Distances smaller then the treshold of {:.3f} are considered to be between same IDs, bigger values are considered different".format(tresh))
        print("Percentage of same ID that could be confused for other ID {:.3f}%".format(100*cross_same/same_dist.size))
        print("Percentage of other ID that could be confused for same ID {:.3f}%".format(100*cross_diff/other_dist.size))
    
    if args.example == True:                
        categories = ["best_sep","worst_sep"]
        query_path,gallery_path = get_imgs_paths()
        output_path = config.IMAGE_PAIRS
        for category in categories:
            output_pairs(output_path,same_ID,other_ID,category,query_path,gallery_path,N_img=16)   
        build_img_grid(output_path)
        