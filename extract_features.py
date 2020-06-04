"""
Created on Fri Mar  6 15:19:33 2020

@author: Nathan Bastien - EPL (31171500) - master thesis "Comparative analysis of re-ID models for matching pairs of Identities"
@file_goal: extract features from the images in either testing or training set
@Needs: - model_path.pth
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import time
import os
import pickle
import argparse

import config
from model import ReID_net


def extract_features(model,dataloaders,device,n_features=2048):
    features = torch.FloatTensor()
    count = 0
    for inputs,labels in dataloaders:
        #Transforms data for CPU/GPU
        inputs = inputs.to(device)
        labels = labels.to(device)
        n, c, h, w = inputs.size()
        count += n
        #print(count)
        #Extract features from the batch
        bf = torch.FloatTensor(n,n_features).zero_().cuda()
        bf = model(inputs)
        #Normalize features
        norm = torch.norm(bf, p=2, dim=1, keepdim=True)
        bf = bf.div(norm.expand_as(bf))
        #Add to output vector
        features = torch.cat((features,bf.data.cpu()), 0)
    return features

def get_ID(imgs_path):
    labels = []
    cameras = []
    for img, v in imgs_path:
        filename = os.path.basename(img)
        ID = filename.split("_")[0]
        if ID == '00-1':
            labels.append(-1)
        else:
            labels.append(int(ID))
        cameras.append(filename.split('c')[1][0])
    return labels,cameras

##START OF THE SCRIPT

if __name__ == '__main__':
    #Command line arguments
    #model_type = classification or clustering based feature extractor
    #dataset = training or testing set 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',dest = "model",default="CL", type=str)
    parser.add_argument('--dataset',dest = "dataset",default="test", type=str)
    args = parser.parse_args()
    dataset_path = config.DATA_PATH
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.dataset == "test":
        #Load Testing data
        
        data_transforms =  transforms.Compose([
            transforms.Resize((256,128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_datasets = {x: datasets.ImageFolder(os.path.sep.join([dataset_path,x]),data_transforms)
                      for x in [config.QUERY, config.GAL]}
        
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128,
                                                 shuffle=False, num_workers=2)
                  for x in [config.QUERY, config.GAL]}
        
        gallery_path = image_datasets['gallery'].imgs
        query_path = image_datasets['query'].imgs
        
        gallery_labels, gallery_cameras = get_ID(gallery_path)
        query_labels, query_cameras = get_ID(query_path)
        N_classes = len(image_datasets[config.QUERY].classes)
        
        #Load model
        if args.model == "CL":
            model = ReID_net(N_classes+1,name = "CL")
            model.load_state_dict(torch.load(config.CL_MODEL_PATH))
            #remove classifier
            model.head.classifier = nn.Sequential()
        elif args.model_type == "ML":
            model = ReID_net(name = "ML")
            model.load_state_dict(torch.load(config.ML_MODEL_PATH))
        elif args.model_type == "CL+ML_ML":
            model = ReID_net(N_classes+1,name = "CL")
            model.load_state_dict(torch.load(config.CL_ML_MODEL_PATH))
            #remove classifier
            model.head = nn.Sequential()
        elif args.model_type == "CL+ML_CL":
            model = ReID_net(N_classes+1,name = "CL")
            model.load_state_dict(torch.load(config.CL_ML_MODEL_PATH))
            #remove classifier
            model.head.classifier = nn.Sequential()
         
        #Actual feature extraction
        model.to(device)
        model.eval()
        
        since = time.time()
        
        with torch.no_grad():
            gallery_features = extract_features(model,dataloaders[config.GAL],device,n_features=512)
            query_features = extract_features(model,dataloaders[config.QUERY],device,n_features=512)
        
        # features are Torch FloatTensor, labels and cameras are lists
        result = {'gallery_f':gallery_features,'gallery_label':gallery_labels,'gallery_cam':gallery_cameras,
                  'query_f':query_features,'query_label':query_labels,'query_cam':query_cameras}
        
        with open(config.TEST_FEATURES, 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        time_elapsed = time.time() - since
        print('Features extraction complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
    
    elif args.dataset == "train":

        data_transforms =  transforms.Compose([
            transforms.Resize((256,128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_datasets = {x: datasets.ImageFolder(os.path.sep.join([dataset_path,x]),data_transforms)
                  for x in [config.TRAIN, config.VAL]}
    
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128,
                                             shuffle=False, num_workers=0)
              for x in [config.TRAIN, config.VAL]}   
        
        image_paths = {x: image_datasets[x].imgs for x in [config.TRAIN, config.VAL]}
        
        train_labels, train_cameras = get_ID(image_paths[config.TRAIN])
        val_labels, val_cameras = get_ID(image_paths[config.VAL])
        N_classes = len(image_datasets[config.TRAIN].classes)
        #Load model
        if args.model == "CL":
            model = ReID_net(N_classes+1,name = "CL")
            model.load_state_dict(torch.load(config.CL_MODEL_PATH))
            #remove classifier
            model.head.classifier = nn.Sequential()
        elif args.model_type == "ML":
            model = ReID_net(name = "ML")
            model.load_state_dict(torch.load(config.ML_MODEL_PATH))
        elif args.model_type == "CL+ML_ML":
            model = ReID_net(N_classes+1,name = "CL")
            model.load_state_dict(torch.load(config.CL_ML_MODEL_PATH))
            #remove classifier
            model.head = nn.Sequential()
        elif args.model_type == "CL+ML_CL":
            model = ReID_net(N_classes+1,name = "CL")
            model.load_state_dict(torch.load(config.CL_ML_MODEL_PATH))
            #remove classifier
            model.head.classifier = nn.Sequential()
        
        #Prepare model for evaluation
        model.to(device)
        model.eval()
        
        since = time.time()
        
        with torch.no_grad():
            train_features = extract_features(model,dataloaders[config.TRAIN],device,n_features=512)
            val_features = extract_features(model,dataloaders[config.VAL],device,n_features=512)
        
        # features are Torch FloatTensor, labels and cameras are lists
        extract = {'train_features':train_features,'train_labels':train_labels,'train_cameras':train_cameras,
                   'val_features':val_features,'val_labels':val_labels,'val_cameras':val_cameras}
        
        with open(config.TRAIN_FEATURES, 'wb') as handle:
            pickle.dump(extract, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        time_elapsed = time.time() - since
        print('Features extraction complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))