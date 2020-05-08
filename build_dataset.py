# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:19:33 2020

@author: Nathan Bastien - EPL (31171500) - master thesis "Comparative analysis of re-ID models for matching pairs of Identities"
@file_goal: Branch Market 1501 into training, validation and testing sets
"""

import config
import os
import shutil
import time
import numpy as np
from imutils import paths


    
#populating training and validation, query and gallery sets


for dataSplit in (config.TRAIN_ALL,config.QUERY,config.GAL):
    print("[INFO] populating '{} set'...".format(dataSplit))
    
    #Choose the appropriate folder of origin and set the maximal number of IDs
    if (dataSplit == config.TRAIN_ALL):
        inputPath = config.INPUT_TRAIN_ALL
        #maxID = config.MAX_ID_TRAIN
    elif (dataSplit == config.QUERY):
        inputPath = config.INPUT_QUERY
        #maxID = config.MAX_ID_TEST
    else:
        inputPath = config.INPUT_GAL
        #maxID = config.MAX_ID_TEST
    
    IDpair = {}
    #extract all image paths from the folder 
    imagePaths = list(paths.list_images(inputPath))
    folder_path = os.path.sep.join([config.DATA_PATH,dataSplit])
    if (not os.path.exists(folder_path)):
        os.makedirs(folder_path)
    #nID = len(os.listdir(folder_path))
    #nImage = 0
    newID = 0
    for imagePath in imagePaths:
        #Extract the ID from the image path
        #imagePath = imagePaths[nImage]
        filename = imagePath.split(os.path.sep)[-1]
        ID = filename.split("_")[0]
        if ID not in IDpair:
          IDpair[ID] = newID
          newID+=1
        #Path to the actual dataset location of the image
        dirPath = os.path.sep.join([config.DATA_PATH,dataSplit,str(IDpair[ID])])
        #Create location if it does not exist and update the number of IDs
        if (not os.path.exists(dirPath)):
          os.makedirs(dirPath)
                
        #Add image to the directory
        p = os.path.sep.join([dirPath,filename])
        if(os.path.exists(p)):
            #nImage+=1
            continue
        else:
            try:
                shutil.copy2(imagePath,p)
                #nImage += 1
            except OSError:
                time.sleep(5)
                shutil.copy2(imagePath,p)
                #nImage += 1
                continue

print("[INFO] Spliting training and validation data...")
train_all_path = os.path.sep.join([config.DATA_PATH,config.TRAIN_ALL])
for ID_folder in os.listdir(train_all_path):
    #Extract all training images per ID
    input_ID_path = os.path.sep.join([train_all_path,ID_folder])
    ID_imagePaths = list(paths.list_images(input_ID_path))
    
    #Create training and validation directories for each ID
    trainDir = os.path.sep.join([config.DATA_PATH, config.TRAIN, ID_folder])
    valDir = os.path.sep.join([config.DATA_PATH, config.VAL, ID_folder])
    if (not os.path.exists(trainDir)):      
        os.makedirs(trainDir)
    if (not os.path.exists(valDir)):
        os.makedirs(valDir)
      
    
    #split all images of the ID between training and validation
    imNumber = 0
    for imagePath in ID_imagePaths:
        filename = imagePath.split(os.path.sep)[-1]
        if(np.mod(imNumber,5)):
            p = os.path.sep.join([trainDir,filename])
        else:
            p = os.path.sep.join([valDir,filename])
        if(os.path.exists(p)):
            continue
            imNumber+=1
        else:
            shutil.copy2(imagePath,p)
            imNumber += 1
        