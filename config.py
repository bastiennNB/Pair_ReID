# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:19:33 2020

@author: Nathan Bastien - EPL (31171500) - master thesis "Comparative analysis of re-ID models for matching pairs of Identities"
@file_goal: list of paths and directory names
"""

import os
import pickle

#The original path to the directory of the database
INPUT_DATASET = os.path.sep.join(["Market-1501","Market"])
#Paths to the directories of the pitcures
INPUT_TRAIN_ALL = os.path.sep.join([INPUT_DATASET,"bounding_box_train"])
INPUT_QUERY = os.path.sep.join([INPUT_DATASET,"query"])
INPUT_GAL = os.path.sep.join([INPUT_DATASET,"bounding_box_test"])

#Path to the actual dataset directory that will be used for training, validation and testing
#DATA_PATH = "miniDataset"
DATA_PATH = "dataset"

#the names of the folders of the dataset directory
TRAIN_ALL = "training_and_validation"
TRAIN = "training"
VAL = "validation"
QUERY = "query"
GAL = "gallery"

BATCH_SIZE = 32

#Max TRAIN + VAL = 751
N_ID_TRAIN = 751
#Max TEST = 750
N_ID_TEST = 750

MINI_DATA = 5

#Paths to the outputs of the program

#Classification
CL_DIR = os.path.sep.join(["output","CL"])
if not os.path.exists(CL_DIR):
    os.makedirs(CL_DIR)
CL_MODEL_PATH = os.path.sep.join([CL_DIR,"CL_model.pth"])
CL_PLOT =  os.path.sep.join([CL_DIR,"train_history"])
if not os.path.exists(CL_PLOT):
    os.makedirs(CL_PLOT)
    
#CL2ML
MOML_DIR = os.path.sep.join(["output","moml"])
if not os.path.exists(MOML_DIR):
    os.makedirs(MOML_DIR)
    
TRIPLET_DIR = os.path.sep.join(["output","triplet"])
if not os.path.exists(TRIPLET_DIR):
    os.makedirs(TRIPLET_DIR)
    
MINKOWSKI_DIR = os.path.sep.join(["output","minkowski"])
if not os.path.exists(MINKOWSKI_DIR):
    os.makedirs(MINKOWSKI_DIR)

DIST_METRICS = os.path.sep.join(["output","metrics_param.pkl"])
if not os.path.exists(DIST_METRICS):
    metrics_param = {}
    with open(DIST_METRICS, 'wb') as handle:
        pickle.dump(metrics_param, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Metric Learning
ML_DIR = os.path.sep.join(["output","ML"])
if not os.path.exists(ML_DIR):
    os.makedirs(ML_DIR)
ML_MODEL_PATH = os.path.sep.join([ML_DIR,"ML_model.pth"])
ML_PLOT =  os.path.sep.join([ML_DIR,"train_history"])
if not os.path.exists(ML_PLOT):
    os.makedirs(ML_PLOT)

#CL_ML 
CL_ML_DIR = os.path.sep.join(["output","CL+ML"])
if not os.path.exists(CL_ML_DIR):
    os.makedirs(CL_ML_DIR)
CL_ML_MODEL_PATH = os.path.sep.join([CL_ML_DIR,"CL+ML_model.pth"])
CL_ML_PLOT =  os.path.sep.join([CL_ML_DIR,"train_history"])
if not os.path.exists(CL_ML_PLOT):
    os.makedirs(CL_ML_PLOT)


#Image features, labels and cameras
TEST_FEATURES = os.path.sep.join(["output","test_features.pkl"])
TRAIN_FEATURES =  os.path.sep.join(["output","train_features.pkl"])

#Evaluation related paths
IMAGE_PAIRS = os.path.sep.join(["output","pairs"])
RESULT_IMG = os.path.sep.join(["output","query"])
DIST = os.path.sep.join(["output","dist.pickle"])
DIST_PLOT = os.path.sep.join(["output","dist_plot"])
SORTED_DIST = os.path.sep.join(["output","sorted_dist.pkl"])

