# A Comparative analysis for Pair Re-ID

Code repository relating to master thesis "A Comparative analysis of re-ID models for matching pairs of Identities" by Nathan Bastien from EPL 2020 MAP promotion (inchallah).

## Problem definition and objective of the models

- Pair Re-ID question: Is this pair of images, each containing a person, from the same identity ?
- Objective of the models: The model learns a feature space where pairs of embeddings of the same identity are separated by a smaller distance than pairs of embeddings of different identities.

## Models

Trains 5 models for Pair Re-ID:
- CL: classification supervision
- ML: metric learning supervision
- CL2ML: adds a metric learning supervised model on top of the CL CNN. 2 versions of this models:
	- Multi-layer perceptron 
	- Mahalanobis distance
- CL+ML model: supervision with classification and metric learning objectives.

## Requirement

- Libraries see env.txt file
- Market-1501 dataset (http://www.liangzheng.com.cn/Project/project_reid.html)
- Keep file structure:  
	-repo  
	--- output  
	--- dataset

## Background

- Master thesis paper (in writing)
- Builds mainly from following papers:
	- Rethinking Person Re-Identification with Confidence (https://arxiv.org/abs/1906.04692)
	- Bag of Tricks and A Strong Baseline for Deep Person Re-identification (https://arxiv.org/abs/1903.07071)
	- In Defense of the Triplet Loss for Person Re-Identification (https://arxiv.org/abs/1703.07737)
	- Online Deep Metric Learning (https://arxiv.org/abs/1805.05510)


## Code Structure

1) config.py: Defines all directories names and paths to outputs of the program
2) build_dataset.py: Splits Market-1501 in training, validation, query and gallery sets
3) train.py: trains the CNNS using train_utils.py and loss.py
4) extract_features.py: Extract features from dataset (training or testing)
If CL2ML model:  
	4.b) go to 5) OR   
	4.b) train_mlp.py: Train MLP from training set extracted features Metric Learning OR  
	4.b) train_moml.py: Train Matrix from extracted features for Mahalanobis distance based Metric Learning  
	4.c) compare_metrics.py: Compare Metric learning training performances on training set
5) evaluate.py: Compute distances between all possible pairs of one query image and one gallery image. Also outputs traditional Re-ID metrics
6) dist_distribution.py: Sorts distances between same ID pairs and different IDs pairs. Output pairs examples from tails of distribution