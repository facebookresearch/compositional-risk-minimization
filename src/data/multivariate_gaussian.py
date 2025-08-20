# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import sys
import argparse
import numpy as np 
import pandas as pd
import torch
from scipy.stats import multivariate_normal
from scipy.linalg import qr
import matplotlib.pyplot as plt


# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate multivariate Gaussian synthetic data')
parser.add_argument('--total_attr', type=int, default=2, help='Total number of attributes')
parser.add_argument('--total_cat', type=int, default=10, help='Total number of categories')
parser.add_argument('--data_path', default='datasets/',help='Path to the datasets directory.')
args = parser.parse_args()

# Set global variables from parsed arguments
TOTAL_ATTR = args.total_attr
TOTAL_CAT = args.total_cat
TOTAL_GROUPS=  (TOTAL_CAT ** TOTAL_ATTR)
# Break down attributes z_1, .., z_m as (class label, spurious attr.), i.e. (z1, (z_2, ..z_m)) 
TOTAL_SPUR_GROUPS= int( TOTAL_GROUPS / TOTAL_CAT )
DATA_DIM=100
GROUP_SIZE= 100
DATASET_SIZE= GROUP_SIZE * TOTAL_GROUPS
VAL_RATIO= 0.2
SPLIT_SIZES={
    'tr': int((1-VAL_RATIO)* DATASET_SIZE),
    'va': int(VAL_RATIO*DATASET_SIZE),
    'te': DATASET_SIZE
}


DATA_DIR= os.path.join(args.data_path, 'multivariate_gaussian/', f'total_attr_{TOTAL_ATTR}_total_cat_{TOTAL_CAT}', '')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

#Initialize mean vectors
feat_dim= DATA_DIM    
# TOTAL_CAT * TOTAL_ATTR mean vectors in feat_dim-dimensional space. Think of them as (mu_i0, mu_i1, ...)_i where is the attribute index
A = np.random.rand(feat_dim, TOTAL_CAT*TOTAL_ATTR)  
# Perform QR decomposition to get orthogonal vectors
G_MEAN, R = qr(A, mode='economic')  # 'economic' ensures Q has minimal shape
G_MEAN= G_MEAN.T.reshape(TOTAL_ATTR, TOTAL_CAT, feat_dim)

def featurize(x):
    feat= np.concatenate( [np.ones((x.shape[0], 1)), x , x**2], axis=1 )
    return feat

def sample_from_conditional(g_idx, group_size):
    """Return tensor of the shape [DATA_DIM]"""    
    # Convert g_idx to base TOTAL_CAT to get individual attribute values
    # g_idx ranges from 0 to TOTAL_CAT^TOTAL_ATTR - 1
    attr_vals = []
    temp_idx = g_idx    
    for _ in range(TOTAL_ATTR):
        attr_vals.append(temp_idx % TOTAL_CAT)
        temp_idx = temp_idx // TOTAL_CAT
    
    # Reverse to get proper order (most significant digit first)
    attr_vals = attr_vals[::-1]
    # Compute mean as average of attribute-specific means
    mu = 0.0
    for attr_idx in range(TOTAL_ATTR):
        attr_val = attr_vals[attr_idx]
        mu += G_MEAN[attr_idx, attr_val]
    mu = mu / TOTAL_ATTR
    
    sigma = np.eye(DATA_DIM)
    x = multivariate_normal.rvs(mean=mu, cov=sigma, size=group_size)

    return x


final_data={}
for data_split in ['tr', 'va', 'te']:
    x=[]
    y=[]
    m=[]
    sample_size= SPLIT_SIZES[data_split]
    #Breakdown groups as (y, a) where y has 2 values and a has 2^{TOTAL_ATTR -1} values
    for z_1 in range(TOTAL_CAT):
        for z_2 in range(TOTAL_SPUR_GROUPS):
            g_idx=  TOTAL_SPUR_GROUPS * z_1 + z_2
            x.append( sample_from_conditional(g_idx, GROUP_SIZE) ) 
            y.append( GROUP_SIZE * [z_1] )
            m.append( GROUP_SIZE * [z_2] )
    
    #Concat along sample axis
    x= np.concatenate(x, axis=0)
    y= np.concatenate(y, axis=0)
    m= np.concatenate(m, axis=0)
    
    #Randomly permute data and featurize; conver to tensors as well
    perm= np.random.permutation(x.shape[0])
    x= torch.Tensor(featurize(x[perm]))
    y= torch.Tensor(y[perm]).view(x.shape[0], 1)
    m= torch.Tensor(m[perm]).view(x.shape[0], 1)

    print(x.shape, y.shape, m.shape)
    final_data[data_split]= {
        'x': x,
        'y': y,
        'm': m 
    }

torch.save(final_data, DATA_DIR+ "features.pt")
for seed in range(3):
    #Dropping Groups
    g_indices= np.linspace(1, TOTAL_GROUPS, num=TOTAL_GROUPS, dtype=int) - 1
    #Randomly shuffle non-zero training groups
    np.random.shuffle(g_indices)

    #Keep decreasing in intervals of 10 percent of total groups
    for idx in range(9):
        total_disc_groups= int(TOTAL_GROUPS*0.1*(1+idx))
        drop_indices= g_indices[:total_disc_groups]
        print('Seed: ', seed, 'Drop Indices', drop_indices.shape)
        np.save( DATA_DIR + 'seed_' + str(seed) + '_discarded_group_scenario_' + str(idx) + '.npy', drop_indices)

    #Some extra hard cases for TOTAL_ATTR>2 case
    for idx in [8.5, 8.7, 8.9]:
        total_disc_groups= int(TOTAL_GROUPS*0.1*(1+idx))
        drop_indices= g_indices[:total_disc_groups]
        print('Seed: ', seed, 'Drop Indices', drop_indices.shape)
        np.save( DATA_DIR + 'seed_' + str(seed) + '_discarded_group_scenario_' + str(idx) + '.npy', drop_indices)