import os 
import sys
import random

import numpy as np
import pandas as pd
import torch
import torchvision

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datasets import get_loaders

def precompute_features(data_path='data/', dataset='celeba_multiattr'):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    net = torchvision.models.resnet.resnet50(pretrained=True).cuda()
    net.fc = torch.nn.Identity()
    net.eval()

    hparams = {
        'precompute_features': False,
        'dataset_name':'CelebAMultiAttr',
        'data_path': data_path,
        'algorithm_name': 'ERM',
        'group_labels': 'yes',
        'batch_size': 16,
        'balanced_batch': False,
        'num_workers': 8, 
        'add_discarded_groups': '',
        'num_y': 2,
        'num_m': 16}
    loaders = get_loaders(hparams)

    dset = {}
    for split in ['tr', 'va', 'te']:
        feats, inds, ys, ms = [], [], [], []
        with torch.no_grad():
            for ind, x, y, a in loaders[split]:
                f = net(x.cuda())
                feats.append(f)
                inds.append(ind)
                ys.append(y)
                ms.append(a)
        inds = torch.cat(inds)

        dset[split] = {'x': torch.cat(feats)[torch.argsort(inds)].cpu(),
                       'y': torch.cat(ys).view(-1, 1)[torch.argsort(inds)].cpu(),
                       'm': torch.cat(ms).view(-1, 1)[torch.argsort(inds)].cpu()}
    torch.save(dset, os.path.join(data_path, dataset, "features.pt"))


def get_attr_values(rel_idx):
    #Relevant index for the attribtue; shift by 1 as values for them have image name at the start
    rel_idx= rel_idx + 1

    rel_outputs=[]
    for idx in range(1, len(data_attr)):
        val_attr= data_attr[idx].replace('\n', '').split(' ')
        #Remove data points with empty strings
        clean_val_attr=[]
        for item in val_attr:
            if item != '':
                clean_val_attr.append(item)

        if clean_val_attr[rel_idx] == '-1':
            rel_outputs.append(0)
        else:
            rel_outputs.append(1)

    rel_outputs= np.array(rel_outputs)    
    return rel_outputs

DATA_DIR= 'data/celeba/'
f_attr= 'list_attr_celeba.txt'
f_metadata= 'metadata_celeba.csv'
f_feat= 'features.pt'

#Load MetaData into pandas df
df_metadata= pd.read_csv(DATA_DIR + f_metadata)

#Read text file and ignore the first line is meaningless so ignore
data_attr= open(DATA_DIR + f_attr, 'r').readlines()[1:]

#This contains the header describing names of the attributes
list_attr= data_attr[0].replace('\n', '').split(' ')
list_attr.remove('')

#Sanity Check: Verify wherther we can recover the correct values for attributes present in the CelebA metadata.csv
rel_idx= list_attr.index('Blond_Hair')
attr_haircol= get_attr_values(rel_idx)
assert np.mean( attr_haircol == df_metadata['y'].to_numpy() ) == 1

rel_idx= list_attr.index('Male')
attr_gender= get_attr_values(rel_idx)
assert np.mean( attr_gender == df_metadata['a'].to_numpy() ) == 1

#Get values for the extra attribute and merge it with the prior spurious attribute 'gender'
rel_idx= list_attr.index('Eyeglasses')
attr_eyeglasses= get_attr_values(rel_idx)

rel_idx= list_attr.index('Wearing_Hat')
attr_hat= get_attr_values(rel_idx)

rel_idx= list_attr.index('Wearing_Earrings')
attr_earrings= get_attr_values(rel_idx)

attr_spur=  8 * attr_earrings + 4 * attr_hat  + 2 * attr_eyeglasses + attr_gender

#Transform metadata.csv file with extra attribute
df_metadata['a']= attr_spur.tolist()
df_metadata.to_csv('data/celeba_multiattr/' + 'metadata_celeba.csv', index=False)

#Transform features.pt file  with extra attribute
precompute_features()