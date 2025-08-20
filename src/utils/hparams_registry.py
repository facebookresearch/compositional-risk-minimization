# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import numpy as np
import torch


def get_hparams(args):
    print(args)
    hparams = {
        'data_path': args['data_path'],
        'dataset_name': args['dataset'],
        'group_labels': args['group_labels'],
        'balanced_batch': False,
        'precompute_features': args['precompute_features'],
        'quick_run': args['quick_run'],
        'algorithm_name': args['algorithm'],
        'hparams_comb': args['hparams_comb'],
        'seed': args['seed'],
        'resume': args['resume'],
        'num_workers': 0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'out_dir': args['out_dir'], 
        'add_discarded_groups': args['add_discarded_groups']
        }        

    if 'SynAED' in args['dataset']:
        dataset_parts = args['dataset'].split('_')
        # SynAED_TOTAL_ATTR_TOTAL_CAT format
        if len(dataset_parts) == 3:  
            syn_aed_total_attr = int(dataset_parts[1])
            syn_aed_total_cat = int(dataset_parts[2])
            hparams['data_path'] = hparams['data_path'] + f'/multivariate_gaussian/total_attr_{syn_aed_total_attr}_total_cat_{syn_aed_total_cat}'
        else:
            raise ValueError("Invalid dataset format for SynAED")

    # Default configurations for different dataset types
    default_configs = {
        'Waterbirds': [5001, 50],
        'CelebA': [10001, 50],
        'CelebAMultiAttr': [10001, 50],
        'MultiNLI': [10001, 50],
        'CivilComments': [10001, 50],
        'MetaShift': [5001, 50],
        'CompositionalNICOpp': [10001, 50],
        'NICOpp': [10001, 50],
        'Simple_2d': [100001, 50]
    }
    
    # Handle SynAED datasets dynamically
    if 'SynAED' in args['dataset']:
        hparams['num_step'], hparams['checkpoint_freq'] = [100001, 50]
    else:
        hparams['num_step'], hparams['checkpoint_freq'] = default_configs[args['dataset']]

    # Default num_m configurations
    default_num_m = {
        'Waterbirds': 2,
        'CelebA': 2,
        'CelebAMultiAttr': 16,
        'MultiNLI': 2,
        'CivilComments': 8,
        'MetaShift': 2,
        'CompositionalNICOpp': 6,
        'NICOpp': 6,
        'Simple_2d': 2
    }
    
    # Handle SynAED datasets dynamically: num_m = TOTAL_CAT^(TOTAL_ATTR-1)
    if 'SynAED' in args['dataset']:
        hparams['num_m'] = syn_aed_total_cat ** (syn_aed_total_attr - 1)
    else:
        hparams['num_m'] = default_num_m[args['dataset']]
    
    # Default num_y configurations
    default_num_y = {
        'Waterbirds': 2,
        'CelebA': 2,
        'CelebAMultiAttr': 2,
        'MultiNLI': 3,
        'CivilComments': 2,
        'MetaShift': 2,
        'CompositionalNICOpp': 60,
        'NICOpp': 60,
        'Simple_2d': 2    
        }
    
    # Handle SynAED datasets dynamically: num_y = TOTAL_CAT
    if 'SynAED' in args['dataset']:
        hparams['num_y'] = syn_aed_total_cat
    else:
        hparams['num_y'] = default_num_y[args['dataset']]

    #Classification network configuration
    if hparams['algorithm_name'] in ['CRM']:
        hparams['fc_type']= 'add_energy'
    elif hparams['algorithm_name'] in ['CRM_Multi_Attr']:
        hparams['fc_type']= 'add_energy_multi_attr'
    else:
        hparams['fc_type']= 'standard'

    rs = np.random.RandomState(args['hparams_comb'])
    if args['dataset'] in ["MultiNLI", "CivilComments"]:
        hparams['net_type'] = 'bert'
        hparams['balanced_batch'] = args['dataset'] == "CivilComments"
        hparams['lr'] = 10 ** rs.uniform(-6, -4)
        hparams['weight_decay'] = 10 ** rs.uniform(-6, -3)
        hparams['batch_size'] = int(2 ** rs.uniform(4, 6))
        hparams['last_layer_dropout'] = rs.choice([0., 0.1, 0.5])
    elif 'SynAED' in args['dataset']:
        hparams['net_type'] = 'linear'
        hparams['lr'] = 10 ** rs.uniform(-5, -3)
        hparams['weight_decay'] = 10 ** rs.uniform(-6, -3)
        hparams['batch_size'] = int(2 ** rs.uniform(3, 6))
    elif 'Simple_2d' == args['dataset']:
        hparams['net_type'] = 'linear'
        hparams['lr'] = 10 ** rs.uniform(-5, -3)
        hparams['weight_decay'] = 10 ** rs.uniform(-6, -3)
        hparams['batch_size'] = int(2 ** rs.uniform(3, 6))
    else:
        hparams['net_type'] = 'resnet'
        hparams['lr'] = 10 ** rs.uniform(-5, -3)
        hparams['weight_decay'] = 10 ** rs.uniform(-6, -3)
        hparams['batch_size'] = int(2 ** rs.uniform(5, 7))

    if args['algorithm'] == 'GroupDRO':
        hparams['eta'] = 10 ** rs.uniform(-3, -1)

    if args['algorithm'] in ['IRM', 'VREx', 'Fishr']:
        hparams['lambda'] = 10 ** rs.uniform(-1, 5)
        hparams['penalty_anneal_iters'] = int(10 ** rs.uniform(0, 4))

    if hparams['algorithm_name'] in ['LA_Group', 'LA_Cond']:
            hparams['temp'] = rs.choice([0.01, 0.1, 1.0])
            hparams['use_true_m'] = True
            if hparams['algorithm_name'] == 'LA_Group':
                hparams['adjustment_method'] = 'LC'
            elif hparams['algorithm_name'] == 'LA_Cond':
                hparams['adjustment_method'] = 'uLA'
    
    if args['algorithm'] in ['Mixup']:
        hparams['alpha']= 10 ** rs.uniform(0, 4)
 
    # for debug only
    if hparams['quick_run']:
        if 'SynAED' in args['dataset']:
            hparams['num_step'] = 50001
            hparams['lr'] = 0.01
            hparams['batch_size'] = 2048
            hparams['checkpoint_freq'] = 1000
        else:
            hparams['resume'] = False
            hparams['num_step'] = 5001
            hparams['num_workers'] = 0
            hparams['lr'] = 0.01
            hparams['weight_decay'] = 0
            hparams['batch_size'] = 2048
            hparams['last_layer_dropout'] = 0.1        
            hparams['eta'] = 0.04
            hparams['temp'] = 0.1
            hparams['alpha']= 10

    ext = f'hpcomb_{args["hparams_comb"]}_seed{args["seed"]}'
    hparams['ckpt_file'] = os.path.join(hparams['out_dir'], f'ckpt_{ext}.pt')
    hparams['results_file'] = os.path.join(hparams['out_dir'], f'results_{ext}.json')

    return hparams
