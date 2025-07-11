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
        hparams['data_path'] = hparams['data_path'] + '/syn_total_cat_' +  args['dataset'].split('_')[-1]

    if 'SynGenAED' in args['dataset']:
        hparams['data_path'] = hparams['data_path'] + '/syn_general_total_cat_' +  args['dataset'].split('_')[-1]

    hparams['num_step'], hparams['checkpoint_freq'] = {
        'Waterbirds': [5001, 50],
        'CelebA': [10001, 50],
        'CelebAMultiAttr': [10001, 50],
        'MultiNLI': [10001, 50],
        'CivilComments': [10001, 50],
        'MetaShift': [5001, 50],
        'CompositionalNICOpp': [10001, 50],
        'NICOpp': [10001, 50],
        'Simple_2d': [100001, 50],
        'SynAED_2': [100001, 50],
        'SynAED_10': [100001, 50],
        'SynAED_20': [100001, 50],
        'SynAED_30': [100001, 50],
        'SynAED_40': [100001, 50],
        'SynAED_50': [100001, 50],
        'SynGenAED_2': [100001, 50],
        'SynGenAED_7': [100001, 50],
        'SynGenAED_8': [100001, 50],
        'SynGenAED_9': [100001, 50],
        'SynGenAED_10': [100001, 50],
        'SynGenAED_11': [100001, 50]}[args['dataset']]

    hparams['num_m'] = {
        'Waterbirds': 2,
        'CelebA': 2,
        'CelebAMultiAttr': 16,
        'MultiNLI': 2,
        'CivilComments': 8,
        'MetaShift': 2,
        'CompositionalNICOpp': 6,
        'NICOpp': 6,
        'ImagenetBG': 2,
        'Simple_2d': 2,
        'SynAED_2': 2,
        'SynAED_10': 10,
        'SynAED_20': 20,
        'SynAED_30': 30,
        'SynAED_40': 40,
        'SynAED_50': 50,
        'SynGenAED_2': 2,
        'SynGenAED_7': 64,
        'SynGenAED_8': 128,
        'SynGenAED_9': 256,
        'SynGenAED_10': 512,
        'SynGenAED_11': 1024}[args['dataset']]
    
    hparams['num_y'] = {
        'Waterbirds': 2,
        'CelebA': 2,
        'CelebAMultiAttr': 2,
        'MultiNLI': 3,
        'CivilComments': 2,
        'MetaShift': 2,
        'CompositionalNICOpp': 60,
        'NICOpp': 60,
        'ImagenetBG': 9,
        'Simple_2d': 2,
        'SynAED_2': 2,
        'SynAED_10': 10,
        'SynAED_20': 20,
        'SynAED_30': 30,
        'SynAED_40': 40,
        'SynAED_50': 50,
        'SynGenAED_2': 2,
        'SynGenAED_7': 2,
        'SynGenAED_8': 2,
        'SynGenAED_9': 2,
        'SynGenAED_10': 2,
        'SynGenAED_11': 2}[args['dataset']]

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
    elif 'SynAED' in args['dataset'] or 'SynGenAED' in args['dataset']:
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

        #Hparams for group complexity Experiments
        # hparams['num_step'] = 50001
        # hparams['lr'] = 0.01
        # hparams['batch_size'] = 2048
        # hparams['checkpoint_freq'] = 1000
        
        # #Simple 2D Experiments
        # hparams['num_step'] = 15001
        # hparams['lr'] = 0.0005
        # hparams['batch_size'] = 128
        # hparams['checkpoint_freq'] = 50

    ext = f'hpcomb_{args["hparams_comb"]}_seed{args["seed"]}'
    hparams['ckpt_file'] = os.path.join(hparams['out_dir'], f'ckpt_{ext}.pt')
    hparams['results_file'] = os.path.join(hparams['out_dir'], f'results_{ext}.json')

    return hparams
