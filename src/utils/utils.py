import os
import glob
import re
import random
import numpy as np
import pandas as pd
import json
import torch
import pickle
import copy
from sklearn.metrics import f1_score

def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def prepare_out_dir(hparams):
    # writing the hparams into results.json file
    os.makedirs(hparams['out_dir'], exist_ok=True)
    if not hparams['resume'] or not os.path.exists(hparams['results_file']):
        with open(hparams['results_file'], 'w') as f:
            json.dump(hparams, f)
    print(hparams)


def read_results_file(file):
    results = {}
    with open(file, "r") as file:
        for line in file:
            res = json.loads(line)
            if 'step' in res.keys() and res['step'] > 0:
                for key, val in res.items():
                    results.setdefault(key, []).append(val)
    return results


def get_star_value_from_file_name(pattern, file_name):
    pattern= pattern.replace('[', '\[').replace(']', '\]')
    return int(re.match(pattern.replace('*', r'(\d+)'), file_name).group(1))


def load_checkpoint(hparams, algorithm, sel_metric='wga'):
    fname= hparams['results_file'][:-5] + '_sel_metric_' + str(sel_metric) +  '.pt'
    print(fname)
    if os.path.exists(fname):
        ckpt = torch.load(fname)
        algorithm.net.load_state_dict(ckpt['net_state_dict'])
        algorithm.optim.load_state_dict(ckpt['optim_state_dict'])
        if 'lr_scheduler' in ckpt.keys():
            algorithm.optim.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

        print('Checkpoint saved at step: ', ckpt['last_step'])
        return ckpt['last_step'] + 1
    raise AssertionError('Checkpoint not present for this configuration!')
    return 0


def save_checkpoint(hparams, algorithm, step, sel_metric= 'wga'):
    fname= hparams['results_file'][:-5] + '_sel_metric_' + str(sel_metric) +  '.pt'
    to_save = {'net_state_dict': algorithm.net.state_dict(),
               'optim_state_dict': algorithm.optim.state_dict(),
               'last_step': step}
    if algorithm.optim.lr_scheduler is not None:
        to_save['lr_scheduler'] = algorithm.optim.lr_scheduler.state_dict()
    torch.save(to_save, fname)

def report_stats(hparams, algorithm, loaders, step, num_y):
    stats = {'step': step}
    for split in ['va', 'te']:
        if hparams['algorithm_name'] in ['CRM', 'CRM_Multi_Attr']:                
            log_prior_prob= torch.zeros_like( algorithm.train_log_prior_prob )
            ys, y_hats, ms = algorithm.evaluate(loaders[split], log_prior_prob= log_prior_prob, extrapolate= False)                
        else:
            ys, y_hats, ms = algorithm.evaluate(loaders[split])
        metrics = get_metrics(hparams, ys, y_hats, ms, num_y= num_y)
        for metric, value in metrics.items():
            stats[f'{split}_{metric}'] = value

    with open(hparams['results_file'], 'a') as f:
        f.write('\n')
        json.dump(stats, f)

    #Printing
    stats_strs = []
    for key, val in stats.items():
        #Do not print per group accuracy
        if 'g_acc_' in key:
            continue
        if isinstance(val, float):
            stats_strs.append(f"{key}: {val:.3f}")
        else:
            stats_strs.append(f"{key}: {val}")
    print(', '.join(stats_strs))

    return stats

def get_metrics(hparams, ys, y_hats, ms, num_y, acc_per_group= False):
    gs = (hparams['num_m'] * ys + ms).view(-1)
    avg_acc = y_hats.argmax(1).eq(ys).float().mean().item()
    wga = min([y_hats.argmax(1).eq(ys)[gs == g_i].float().mean().item()
               for g_i in torch.unique(gs)])
    
    #Uniform accuracy across groups
    stats={}
    acc= y_hats.argmax(1).eq(ys) 
    for g_i in torch.unique(gs):
        indices= gs == g_i
        #Only compute these statistics for groups present during training
        if torch.sum(indices):
            # print(g_i, acc[indices].float().mean().item())
            stats['g_acc_' + str(g_i.item())]= acc[indices].float().mean().item()
    stats['unf_acc']= np.mean([*stats.values()])

    #Worst Case and Average Acc
    stats['wga'] = wga
    stats['avg_acc']= avg_acc

    if not acc_per_group:
        for key in list(stats.keys()):
            if 'g_acc_' in key:
                del stats[key]

    return stats


class Iter:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(self.loader)

    def next(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        return batch


def process_json_files(pattern, json_files, selection_criterion=None):
    all_values = {
        'hp_comb': [],
        'flip_rate': [],
        'va_wga': [],
        'te_wga': [],
        'va_unf_acc': [],
        'te_unf_acc': [],
        'va_avg_acc': [],
        'te_avg_acc': []
    }

    for file in json_files:
        results = read_results_file(file)
        if len(results) == 0:
            print('File empty..!')
            continue
        all_values['hp_comb'].append(
            get_star_value_from_file_name(pattern, file))

        ind = np.argmax(results[selection_criterion])
        all_values['va_wga'].append(results['va_wga'][ind])
        # all_values['te_wga'].append(results['te_wga'][ind])
        all_values['va_unf_acc'].append(results['va_unf_acc'][ind])
        # all_values['te_unf_acc'].append(results['te_unf_acc'][ind])
        all_values['va_avg_acc'].append(results['va_avg_acc'][ind])
        # all_values['te_avg_acc'].append(results['te_avg_acc'][ind])

    return all_values


def is_degenerate(algorithm, inferred_groups):
    ys = algorithm.tr_loader.y
    m_hats = inferred_groups['tr']
    g_hats = 2 * ys + m_hats
    # each class y should be grouped into 2 envs
    # if not, the flip_rate is reported -1
    if len(g_hats.unique()) != 2 * len(ys.unique()):##################
        algorithm.degenerate = True
        return True
    return False


def sort_and_remove_empty(all_values):
    inds = np.argsort(all_values['hp_comb'])
    to_remove = []

    for key in all_values.keys():
        if len(all_values[key]) == len(inds):
            all_values[key] = np.array(all_values[key])[inds]
        else:
            assert len(all_values[key]) == 0
            to_remove.append(key)  # remove empty

    for key in to_remove:
        all_values.pop(key)

    return all_values


def find_best_hparams_comb(dir_, selection_metric, seed):
    pattern = os.path.join(dir_, f'results_hpcomb_*_seed{seed}.json')
    json_files = glob.glob(pattern)
    assert len(json_files) > 0
    all_values = process_json_files(pattern, json_files, selection_metric)
    all_values = sort_and_remove_empty(all_values)
    best_hparams_comb = int(all_values['hp_comb'][
        np.argmax(all_values[selection_metric])])
    return best_hparams_comb
